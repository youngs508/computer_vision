import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from edgeconnect_data import EdgeConnectFolderDataset
from edgeconnect_models import (
    EdgeGenerator, InpaintGenerator,
    PatchDiscriminator, VGG16FeatureExtractor,
    gan_loss_logits, feature_matching_loss
)

# Put these near the top of train_edgeconnect_gan.py
import os
from torchvision.utils import save_image

@torch.no_grad()
def build_fixed_viz_batch(dataset, device, indices, pin_size=None):
    """
    Load a fixed set of samples from dataset by index and stack into a batch.
    Returns tensors on `device`.
    """
    samples = [dataset[i] for i in indices]

    img   = torch.stack([s["image"]  for s in samples], dim=0).to(device)
    mask  = torch.stack([s["mask"]   for s in samples], dim=0).to(device)
    masked= torch.stack([s["masked"] for s in samples], dim=0).to(device)

    return {"image": img, "mask": mask, "masked": masked}

@torch.no_grad()
def save_joint_epoch_viz(out_dir, epoch, masked, mask, edge_pred, comp, img_gt, max_n=6):
    """
    Saves a grid image with rows of:
      [GT | mask | masked | edge_pred | comp]
    """
    os.makedirs(out_dir, exist_ok=True)
    b = min(masked.size(0), max_n)

    gt = img_gt[:b].clamp(0, 1)
    m  = mask[:b].repeat(1, 3, 1, 1).clamp(0, 1)
    mi = masked[:b].clamp(0, 1)
    e  = edge_pred[:b].repeat(1, 3, 1, 1).clamp(0, 1)
    cp = comp[:b].clamp(0, 1)

    row = torch.cat([gt, m, mi, e, cp], dim=3)  # concat along width
    save_path = os.path.join(out_dir, f"joint_epoch{epoch:03d}.png")
    save_image(row, save_path, nrow=1)
    print(f"[viz] saved: {save_path}")

def save_ckpt(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)

def linear_schedule(step, start_step, end_step, start_val, end_val):
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    t = (step - start_step) / max(1, (end_step - start_step))
    return start_val + t * (end_val - start_val)

@torch.no_grad()
def make_pred_edge(edgeG, masked, mask):
    logits = edgeG(masked, mask)
    pred = torch.sigmoid(logits)
    # binarize for conditioning (EdgeConnect typically uses binary edge map)
    pred = (pred > 0.5).float()
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/waymo_data/masks")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lrG", type=float, default=2e-4)
    ap.add_argument("--lrD", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stage", choices=["edge", "inpaint", "joint"], required=True)
    ap.add_argument("--ckpt_dir", default="./checkpoints_edgeconnect_gan")

    # loss weights (reasonable starting points)
    ap.add_argument("--w_gan", type=float, default=1.0)
    ap.add_argument("--w_fm", type=float, default=10.0)
    ap.add_argument("--w_bce", type=float, default=1.0)     # edge BCE
    ap.add_argument("--w_l1", type=float, default=1.0)      # inpaint L1
    ap.add_argument("--w_perc", type=float, default=0.1)    # perceptual

    # Edge mixing schedule for stage 2 / joint:
    # prob of using GT edges (teacher forcing) linearly decays.
    ap.add_argument("--gt_edge_start", type=float, default=1.0)
    ap.add_argument("--gt_edge_end", type=float, default=0.0)
    ap.add_argument("--gt_edge_decay_steps", type=int, default=20000)

    # optionally resume
    ap.add_argument("--resume_edgeG", default="")
    ap.add_argument("--resume_inpaintG", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EdgeConnectFolderDataset(args.root, size=args.size, use_masked=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    # ---- Fixed visualization samples (same every epoch) ----
    # Pick deterministic indices (change these if you want different samples)
    viz_indices = [0, 1, 2, 3, 4, 5]  # must be < len(ds)

    # If dataset is smaller, clamp
    viz_indices = [i for i in viz_indices if i < len(ds)]
    assert len(viz_indices) > 0, "No valid viz indices."

    fixed_viz = build_fixed_viz_batch(ds, device, viz_indices)
    # models
    edgeG = EdgeGenerator().to(device)
    inpaintG = InpaintGenerator().to(device)

    # discriminators:
    # D_edge sees (edge + mask) or (edge + masked image + mask) in some implementations.
    # We'll feed: [edge(1) + mask(1)] => 2 channels
    D_edge = PatchDiscriminator(in_ch=2).to(device)

    # D_img sees completed image + mask or image+masked+mask; we'll feed: [rgb(3)+mask(1)] => 4 channels
    D_img = PatchDiscriminator(in_ch=4).to(device)

    # VGG perceptual
    vgg = VGG16FeatureExtractor().to(device).eval()

    # optionally resume
    if args.resume_edgeG:
        ck = torch.load(args.resume_edgeG, map_location="cpu")
        edgeG.load_state_dict(ck["model"])
    if args.resume_inpaintG:
        ck = torch.load(args.resume_inpaintG, map_location="cpu")
        inpaintG.load_state_dict(ck["model"])

    # optimizers
    opt_edgeG = torch.optim.Adam(edgeG.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    opt_inpaintG = torch.optim.Adam(inpaintG.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    opt_D_edge = torch.optim.Adam(D_edge.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    opt_D_img = torch.optim.Adam(D_img.parameters(), lr=args.lrD, betas=(0.5, 0.999))

    bce_logits = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(args.epochs):
        edgeG.train()
        inpaintG.train()
        D_edge.train()
        D_img.train()

        pbar = tqdm(dl, desc=f"{args.stage} epoch {epoch+1}/{args.epochs}")

        # -----------------------------
        # Inner batch loop (training)
        # -----------------------------
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)     # [B,3,H,W]
            mask = batch["mask"].to(device, non_blocking=True)     # [B,1,H,W], 1=hole
            masked = batch["masked"].to(device, non_blocking=True) # [B,3,H,W]
            edge_gt = batch["edge"].to(device, non_blocking=True)  # [B,1,H,W]

            if args.stage == "edge":
                # ---- Train D_edge ----
                with torch.no_grad():
                    edge_fake_logits = edgeG(masked, mask)
                    edge_fake = (torch.sigmoid(edge_fake_logits) > 0.5).float()

                real_in = torch.cat([edge_gt, mask], dim=1)
                fake_in = torch.cat([edge_fake, mask], dim=1)

                opt_D_edge.zero_grad(set_to_none=True)
                real_logits, real_feats = D_edge(real_in)
                fake_logits, fake_feats = D_edge(fake_in)
                d_loss = gan_loss_logits(real_logits, True) + gan_loss_logits(fake_logits, False)
                d_loss.backward()
                opt_D_edge.step()

                # ---- Train EdgeG ----
                opt_edgeG.zero_grad(set_to_none=True)
                edge_pred_logits = edgeG(masked, mask)
                bce_loss = bce_logits(edge_pred_logits, edge_gt)

                edge_pred = (torch.sigmoid(edge_pred_logits) > 0.5).float()
                fake_in = torch.cat([edge_pred, mask], dim=1)
                fake_logits, fake_feats = D_edge(fake_in)

                g_gan = gan_loss_logits(fake_logits, True)
                with torch.no_grad():
                    real_logits, real_feats = D_edge(real_in)
                g_fm = feature_matching_loss(fake_feats, real_feats)

                g_loss = args.w_bce * bce_loss + args.w_gan * g_gan + args.w_fm * g_fm
                g_loss.backward()
                opt_edgeG.step()

                global_step += 1
                pbar.set_postfix(d=float(d_loss.item()), g=float(g_loss.item()), bce=float(bce_loss.item()))

            elif args.stage == "inpaint":
                # GT edge teacher forcing schedule
                gt_prob = linear_schedule(
                    global_step,
                    start_step=0,
                    end_step=args.gt_edge_decay_steps,
                    start_val=args.gt_edge_start,
                    end_val=args.gt_edge_end
                )

                with torch.no_grad():
                    edge_pred = make_pred_edge(edgeG.eval(), masked, mask)
                edgeG.train()

                use_gt = (torch.rand(img.size(0), device=device) < gt_prob).float().view(-1,1,1,1)
                edge_cond = use_gt * edge_gt + (1.0 - use_gt) * edge_pred

                # ---- Train D_img ----
                with torch.no_grad():
                    out_logits = inpaintG(masked, edge_cond, mask)
                    out = torch.sigmoid(out_logits)
                    comp = out * mask + img * (1.0 - mask)

                real_in = torch.cat([img, mask], dim=1)
                fake_in = torch.cat([comp, mask], dim=1)

                opt_D_img.zero_grad(set_to_none=True)
                real_logits, real_feats = D_img(real_in)
                fake_logits, fake_feats = D_img(fake_in)
                d_loss = gan_loss_logits(real_logits, True) + gan_loss_logits(fake_logits, False)
                d_loss.backward()
                opt_D_img.step()

                # ---- Train InpaintG ----
                opt_inpaintG.zero_grad(set_to_none=True)

                out_logits = inpaintG(masked, edge_cond, mask)
                out = torch.sigmoid(out_logits)
                comp = out * mask + img * (1.0 - mask)

                l1_loss = F.l1_loss(out * mask, img * mask)

                vgg_fake = vgg(comp)
                vgg_real = vgg(img)
                perc_loss = sum(F.l1_loss(a, b.detach()) for a, b in zip(vgg_fake, vgg_real))

                fake_in = torch.cat([comp, mask], dim=1)
                fake_logits, fake_feats = D_img(fake_in)
                g_gan = gan_loss_logits(fake_logits, True)
                with torch.no_grad():
                    real_logits, real_feats = D_img(real_in)
                g_fm = feature_matching_loss(fake_feats, real_feats)

                g_loss = args.w_l1 * l1_loss + args.w_perc * perc_loss + args.w_gan * g_gan + args.w_fm * g_fm
                g_loss.backward()
                opt_inpaintG.step()

                global_step += 1
                pbar.set_postfix(d=float(d_loss.item()), g=float(g_loss.item()),
                                l1=float(l1_loss.item()), perc=float(perc_loss.item()),
                                gt_prob=float(gt_prob))

            else:  # args.stage == "joint"
                # ---- forward edge ----
                edge_pred = make_pred_edge(edgeG, masked, mask)  # binary

                # ---- Train D_edge ----
                real_edge_in = torch.cat([edge_gt, mask], dim=1)
                fake_edge_in = torch.cat([edge_pred.detach(), mask], dim=1)

                opt_D_edge.zero_grad(set_to_none=True)
                real_logits, _ = D_edge(real_edge_in)
                fake_logits, _ = D_edge(fake_edge_in)
                d_edge_loss = gan_loss_logits(real_logits, True) + gan_loss_logits(fake_logits, False)
                d_edge_loss.backward()
                opt_D_edge.step()

                # ---- Train D_img ----
                with torch.no_grad():
                    out_logits = inpaintG(masked, edge_pred, mask)
                    out = torch.sigmoid(out_logits)
                    comp = out * mask + img * (1.0 - mask)

                real_img_in = torch.cat([img, mask], dim=1)
                fake_img_in = torch.cat([comp, mask], dim=1)

                opt_D_img.zero_grad(set_to_none=True)
                real_logits, _ = D_img(real_img_in)
                fake_logits, _ = D_img(fake_img_in)
                d_img_loss = gan_loss_logits(real_logits, True) + gan_loss_logits(fake_logits, False)
                d_img_loss.backward()
                opt_D_img.step()

                # ---- Train both generators ----
                opt_edgeG.zero_grad(set_to_none=True)
                opt_inpaintG.zero_grad(set_to_none=True)

                # EdgeG losses
                edge_logits = edgeG(masked, mask)
                bce_loss = bce_logits(edge_logits, edge_gt)
                edge_bin = (torch.sigmoid(edge_logits) > 0.5).float()

                fake_edge_in = torch.cat([edge_bin, mask], dim=1)
                fake_logits, fake_feats = D_edge(fake_edge_in)
                g_edge_gan = gan_loss_logits(fake_logits, True)
                with torch.no_grad():
                    real_logits, real_feats = D_edge(real_edge_in)
                g_edge_fm = feature_matching_loss(fake_feats, real_feats)

                # InpaintG losses (condition on predicted edge_bin)
                out_logits = inpaintG(masked, edge_bin, mask)
                out = torch.sigmoid(out_logits)
                comp = out * mask + img * (1.0 - mask)

                l1_loss = F.l1_loss(out * mask, img * mask)

                vgg_fake = vgg(comp)
                vgg_real = vgg(img)
                perc_loss = sum(F.l1_loss(a, b.detach()) for a, b in zip(vgg_fake, vgg_real))

                fake_img_in = torch.cat([comp, mask], dim=1)
                fake_logits, fake_feats = D_img(fake_img_in)
                g_img_gan = gan_loss_logits(fake_logits, True)
                with torch.no_grad():
                    real_logits, real_feats = D_img(real_img_in)
                g_img_fm = feature_matching_loss(fake_feats, real_feats)

                g_loss = (
                    args.w_bce * bce_loss +
                    args.w_gan * g_edge_gan + args.w_fm * g_edge_fm +
                    args.w_l1 * l1_loss + args.w_perc * perc_loss +
                    args.w_gan * g_img_gan + args.w_fm * g_img_fm
                )
                g_loss.backward()
                opt_edgeG.step()
                opt_inpaintG.step()

                global_step += 1
                pbar.set_postfix(d_edge=float(d_edge_loss.item()),
                                d_img=float(d_img_loss.item()),
                                g=float(g_loss.item()),
                                l1=float(l1_loss.item()),
                                perc=float(perc_loss.item()))

        # -----------------------------
        # End-of-epoch visualization (ONCE PER EPOCH)
        # -----------------------------
        if args.stage == "joint":
            edgeG.eval()
            inpaintG.eval()

            img0 = fixed_viz["image"]
            mask0 = fixed_viz["mask"]
            masked0 = fixed_viz["masked"]

            edge_pred0 = make_pred_edge(edgeG, masked0, mask0)
            out_logits0 = inpaintG(masked0, edge_pred0, mask0)
            out0 = torch.sigmoid(out_logits0)
            comp0 = out0 * mask0 + img0 * (1.0 - mask0)

            save_joint_epoch_viz(
                out_dir=os.path.join(args.ckpt_dir, "viz_joint"),
                epoch=epoch + 1,
                masked=masked0.detach().cpu(),
                mask=mask0.detach().cpu(),
                edge_pred=edge_pred0.detach().cpu(),
                comp=comp0.detach().cpu(),
                img_gt=img0.detach().cpu(),
                max_n=len(viz_indices)
            )

            edgeG.train()
            inpaintG.train()

        # -----------------------------
        # End-of-epoch checkpoint saving
        # -----------------------------
        save_ckpt(os.path.join(args.ckpt_dir, f"{args.stage}_edgeG_epoch{epoch+1}.pt"),
                {"model": edgeG.state_dict(), "epoch": epoch+1, "step": global_step})
        save_ckpt(os.path.join(args.ckpt_dir, f"{args.stage}_inpaintG_epoch{epoch+1}.pt"),
                {"model": inpaintG.state_dict(), "epoch": epoch+1, "step": global_step})
        save_ckpt(os.path.join(args.ckpt_dir, f"{args.stage}_Dedge_epoch{epoch+1}.pt"),
                {"model": D_edge.state_dict(), "epoch": epoch+1, "step": global_step})
        save_ckpt(os.path.join(args.ckpt_dir, f"{args.stage}_Dimg_epoch{epoch+1}.pt"),
                {"model": D_img.state_dict(), "epoch": epoch+1, "step": global_step})

        print(f"Saved epoch {epoch+1} checkpoints + (if joint) viz to {args.ckpt_dir}")

if __name__ == "__main__":
    main()
