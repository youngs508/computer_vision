import os
import argparse
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import save_image

from edgeconnect_data import EdgeConnectFolderDataset
from edgeconnect_models import EdgeGenerator, InpaintGenerator

@torch.no_grad()
def make_pred_edge(edgeG, masked, mask, thr=0.5):
    logits = edgeG(masked, mask)
    prob = torch.sigmoid(logits)
    edge = (prob > thr).float()
    return edge

def tensor_to_uint8_img(x):
    # x: [3,H,W] float in [0,1]
    x = x.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # HWC
    return x

def tensor_to_uint8_gray(x):
    # x: [1,H,W] float in [0,1]
    x = x.detach().cpu().clamp(0, 1).numpy()[0]
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def compute_psnr_ssim(pred_rgb, gt_rgb):
    # pred_rgb, gt_rgb: uint8 HWC
    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)
    ssim = structural_similarity(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
    return psnr, ssim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/waymo_data/masks")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--edge_ckpt", required=True, help="path to EdgeG checkpoint .pt")
    ap.add_argument("--inpaint_ckpt", required=True, help="path to InpaintG checkpoint .pt")

    ap.add_argument("--save_dir", default="./eval_outputs")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--max_save", type=int, default=50)

    ap.add_argument("--edge_thr", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (uses your images/masks/masked; if masked missing, set use_masked=False in dataset)
    ds = EdgeConnectFolderDataset(args.root, size=args.size, use_masked=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load models
    edgeG = EdgeGenerator().to(device).eval()
    inpaintG = InpaintGenerator().to(device).eval()

    ck = torch.load(args.edge_ckpt, map_location="cpu")
    edgeG.load_state_dict(ck["model"] if "model" in ck else ck)

    ck = torch.load(args.inpaint_ckpt, map_location="cpu")
    inpaintG.load_state_dict(ck["model"] if "model" in ck else ck)

    os.makedirs(args.save_dir, exist_ok=True)

    psnr_all, ssim_all = [], []
    psnr_hole, ssim_hole = [], []

    saved = 0

    for batch in dl:
        img = batch["image"].to(device, non_blocking=True)     # [B,3,H,W] in [0,1]
        mask = batch["mask"].to(device, non_blocking=True)     # [B,1,H,W] in {0,1}
        masked = batch["masked"].to(device, non_blocking=True) # [B,3,H,W]
        names = batch["name"]

        # Predict edge then inpaint
        edge_pred = make_pred_edge(edgeG, masked, mask, thr=args.edge_thr)  # [B,1,H,W]
        out_logits = inpaintG(masked, edge_pred, mask)                      # [B,3,H,W] logits
        out = torch.sigmoid(out_logits)                                     # [0,1]

        # Composite: fill only holes, keep known pixels from GT (or from masked input)
        comp = out * mask + img * (1.0 - mask)

        # Metrics (overall + hole-only)
        for i in range(img.size(0)):
            gt_u8 = tensor_to_uint8_img(img[i])
            comp_u8 = tensor_to_uint8_img(comp[i])

            p, s = compute_psnr_ssim(comp_u8, gt_u8)
            psnr_all.append(p); ssim_all.append(s)

            # Hole-only metrics: evaluate only on mask==1 region
            m = mask[i].detach().cpu().numpy()[0]  # float 0/1
            m_bool = m > 0.5

            # If mask has no hole pixels, skip hole metrics
            if m_bool.sum() > 0:
                # compute PSNR hole-only by MSE on hole pixels
                gt_f = gt_u8.astype(np.float32)
                pr_f = comp_u8.astype(np.float32)
                diff = (gt_f - pr_f)
                mse = np.mean((diff[m_bool] ** 2))
                if mse <= 1e-12:
                    psnr_h = 99.0
                else:
                    psnr_h = 10.0 * np.log10((255.0 ** 2) / mse)

                # SSIM hole-only is not well-defined directly (SSIM needs full image),
                # so we compute SSIM on the full image but report PSNR_hole separately.
                psnr_hole.append(psnr_h)

            # Save images
            if args.save_images and saved < args.max_save:
                name = names[i] if isinstance(names[i], str) else str(names[i])
                # Save a strip: GT | mask | masked | edge | comp
                gt = img[i:i+1].detach().cpu()
                mk = mask[i:i+1].detach().cpu().repeat(1, 3, 1, 1)
                ms = masked[i:i+1].detach().cpu()
                ed = edge_pred[i:i+1].detach().cpu().repeat(1, 3, 1, 1)
                cp = comp[i:i+1].detach().cpu()

                strip = torch.cat([gt, mk, ms, ed, cp], dim=3)
                save_image(strip, os.path.join(args.save_dir, f"{name}_viz.png"), nrow=1)
                save_image(cp, os.path.join(args.save_dir, f"{name}_comp.png"))
                saved += 1

    # Summaries
    psnr_all = np.array(psnr_all); ssim_all = np.array(ssim_all)
    psnr_hole = np.array(psnr_hole) if len(psnr_hole) > 0 else None

    print("=== Evaluation Results ===")
    print(f"Count: {len(psnr_all)}")
    print(f"PSNR (all pixels): mean={psnr_all.mean():.3f}  std={psnr_all.std():.3f}")
    print(f"SSIM (all pixels): mean={ssim_all.mean():.4f} std={ssim_all.std():.4f}")
    if psnr_hole is not None and len(psnr_hole) > 0:
        print(f"PSNR (hole only):  mean={psnr_hole.mean():.3f}  std={psnr_hole.std():.3f}")
    else:
        print("PSNR (hole only):  (no hole pixels found in masks?)")

if __name__ == "__main__":
    main()