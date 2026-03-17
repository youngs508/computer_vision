import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgeconnect_data import EdgeConnectFolderDataset

# ---------- Simple UNet blocks ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = ConvBlock(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m  = self.mid(self.pool3(e3))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# ---------- Training helpers ----------
def save_ckpt(path, model, opt, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/waymo_data/masks")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stage", choices=["edge", "inpaint"], required=True)
    ap.add_argument("--ckpt_dir", default="./checkpoints_edgeconnect_baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EdgeConnectFolderDataset(args.root, size=args.size, use_masked=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.stage == "edge":
        # input: masked(3) + mask(1)  -> predict edge(1)
        model = UNet(in_ch=4, out_ch=1).to(device)
        # BCEWithLogits for edge maps
        crit = nn.BCEWithLogitsLoss()
    else:
        # input: masked(3) + edge(1) + mask(1) -> predict image(3)
        model = UNet(in_ch=5, out_ch=3).to(device)
        crit = nn.L1Loss()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"{args.stage} epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)    # [B,3,H,W]
            msk = batch["mask"].to(device, non_blocking=True)     # [B,1,H,W]
            mimg = batch["masked"].to(device, non_blocking=True)  # [B,3,H,W]
            edge = batch["edge"].to(device, non_blocking=True)    # [B,1,H,W]

            opt.zero_grad(set_to_none=True)

            if args.stage == "edge":
                x = torch.cat([mimg, msk], dim=1)     # [B,4,H,W]
                pred = model(x)                       # logits [B,1,H,W]
                # supervise edges, optionally focus more on missing region:
                # loss = crit(pred, edge) + crit(pred * msk, edge * msk)
                loss = crit(pred, edge)
            else:
                # NOTE: during real EdgeConnect stage2 you’d use predicted edges;
                # baseline uses GT edges for stability. You can switch later.
                x = torch.cat([mimg, edge, msk], dim=1)  # [B,5,H,W]
                pred = torch.sigmoid(model(x))           # [0,1]
                loss = crit(pred, img)

            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), step=global_step)

        ckpt_path = os.path.join(args.ckpt_dir, f"{args.stage}_epoch{epoch+1}.pt")
        save_ckpt(ckpt_path, model, opt, global_step)
        print(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()
