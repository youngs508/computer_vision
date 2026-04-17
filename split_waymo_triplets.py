import os
import shutil
import random
import argparse
from glob import glob

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def mkdirs(base_out: str):
    for split in ["train", "val"]:
        for sub in ["images", "masks", "masked"]:
            os.makedirs(os.path.join(base_out, split, sub), exist_ok=True)

def find_images(img_dir: str):
    files = []
    for ext in IMG_EXTS:
        files += glob(os.path.join(img_dir, f"*{ext}"))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Source root containing images/, masks/, masked/")
    ap.add_argument("--dst", required=True,
                    help="Destination root to create train/ and val/ under")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copy (default: copy)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print actions without copying/moving")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)

    img_dir = os.path.join(src, "images")
    mask_dir = os.path.join(src, "masks")
    masked_dir = os.path.join(src, "masked")

    assert os.path.isdir(img_dir), f"Missing folder: {img_dir}"
    assert os.path.isdir(mask_dir), f"Missing folder: {mask_dir}"
    assert os.path.isdir(masked_dir), f"Missing folder: {masked_dir}"

    mkdirs(dst)

    # Index masks/masked by stem
    masks = {stem(p): p for p in glob(os.path.join(mask_dir, "*.png"))}
    maskeds = {stem(p): p for p in glob(os.path.join(masked_dir, "*.png"))}

    # Collect triplets
    images = find_images(img_dir)
    triplets = []
    missing = []

    for ip in images:
        s = stem(ip)
        mp = masks.get(s)
        mdp = maskeds.get(s)
        if mp and mdp:
            triplets.append((s, ip, mp, mdp))
        else:
            missing.append((s, bool(mp), bool(mdp)))

    print(f"Found images: {len(images)}")
    print(f"Matched triplets: {len(triplets)}")
    print(f"Missing pairs: {len(missing)}")
    if missing:
        print("Examples of missing (stem, has_mask, has_masked):")
        for x in missing[:10]:
            print("  ", x)

    assert len(triplets) > 0, "No matched triplets found. Check filenames/extensions."

    # Split
    random.seed(args.seed)
    random.shuffle(triplets)
    n_val = int(round(len(triplets) * args.val_ratio))
    val_set = triplets[:n_val]
    train_set = triplets[n_val:]

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  (val_ratio={args.val_ratio})")

    # Copy/move
    op = shutil.move if args.move else shutil.copy2

    def do_one(split, s, ip, mp, mdp):
        out_img = os.path.join(dst, split, "images", os.path.basename(ip))
        out_m   = os.path.join(dst, split, "masks",  os.path.basename(mp))
        out_md  = os.path.join(dst, split, "masked", os.path.basename(mdp))

        if args.dry_run:
            print(f"[{split}] {ip} -> {out_img}")
            print(f"[{split}] {mp} -> {out_m}")
            print(f"[{split}] {mdp} -> {out_md}")
        else:
            op(ip, out_img)
            op(mp, out_m)
            op(mdp, out_md)

    for (s, ip, mp, mdp) in train_set:
        do_one("train", s, ip, mp, mdp)
    for (s, ip, mp, mdp) in val_set:
        do_one("val", s, ip, mp, mdp)

    print("Done.")
    print("Output structure:")
    print(f"  {dst}/train/images  {dst}/train/masks  {dst}/train/masked")
    print(f"  {dst}/val/images    {dst}/val/masks    {dst}/val/masked")

if __name__ == "__main__":
    main()