import os
from glob import glob
import cv2
import numpy as np
from PIL import Image 
import torch
from torch.utils.data import Dataset

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # HWC -> CHW, uint8 -> float32 [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t

class EdgeConnectFolderDataset(Dataset):
    """
    Expects:
      root/images/*.png
      root/masks/*.png
      root/masked/*.png (optional; can be ignored if you want to re-mask on the fly)
    """
    def __init__(self, root, size=256, canny_low=100, canny_high=200, use_masked=True):
        self.root = os.path.expanduser(root)
        self.size = size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.use_masked = use_masked

        self.img_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")
        self.masked_dir = os.path.join(self.root, "masked")

        self.images = sorted(glob(os.path.join(self.img_dir, "*")))
        assert len(self.images) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.images)

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    def _load_mask(self, path):
        # Load as single channel, resize with NEAREST so mask stays binary-ish
        m = Image.open(path).convert("L")
        m = m.resize((self.size, self.size), Image.NEAREST)
        return m

    def _mask_to_tensor(self, m: Image.Image):
        # Convert 0..255 -> {0,1}
        mt = pil_to_tensor(m)  # [1,H,W] in [0,1]
        mt = (mt > 0.5).float()
        return mt

    def _compute_edge(self, rgb: Image.Image):
        # Canny expects uint8 grayscale
        arr = np.array(rgb)  # HWC RGB uint8
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, self.canny_low, self.canny_high)  # 0..255
        edge = Image.fromarray(edge).convert("L")
        edge_t = pil_to_tensor(edge)  # [1,H,W] in [0,1]
        # binarize edges
        edge_t = (edge_t > 0.1).float()
        return edge_t

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)                 # ... .jpg
        stem, _ = os.path.splitext(img_name)                  # remove extension
        png_name = stem + ".png"                              # ... .png

        mask_path = os.path.join(self.mask_dir, png_name)
        assert os.path.exists(mask_path), f"Missing mask for {img_name}: {mask_path}"

        img = self._load_rgb(img_path)
        mask = self._load_mask(mask_path)

        img_t = pil_to_tensor(img)
        mask_t = self._mask_to_tensor(mask)
        edge_t = self._compute_edge(img)

        if self.use_masked:
            masked_path = os.path.join(self.masked_dir, png_name)
            assert os.path.exists(masked_path), f"Missing masked for {img_name}: {masked_path}"
            masked = self._load_rgb(masked_path)
            masked_t = pil_to_tensor(masked)
        else:
            masked_t = img_t * (1.0 - mask_t)

        return {
            "image": img_t,
            "mask": mask_t,
            "masked": masked_t,
            "edge": edge_t,
            "name": stem
        }
