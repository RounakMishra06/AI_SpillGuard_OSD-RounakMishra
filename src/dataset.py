# dataset.py
import os
import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    raise

from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, to_binary=True, target_size=(256,256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Get image files (jpg) and find corresponding mask files (png)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
        self.transform = transform
        self.to_binary = to_binary
        self.target_size = target_size

    def _get_mask_path(self, img_name):
        """Convert image filename to corresponding mask filename"""
        # Change extension from .jpg to .png
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        return os.path.join(self.mask_dir, mask_name)

    def _rgb_to_binary(self, mask_rgb):
        # Many Kaggle masks are colored; treat any non-black pixel as spill.
        if len(mask_rgb.shape) == 3:
            gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = mask_rgb
        bin_mask = (gray > 0).astype(np.uint8)
        return bin_mask

    def __len__(self): 
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = self._get_mask_path(img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Convert BGR to RGB for consistency
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize (keep it simple for milestone 1)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.to_binary:
            mask = self._rgb_to_binary(mask)  # [H,W] 0/1
        else:
            # If later you do multi-class, convert RGB->class indices here.
            pass

        # Albumentations expects dict; we'll apply transforms later
        sample = {"image": img, "mask": mask}
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            sample["image"] = augmented["image"]
            sample["mask"] = augmented["mask"]

        return sample, img_name
