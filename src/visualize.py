# visualize.py
import os, cv2, numpy as np, matplotlib.pyplot as plt
from src.dataset import OilSpillDataset
from src.transforms import get_preproc_transforms, get_train_aug_transforms

def show_panel(root_split="data/train"):
    img_dir = os.path.join(root_split, "images")
    mask_dir = os.path.join(root_split, "masks")

    # 1) Load raw (no transform) to display original and binary
    raw_ds = OilSpillDataset(img_dir, mask_dir, transform=None, to_binary=True, target_size=(256,256))
    (sample, name) = raw_ds[0]
    
    # Load original files with correct extensions
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    if not img_files:
        print("No image files found!")
        return
    
    first_img = img_files[0]
    img_path = os.path.join(img_dir, first_img)
    mask_name = first_img.rsplit('.', 1)[0] + '.png'
    mask_path = os.path.join(mask_dir, mask_name)
    
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask_raw_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    
    if img_raw is None or mask_raw_rgb is None:
        print(f"Failed to load files: {img_path}, {mask_path}")
        return
        
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    mask_raw_rgb = cv2.cvtColor(mask_raw_rgb, cv2.COLOR_BGR2RGB)

    # Binary (after our conversion)
    bin_mask = sample["mask"]  # 0/1 already sized to 256x256

    # 2) Preprocessed
    preproc_ds = OilSpillDataset(img_dir, mask_dir, transform=get_preproc_transforms(), to_binary=True)
    preproc_sample, _ = preproc_ds[0]
    img_pre = preproc_sample["image"].permute(1,2,0).cpu().numpy()
    mask_pre = preproc_sample["mask"].cpu().numpy()

    # 3) Augmented example
    aug_ds = OilSpillDataset(img_dir, mask_dir, transform=get_train_aug_transforms(), to_binary=True)
    aug_sample, _ = aug_ds[0]
    img_aug = aug_sample["image"].permute(1,2,0).cpu().numpy()
    mask_aug = aug_sample["mask"].cpu().numpy()

    # Plot & SAVE for submission
    plt.figure(figsize=(12,10))

    plt.subplot(2,3,1); plt.title("Original image"); plt.imshow(img_raw); plt.axis('off')
    plt.subplot(2,3,2); plt.title("Original mask (RGB)"); plt.imshow(mask_raw_rgb); plt.axis('off')
    plt.subplot(2,3,3); plt.title("Binary mask (0/1)"); plt.imshow(bin_mask, cmap='gray'); plt.axis('off')

    plt.subplot(2,3,4); plt.title("Preprocessed image"); plt.imshow(np.clip(img_pre,0,1)); plt.axis('off')
    plt.subplot(2,3,5); plt.title("Preprocessed mask"); plt.imshow(mask_pre, cmap='gray'); plt.axis('off')
    plt.subplot(2,3,6); plt.title("Augmented image + mask")
    plt.imshow(np.clip(img_aug,0,1)); plt.imshow(mask_aug, cmap='jet', alpha=0.4); plt.axis('off')

    os.makedirs("outputs/milestone1_panels", exist_ok=True)
    out = "outputs/milestone1_panels/panel_example.png"
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
