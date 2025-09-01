# transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_preproc_transforms():
    return A.Compose([
        # Optional SAR-specific: slight blur to reduce speckle (light touch)
        A.GaussianBlur(blur_limit=(3,5), p=0.15),
        A.Resize(256, 256),  # safe even if already resized
        A.Normalize(),       # mean/std per ImageNet; fine for milestone 1
        ToTensorV2()
    ])

def get_train_aug_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=0.3),
        # Keep masks crisp:
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])
