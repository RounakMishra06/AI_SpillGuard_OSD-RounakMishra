#!/usr/bin/env python3
"""
Simple Week 1 Test - Oil Spill Dataset Preprocessing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def main():
    print(" Week 1 Oil Spill Preprocessing Test")
    print("=" * 60)
    
    # Check dataset structure
    data_root = Path('data')
    print("1. Dataset Structure:")
    for split in ['train', 'val', 'test']:
        img_dir = data_root / split / 'images'
        mask_dir = data_root / split / 'masks'
        
        img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
        mask_count = len(list(mask_dir.glob('*.png'))) if mask_dir.exists() else 0
        
        print(f"   {split}: {img_count} images, {mask_count} masks")
    
    # Load sample image and mask
    print("\n2. Loading Sample Image-Mask Pair:")
    train_images = data_root / 'train' / 'images'
    train_masks = data_root / 'train' / 'masks'
    
    image_files = list(train_images.glob('*.jpg'))
    if not image_files:
        print(" No image files found!")
        return
    
    sample_name = image_files[0].name
    print(f"   Sample: {sample_name}")
    
    # Load image (jpg) and corresponding mask (png)
    img_path = train_images / sample_name
    mask_name = sample_name.rsplit('.', 1)[0] + '.png'
    mask_path = train_masks / mask_name
    
    print(f"   Image path: {img_path}")
    print(f"   Mask path: {mask_path}")
    
    if not img_path.exists():
        print(" Image file not found!")
        return
    if not mask_path.exists():
        print(" Mask file not found!")
        return
    
    # Load files
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path))
    
    if image is None or mask is None:
        print(" Failed to load files!")
        return
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    print(f"    Loaded successfully!")
    print(f"   Image shape: {image_rgb.shape}")
    print(f"   Mask shape: {mask_rgb.shape}")
    
    # Convert mask to binary
    print("\n3. Binary Conversion:")
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    binary_mask = (gray > 0).astype(np.uint8)
    oil_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    coverage = (oil_pixels / total_pixels) * 100
    
    print(f"   Oil spill coverage: {coverage:.2f}%")
    print(f"   Unique values: {np.unique(binary_mask)}")
    
    # Preprocessing
    print("\n4. Preprocessing:")
    target_size = (256, 256)
    
    # Resize
    img_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Normalize image
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    print(f"   Resized to: {img_resized.shape}")
    print(f"   Image range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    
    # Create visualization
    print("\n5. Creating Visualization:")
    
    try:
        import albumentations as A
        
        # Simple augmentation
        augment = A.Compose([
            A.HorizontalFlip(p=1.0),  # Force flip for demo
            A.RandomBrightnessContrast(p=1.0),
            A.Resize(256, 256),
        ])
        
        # Apply augmentation
        augmented = augment(image=img_normalized, mask=mask_resized)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        print("   Augmentation applied")
        
    except ImportError:
        print("    Albumentations not available, using simple flip")
        aug_img = np.fliplr(img_normalized)
        aug_mask = np.fliplr(mask_resized)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Original data
    plt.subplot(3, 3, 1)
    plt.imshow(image_rgb)
    plt.title('1. Original Satellite Image', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(mask_rgb)
    plt.title('2. Original Mask (RGB)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('3. Binary Mask (0/1)', fontweight='bold')
    plt.axis('off')
    
    # Row 2: Preprocessed data
    plt.subplot(3, 3, 4)
    plt.imshow(img_normalized)
    plt.title('4. Preprocessed Image\\n(256×256, Normalized)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(mask_resized, cmap='gray')
    plt.title('5. Preprocessed Mask', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(img_normalized)
    plt.imshow(mask_resized, cmap='Reds', alpha=0.4)
    plt.title('6. Preprocessed Overlay', fontweight='bold')
    plt.axis('off')
    
    # Row 3: Augmented data
    plt.subplot(3, 3, 7)
    plt.imshow(aug_img)
    plt.title('7. Augmented Image', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(aug_mask, cmap='gray')
    plt.title('8. Augmented Mask', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(aug_img)
    plt.imshow(aug_mask, cmap='jet', alpha=0.4)
    plt.title('9. Final Augmented Overlay', fontweight='bold')
    plt.axis('off')
    
    # Add title
    fig.suptitle(f'Week 1: Complete Oil Spill Preprocessing Pipeline\\n' +
                 f'Sample: {sample_name} | Oil Spill Coverage: {coverage:.2f}%',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    os.makedirs('outputs/week1_final', exist_ok=True)
    output_path = 'outputs/week1_final/preprocessing_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"   Saved to: {output_path}")
    
    # Show if in interactive environment
    try:
        plt.show()
    except:
        pass
    
    print("\n" + "=" * 60)
    print("Week 1 Preprocessing Pipeline Complete!")
    print("All tasks implemented successfully:")
    print("   1. Dataset structure verified")
    print("   2. Image-mask loading working")
    print("   3. RGB to binary conversion implemented")
    print("   4. Resize and preprocessing applied")
    print("   5. Data augmentation demonstrated")
    print("   6. Comprehensive visualization created")
    print(f"\nSubmit this image: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
