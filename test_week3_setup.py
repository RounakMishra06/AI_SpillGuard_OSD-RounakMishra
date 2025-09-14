#!/usr/bin/env python3
"""
Quick test script to verify the Week 3 model development setup
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import cv2
    print("✅ All required packages imported successfully")
    print(f"🖥️ PyTorch version: {torch.__version__}")
    print(f"🖥️ Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Check dataset
    data_root = Path('data')
    for split in ['train', 'val', 'test']:
        for folder in ['images', 'masks']:
            path = data_root / split / folder
            if path.exists():
                count = len(list(path.glob('*.*')))
                print(f"📂 {split}/{folder}/: {count} files")
            else:
                print(f"❌ Missing: {split}/{folder}/")
    
    # Test basic U-Net forward pass
    print("\n🏗️ Testing U-Net architecture...")
    
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ConvBlock, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            return self.conv(x)

    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super(SimpleUNet, self).__init__()
            self.encoder1 = ConvBlock(in_channels, 64)
            self.encoder2 = ConvBlock(64, 128)
            self.decoder1 = ConvBlock(128, 64)
            self.final = nn.Conv2d(64, out_channels, kernel_size=1)
            self.pool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            d1 = self.decoder1(self.upsample(e2))
            return self.final(d1)
    
    # Test model
    model = SimpleUNet()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test data loading
    print("\n📁 Testing data loading...")
    train_images = Path('data/train/images')
    train_masks = Path('data/train/masks')
    
    if train_images.exists() and train_masks.exists():
        image_files = list(train_images.glob('*.jpg'))
        mask_files = list(train_masks.glob('*.png'))
        
        if len(image_files) > 0 and len(mask_files) > 0:
            # Load first sample
            img = cv2.imread(str(image_files[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            
            print(f"✅ Sample data loaded successfully!")
            print(f"   Image shape: {img.shape}")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Image dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
            print(f"   Mask dtype: {mask.dtype}, range: [{mask.min()}, {mask.max()}]")
        else:
            print("❌ No image or mask files found")
    else:
        print("❌ Data directories not found")
    
    print("\n🎉 Environment setup verification complete!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
