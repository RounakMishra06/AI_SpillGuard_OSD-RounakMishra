"""
Quick Training Script for AI SpillGuard Model
===========================================

This script will train the U-Net model quickly with basic transforms.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Set paths
data_root = Path('data')

# U-Net Model (same as in notebook)
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

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)

# Dataset Class
class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        self.mask_paths = []
        
        for img_path in self.image_paths:
            mask_filename = img_path.stem + ".png"
            mask_path = self.mask_dir / mask_filename
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                self.image_paths.remove(img_path)
                
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in number of images and masks"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.float32)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        predictions = torch.sigmoid(predictions)
        
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss

# Metrics
def calculate_iou(pred_mask, gt_mask, threshold=0.5, smooth=1e-5):
    pred_binary = (pred_mask > threshold).float()
    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_dice(pred_mask, gt_mask, threshold=0.5, smooth=1e-5):
    pred_binary = (pred_mask > threshold).float()
    intersection = (pred_binary * gt_mask).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_binary.sum() + gt_mask.sum() + smooth
    )
    return dice.item()

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_ious = []
        val_dices = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                pred_probs = torch.sigmoid(outputs)
                for i in range(images.size(0)):
                    iou = calculate_iou(pred_probs[i], masks[i])
                    dice = calculate_dice(pred_probs[i], masks[i])
                    val_ious.append(iou)
                    val_dices.append(dice)
        
        val_loss /= len(val_loader)
        val_iou = np.mean(val_ious)
        val_dice = np.mean(val_dices)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            print(f"🎉 New best model! Dice: {best_dice:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_dice,
                'val_iou': val_iou,
            }, 'models/best_model.pth')
    
    return model, history

# Main training
def main():
    print("🚀 Starting Oil Spill Detection Model Training...")
    
    # Hyperparameters
    BATCH_SIZE = 4  # Smaller batch size for stability
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 3  # Quick training
    
    # Create datasets
    print("📁 Loading datasets...")
    train_dataset = OilSpillDataset(data_root / 'train' / 'images', data_root / 'train' / 'masks')
    val_dataset = OilSpillDataset(data_root / 'val' / 'images', data_root / 'val' / 'masks')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Create model
    print("🧠 Creating U-Net model...")
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    
    print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("🏃‍♂️ Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    print("✅ Training complete!")
    print(f"🏆 Best validation Dice: {max(history['val_dice']):.4f}")
    print(f"💾 Model saved to: models/best_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📈 Training curves saved to: models/training_curves.png")

if __name__ == "__main__":
    main()
