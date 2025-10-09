"""
Enhanced Training Script for AI SpillGuard Models
===============================================

Advanced training pipeline with data augmentation, model comparison,
and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import our models
from enhanced_models import (
    MultiScaleUNet, 
    OilSpillSeverityNet, 
    EfficientOilSpillNet
)
from unet import UNet  # Original model
from dataset import OilSpillDataset
from losses import DiceLoss, FocalLoss
from metrics import iou_score, dice_score

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """Enhanced trainer with model comparison and advanced metrics"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_data_loaders()
        
        # Results storage
        self.results = {}
        
    def _initialize_models(self):
        """Initialize all model architectures for comparison"""
        models = {
            'unet': UNet(3, 1),
            'multiscale_unet': MultiScaleUNet(3, 1),
            'efficient_net': EfficientOilSpillNet(3, 1),
            'severity_net': OilSpillSeverityNet(3, 4)
        }
        
        # Move models to device
        for name, model in models.items():
            model = model.to(self.device)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"{name}: {total_params:,} parameters")
            
        return models
    
    def _setup_data_loaders(self):
        """Setup enhanced data loaders with augmentation"""
        
        # Enhanced augmentation pipeline
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = OilSpillDataset(
            image_dir=self.config['train_image_dir'],
            mask_dir=self.config['train_mask_dir'],
            transform=train_transform
        )
        
        val_dataset = OilSpillDataset(
            image_dir=self.config['val_image_dir'],
            mask_dir=self.config['val_mask_dir'],
            transform=val_transform
        )
        
        test_dataset = OilSpillDataset(
            image_dir=self.config['test_image_dir'],
            mask_dir=self.config['test_mask_dir'],
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _get_loss_function(self, model_name):
        """Get appropriate loss function for each model"""
        if model_name == 'severity_net':
            # Multi-task loss for severity network
            return {
                'segmentation': DiceLoss(),
                'severity': nn.CrossEntropyLoss(),
                'size': nn.MSELoss()
            }
        else:
            # Standard segmentation loss
            dice_loss = DiceLoss()
            focal_loss = FocalLoss()
            
            def combined_loss(pred, target):
                return 0.7 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target)
            
            return combined_loss
    
    def train_model(self, model_name, epochs=None):
        """Train a specific model"""
        if epochs is None:
            epochs = self.config['epochs']
            
        logger.info(f"Training {model_name}...")
        
        model = self.models[model_name]
        loss_fn = self._get_loss_function(model_name)
        
        # Setup optimizer with different learning rates for different models
        if model_name == 'efficient_net':
            lr = self.config['learning_rate'] * 2  # Higher LR for efficient model
        else:
            lr = self.config['learning_rate']
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
            for batch_idx, (images, masks) in enumerate(train_pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                
                if model_name == 'severity_net':
                    # Multi-task training
                    outputs = model(images)
                    
                    seg_loss = loss_fn['segmentation'](outputs['segmentation'], masks)
                    # For severity, we need ground truth labels (simplified here)
                    severity_labels = self._generate_severity_labels(masks)
                    severity_loss = loss_fn['severity'](outputs['severity_logits'], severity_labels)
                    size_labels = self._calculate_spill_size(masks)
                    size_loss = loss_fn['size'](outputs['estimated_size'].squeeze(), size_labels)
                    
                    loss = seg_loss + 0.3 * severity_loss + 0.1 * size_loss
                else:
                    # Standard segmentation training
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_ious = []
            val_dices = []
            
            with torch.no_grad():
                for images, masks in tqdm(self.val_loader, desc="Validation"):
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    if model_name == 'severity_net':
                        outputs = model(images)
                        predictions = outputs['segmentation']
                        
                        seg_loss = loss_fn['segmentation'](predictions, masks)
                        severity_labels = self._generate_severity_labels(masks)
                        severity_loss = loss_fn['severity'](outputs['severity_logits'], severity_labels)
                        size_labels = self._calculate_spill_size(masks)
                        size_loss = loss_fn['size'](outputs['estimated_size'].squeeze(), size_labels)
                        
                        loss = seg_loss + 0.3 * severity_loss + 0.1 * size_loss
                    else:
                        predictions = model(images)
                        loss = loss_fn(predictions, masks)
                    
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    batch_iou = iou_score(predictions, masks)
                    batch_dice = dice_score(predictions, masks)
                    
                    val_ious.extend(batch_iou)
                    val_dices.extend(batch_dice)
            
            # Update history
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            avg_val_iou = np.mean(val_ious)
            avg_val_dice = np.mean(val_dices)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_iou'].append(avg_val_iou)
            history['val_dice'].append(avg_val_dice)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'val_iou': avg_val_iou,
                    'val_dice': avg_val_dice
                }, self.output_dir / 'models' / f'best_{model_name}.pth')
            else:
                patience_counter += 1
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val IoU: {avg_val_iou:.4f}, "
                f"Val Dice: {avg_val_dice:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model and history
        torch.save(model.state_dict(), 
                  self.output_dir / 'models' / f'final_{model_name}.pth')
        
        with open(self.output_dir / 'logs' / f'{model_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.results[model_name] = history
        
        return history
    
    def _generate_severity_labels(self, masks):
        """Generate severity labels based on spill area (simplified)"""
        spill_areas = torch.sum(masks.view(masks.size(0), -1), dim=1)
        total_pixels = masks.size(-1) * masks.size(-2)
        spill_ratios = spill_areas / total_pixels
        
        # Simple severity classification based on coverage
        severity = torch.zeros(masks.size(0), dtype=torch.long, device=masks.device)
        severity[spill_ratios > 0.1] = 1  # Low
        severity[spill_ratios > 0.2] = 2  # Medium  
        severity[spill_ratios > 0.4] = 3  # High
        
        return severity
    
    def _calculate_spill_size(self, masks):
        """Calculate estimated spill size in km² (simplified)"""
        spill_areas = torch.sum(masks.view(masks.size(0), -1), dim=1).float()
        # Convert pixel area to km² (this would depend on image resolution and coverage)
        # Simplified conversion factor
        conversion_factor = 0.001  # Adjust based on actual image scale
        return spill_areas * conversion_factor
    
    def evaluate_model(self, model_name):
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        model = self.models[model_name]
        model.eval()
        
        # Load best model weights
        checkpoint = torch.load(self.output_dir / 'models' / f'best_{model_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = {
            'ious': [],
            'dices': [],
            'accuracies': [],
            'precisions': [],
            'recalls': []
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc=f"Evaluating {model_name}"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                if model_name == 'severity_net':
                    outputs = model(images)
                    predictions = outputs['segmentation']
                else:
                    predictions = model(images)
                
                # Convert to binary predictions
                binary_preds = (predictions > 0.5).float()
                
                # Calculate metrics
                batch_iou = iou_score(binary_preds, masks)
                batch_dice = dice_score(binary_preds, masks)
                
                test_metrics['ious'].extend(batch_iou)
                test_metrics['dices'].extend(batch_dice)
                
                # Store for confusion matrix
                all_predictions.extend(binary_preds.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        # Calculate average metrics
        avg_metrics = {
            'mean_iou': np.mean(test_metrics['ious']),
            'std_iou': np.std(test_metrics['ious']),
            'mean_dice': np.mean(test_metrics['dices']),
            'std_dice': np.std(test_metrics['dices'])
        }
        
        logger.info(f"{model_name} Test Results:")
        logger.info(f"  IoU: {avg_metrics['mean_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
        logger.info(f"  Dice: {avg_metrics['mean_dice']:.4f} ± {avg_metrics['std_dice']:.4f}")
        
        return avg_metrics, test_metrics
    
    def compare_models(self):
        """Compare all trained models"""
        logger.info("Comparing all models...")
        
        comparison_results = {}
        
        for model_name in self.models.keys():
            if model_name in self.results:
                avg_metrics, _ = self.evaluate_model(model_name)
                comparison_results[model_name] = avg_metrics
        
        # Create comparison plots
        self._plot_model_comparison(comparison_results)
        
        # Save comparison results
        with open(self.output_dir / 'logs' / 'model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        return comparison_results
    
    def _plot_model_comparison(self, results):
        """Create comparison plots"""
        models = list(results.keys())
        ious = [results[model]['mean_iou'] for model in models]
        dices = [results[model]['mean_dice'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IoU comparison
        bars1 = ax1.bar(models, ious, color=['blue', 'green', 'red', 'orange'][:len(models)])
        ax1.set_title('Model Comparison - IoU Score')
        ax1.set_ylabel('IoU Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, iou in zip(bars1, ious):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{iou:.3f}', ha='center', va='bottom')
        
        # Dice comparison
        bars2 = ax2.bar(models, dices, color=['blue', 'green', 'red', 'orange'][:len(models)])
        ax2.set_title('Model Comparison - Dice Score')
        ax2.set_ylabel('Dice Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, dice in zip(bars2, dices):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{dice:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_all_models(self):
        """Train all models and compare results"""
        logger.info("Starting comprehensive model training...")
        
        for model_name in self.models.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                history = self.train_model(model_name)
                logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Compare all trained models
        comparison_results = self.compare_models()
        
        # Find best model
        best_model = max(comparison_results.keys(), 
                        key=lambda x: comparison_results[x]['mean_iou'])
        
        logger.info(f"\nBest performing model: {best_model}")
        logger.info(f"Best IoU: {comparison_results[best_model]['mean_iou']:.4f}")
        
        return comparison_results

def main():
    """Main training pipeline"""
    
    # Configuration
    config = {
        'train_image_dir': 'data/train/images',
        'train_mask_dir': 'data/train/masks',
        'val_image_dir': 'data/val/images',
        'val_mask_dir': 'data/val/masks',
        'test_image_dir': 'data/test/images',
        'test_mask_dir': 'data/test/masks',
        'output_dir': 'outputs/enhanced_training',
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 1e-4,
        'num_workers': 4,
        'patience': 10
    }
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\nTraining completed!")
    print("Check outputs/enhanced_training/ for results, models, and plots.")

if __name__ == "__main__":
    main()