"""
Enhanced U-Net Models for AI SpillGuard
======================================

Advanced model architectures for improved oil spill detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """Double Convolution Block"""
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

class AttentionBlock(nn.Module):
    """Attention mechanism for better feature focus"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class MultiScaleUNet(nn.Module):
    """Enhanced U-Net with multi-scale processing"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(MultiScaleUNet, self).__init__()
        
        # Multi-scale input branches
        self.scale_branch_1 = self._make_scale_branch(in_channels, features[0])  # Full resolution
        self.scale_branch_2 = self._make_scale_branch(in_channels, features[0])  # Half resolution
        self.scale_branch_3 = self._make_scale_branch(in_channels, features[0])  # Quarter resolution
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(features[0] * 3, features[0], kernel_size=1)
        
        # U-Net encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        in_ch = features[0]
        for feature in features:
            self.downs.append(ConvBlock(in_ch, feature))
            in_ch = feature
        
        # Bottleneck with attention
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.bottleneck_attention = AttentionBlock(features[-1] * 2)
        
        # U-Net decoder
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _make_scale_branch(self, in_channels, out_channels):
        """Create scale-specific feature extraction branch"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Multi-scale processing
        scale_1 = self.scale_branch_1(x)  # Full resolution
        
        # Half resolution processing
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        scale_2 = self.scale_branch_2(x_half)
        scale_2 = F.interpolate(scale_2, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Quarter resolution processing
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        scale_3 = self.scale_branch_3(x_quarter)
        scale_3 = F.interpolate(scale_3, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Fuse multi-scale features
        fused = torch.cat([scale_1, scale_2, scale_3], dim=1)
        x = self.fusion_conv(fused)
        
        # Standard U-Net processing
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.bottleneck_attention(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))

class OilSpillSeverityNet(nn.Module):
    """Oil spill detection with severity classification"""
    def __init__(self, in_channels=3, num_severity_classes=4):
        super(OilSpillSeverityNet, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = MultiScaleUNet(in_channels, 64)
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # Severity classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.severity_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_severity_classes)  # No spill, Low, Medium, High, Critical
        )
        
        # Size estimation head
        self.size_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Estimated area in km²
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor.feature_extractor(x)  # Get intermediate features
        
        # Segmentation prediction
        segmentation = torch.sigmoid(self.segmentation_head(features))
        
        # Global features for classification
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)
        
        # Severity classification
        severity_logits = self.severity_classifier(global_features)
        
        # Size estimation
        estimated_size = self.size_estimator(global_features)
        
        return {
            'segmentation': segmentation,
            'severity_logits': severity_logits,
            'estimated_size': estimated_size
        }

class EfficientOilSpillNet(nn.Module):
    """Lightweight model for real-time processing"""
    def __init__(self, in_channels=3, out_channels=1):
        super(EfficientOilSpillNet, self).__init__()
        
        # Efficient encoder with depthwise separable convolutions
        self.encoder = nn.ModuleList([
            self._depthwise_separable_conv(in_channels, 32, stride=1),
            self._depthwise_separable_conv(32, 64, stride=2),
            self._depthwise_separable_conv(64, 128, stride=2),
            self._depthwise_separable_conv(128, 256, stride=2),
        ])
        
        # Efficient decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(256, 64, 2, stride=2),  # 256 = 128 + 128 (skip connection)
            nn.ConvTranspose2d(128, 32, 2, stride=2),  # 128 = 64 + 64
        ])
        
        # Final prediction
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 64 = 32 + 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )
        
    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """Depthwise separable convolution for efficiency"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        # Decoder with skip connections
        skip_connections = skip_connections[:-1][::-1]  # Reverse and exclude last
        
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(skip_connections):
                # Concatenate with corresponding skip connection
                skip = skip_connections[i]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
        
        return self.final_conv(x)

# Utility functions for enhanced models
def get_model_summary(model, input_size=(1, 3, 256, 256)):
    """Get model parameter count and FLOPs estimate"""
    from torchsummary import summary
    
    print(f"Model: {model.__class__.__name__}")
    summary(model, input_size[1:])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def compare_models():
    """Compare different model architectures"""
    models = {
        'Standard U-Net': UNet(3, 1),
        'Multi-Scale U-Net': MultiScaleUNet(3, 1),
        'Efficient Oil Spill Net': EfficientOilSpillNet(3, 1),
        'Severity Net': OilSpillSeverityNet(3, 4)
    }
    
    print("Model Comparison:")
    print("-" * 50)
    
    for name, model in models.items():
        total_params, _ = get_model_summary(model)
        print(f"{name}: {total_params:,} parameters")
    
    return models

if __name__ == "__main__":
    # Example usage
    print("AI SpillGuard Enhanced Models")
    print("=" * 40)
    
    # Create enhanced model
    model = MultiScaleUNet(in_channels=3, out_channels=1)
    
    # Test with dummy input
    x = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Model comparison
    compare_models()