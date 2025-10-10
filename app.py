"""
AI SpillGuard - Oil Spill Detection Streamlit App
=================================================

A web-based interface for real-time oil spill detection using satellite imagery.
Upload satellite images and get instant segmentation results with oil spill detection.

Features:
- Real-time image upload and processing
- U-Net model inference
- Interactive visualization with overlays
- Alert system for detected oil spills
- Download results functionality
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
import base64
from datetime import datetime
import json

# Set page config
st.set_page_config(
    page_title="AI SpillGuard - Oil Spill Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# U-Net Model Definition (same as in your notebook)
class ConvBlock(nn.Module):
    """Double Convolution Block for U-Net"""
    
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
    """U-Net architecture for oil spill segmentation"""
    
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

# Utility functions
@st.cache_resource
def load_model():
    """Load the trained U-Net model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=3, out_channels=1)
        
        # Try to load the best model
        model_path = Path("models/best_model.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"✅ Model loaded successfully from {model_path}")
            return model.to(device), device
        else:
            st.warning("⚠️ No trained model found. Using untrained model for demo.")
            return model.to(device), device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess uploaded image for model inference"""
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Resize image
    image_resized = cv2.resize(image_np, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_resized

def calculate_metrics(pred_mask, threshold=0.5):
    """Calculate detection metrics"""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    
    # Calculate basic statistics
    total_pixels = pred_mask.size
    oil_pixels = np.sum(pred_binary)
    oil_percentage = (oil_pixels / total_pixels) * 100
    
    # Calculate affected area (assuming each pixel represents 1 square meter for demo)
    affected_area_sqm = oil_pixels  # This would be scaled based on actual image resolution
    affected_area_sqkm = affected_area_sqm / 1_000_000
    
    return {
        'oil_percentage': oil_percentage,
        'affected_area_sqm': affected_area_sqm,
        'affected_area_sqkm': affected_area_sqkm,
        'severity': get_severity_level(oil_percentage)
    }

def get_severity_level(oil_percentage):
    """Determine severity level based on oil spill percentage"""
    if oil_percentage < 5:
        return "Low"
    elif oil_percentage < 15:
        return "Medium"
    elif oil_percentage < 30:
        return "High"
    else:
        return "Critical"

def create_overlay_image(original, mask, alpha=0.6):
    """Create overlay of original image with oil spill mask"""
    overlay = original.copy()
    
    # Create red overlay for oil spill areas
    red_overlay = np.zeros_like(original)
    red_overlay[:, :, 0] = 255  # Red channel
    
    # Apply mask
    mask_3d = np.stack([mask, mask, mask], axis=2)
    overlay = np.where(mask_3d > 0.5, 
                      alpha * red_overlay + (1 - alpha) * original,
                      original)
    
    return overlay.astype(np.uint8)

def save_results(original_image, pred_mask, metrics, filename_prefix="oil_spill_detection"):
    """Save detection results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save images
    cv2.imwrite(f"results/{filename_prefix}_{timestamp}_original.jpg", 
                cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"results/{filename_prefix}_{timestamp}_mask.png", 
                (pred_mask * 255).astype(np.uint8))
    
    # Save metrics
    with open(f"results/{filename_prefix}_{timestamp}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return f"{filename_prefix}_{timestamp}"

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ AI SpillGuard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Real-time Oil Spill Detection from Satellite Imagery</h2>', unsafe_allow_html=True)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Cannot proceed without a model. Please check your model file.")
        return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Detection threshold
    threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1,
        help="Threshold for binary segmentation. Lower values detect more areas as oil spills."
    )
    
    # Alert settings
    st.sidebar.subheader("🚨 Alert Settings")
    alert_threshold = st.sidebar.slider(
        "Alert Threshold (%)", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Percentage of oil spill coverage to trigger alerts"
    )
    
    enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)
    
    # File upload
    st.subheader("📤 Upload Satellite Image")
    uploaded_file = st.file_uploader(
        "Choose a satellite image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a satellite image for oil spill detection"
    )
    
    # Check for sample image in session state
    sample_image_path = st.session_state.get('uploaded_file', None)
    
    if uploaded_file is not None or sample_image_path is not None:
        # Display original image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = "Uploaded Image"
        else:
            image = Image.open(sample_image_path)
            image_source = f"Sample Image: {Path(sample_image_path).name}"
            # Clear the session state after using the sample
            if 'uploaded_file' in st.session_state:
                del st.session_state.uploaded_file
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption=image_source, use_container_width=True)
        
        # Preprocess and run inference
        with st.spinner("🔄 Processing image..."):
            image_tensor, image_resized = preprocess_image(image)
            
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Run inference
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, threshold)
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Create binary mask for visualization
            pred_binary = (pred_mask > threshold).astype(np.float32)
            
            # Display predicted mask
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(pred_binary, cmap='Reds', alpha=0.8)
            ax.set_title("Oil Spill Detection Mask")
            ax.axis('off')
            st.pyplot(fig)
        
        # Display metrics
        st.subheader("📊 Detection Metrics")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Oil Coverage</h3>'
                f'<h2>{metrics["oil_percentage"]:.2f}%</h2>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        with metric_cols[1]:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Affected Area</h3>'
                f'<h2>{metrics["affected_area_sqkm"]:.3f} km²</h2>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        with metric_cols[2]:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Severity Level</h3>'
                f'<h2>{metrics["severity"]}</h2>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        with metric_cols[3]:
            st.markdown(
                f'<div class="metric-card">'
                f'<h3>Detection Status</h3>'
                f'<h2>{"🚨 Detected" if metrics["oil_percentage"] > 1 else "✅ Clear"}</h2>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        # Alert system
        if enable_alerts and metrics["oil_percentage"] > alert_threshold:
            st.markdown(
                f'<div class="alert-box alert-danger">'
                f'<h3>🚨 ALERT: Oil Spill Detected!</h3>'
                f'<p><strong>Severity:</strong> {metrics["severity"]}</p>'
                f'<p><strong>Coverage:</strong> {metrics["oil_percentage"]:.2f}% of image area</p>'
                f'<p><strong>Affected Area:</strong> {metrics["affected_area_sqkm"]:.3f} km²</p>'
                f'<p>Immediate attention required for environmental protection.</p>'
                f'</div>', 
                unsafe_allow_html=True
            )
        elif metrics["oil_percentage"] > 1:
            st.markdown(
                f'<div class="alert-box alert-info">'
                f'<h3>ℹ️ Oil Spill Detected</h3>'
                f'<p>Small oil spill detected. Monitor for changes.</p>'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-box alert-success">'
                f'<h3>✅ No Oil Spill Detected</h3>'
                f'<p>The analyzed area appears to be clear of oil spills.</p>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        # Overlay visualization
        st.subheader("🎨 Overlay Visualization")
        overlay_image = create_overlay_image(image_resized, pred_binary)
        
        col3, col4 = st.columns(2)
        with col3:
            st.image(image_resized, caption="Original", use_container_width=True)
        with col4:
            st.image(overlay_image, caption="Oil Spill Overlay (Red Areas)", use_container_width=True)
        
        # Interactive plot with Plotly
        st.subheader("📈 Interactive Analysis")
        
        # Create heatmap
        fig_heatmap = px.imshow(
            pred_mask, 
            color_continuous_scale='Reds',
            title="Oil Spill Probability Heatmap"
        )
        fig_heatmap.update_layout(coloraxis_colorbar=dict(title="Probability"))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Save results option
        st.subheader("💾 Save Results")
        if st.button("Save Detection Results"):
            filename = save_results(image_resized, pred_binary, metrics)
            st.success(f"✅ Results saved as: {filename}")
        
        # Download options
        col5, col6 = st.columns(2)
        
        with col5:
            # Create downloadable mask
            mask_pil = Image.fromarray((pred_binary * 255).astype(np.uint8))
            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format='PNG')
            mask_bytes.seek(0)
            
            st.download_button(
                label="📥 Download Mask",
                data=mask_bytes.getvalue(),
                file_name="oil_spill_mask.png",
                mime="image/png"
            )
        
        with col6:
            # Create downloadable overlay
            overlay_pil = Image.fromarray(overlay_image)
            overlay_bytes = io.BytesIO()
            overlay_pil.save(overlay_bytes, format='PNG')
            overlay_bytes.seek(0)
            
            st.download_button(
                label="📥 Download Overlay",
                data=overlay_bytes.getvalue(),
                file_name="oil_spill_overlay.png",
                mime="image/png"
            )
    
    # Sample images section
    st.subheader("🖼️ Try Sample Images")
    st.write("Don't have satellite images? Try these sample images from the test dataset:")
    
    sample_cols = st.columns(3)
    
    # Try to load sample images from test dataset
    test_images_path = Path("data/test/images")
    if test_images_path.exists():
        sample_images = list(test_images_path.glob("*.jpg"))[:3]
        
        for i, (col, img_path) in enumerate(zip(sample_cols, sample_images)):
            with col:
                sample_img = Image.open(img_path)
                st.image(sample_img, caption=f"Sample {i+1}", use_container_width=True)
                
                # Use sample image for detection
                if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                    # Store the selected sample image in session state
                    st.session_state.uploaded_file = img_path
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🛰️ AI SpillGuard - Powered by Deep Learning | Built with Streamlit</p>
        <p>For environmental monitoring and marine protection</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
