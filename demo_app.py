"""
AI SpillGuard - Oil Spill Detection Demo App
===========================================

A simplified demo version that works without requiring a trained model.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI SpillGuard - Oil Spill Detection Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
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
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)

class OilSpillDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=1)
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load trained model, but work without it if not available
        model_path = Path("models/best_model.pth")
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_loaded = True
                st.success("✅ Trained model loaded successfully!")
            except Exception as e:
                self.model_loaded = False
                st.warning(f"⚠️ Could not load trained model: {e}")
                st.info("🔄 Using demo mode with simulated predictions")
        else:
            self.model_loaded = False
            st.info("🔄 No trained model found. Using demo mode with simulated predictions.")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize to model input size
        image_resized = cv2.resize(image, (256, 256))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """Make prediction on input image"""
        try:
            with torch.no_grad():
                image_tensor = self.preprocess_image(image)
                
                if self.model_loaded:
                    # Use actual trained model
                    output = self.model(image_tensor)
                    prediction = torch.sigmoid(output).cpu().numpy().squeeze()
                else:
                    # Generate simulated prediction for demo
                    prediction = self.generate_demo_prediction(image)
                
                return prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return self.generate_demo_prediction(image)
    
    def generate_demo_prediction(self, image):
        """Generate a demo prediction for visualization"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create a simple demo mask based on image characteristics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simulate oil spill detection based on dark areas
        _, binary = cv2.threshold(gray, 60, 1, cv2.THRESH_BINARY_INV)
        
        # Add some noise and smooth the mask
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.GaussianBlur(binary.astype(np.float32), (7,7), 0)
        
        # Resize to standard size
        prediction = cv2.resize(binary, (256, 256))
        
        return prediction

def create_overlay(original_image, mask, threshold=0.5):
    """Create overlay visualization"""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Resize mask to match original image
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    binary_mask = (mask_resized > threshold).astype(np.uint8)
    
    # Create red overlay for oil spills
    overlay = original_image.copy()
    overlay[binary_mask == 1] = [255, 0, 0]  # Red color
    
    # Blend with original
    result = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    
    return result, binary_mask

def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ AI SpillGuard - Oil Spill Detection</h1>', unsafe_allow_html=True)
    st.markdown("**Upload a satellite image to detect oil spills using AI-powered computer vision**")
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return OilSpillDetector()
    
    detector = load_detector()
    
    # Sidebar
    st.sidebar.header("🔧 Settings")
    confidence_threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5)
    show_probability_map = st.sidebar.checkbox("Show Probability Map", True)
    
    st.sidebar.header("📊 Model Info")
    st.sidebar.info(f"""
    **Device**: {detector.device}
    **Model Status**: {'✅ Trained' if detector.model_loaded else '🔄 Demo Mode'}
    **Input Size**: 256x256
    **Architecture**: U-Net
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Choose a satellite image...",
        type=['png', 'jpg', 'jpeg', 'tiff'],
        help="Upload a satellite image to analyze for oil spills"
    )
    
    # Demo images section
    st.subheader("🖼️ Or try with sample images:")
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    demo_option = None
    with col_demo1:
        if st.button("🌊 Ocean Sample"):
            demo_option = "ocean"
    with col_demo2:
        if st.button("🛢️ Oil Spill Sample"):
            demo_option = "oil_spill"
    with col_demo3:
        if st.button("🏝️ Coastal Sample"):
            demo_option = "coastal"
    
    # Generate demo image if requested
    if demo_option:
        uploaded_file = create_demo_image(demo_option)
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded Satellite Image", use_container_width=True)
        
        # Make prediction
        with st.spinner("🔍 Analyzing image for oil spills..."):
            prediction = detector.predict(image)
            overlay, binary_mask = create_overlay(image, prediction, confidence_threshold)
        
        with col2:
            st.subheader("🎯 Detection Results")
            st.image(overlay, caption="Oil Spill Detection (Red Areas)", use_container_width=True)
        
        # Calculate metrics
        oil_spill_percentage = (binary_mask.sum() / binary_mask.size) * 100
        confidence = prediction.max()
        affected_pixels = int(binary_mask.sum())
        
        # Display metrics
        st.subheader("📊 Analysis Results")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("🛢️ Oil Spill Coverage", f"{oil_spill_percentage:.2f}%")
        
        with col4:
            st.metric("🎯 Max Confidence", f"{confidence:.3f}")
        
        with col5:
            st.metric("📐 Affected Pixels", f"{affected_pixels:,}")
        
        with col6:
            severity = "Low" if oil_spill_percentage < 5 else "Medium" if oil_spill_percentage < 15 else "High" if oil_spill_percentage < 30 else "Critical"
            color = "normal" if severity == "Low" else "inverse"
            st.metric("⚠️ Severity Level", severity, delta=None)
        
        # Probability map
        if show_probability_map:
            st.subheader("🗺️ Oil Spill Probability Map")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(prediction, cmap='Reds', vmin=0, vmax=1)
            ax.set_title("Oil Spill Probability Heatmap")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
        
        # Alert system
        st.subheader("🚨 Alert Status")
        if oil_spill_percentage > 1.0:
            st.error(f"🚨 **ALERT**: Oil spill detected! {oil_spill_percentage:.2f}% of the image shows potential oil contamination.")
            st.markdown("**Recommended Actions:**")
            st.markdown("- 📞 Contact maritime authorities")
            st.markdown("- 🗺️ Record GPS coordinates")
            st.markdown("- 📊 Document extent and severity")
        elif oil_spill_percentage > 0.1:
            st.warning(f"⚠️ **CAUTION**: Possible oil contamination detected ({oil_spill_percentage:.2f}% coverage).")
        else:
            st.success("✅ **ALL CLEAR**: No significant oil spill detected in this image.")
        
        # Download results
        st.subheader("💾 Download Results")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Convert overlay to bytes for download
            overlay_pil = Image.fromarray(overlay)
            img_bytes = io.BytesIO()
            overlay_pil.save(img_bytes, format='PNG')
            
            st.download_button(
                label="📥 Download Detection Image",
                data=img_bytes.getvalue(),
                file_name=f"oil_spill_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        
        with col_dl2:
            # Create report
            report = f"""Oil Spill Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Detection Results:
- Oil Spill Coverage: {oil_spill_percentage:.2f}%
- Max Confidence: {confidence:.3f}
- Affected Pixels: {affected_pixels:,}
- Severity Level: {severity}

Alert Status: {'DETECTED' if oil_spill_percentage > 0.1 else 'CLEAR'}
"""
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"oil_spill_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def create_demo_image(demo_type):
    """Create demo images for testing"""
    # Create a simple demo image
    img = np.random.randint(0, 100, (300, 400, 3), dtype=np.uint8)
    
    if demo_type == "ocean":
        # Blue ocean image
        img[:, :, 0] = np.random.randint(10, 50, (300, 400))  # Low red
        img[:, :, 1] = np.random.randint(30, 80, (300, 400))  # Medium green
        img[:, :, 2] = np.random.randint(80, 150, (300, 400))  # High blue
    
    elif demo_type == "oil_spill":
        # Ocean with dark oil spill areas
        img[:, :, 0] = np.random.randint(10, 50, (300, 400))
        img[:, :, 1] = np.random.randint(30, 80, (300, 400))
        img[:, :, 2] = np.random.randint(80, 150, (300, 400))
        # Add dark oil spill patches
        img[100:200, 150:250] = np.random.randint(0, 30, (100, 100, 3))
        img[50:100, 300:350] = np.random.randint(0, 20, (50, 50, 3))
    
    elif demo_type == "coastal":
        # Mixed coastal image
        img[:, :, 0] = np.random.randint(50, 120, (300, 400))
        img[:, :, 1] = np.random.randint(60, 130, (300, 400))
        img[:, :, 2] = np.random.randint(40, 100, (300, 400))
    
    # Convert to PIL Image and then to BytesIO
    pil_img = Image.fromarray(img.astype(np.uint8))
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

if __name__ == "__main__":
    main()