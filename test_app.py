import streamlit as st
import numpy as np
from PIL import Image
import torch

# Test app to verify Streamlit is working
st.title("🛰️ AI SpillGuard - Oil Spill Detection")
st.write("Welcome to the AI SpillGuard oil spill detection system!")

# Test basic functionality
st.header("System Status")
st.success("✅ Streamlit is working!")
st.info(f"🔧 PyTorch version: {torch.__version__}")
st.info(f"🖥️ Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Simple file uploader test
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")
    
    # Create a simple demo prediction
    st.header("Demo Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Oil Spill Coverage", "2.3%")
        st.metric("Confidence", "0.847")
    
    with col2:
        st.metric("Severity", "Low")
        st.metric("Status", "⚠️ Detected")

# Demo buttons
st.header("Demo Features")
if st.button("🌊 Test Ocean Analysis"):
    st.balloons()
    st.success("Ocean analysis complete!")

if st.button("🔍 Run Detection"):
    progress = st.progress(0)
    for i in range(100):
        progress.progress(i + 1)
    st.success("Detection completed!")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5)
st.sidebar.info(f"Current threshold: {threshold}")