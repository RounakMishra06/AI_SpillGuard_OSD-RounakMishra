"""
AI SpillGuard Demo Script
========================

This script demonstrates how to use the AI SpillGuard Streamlit application
for oil spill detection from satellite imagery.

Usage:
1. Run this script to see example usage
2. Or follow the instructions to use the web interface
"""

import os
import sys
from pathlib import Path

def demo_instructions():
    """Print demo instructions for using the Streamlit app"""
    
    print("🛰️ AI SpillGuard - Oil Spill Detection Demo")
    print("=" * 50)
    
    print("\n📋 Quick Start Guide:")
    print("1. Ensure your virtual environment is activated")
    print("2. Start the Streamlit app:")
    print("   ./run_app.sh")
    print("   OR")
    print("   streamlit run app.py")
    
    print("\n🌐 Access the Application:")
    print("   Open your browser and go to: http://localhost:8501")
    
    print("\n📤 Using the App:")
    print("   1. Upload a satellite image (JPG/PNG)")
    print("   2. Adjust detection threshold in the sidebar")
    print("   3. Configure alert settings")
    print("   4. View detection results and metrics")
    print("   5. Download results if needed")
    
    print("\n🖼️ Test Images Available:")
    test_images_path = Path("data/test/images")
    if test_images_path.exists():
        test_images = list(test_images_path.glob("*.jpg"))
        for i, img in enumerate(test_images[:5], 1):
            print(f"   {i}. {img.name}")
    else:
        print("   ⚠️ No test images found. Train your model first!")
    
    print("\n📊 Features:")
    print("   ✅ Real-time oil spill detection")
    print("   ✅ Interactive visualizations")
    print("   ✅ Metrics dashboard")
    print("   ✅ Alert system")
    print("   ✅ Download results")
    print("   ✅ Sample image testing")
    
    print("\n🔧 Troubleshooting:")
    print("   - If model not found: Run week3_model_development.ipynb first")
    print("   - If port busy: Use different port (--server.port=8502)")
    print("   - If slow processing: Model runs on CPU (normal for large images)")
    
    print("\n📝 Configuration Options:")
    print("   - Detection Threshold: 0.1-0.9 (default: 0.5)")
    print("   - Alert Threshold: 1-50% (default: 10%)")
    print("   - Enable/Disable Alerts: Toggle in sidebar")
    
    print("\n🎯 Expected Results:")
    print("   - Oil Coverage: Percentage of detected oil spill area")
    print("   - Affected Area: Estimated area in km²")
    print("   - Severity: Low/Medium/High/Critical")
    print("   - Visual: Red overlay on detected areas")
    
    print("\n" + "=" * 50)
    print("🚀 Ready to detect oil spills! Launch the app and start testing.")

def check_setup():
    """Check if the setup is complete"""
    
    print("\n🔍 Checking Setup...")
    
    # Check if model exists
    model_path = Path("models/best_model.pth")
    if model_path.exists():
        print("✅ Model found: models/best_model.pth")
    else:
        print("⚠️  Model not found: models/best_model.pth")
        print("   Run week3_model_development.ipynb to train the model first")
    
    # Check if test data exists
    test_path = Path("data/test/images")
    if test_path.exists() and list(test_path.glob("*.jpg")):
        print(f"✅ Test images found: {len(list(test_path.glob('*.jpg')))} images")
    else:
        print("⚠️  Test images not found")
    
    # Check if app.py exists
    app_path = Path("app.py")
    if app_path.exists():
        print("✅ Streamlit app found: app.py")
    else:
        print("❌ Streamlit app not found: app.py")
    
    # Check if requirements are met
    try:
        import streamlit
        print("✅ Streamlit installed")
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit")
    
    try:
        import plotly
        print("✅ Plotly installed")
    except ImportError:
        print("❌ Plotly not installed. Run: pip install plotly")
    
    print("\n" + "=" * 30)

def main():
    """Main demo function"""
    
    print("🎬 AI SpillGuard Demo Starting...")
    
    # Check setup
    check_setup()
    
    # Show instructions
    demo_instructions()
    
    # Ask if user wants to launch the app
    print("\n🚀 Launch Options:")
    print("1. Run './run_app.sh' to start the app")
    print("2. Run 'streamlit run app.py' manually")
    print("3. Visit http://localhost:8501 after launching")
    
    print("\n📱 Mobile Testing:")
    print("For mobile testing, use your computer's IP address:")
    print("http://YOUR_IP_ADDRESS:8501")
    
    print("\n🎯 Demo Complete! You're ready to detect oil spills.")

if __name__ == "__main__":
    main()
