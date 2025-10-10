import gradio as gr
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def detect_oil_spill(image):
    """
    Simplified oil spill detection demo function
    """
    if image is None:
        return None, None, "Please upload an image first."
    
    # Convert PIL to numpy
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Create a realistic demo mask
    demo_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create some realistic "oil spill" regions
    center_y, center_x = height // 2, width // 2
    
    # Create multiple spill regions
    spill_regions = [
        (center_y - 50, center_x - 80, 60, 40),  # Main spill
        (center_y + 30, center_x + 20, 30, 25),  # Secondary spill
        (center_y - 20, center_x + 60, 20, 15),  # Small spill
    ]
    
    total_spill_pixels = 0
    for y, x, h, w in spill_regions:
        if y >= 0 and x >= 0 and y + h < height and x + w < width:
            # Create elliptical spill shape
            for dy in range(h):
                for dx in range(w):
                    if ((dy - h//2)**2 / (h//2)**2 + (dx - w//2)**2 / (w//2)**2) <= 1:
                        demo_mask[y + dy, x + dx] = 255
                        total_spill_pixels += 1
    
    # Create overlay visualization
    overlay = img_array.copy()
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = demo_mask
    
    # Blend original image with red overlay
    alpha = 0.4
    overlay = (1 - alpha) * overlay + alpha * red_overlay
    overlay = overlay.astype(np.uint8)
    
    # Calculate statistics
    total_pixels = height * width
    spill_percentage = (total_spill_pixels / total_pixels) * 100
    
    # Determine severity
    if spill_percentage < 1:
        severity = "Low"
        risk_level = "🟢 Minimal Environmental Impact"
    elif spill_percentage < 5:
        severity = "Moderate" 
        risk_level = "🟡 Moderate Environmental Concern"
    else:
        severity = "High"
        risk_level = "🔴 Severe Environmental Threat"
    
    # Create results text
    results = f"""
🛢️ **AI SpillGuard Detection Results**

**Spill Detection:** {'✅ Oil Spill Detected' if total_spill_pixels > 0 else '❌ No Oil Spill Detected'}
**Affected Area:** {spill_percentage:.2f}% of image
**Severity Level:** {severity}
**Environmental Risk:** {risk_level}
**Total Spill Pixels:** {total_spill_pixels:,}

📊 **Analysis Summary:**
- Image Resolution: {width} × {height} pixels
- Detection Confidence: 89.3%
- Processing Time: 0.45 seconds

⚠️ **Note:** This is a demonstration using simulated detection results.
"""
    
    return overlay, demo_mask, results

# Create simple interface for local testing
with gr.Blocks(title="AI SpillGuard - Oil Spill Detection") as demo:
    
    gr.Markdown("# 🛢️ AI SpillGuard - Oil Spill Detection System")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Satellite Image")
            detect_btn = gr.Button("🔍 Detect Oil Spills", variant="primary")
        
        with gr.Column():
            output_overlay = gr.Image(label="Detection Overlay")
            output_mask = gr.Image(label="Spill Mask") 
            results_text = gr.Markdown(label="Results")
    
    detect_btn.click(
        fn=detect_oil_spill,
        inputs=[input_image],
        outputs=[output_overlay, output_mask, results_text]
    )

# Launch for local testing
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Local only
        server_port=7860,
        show_error=True,
        share=False  # No sharing for local testing
    )