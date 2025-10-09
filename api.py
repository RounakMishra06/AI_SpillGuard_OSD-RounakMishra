"""
AI SpillGuard FastAPI Backend
============================

REST API for oil spill detection using satellite imagery.
Provides endpoints for image upload, prediction, and result retrieval.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
from pathlib import Path
import sys
import uvicorn
from typing import Dict, Any

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import your model
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = FastAPI(
    title="AI SpillGuard API",
    description="Oil Spill Detection API using Satellite Imagery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OilSpillAPI:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=1)
        
        # Load the trained model
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file {model_path} not found!")
            
        # Image preprocessing pipeline
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image):
        """Preprocess the input image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def predict(self, image):
        """Make prediction on the input image"""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image).to(self.device)
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output).cpu().numpy()
            return prediction.squeeze()

# Initialize the API
detector = OilSpillAPI()

@app.get("/")
async def root():
    return {"message": "AI SpillGuard Oil Spill Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(detector.device)}

@app.post("/predict")
async def predict_oil_spill(file: UploadFile = File(...)):
    """
    Upload an image and get oil spill detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        prediction = detector.predict(image)
        
        # Calculate metrics
        binary_mask = (prediction > 0.5).astype(np.uint8)
        oil_spill_percentage = (binary_mask.sum() / binary_mask.size) * 100
        confidence = float(prediction.max())
        affected_pixels = int(binary_mask.sum())
        
        # Convert prediction to base64 for visualization
        prediction_img = (prediction * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', prediction_img)
        prediction_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create overlay
        original_array = np.array(image)
        mask_resized = cv2.resize(prediction, (original_array.shape[1], original_array.shape[0]))
        binary_mask_resized = (mask_resized > 0.5).astype(np.uint8)
        
        overlay = original_array.copy()
        overlay[binary_mask_resized == 1] = [255, 0, 0]  # Red for oil spill
        result = cv2.addWeighted(original_array, 0.7, overlay, 0.3, 0)
        
        _, overlay_buffer = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        return {
            "success": True,
            "results": {
                "oil_spill_detected": oil_spill_percentage > 0.1,
                "oil_spill_percentage": round(oil_spill_percentage, 2),
                "max_confidence": round(confidence, 3),
                "affected_pixels": affected_pixels,
                "prediction_mask": prediction_base64,
                "overlay_image": overlay_base64
            },
            "alert": "Oil spill detected!" if oil_spill_percentage > 0.1 else "No significant oil spill detected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Process multiple images for oil spill detection
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            prediction = detector.predict(image)
            
            binary_mask = (prediction > 0.5).astype(np.uint8)
            oil_spill_percentage = (binary_mask.sum() / binary_mask.size) * 100
            
            results.append({
                "filename": file.filename,
                "oil_spill_detected": oil_spill_percentage > 0.1,
                "oil_spill_percentage": round(oil_spill_percentage, 2),
                "max_confidence": round(float(prediction.max()), 3)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"success": True, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)