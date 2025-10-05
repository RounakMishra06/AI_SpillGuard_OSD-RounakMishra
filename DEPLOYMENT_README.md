# AI SpillGuard - Streamlit Deployment Guide

## 🛰️ Oil Spill Detection Web Application

This Streamlit app provides a user-friendly web interface for real-time oil spill detection from satellite imagery using your trained U-Net model.

## 🚀 Features

- **Real-time Image Upload**: Upload satellite images in JPG/PNG format
- **AI-Powered Detection**: Uses your trained U-Net model for segmentation
- **Interactive Visualization**: 
  - Original image display
  - Binary mask visualization
  - Red overlay highlighting detected oil spills
  - Interactive heatmaps with Plotly
- **Metrics Dashboard**:
  - Oil coverage percentage
  - Affected area calculation
  - Severity level classification
- **Alert System**: Configurable alerts based on oil spill coverage
- **Download Results**: Save detection masks and overlay images
- **Sample Images**: Test with provided satellite images

## 📦 Installation & Setup

### 1. Install Dependencies

```bash
# Install additional packages for Streamlit app
pip install streamlit plotly opencv-python-headless
```

### 2. Ensure Model is Available

Make sure your trained model is saved at:
```
models/best_model.pth
```

If you haven't trained the model yet, run the notebook `week3_model_development.ipynb` first.

### 3. Create Required Directories

```bash
mkdir -p models results
```

## 🏃‍♂️ Running the Application

### Method 1: Using the Launch Script

```bash
# Make script executable (if not already done)
chmod +x run_app.sh

# Launch the app
./run_app.sh
```

### Method 2: Direct Streamlit Command

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the app
streamlit run app.py
```

### Method 3: Custom Port/Host

```bash
streamlit run app.py --server.port=8502 --server.address=0.0.0.0
```

## 🌐 Accessing the Application

Once running, open your web browser and navigate to:
- **Local access**: http://localhost:8501
- **Network access**: http://your-ip-address:8501

## 📱 How to Use

1. **Upload Image**: Click "Choose a satellite image..." and select your image
2. **Adjust Settings**: Use the sidebar to configure:
   - Detection threshold (0.1 - 0.9)
   - Alert threshold percentage
   - Enable/disable alerts
3. **View Results**: The app will display:
   - Original image
   - Detection mask
   - Overlay visualization
   - Metrics dashboard
4. **Download Results**: Save detection masks and overlays
5. **Try Samples**: Use provided test images if you don't have your own

## ⚙️ Configuration Options

### Detection Threshold
- **Range**: 0.1 - 0.9
- **Default**: 0.5
- **Purpose**: Controls sensitivity of oil spill detection
- **Lower values**: More sensitive, detects smaller spills
- **Higher values**: Less sensitive, only detects clear spills

### Alert System
- **Alert Threshold**: Percentage of image coverage to trigger alerts
- **Default**: 10%
- **Colors**:
  - 🟢 Green: No oil spill detected
  - 🔵 Blue: Small spill detected (below alert threshold)
  - 🔴 Red: Alert triggered (above alert threshold)

### Severity Levels
- **Low**: < 5% coverage
- **Medium**: 5-15% coverage
- **High**: 15-30% coverage
- **Critical**: > 30% coverage

## 📊 Understanding the Results

### Metrics Explained
1. **Oil Coverage**: Percentage of image pixels classified as oil spill
2. **Affected Area**: Estimated area in square kilometers
3. **Severity Level**: Classification based on coverage percentage
4. **Detection Status**: Binary detection result

### Visualizations
1. **Detection Mask**: Binary mask showing detected oil spill areas
2. **Overlay**: Original image with red highlights on detected areas
3. **Heatmap**: Probability map showing confidence levels

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ⚠️ No trained model found. Using untrained model for demo.
   ```
   **Solution**: Train your model using `week3_model_development.ipynb`

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: The app automatically uses CPU if GPU memory is insufficient

3. **Port Already in Use**
   ```
   Port 8501 is already in use
   ```
   **Solution**: Use a different port: `streamlit run app.py --server.port=8502`

4. **Package Import Errors**
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   **Solution**: Install missing packages: `pip install streamlit plotly`

### Performance Tips

1. **Image Size**: Larger images take longer to process
2. **CPU vs GPU**: GPU processing is faster but uses more memory
3. **Browser**: Use Chrome or Firefox for best performance
4. **Network**: For remote access, ensure firewall allows the port

## 🔒 Security Considerations

### For Production Deployment

1. **Authentication**: Add user authentication if needed
2. **File Upload Limits**: Configure maximum file sizes
3. **Rate Limiting**: Implement request rate limiting
4. **HTTPS**: Use SSL/TLS for secure connections
5. **Firewall**: Restrict access to specific IP addresses

### Example Production Config

```python
# In app.py, add these configurations for production:
st.set_page_config(
    page_title="AI SpillGuard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Add file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

## 🌐 Deployment Options

### 1. Local Development
- Use for testing and development
- Access via localhost only

### 2. Network Deployment
- Access from other devices on the same network
- Use `--server.address=0.0.0.0`

### 3. Cloud Deployment

#### Streamlit Cloud
```bash
# Push to GitHub and deploy via Streamlit Cloud
# Visit: https://streamlit.io/cloud
```

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📈 Future Enhancements

### Potential Features
1. **Real-time Monitoring**: Connect to satellite data feeds
2. **Historical Analysis**: Track oil spills over time
3. **API Integration**: RESTful API for external systems
4. **Mobile App**: React Native or Flutter mobile version
5. **Multi-language Support**: Internationalization
6. **Advanced Analytics**: Statistical analysis and reporting
7. **User Management**: Multi-user support with roles
8. **Data Export**: CSV/Excel export functionality

### API Development
```python
# Example FastAPI backend integration
from fastapi import FastAPI, UploadFile
app = FastAPI()

@app.post("/detect")
async def detect_oil_spill(file: UploadFile):
    # Process image and return results
    pass
```

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure your model is properly trained and saved
4. Check the Streamlit logs in the terminal

For additional help, refer to:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Your project notebooks for model details

---

**Happy Oil Spill Detecting! 🛰️🌊**
