# AI SpillGuard - Oil Spill Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

🛰️ **AI-powered real-time oil spill detection from satellite imagery using Deep Learning**

## 🚀 Live Demo

**Streamlit App**: [Deploy to get link](STREAMLIT_DEPLOYMENT.md)

## 📋 Project Overview

AI SpillGuard is an end-to-end machine learning project that detects and segments oil spills in satellite imagery using a U-Net deep learning architecture. The system provides:

- ✅ Real-time oil spill detection and segmentation
- ✅ Interactive web interface for image upload and analysis
- ✅ Metrics dashboard showing coverage, affected area, and severity
- ✅ Alert system for environmental monitoring
- ✅ Downloadable results and visualizations

## 🎯 Features

### Core Functionality
- **Image Upload**: Support for JPG/PNG satellite images
- **AI Detection**: U-Net model for precise segmentation
- **Visualization**: Interactive overlays, masks, and heatmaps
- **Metrics**: Oil coverage %, affected area (km²), severity levels
- **Alerts**: Configurable threshold-based alert system
- **Export**: Download detection masks and overlay images

### Technical Stack
- **Deep Learning**: PyTorch, U-Net architecture
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Plotly
- **Image Processing**: OpenCV, Albumentations
- **Data Science**: NumPy, Pandas

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RounakMishra06/AI_SpillGuard_OSD-RounakMishra.git
cd AI_SpillGuard_OSD-RounakMishra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### Run Locally

```bash
# Launch the Streamlit app
./run_app.sh

# Or manually
streamlit run app.py
```

Visit: `http://localhost:8501`

### Deploy to Cloud

See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for detailed deployment instructions.

## 📊 Project Structure

```
AI_SpillGuard_OSD-RounakMishra/
├── app.py                          # Streamlit web application
├── notebooks/
│   ├── week1_complete_preprocessing.ipynb
│   └── week3_model_development.ipynb
├── src/
│   ├── dataset.py                  # Dataset classes
│   ├── unet.py                     # U-Net model
│   ├── losses.py                   # Loss functions
│   ├── metrics.py                  # Evaluation metrics
│   ├── train.py                    # Training pipeline
│   └── visualize.py                # Visualization utilities
├── data/
│   ├── train/                      # Training data
│   ├── val/                        # Validation data
│   └── test/                       # Test data
├── models/                         # Trained model checkpoints
├── outputs/                        # Visualization outputs
└── requirements.txt                # Python dependencies
```

## 🧠 Model Architecture

**U-Net** - Convolutional Neural Network for Image Segmentation
- **Encoder**: 4 downsampling blocks (64, 128, 256, 512 filters)
- **Bottleneck**: 1024 filters
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Binary segmentation mask

**Loss Function**: Combined BCE-Dice Loss (50/50 weight)

**Metrics**: IoU, Dice Coefficient, Accuracy, Precision, Recall

## 📈 Performance

- **Validation IoU**: ~0.85 (target)
- **Validation Dice**: ~0.90 (target)
- **Inference Time**: ~2-5 seconds per image (CPU)
- **Image Size**: 256x256 pixels (resized)

## 🎓 Implementation Milestones

### ✅ Milestone 1 (Week 1-2): Data Collection & Preprocessing
- Data acquisition from Kaggle Oil Spill Dataset
- Exploratory Data Analysis (EDA)
- Image preprocessing and augmentation pipeline
- Dataset split (train/val/test)

### ✅ Milestone 2 (Week 3-4): Model Development & Training
- U-Net architecture implementation
- Custom loss functions (Dice, BCE-Dice)
- Training pipeline with validation
- Hyperparameter tuning
- Model evaluation with metrics

### ✅ Milestone 3 (Week 5-6): Deployment & Visualization
- Streamlit web application
- Interactive visualizations
- Alert system
- Results export functionality
- Cloud deployment configuration

## 🔧 Usage Examples

### Upload and Analyze

1. Open the web app
2. Upload a satellite image
3. Adjust detection threshold (0.1-0.9)
4. Configure alert settings
5. View results and download

### Customize Settings

- **Detection Threshold**: Control sensitivity
- **Alert Threshold**: Set coverage % for alerts
- **Severity Levels**: Low, Medium, High, Critical

## 📸 Screenshots

*Coming soon - Add screenshots of your app in action*

## 🤝 Contributing

This is an academic project. For suggestions or issues:
1. Open an issue
2. Fork the repository
3. Submit a pull request

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 👨‍💻 Author

**Rounak Mishra**
- GitHub: [@RounakMishra06](https://github.com/RounakMishra06)
- Repository: [AI_SpillGuard_OSD-RounakMishra](https://github.com/RounakMishra06/AI_SpillGuard_OSD-RounakMishra)

## 🙏 Acknowledgments

- Oil Spill Detection Dataset - Kaggle
- U-Net Architecture - Ronneberger et al.
- Streamlit Framework
- PyTorch Community

## 📚 Documentation

- [Deployment Guide](STREAMLIT_DEPLOYMENT.md)
- [Deployment Summary](DEPLOYMENT_SUMMARY.md)
- [Week 1 Complete](WEEK1_COMPLETE.md)
- [Week 3 Model Development](WEEK3_MODEL_DEVELOPMENT_WITH_OUTPUTS.md)

## 🔮 Future Enhancements

- [ ] Real-time satellite feed integration
- [ ] Historical tracking and trend analysis
- [ ] Multi-model ensemble for better accuracy
- [ ] Mobile app development
- [ ] RESTful API for external integrations
- [ ] Advanced analytics and reporting

---

**🛰️ Built with ❤️ for environmental protection and marine conservation 🌊**
