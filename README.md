# AI SpillGuard Oil Spill Detection 🛢️🌊

**Project by:** Rounak Mishra  
**Repository:** [AI_SpillGuard_OSD-RounakMishra](https://github.com/RounakMishra06/AI_SpillGuard_OSD-RounakMishra)  
**Status:** ✅ **Complete**

## 🎯 Project Overview

AI SpillGuard is a comprehensive deep learning solution for automatic oil spill detection in satellite imagery. This project implements state-of-the-art computer vision and deep learning techniques to identify and segment oil spill regions from SAR (Synthetic Aperture Radar) and optical satellite data.

## 🏆 Milestones Completed

### ✅ **Milestone 1: Week 1-2 - Data Pipeline Implementation**
- **Module 1: Data Collection** - Dataset acquisition and organization
- **Module 2: Data Exploration & Preprocessing** - Comprehensive preprocessing pipeline

### ✅ **Milestone 2: Week 3-4 - Model Development & Training**
- **Module 3: Model Development** - U-Net architecture implementation
- **Module 4: Training & Evaluation** - Complete training pipeline with metrics

## 📁 Project Structure

```
AI_SpillGuard_OSD-RounakMishra/
├── 📓 notebooks/
│   ├── week1_complete_preprocessing.ipynb    # Week 1-2 Implementation
│   └── week3_model_development.ipynb         # Week 3-4 Implementation
├── 🔧 src/
│   ├── dataset.py                           # Dataset loading utilities
│   ├── unet.py                              # U-Net model architecture
│   ├── losses.py                            # Loss functions (BCE, Dice)
│   ├── metrics.py                           # Evaluation metrics
│   ├── transforms.py                        # Data augmentation
│   ├── train.py                             # Training script
│   └── visualize.py                         # Visualization utilities
├── 📊 outputs/
│   └── week1_panels/                        # Generated visualizations
├── 🗂️ data/                                # Dataset (structured)
│   ├── train/ (811 images + masks)
│   ├── val/ (203 images + masks)
│   └── test/ (254 images + masks)
├── 📝 WEEK1_COMPLETE.md                     # Week 1 completion report
├── 🧪 simple_week1_test.py                  # Standalone test script
└── 📋 requirements.txt                      # Dependencies
```

## 🚀 Implementation Highlights

### **Week 1-2: Data Pipeline**
- ✅ **Dataset Organization**: 1,268 satellite image-mask pairs
- ✅ **Preprocessing Pipeline**: Resize, normalize, SAR filtering
- ✅ **Data Augmentation**: Advanced transformations for class balance
- ✅ **Visualization**: 9-panel comprehensive pipeline demonstration

### **Week 3-4: Model Development**
- ✅ **U-Net Architecture**: Customizable encoder-decoder for segmentation
- ✅ **Loss Functions**: Dice Loss, BCE Loss, Combined BCE-Dice Loss
- ✅ **Evaluation Metrics**: IoU, Dice Coefficient, Precision, Recall
- ✅ **Training Pipeline**: Real-time augmentation, validation, checkpointing
- ✅ **Hyperparameter Tuning**: Learning rate scheduling, optimization

## 💡 Key Features

### 🔬 **Advanced Preprocessing**
- Binary mask conversion from RGB annotations
- SAR-specific noise filtering (Gaussian, Bilateral)
- Multi-technique data augmentation
- Statistical analysis and visualization

### 🧠 **Deep Learning Model**
- U-Net architecture optimized for satellite imagery
- Multi-channel input support (RGB/SAR)
- Custom loss functions for imbalanced segmentation
- GPU acceleration and memory optimization

### 📊 **Comprehensive Evaluation**
- IoU (Intersection over Union) metrics
- Dice coefficient for segmentation quality
- Training/validation curves visualization
- Prediction overlay visualization

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Image Processing**: OpenCV, PIL, albumentations
- **Visualization**: Matplotlib, seaborn
- **Data Science**: NumPy, pandas
- **Environment**: Jupyter Notebooks, Python 3.11+

## 📈 Results & Performance

- **Dataset Size**: 1,268 image-mask pairs
- **Image Resolution**: 256×256 pixels (standardized)
- **Model Architecture**: U-Net with 4 encoder/decoder levels
- **Training Framework**: PyTorch with GPU acceleration
- **Metrics**: IoU and Dice coefficient tracking

## 🎓 Academic Requirements Met

### **Week 1-2 Deliverables** ✅
- [x] Dataset acquisition and organization
- [x] Image-mask loading and preprocessing
- [x] Binary conversion and normalization
- [x] Data augmentation implementation
- [x] Complete visualization pipeline
- [x] Statistical analysis and reporting

### **Week 3-4 Deliverables** ✅
- [x] U-Net model architecture design
- [x] Custom dataset class implementation
- [x] Loss function implementation (BCE, Dice)
- [x] Training pipeline with validation
- [x] Evaluation metrics (IoU, Dice)
- [x] Hyperparameter tuning framework

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RounakMishra06/AI_SpillGuard_OSD-RounakMishra.git
   cd AI_SpillGuard_OSD-RounakMishra
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing pipeline:**
   ```bash
   python simple_week1_test.py
   ```

4. **Open Jupyter notebooks:**
   ```bash
   jupyter notebook notebooks/week1_complete_preprocessing.ipynb
   jupyter notebook notebooks/week3_model_development.ipynb
   ```

## 📝 Documentation

- **[WEEK1_COMPLETE.md](WEEK1_COMPLETE.md)** - Detailed Week 1 completion report
- **[week1_complete_preprocessing.ipynb](notebooks/week1_complete_preprocessing.ipynb)** - Interactive preprocessing pipeline
- **[week3_model_development.ipynb](notebooks/week3_model_development.ipynb)** - Model training and evaluation

## 🏅 Achievement Summary

**🎯 All Milestones Completed Successfully!**

- ✅ **Module 1**: Data Collection & Organization
- ✅ **Module 2**: Data Exploration & Preprocessing  
- ✅ **Module 3**: Model Development & Architecture
- ✅ **Module 4**: Training & Evaluation Pipeline

**📊 Final Status:** Ready for deployment and real-world testing!

---

## 📌 Submission Details

**Student:** Rounak Mishra  
**Project:** AI SpillGuard Oil Spill Detection  
**Implementation Period:** September 2025  
**All Requirements:** ✅ **COMPLETE**
