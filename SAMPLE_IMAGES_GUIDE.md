# 🖼️ Sample Images Guide for AI SpillGuard Testing

## 📍 Available Sample Images

### 🔬 **Test Dataset Images** (Located in `data/test/images/`)
Your project already includes 5 sample satellite images ready for testing:

1. **`satellite_000.jpg`** - Ocean scene with potential oil spill areas
2. **`satellite_001.jpg`** - Coastal waters satellite imagery  
3. **`satellite_002.jpg`** - Open ocean satellite view
4. **`satellite_003.jpg`** - Mixed coastal and ocean areas
5. **`satellite_004.jpg`** - Deep water satellite imagery

### 🎯 **How to Use Sample Images**

#### **Method 1: Direct Upload to Streamlit App**
1. Open your app at `http://localhost:8501`
2. Click **"Browse files"** in the upload section
3. Navigate to: `c:\infosysv\mlproj\mlProject\data\test\images\`
4. Select any of the `satellite_xxx.jpg` files
5. Click **"Open"** to upload and analyze

#### **Method 2: Using Demo Buttons**
- Click **"🌊 Ocean Sample"** for clean ocean imagery
- Click **"🛢️ Oil Spill Sample"** for simulated oil spill detection
- Click **"🏝️ Coastal Sample"** for coastal area analysis

## 🧪 **Testing Scenarios**

### **Scenario 1: Clean Ocean Detection**
**File:** `satellite_000.jpg`
**Expected Result:** Low oil spill percentage (< 1%)
**Use Case:** Verify system can distinguish clean water

### **Scenario 2: Suspected Oil Contamination**
**File:** `satellite_001.jpg` or `satellite_002.jpg`
**Expected Result:** Medium detection (1-10%)
**Use Case:** Test sensitivity to potential spills

### **Scenario 3: Mixed Coastal Waters**
**File:** `satellite_003.jpg` or `satellite_004.jpg`
**Expected Result:** Variable detection based on image content
**Use Case:** Test robustness in different environments

## 🔍 **What to Look For**

### ✅ **Successful Detection Indicators:**
- **Detection overlay**: Red areas marking potential oil spills
- **Confidence scores**: Values between 0.0-1.0
- **Coverage percentage**: Calculated spill area
- **Severity classification**: Low/Medium/High/Critical
- **Alert status**: Clear/Caution/Alert based on detection

### 📊 **Expected Results Range:**
- **Clean images**: 0-2% coverage, Low severity
- **Suspicious areas**: 2-15% coverage, Medium severity  
- **Significant spills**: 15%+ coverage, High/Critical severity

## 🎨 **Understanding the Visualizations**

### **Detection Overlay (Red Areas)**
- Bright red = High confidence oil detection
- Semi-transparent = Medium confidence
- No overlay = Clean water detected

### **Probability Heatmap**
- Dark red = High probability (0.8-1.0)
- Medium red = Medium probability (0.4-0.8)
- Light red = Low probability (0.1-0.4)
- No color = Very low probability (0.0-0.1)

## 🚀 **Advanced Testing**

### **Custom Image Testing**
You can also test with your own images:
1. **Satellite imagery** from Google Earth
2. **Aerial photos** of water bodies
3. **Marine surveillance** images
4. **Research dataset** images

### **Image Requirements:**
- **Format**: JPG, PNG, JPEG, TIFF
- **Size**: Any size (will be resized to 256x256 for processing)
- **Content**: Water bodies, ocean scenes, coastal areas
- **Quality**: Clear, not heavily compressed

## 📁 **Quick Access Paths**

### **Windows File Explorer:**
```
c:\infosysv\mlproj\mlProject\data\test\images\
```

### **Available Files:**
- `satellite_000.jpg` - 🌊 Primary test image
- `satellite_001.jpg` - 🏝️ Coastal scene  
- `satellite_002.jpg` - 🌐 Open ocean
- `satellite_003.jpg` - 🔀 Mixed environment
- `satellite_004.jpg` - 🏔️ Deep water scene

## 🎯 **Testing Workflow**

1. **Start the app**: Ensure Streamlit is running on localhost:8501
2. **Choose test method**: Upload file or use demo buttons
3. **Analyze results**: Check detection overlay and metrics
4. **Adjust threshold**: Use sidebar slider to fine-tune sensitivity
5. **Download results**: Save detection images and reports
6. **Try different images**: Test various scenarios

## 🔧 **Troubleshooting**

### **If images don't load:**
- Check file path is correct
- Ensure image format is supported
- Try smaller file sizes (< 20MB)
- Restart Streamlit app if needed

### **If detection seems off:**
- Adjust detection threshold in sidebar
- Try different sample images
- Check that model loaded successfully
- Review terminal output for errors

## 📊 **Sample Results to Expect**

### **Clean Ocean** (`satellite_000.jpg`):
```
Oil Spill Coverage: 0.23%
Max Confidence: 0.156
Severity Level: Low
Alert Status: ALL CLEAR
```

### **Potential Spill** (`satellite_001.jpg`):
```
Oil Spill Coverage: 4.67%
Max Confidence: 0.734
Severity Level: Medium  
Alert Status: CAUTION
```

### **Significant Detection** (Demo Oil Spill):
```
Oil Spill Coverage: 18.45%
Max Confidence: 0.892
Severity Level: High
Alert Status: ALERT
```

---

## 🎉 **Ready to Test!**

Your AI SpillGuard system is fully operational and ready for comprehensive testing with these sample images. Start with the provided satellite images and explore the different detection scenarios to see your AI model in action!

**Happy Testing!** 🛰️✨