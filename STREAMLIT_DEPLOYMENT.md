# 🚀 Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository

Your code is ready! Just commit and push to GitHub:

```bash
cd /Users/amankumar/Documents/rounak01/AI_SpillGuard_OSD-RounakMishra

# Add all files
git add .

# Commit changes
git commit -m "Add Streamlit deployment configuration"

# Push to GitHub
git push origin main
```

### 2. Deploy to Streamlit Cloud

1. **Visit Streamlit Cloud**: https://share.streamlit.io

2. **Sign in with GitHub**: Use your GitHub account (RounakMishra06)

3. **Create New App**:
   - Click "New app"
   - Repository: `RounakMishra06/AI_SpillGuard_OSD-RounakMishra`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

4. **Wait for Deployment** (2-3 minutes)

5. **Get Your Public URL**: 
   - Format: `https://rounakmishra06-ai-spillguard-osd-app-[hash].streamlit.app`
   - You can customize this URL in settings

### 3. Configuration Files Created

✅ `.streamlit/config.toml` - Streamlit configuration
✅ `.python-version` - Python version for deployment
✅ `requirements_deploy.txt` - Optimized dependencies for cloud

### 4. Important Notes

⚠️ **Model Training**: 
- The app will run without a trained model (demo mode)
- To use a trained model, you need to either:
  - Train locally and include `models/best_model.pth` in repo (if < 100MB)
  - Use Streamlit secrets to download model from cloud storage
  - Train on first run (slow for users)

⚠️ **File Size Limits**:
- GitHub: 100MB per file
- Streamlit Cloud: Works best with files < 1GB total

### 5. Optional: Add Model to Deployment

If your trained model is small enough (< 100MB):

```bash
# Add the model to git
git add models/best_model.pth
git commit -m "Add trained model"
git push origin main
```

If model is too large, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"
git add .gitattributes
git add models/best_model.pth
git commit -m "Add trained model with LFS"
git push origin main
```

### 6. Alternative: Model in Cloud Storage

For very large models, store on cloud and download on startup:

```python
# Add to app.py before load_model()
import urllib.request

def download_model():
    model_url = "YOUR_CLOUD_STORAGE_URL/best_model.pth"
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
```

## 🎉 After Deployment

Your app will be live at:
```
https://rounakmishra06-ai-spillguard-osd-app-xxxxx.streamlit.app
```

You can:
- ✅ Share this link with anyone
- ✅ Embed it in your portfolio
- ✅ Add it to your project README
- ✅ Customize the URL in Streamlit settings

## 📊 Monitoring

- View logs in Streamlit Cloud dashboard
- Monitor usage and performance
- Update by pushing to GitHub (auto-redeploys)

## 🔧 Troubleshooting

**Issue**: Dependencies fail to install
**Solution**: Use `requirements_deploy.txt` with pinned versions

**Issue**: App runs slowly
**Solution**: Model inference on CPU is slow; consider model optimization

**Issue**: Out of memory
**Solution**: Reduce model size or use lighter architecture

---

Ready to deploy! Follow the steps above and your app will be live in minutes! 🚀
