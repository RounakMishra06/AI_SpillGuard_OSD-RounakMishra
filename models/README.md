# Models Directory

This directory contains the trained model files.

## Model File

The trained model `best_model.pth` is too large for GitHub (355 MB > 100 MB limit).

### For Local Development:
- Train the model using `notebooks/week3_model_development.ipynb`
- Or run `quick_train.py` to train a model
- Model will be saved here automatically

### For Streamlit Cloud Deployment:
The app will work in demo mode without a trained model. To use a trained model:

**Option 1**: Upload to cloud storage and download on startup
**Option 2**: Train a smaller model (< 100 MB)
**Option 3**: Use Git LFS (requires installation)

See `DEPLOYMENT_README.md` for details.
