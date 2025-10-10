@echo off
echo 🚀 Preparing files for Hugging Face upload...
echo.

echo ✅ Files ready for upload to: https://huggingface.co/spaces/Rounak-06/ai-spillguard-oil-detection/
echo.

echo 📁 Upload these files in this order:
echo.

echo 1. README.md
echo    📂 Source: hf_README.md
echo    📤 Upload as: README.md
echo    💬 Commit: "Add README with model description"
echo.

echo 2. app.py  
echo    📂 Source: app_huggingface.py
echo    📤 Upload as: app.py
echo    💬 Commit: "Add main Gradio application"
echo.

echo 3. requirements.txt
echo    📂 Source: hf_requirements.txt  
echo    📤 Upload as: requirements.txt
echo    💬 Commit: "Add dependencies"
echo.

echo 4. examples/satellite_1.jpg
echo    📂 Source: examples/satellite_1.jpg
echo    📤 Upload as: examples/satellite_1.jpg
echo    💬 Commit: "Add example satellite image 1"
echo.

echo 5. examples/satellite_2.jpg
echo    📂 Source: examples/satellite_2.jpg
echo    📤 Upload as: examples/satellite_2.jpg  
echo    💬 Commit: "Add example satellite image 2"
echo.

echo 🎯 Upload Process:
echo 1. Go to: https://huggingface.co/spaces/Rounak-06/ai-spillguard-oil-detection/
echo 2. Click "Files" tab
echo 3. Click "Add file" → "Upload file"
echo 4. Select source file and rename as specified
echo 5. Add commit message and click "Commit new file"
echo 6. Repeat for each file
echo.

echo 🧪 After upload:
echo - Wait 2-3 minutes for build
echo - Check "Logs" tab for any errors
echo - Test your app with example images
echo.

echo ✨ Your AI SpillGuard will be live on Hugging Face! ✨
pause