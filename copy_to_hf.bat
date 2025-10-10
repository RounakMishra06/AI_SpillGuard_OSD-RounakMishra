@echo off
echo 🚀 Copying files to Hugging Face space...
echo.

REM Copy README
copy hf_README.md hf_space\README.md
echo ✅ Copied README.md

REM Copy main app
copy app_huggingface.py hf_space\app.py
echo ✅ Copied app.py

REM Copy requirements
copy hf_requirements.txt hf_space\requirements.txt
echo ✅ Copied requirements.txt

REM Create examples directory and copy images
mkdir hf_space\examples 2>nul
copy examples\satellite_1.jpg hf_space\examples\satellite_1.jpg
copy examples\satellite_2.jpg hf_space\examples\satellite_2.jpg
echo ✅ Copied example images

echo.
echo 📁 Files copied to hf_space directory!
echo.
echo 🎯 Next steps:
echo 1. cd hf_space
echo 2. git add .
echo 3. git commit -m "Add AI SpillGuard application files"
echo 4. git push
echo.
pause