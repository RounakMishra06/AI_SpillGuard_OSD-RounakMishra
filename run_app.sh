#!/bin/bash

# AI SpillGuard Streamlit App Launcher
echo "🛰️ Starting AI SpillGuard Oil Spill Detection App..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  Activating virtual environment..."
    source .venv/bin/activate
fi

# Install required packages if not already installed
echo "📦 Checking dependencies..."
pip install streamlit plotly opencv-python-headless > /dev/null 2>&1

# Create necessary directories
mkdir -p models results

echo "🚀 Launching Streamlit app..."
echo "📍 App will be available at: http://localhost:8501"
echo "🔧 Use Ctrl+C to stop the app"
echo ""

# Launch the Streamlit app
streamlit run app.py --server.port=8501 --server.address=localhost
