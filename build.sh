#!/bin/bash
# Build script for Railway deployment

# Install dependencies
pip install -r requirements.txt

# Fix OpenCV installation (remove GUI version, install headless)
pip uninstall -y opencv-contrib-python opencv-python 2>/dev/null || true
pip install opencv-python-headless --force-reinstall --no-deps 2>/dev/null || true

echo "Build complete!"
