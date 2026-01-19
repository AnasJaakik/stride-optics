#!/usr/bin/env python3
"""
Railway startup script
This ensures dependencies are installed before starting the app
"""
import subprocess
import sys
import os

# Check if Flask is installed, if not install dependencies
try:
    import flask
except ImportError:
    print("Dependencies not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Fix OpenCV installation
    print("Fixing OpenCV installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-contrib-python", "opencv-python"], 
                       capture_output=True, check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--force-reinstall", "--no-deps"], 
                       capture_output=True, check=False)
    except:
        pass

# Change to backend directory and run app
os.chdir("backend")
print("Starting Flask application...")
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
