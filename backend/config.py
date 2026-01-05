"""
Configuration settings for Gait Analysis Web App
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

# Database configuration
DATABASE_PATH = BASE_DIR / 'gait_analysis.db'
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# Upload configuration
UPLOAD_FOLDER = BASE_DIR / 'backend' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# Analysis configuration
ANALYSIS_TIMEOUT = 600  # 10 minutes timeout for analysis

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

