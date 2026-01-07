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
# Use PostgreSQL if DATABASE_URL is provided (Railway, etc.), otherwise SQLite
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    # Use /tmp for Vercel serverless, or project directory for local/Railway
    if os.environ.get('VERCEL'):
        DATABASE_PATH = Path('/tmp/gait_analysis.db')
        UPLOAD_FOLDER = Path('/tmp/uploads')
    else:
        DATABASE_PATH = BASE_DIR / 'gait_analysis.db'
        UPLOAD_FOLDER = BASE_DIR / 'backend' / 'uploads'
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
else:
    # PostgreSQL - uploads still go to local directory
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        UPLOAD_FOLDER = Path('/tmp/uploads')
    else:
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

