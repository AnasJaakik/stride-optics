"""
Vercel Serverless Function Handler for StrideOptics
"""
import os
import sys
from pathlib import Path

# Set Vercel environment flag
os.environ['VERCEL'] = '1'

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

# Import Flask app
from app import app

# Vercel Python runtime expects the app to be exported
# The handler will be automatically created by Vercel
# For WSGI compatibility, we export the app directly
handler = app

# For local development
if __name__ == '__main__':
    os.environ.pop('VERCEL', None)  # Remove Vercel flag for local dev
    app.run(debug=True, port=5001)
