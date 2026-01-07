"""
Flask Web Application for Gait Analysis
"""
import os
# Set environment variables BEFORE any OpenCV/MediaPipe imports
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_DISABLE_OPENCL', '1')

import threading
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from config import (
    UPLOAD_FOLDER, MAX_CONTENT_LENGTH, allowed_file,
    DATABASE_URL, SECRET_KEY, DEBUG
)
from models import init_db, get_session, Analysis, Result
from gait_analyzer import analyze_video

app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Initialize database
db_engine = init_db(DATABASE_URL)

# Store active analysis threads
active_analyses = {}

def process_video(analysis_id, video_path):
    """Background task to process video analysis"""
    session = get_session(db_engine)
    try:
        analysis = session.query(Analysis).filter_by(id=analysis_id).first()
        if not analysis:
            return
        
        # Update status to processing
        analysis.status = 'processing'
        analysis.progress = 0
        session.commit()
        
        # Progress callback
        def progress_callback(processed, total):
            if total > 0:
                progress = int((processed / total) * 100)
                analysis.progress = progress
                session.commit()
        
        # Run analysis
        results_data = analyze_video(video_path, progress_callback=progress_callback)
        
        # Validate results
        if not results_data.get('left_leg') and not results_data.get('right_leg'):
            raise ValueError("Unable to detect gait patterns. Please ensure the video clearly shows both legs while running.")
        
        # Create result record
        result = Result(
            analysis_id=analysis_id,
            left_leg_data=results_data.get('left_leg'),
            right_leg_data=results_data.get('right_leg'),
            recommendation_type=results_data.get('recommendation', {}).get('type'),
            recommendation_examples=results_data.get('recommendation', {}).get('examples', []),
            recommendation_description=results_data.get('recommendation', {}).get('description'),
            asymmetry_detected='true' if results_data.get('asymmetry_detected') else 'false',
            asymmetry_details=results_data.get('asymmetry_details'),
            total_frames=results_data.get('total_frames'),
            frames_analyzed=results_data.get('frames_analyzed')
        )
        
        session.add(result)
        analysis.status = 'complete'
        analysis.progress = 100
        session.commit()
        
    except Exception as e:
        analysis = session.query(Analysis).filter_by(id=analysis_id).first()
        if analysis:
            analysis.status = 'error'
            analysis.error_message = str(e)[:500]  # Limit error message length
            session.commit()
    finally:
        session.close()
        # Remove from active analyses
        if analysis_id in active_analyses:
            del active_analyses[analysis_id]
        
        # Clean up video file after processing (optional - comment out if you want to keep files)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass  # Ignore cleanup errors

@app.route('/')
def index():
    """Home page with video upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm, flv)', 'error')
        return redirect(url_for('index'))
    
    # Note: File size validation is handled by Flask's MAX_CONTENT_LENGTH config
    # Additional validation can be done here if needed
    
    try:
        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create analysis record
        session = get_session(db_engine)
        analysis = Analysis(
            filename=unique_filename,
            original_filename=secure_filename(file.filename),
            status='pending',
            progress=0
        )
        session.add(analysis)
        session.commit()
        analysis_id = analysis.id
        session.close()
        
        # Start background processing
        thread = threading.Thread(target=process_video, args=(analysis_id, filepath))
        thread.daemon = True
        thread.start()
        active_analyses[analysis_id] = thread
        
        return redirect(url_for('results', id=analysis_id))
    
    except RequestEntityTooLarge:
        flash('File too large. Maximum size is 500MB', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error uploading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analyze/<int:id>/status')
def analyze_status(id):
    """Get analysis status (JSON endpoint for polling)"""
    session = get_session(db_engine)
    try:
        analysis = session.query(Analysis).filter_by(id=id).first()
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify({
            'status': analysis.status,
            'progress': analysis.progress,
            'error_message': analysis.error_message
        })
    finally:
        session.close()

@app.route('/results/<int:id>')
def results(id):
    """Display analysis results"""
    session = get_session(db_engine)
    try:
        analysis = session.query(Analysis).filter_by(id=id).first()
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('index'))
        
        result = None
        if analysis.result:
            result = analysis.result.to_dict()
        
        return render_template('results.html', analysis=analysis.to_dict(), result=result)
    finally:
        session.close()

@app.route('/history')
def history():
    """Display analysis history"""
    session = get_session(db_engine)
    try:
        analyses = session.query(Analysis).order_by(Analysis.upload_date.desc()).all()
        analyses_data = [a.to_dict() for a in analyses]
        return render_template('history.html', analyses=analyses_data)
    finally:
        session.close()

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Maximum size is 500MB', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found', code=404), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error', code=500), 500

if __name__ == '__main__':
    # Railway and other platforms set PORT environment variable
    port = int(os.environ.get('PORT', 5001))
    # Disable debug mode in production
    debug_mode = DEBUG and os.environ.get('RAILWAY_ENVIRONMENT') is None
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

