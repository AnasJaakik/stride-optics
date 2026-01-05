# Gait Analysis Web Application

A web application that analyzes running gait patterns from video uploads and provides personalized shoe recommendations.

## Features

- **Video Upload**: Drag-and-drop or click to upload running videos
- **AI-Powered Analysis**: Uses MediaPipe to detect gait patterns and measure ankle angles
- **Real-time Progress**: Track analysis progress with live updates
- **Detailed Results**: View comprehensive analysis including:
  - Left and right leg angle measurements
  - Gait pattern classification (Neutral, Overpronation, Supination, etc.)
  - Shoe recommendations with specific models
  - Asymmetry detection
- **Analysis History**: View all past analyses
- **Modern UI**: Clean, responsive design

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Run the Flask application**:
   ```bash
   cd backend
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload a Video**:
   - Record yourself running from a side view
   - Ensure both legs are clearly visible
   - Upload the video file (MP4, AVI, MOV, MKV, WEBM, FLV)
   - Maximum file size: 500MB

2. **Wait for Analysis**:
   - The video will be processed in the background
   - Progress updates will be shown in real-time
   - Processing typically takes 1-5 minutes depending on video length

3. **View Results**:
   - See detailed analysis of your gait pattern
   - Get personalized shoe recommendations
   - Check for any asymmetry between legs

4. **View History**:
   - Access all your past analyses from the History page

## Project Structure

```
gait-analysis-agent/
├── backend/
│   ├── app.py              # Flask application
│   ├── models.py           # Database models
│   ├── gait_analyzer.py    # Core analysis engine
│   ├── config.py           # Configuration
│   ├── engine.py           # Original analysis script (for reference)
│   ├── templates/          # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── results.html
│   │   ├── history.html
│   │   └── error.html
│   ├── static/             # Static assets
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── uploads/            # Uploaded videos (created automatically)
├── requirements.txt
└── README.md
```

## Gait Patterns Detected

- **Severe Overpronation** (< 150°): Maximum stability shoes recommended
- **Overpronation** (150-170°): Stability shoes recommended
- **Neutral** (170-190°): Neutral cushioned shoes recommended
- **Supination** (190-210°): Extra cushioning recommended
- **Severe Supination** (> 210°): Maximum cushioning recommended

## Technical Details

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Computer Vision**: OpenCV and MediaPipe
- **Processing**: Asynchronous background tasks with threading
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)

## Requirements

- Python 3.9+ (MediaPipe compatibility)
- OpenCV
- MediaPipe
- Flask
- SQLAlchemy
- NumPy

## Notes

- Videos are automatically deleted after processing to save disk space
- The database file (`gait_analysis.db`) is created automatically on first run
- For production use, change the `SECRET_KEY` in `config.py` and set `DEBUG=False`

## Troubleshooting

- **Import errors**: Make sure the virtual environment is activated and all dependencies are installed
- **Video processing fails**: Ensure the video clearly shows both legs from a side view
- **File upload errors**: Check file size (max 500MB) and format (supported video formats only)

## License

This project is provided as-is for educational and personal use.

# stride-optics
