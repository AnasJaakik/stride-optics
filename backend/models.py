"""
Database Models for Gait Analysis Web App
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json

Base = declarative_base()

class Analysis(Base):
    """Analysis record - tracks video uploads and processing status"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default='pending', nullable=False)  # pending, processing, complete, error
    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text, nullable=True)
    
    # Relationship to results
    result = relationship("Result", back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message
        }

class Result(Base):
    """Analysis results - stores the gait analysis data"""
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), unique=True, nullable=False)
    
    # Store leg data as JSON
    left_leg_data = Column(JSON, nullable=True)
    right_leg_data = Column(JSON, nullable=True)
    
    # Recommendation data
    recommendation_type = Column(String(255), nullable=True)
    recommendation_examples = Column(JSON, nullable=True)  # List of shoe examples
    recommendation_description = Column(Text, nullable=True)
    
    # Asymmetry detection
    asymmetry_detected = Column(String(5), default='false')  # 'true' or 'false'
    asymmetry_details = Column(JSON, nullable=True)
    
    # Analysis metadata
    total_frames = Column(Integer, nullable=True)
    frames_analyzed = Column(Integer, nullable=True)
    
    # Relationship back to analysis
    analysis = relationship("Analysis", back_populates="result")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'left_leg': self.left_leg_data,
            'right_leg': self.right_leg_data,
            'recommendation': {
                'type': self.recommendation_type,
                'examples': self.recommendation_examples or [],
                'description': self.recommendation_description
            },
            'asymmetry_detected': self.asymmetry_detected == 'true',
            'asymmetry_details': self.asymmetry_details,
            'total_frames': self.total_frames,
            'frames_analyzed': self.frames_analyzed
        }

# Database setup
def init_db(db_url='sqlite:///gait_analysis.db'):
    """Initialize database and create tables"""
    # Handle PostgreSQL URLs from Railway (postgres:// -> postgresql://)
    if db_url and db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()

