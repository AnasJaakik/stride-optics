"""
Gait Analysis Module
Extracts core analysis logic from engine.py for use in web application
"""
import os
# Disable OpenCV GUI backend before importing (for headless environments)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
from typing import Dict, List, Optional, Callable

# Lazy import cv2 to avoid libGL issues
_cv2 = None

def get_cv2():
    """Lazy import cv2 to avoid GUI library dependencies"""
    global _cv2
    if _cv2 is None:
        try:
            # Set environment variables before import
            os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')
            os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
            
            # Import cv2
            import cv2
            _cv2 = cv2
        except (ImportError, OSError) as e:
            error_msg = str(e)
            if 'libGL' in error_msg or 'libGL.so' in error_msg:
                raise RuntimeError(
                    "OpenCV is trying to load GUI libraries (libGL). "
                    "This usually means opencv-contrib-python is installed instead of opencv-python-headless. "
                    "Please ensure only opencv-python-headless is installed: "
                    "pip uninstall opencv-contrib-python opencv-python && pip install opencv-python-headless"
                )
            else:
                raise ImportError(
                    f"Failed to import OpenCV: {error_msg}. "
                    "Please ensure opencv-python-headless is installed: pip install opencv-python-headless"
                )
    return _cv2

# Gait pattern thresholds (in degrees) - pronation/supination deviation angles
# Positive values = inward (pronation), Negative values = outward (supination)
HIGHLY_PRONATED_THRESHOLD = 10.0  # Angle > 10°: highly pronated
PRONATION_THRESHOLD = 7.0  # 7° to 10°: pronation position
NEUTRAL_MIN = 1.0  # 1° to 7°: neutral
SUPINATION_THRESHOLD = -3.0  # -3° to 0°: hint of supination
# Angle < -3°: highly supinated

# Lazy MediaPipe import and initialization
_pose_instance = None
_mp_pose = None

def get_pose_instance():
    """Get or create MediaPipe Pose instance (lazy import)"""
    global _pose_instance, _mp_pose
    if _pose_instance is None:
        try:
            # Import MediaPipe only when needed
            import mediapipe as mp
            
            # Access solutions - this is where the error occurs if MediaPipe isn't properly installed
            _mp_pose = mp.solutions.pose
            
            # Create Pose instance
            _pose_instance = _mp_pose.Pose(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=1
            )
        except AttributeError as e:
            # This catches the "no attribute 'solutions'" error
            error_msg = str(e)
            if 'solutions' in error_msg:
                raise RuntimeError(
                    "MediaPipe 'solutions' attribute not found. "
                    "This usually indicates MediaPipe is not properly installed or there's a version mismatch. "
                    "Please ensure mediapipe>=0.10.0 is installed correctly. "
                    "If the issue persists, try: pip install --force-reinstall mediapipe"
                )
            else:
                raise RuntimeError(f"MediaPipe initialization error: {error_msg}")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import MediaPipe: {str(e)}. "
                "Please ensure mediapipe>=0.10.0 is installed: pip install mediapipe>=0.10.0"
            )
    return _pose_instance

def enhance_contrast(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This helps the AI see 'black shoes on black treadmill'.
    """
    cv2 = get_cv2()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_pronation_supination_angle(knee, ankle, heel, frame_width, frame_height):
    """
    Calculate pronation/supination deviation angle from neutral.
    This measures the angle of the foot deviation from vertical alignment.
    
    Positive values = inward roll (pronation)
    Negative values = outward roll (supination)
    
    Args:
        knee: [x, y] normalized coordinates of knee
        ankle: [x, y] normalized coordinates of ankle
        heel: [x, y] normalized coordinates of heel
        frame_width: width of the frame in pixels
        frame_height: height of the frame in pixels
    
    Returns:
        Deviation angle in degrees (positive = pronation, negative = supination)
    """
    # Convert normalized coordinates to pixel coordinates
    ankle_px = np.array([ankle[0] * frame_width, ankle[1] * frame_height])
    heel_px = np.array([heel[0] * frame_width, heel[1] * frame_height])
    
    # Calculate the vector from ankle to heel
    ankle_heel_vector = heel_px - ankle_px
    
    # Calculate horizontal and vertical components
    horizontal_component = ankle_heel_vector[0]  # x component
    vertical_component = np.abs(ankle_heel_vector[1])  # y component (always positive, pointing down)
    
    # Avoid division by zero
    if vertical_component < 1.0:
        return 0.0
    
    # Calculate the angle of deviation from vertical
    # arctan2(horizontal, vertical) gives us the angle from vertical
    # Positive horizontal = pronation (inward), Negative horizontal = supination (outward)
    deviation_angle = np.arctan2(horizontal_component, vertical_component) * 180.0 / np.pi
    
    return deviation_angle

def classify_gait_pattern(deviation_angle):
    """
    Classifies gait pattern based on pronation/supination deviation angle.
    Positive values = pronation (inward), Negative values = supination (outward)
    
    Returns: (pattern_name, severity_level)
    """
    if deviation_angle > HIGHLY_PRONATED_THRESHOLD:
        return ("SEVERE OVERPRONATION", 4)
    elif deviation_angle > PRONATION_THRESHOLD:
        return ("OVERPRONATION", 3)
    elif deviation_angle >= NEUTRAL_MIN:
        return ("NEUTRAL", 1)
    elif deviation_angle >= SUPINATION_THRESHOLD:
        return ("SUPINATION", 2)
    else:  # deviation_angle < -3°
        return ("SEVERE SUPINATION", 4)

def get_shoe_recommendation(pattern):
    """Returns shoe recommendation based on gait pattern"""
    if "SEVERE OVERPRONATION" in pattern:
        return {
            "type": "Maximum Stability Shoes",
            "examples": ["Brooks Beast", "Asics Gel-Kayano", "New Balance 1340v3"],
            "description": "These shoes provide maximum support and motion control for severe overpronation."
        }
    elif "OVERPRONATION" in pattern:
        return {
            "type": "Stability Shoes",
            "examples": ["Brooks Adrenaline", "Asics GT-2000", "Saucony Guide"],
            "description": "Stability shoes help control excessive inward rolling of the foot."
        }
    elif "NEUTRAL" in pattern:
        return {
            "type": "Neutral Cushioned Shoes",
            "examples": ["Brooks Ghost", "Nike Pegasus", "Asics Gel-Nimbus"],
            "description": "Neutral shoes provide cushioning without extra stability features."
        }
    elif "SUPINATION" in pattern:
        return {
            "type": "Neutral Cushioned Shoes with Extra Padding",
            "examples": ["Brooks Glycerin", "Hoka Clifton", "Asics Gel-Cumulus"],
            "description": "Extra cushioning helps absorb impact for supinated feet."
        }
    elif "SEVERE SUPINATION" in pattern:
        return {
            "type": "Maximum Cushioning Shoes",
            "examples": ["Hoka Bondi", "Brooks Glycerin", "Asics Gel-Nimbus"],
            "description": "Maximum cushioning provides superior shock absorption."
        }
    return {
        "type": "Consult a podiatrist",
        "examples": [],
        "description": "Professional assessment recommended for your specific needs."
    }

def analyze_video(video_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict:
    """
    Analyze a video file for gait patterns.
    
    Args:
        video_path: Path to the video file
        progress_callback: Optional callback function(processed_frames, total_frames)
    
    Returns:
        Dictionary containing analysis results
    """
    cv2 = get_cv2()
    pose = get_pose_instance()
    left_angles = []
    right_angles = []
    frame_count = 0
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update progress
        if progress_callback and total_frames > 0:
            progress_callback(frame_count, total_frames)
        
        # Enhance contrast for better detection
        ai_frame = enhance_contrast(frame)
        image_rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image_rgb)
        frame_height, frame_width, _ = frame.shape
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # LEFT LEG
            try:
                l_knee = [landmarks[25].x, landmarks[25].y]
                l_ankle = [landmarks[27].x, landmarks[27].y]
                l_heel = [landmarks[29].x, landmarks[29].y]
                
                if landmarks[27].visibility > 0.5:
                    # Calculate pronation/supination deviation angle
                    deviation_angle_l = calculate_pronation_supination_angle(
                        l_knee, l_ankle, l_heel, frame_width, frame_height
                    )
                    # For left leg, positive = pronation (inward), negative = supination (outward)
                    # Store the deviation angle
                    left_angles.append(deviation_angle_l)
            except:
                pass
            
            # RIGHT LEG
            try:
                r_knee = [landmarks[26].x, landmarks[26].y]
                r_ankle = [landmarks[28].x, landmarks[28].y]
                r_heel = [landmarks[30].x, landmarks[30].y]
                
                if landmarks[28].visibility > 0.5:
                    # Calculate pronation/supination deviation angle
                    deviation_angle_r = calculate_pronation_supination_angle(
                        r_knee, r_ankle, r_heel, frame_width, frame_height
                    )
                    # For right leg, we need to flip the sign (opposite of left leg)
                    # Positive x deviation for right leg = supination, negative = pronation
                    deviation_angle_r = -deviation_angle_r
                    right_angles.append(deviation_angle_r)
            except:
                pass
    
    cap.release()
    
    # Process results
    result = {
        "left_leg": None,
        "right_leg": None,
        "recommendation": None,
        "asymmetry_detected": False,
        "total_frames": frame_count,
        "frames_analyzed": len(left_angles) + len(right_angles)
    }
    
    # Analyze left leg
    if left_angles:
        # Use average deviation angle for classification (more stable than min)
        avg_angle_l = sum(left_angles) / len(left_angles)
        min_angle_l = min(left_angles)
        max_angle_l = max(left_angles)
        pattern_l, severity_l = classify_gait_pattern(avg_angle_l)
        
        result["left_leg"] = {
            "angles": {
                "min": float(min_angle_l),
                "max": float(max_angle_l),
                "avg": float(avg_angle_l),
                "deviation": float(avg_angle_l),  # Main deviation angle
                "all": [float(a) for a in left_angles]
            },
            "pattern": pattern_l,
            "severity": severity_l,
            "warning": severity_l >= 3
        }
    
    # Analyze right leg
    if right_angles:
        # Use average deviation angle for classification (more stable than min)
        avg_angle_r = sum(right_angles) / len(right_angles)
        min_angle_r = min(right_angles)
        max_angle_r = max(right_angles)
        pattern_r, severity_r = classify_gait_pattern(avg_angle_r)
        
        result["right_leg"] = {
            "angles": {
                "min": float(min_angle_r),
                "max": float(max_angle_r),
                "avg": float(avg_angle_r),
                "deviation": float(avg_angle_r),  # Main deviation angle
                "all": [float(a) for a in right_angles]
            },
            "pattern": pattern_r,
            "severity": severity_r,
            "warning": severity_r >= 3
        }
    
    # Determine recommendation
    if left_angles and right_angles:
        # Use the more severe pattern for recommendation
        if severity_l >= severity_r:
            result["recommendation"] = get_shoe_recommendation(pattern_l)
        else:
            result["recommendation"] = get_shoe_recommendation(pattern_r)
        
        # Check for asymmetry
        if abs(severity_l - severity_r) >= 2:
            result["asymmetry_detected"] = True
            result["asymmetry_details"] = {
                "left_pattern": pattern_l,
                "right_pattern": pattern_r,
                "severity_difference": abs(severity_l - severity_r)
            }
    elif left_angles:
        result["recommendation"] = get_shoe_recommendation(pattern_l)
    elif right_angles:
        result["recommendation"] = get_shoe_recommendation(pattern_r)
    else:
        result["recommendation"] = {
            "type": "Unable to determine",
            "examples": [],
            "description": "Insufficient data collected. Please ensure the video clearly shows both legs while running."
        }
    
    return result

