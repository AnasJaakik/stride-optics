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
import os

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
_using_tasks_api = False  # Track which API we're using

def get_pose_instance():
    """Get or create MediaPipe Pose instance (lazy import)"""
    global _pose_instance, _mp_pose, _using_tasks_api
    if _pose_instance is None:
        try:
            # Import MediaPipe only when needed
            import mediapipe as mp
            
            # Try old solutions API first (most compatible)
            try:
                _mp_pose = mp.solutions.pose
                _pose_instance = _mp_pose.Pose(
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                    model_complexity=1
                )
            except AttributeError:
                # If solutions API not available, use Tasks API with downloaded model
                try:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    import os
                    import urllib.request
                    from pathlib import Path
                    
                    # Download model file if not exists
                    model_dir = Path(__file__).parent.parent / 'models'
                    model_dir.mkdir(exist_ok=True)
                    model_path = model_dir / 'pose_landmarker.task'
                    
                    if not model_path.exists():
                        print("Downloading MediaPipe pose landmarker model...")
                        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                        import ssl
                        ssl_context = ssl.create_default_context()
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE
                        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                        urllib.request.install_opener(opener)
                        urllib.request.urlretrieve(model_url, model_path)
                        print(f"Model downloaded to {model_path}")
                    
                    # Create Pose Landmarker using Tasks API
                    base_options = python.BaseOptions(
                        model_asset_path=str(model_path)
                    )
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        min_pose_detection_confidence=0.3,
                        min_pose_presence_confidence=0.3,
                        min_tracking_confidence=0.3,
                        output_segmentation_masks=False
                    )
                    _pose_instance = vision.PoseLandmarker.create_from_options(options)
                    _mp_pose = mp  # Store for compatibility
                    _using_tasks_api = True  # Mark that we're using Tasks API
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize MediaPipe with Tasks API: {str(e)}. "
                        "Please ensure you have internet connection for first-time model download, "
                        "or use Python 3.9-3.11 to install MediaPipe 0.10.14 which supports the old API."
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize MediaPipe with Tasks API: {str(e)}. "
                        "Please ensure you have internet connection for first-time model download, "
                        "or use Python 3.9-3.11 to install MediaPipe 0.10.14 which supports the old API."
                    )
        except AttributeError as e:
            error_msg = str(e)
            raise RuntimeError(
                f"MediaPipe initialization error: {error_msg}. "
                "Please ensure mediapipe>=0.10.0 is installed correctly. "
                "If the issue persists, try: pip install --force-reinstall mediapipe"
            )
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
    Calculate pronation/supination deviation angle from neutral (REAR VIEW).
    This measures the medial-lateral deviation of the heel relative to the ankle.
    
    For rear view footage:
    - Positive values = inward roll (pronation) - heel moves toward centerline
    - Negative values = outward roll (supination) - heel moves away from centerline
    
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
    
    # Calculate horizontal (medial-lateral) and vertical components
    # For rear view: x-axis represents medial-lateral direction
    horizontal_component = ankle_heel_vector[0]  # x component (medial-lateral)
    vertical_component = np.abs(ankle_heel_vector[1])  # y component (always positive, pointing down)
    
    # Avoid division by zero - require minimum vertical distance for accuracy
    if vertical_component < 2.0:  # Increased threshold for better accuracy
        return None  # Return None instead of 0 to indicate invalid measurement
    
    # Calculate the angle of deviation from vertical
    # arctan2(horizontal, vertical) gives us the angle from vertical
    # For rear view:
    # - Left leg: positive horizontal (heel right of ankle) = pronation (inward)
    # - Right leg: negative horizontal (heel left of ankle) = pronation (inward)
    deviation_angle = np.arctan2(horizontal_component, vertical_component) * 180.0 / np.pi
    
    # Filter out extreme outliers (likely detection errors)
    # Normal pronation/supination angles should be within reasonable bounds
    if abs(deviation_angle) > 45.0:  # Filter angles > 45 degrees (likely errors)
        return None
    
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

def draw_overlay(frame, frame_num, left_angle, right_angle, left_in_contact, right_in_contact, 
                  left_pattern, right_pattern, cadence, fps, frame_width, frame_height):
    """
    Draw analysis overlay on video frame.
    
    Args:
        frame: OpenCV frame (BGR)
        frame_num: Current frame number
        left_angle: Current left leg deviation angle (or None)
        right_angle: Current right leg deviation angle (or None)
        left_in_contact: Whether left foot is in contact
        right_in_contact: Whether right foot is in contact
        left_pattern: Left leg gait pattern string
        right_pattern: Right leg gait pattern string
        cadence: Current cadence (steps/min)
        fps: Video frame rate
        frame_width: Frame width
        frame_height: Frame height
    """
    cv2 = get_cv2()
    
    # Convert BGR to RGB for drawing, then back
    overlay = frame.copy()
    
    # Define colors (BGR format for OpenCV)
    volt_green = (0, 255, 150)  # Volt green
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    black = (0, 0, 0)
    
    # Semi-transparent background for text areas
    overlay_alpha = 0.7
    
    # Handle None patterns
    if left_pattern is None:
        left_pattern = "N/A"
    if right_pattern is None:
        right_pattern = "N/A"
    
    # Top left: Frame info and cadence
    info_bg = np.zeros((100, 280, 3), dtype=np.uint8)
    info_bg[:] = (5, 5, 5)
    overlay[10:110, 10:290] = cv2.addWeighted(overlay[10:110, 10:290], 1-overlay_alpha, info_bg, overlay_alpha, 0)
    
    # Frame number and time
    time_seconds = frame_num / fps if fps > 0 else 0
    cv2.putText(overlay, f"Frame: {frame_num}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, volt_green, 2)
    cv2.putText(overlay, f"Time: {time_seconds:.2f}s", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    if cadence > 0:
        cv2.putText(overlay, f"Cadence: {cadence:.1f} spm", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
    
    # Left leg info (left side, below info)
    if left_angle is not None:
        left_bg = np.zeros((140, 280, 3), dtype=np.uint8)
        left_bg[:] = (5, 5, 5)
        overlay[120:260, 10:290] = cv2.addWeighted(overlay[120:260, 10:290], 1-overlay_alpha, left_bg, overlay_alpha, 0)
        
        cv2.putText(overlay, "LEFT LEG", (20, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, volt_green, 2)
        angle_text = f"Angle: {left_angle:.1f}"
        cv2.putText(overlay, angle_text, (20, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        # Truncate pattern if too long
        pattern_text = left_pattern[:20] if len(left_pattern) > 20 else left_pattern
        cv2.putText(overlay, pattern_text, (20, 205), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
        if left_in_contact:
            cv2.putText(overlay, "CONTACT", (20, 235), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    
    # Right leg info (right side)
    if right_angle is not None:
        right_x = frame_width - 300
        right_bg = np.zeros((140, 280, 3), dtype=np.uint8)
        right_bg[:] = (5, 5, 5)
        overlay[10:150, right_x:right_x+280] = cv2.addWeighted(
            overlay[10:150, right_x:right_x+280], 1-overlay_alpha, right_bg, overlay_alpha, 0)
        
        cv2.putText(overlay, "RIGHT LEG", (right_x + 10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, volt_green, 2)
        angle_text = f"Angle: {right_angle:.1f}"
        cv2.putText(overlay, angle_text, (right_x + 10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        # Truncate pattern if too long
        pattern_text = right_pattern[:20] if len(right_pattern) > 20 else right_pattern
        cv2.putText(overlay, pattern_text, (right_x + 10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
        if right_in_contact:
            cv2.putText(overlay, "CONTACT", (right_x + 10, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    
    return overlay

def detect_ground_contact(ankle_y, previous_ankle_y, previous_velocity, contact_threshold=0.003):
    """
    Detect if foot is in contact with ground based on vertical position and velocity.
    For rear view: ankle at maximum y (lowest point) = ground contact.
    
    Improved algorithm with better accuracy:
    - Uses velocity and position thresholds
    - Requires ankle to be in lower portion of frame
    - Filters out noise with stricter criteria
    
    Args:
        ankle_y: Current ankle vertical position (normalized 0-1, higher = lower in frame)
        previous_ankle_y: Previous frame ankle vertical position
        previous_velocity: Previous vertical velocity
        contact_threshold: Minimum velocity change to detect contact
    
    Returns:
        Tuple: (is_in_contact, is_foot_strike, is_toe_off)
    """
    if previous_ankle_y is None:
        return (False, False, False)
    
    # Calculate vertical velocity (positive = moving down, negative = moving up)
    # In normalized coordinates, higher y = lower in frame
    velocity = ankle_y - previous_ankle_y
    
    # Define thresholds for better accuracy
    # Ankle must be in lower 60% of frame for ground contact (more accurate than 50%)
    ground_level_threshold = 0.6
    # Require more significant velocity change for foot strike (reduces false positives)
    strike_velocity_threshold = contact_threshold * 1.5
    
    # Foot strike: velocity changes from negative/zero to positive (foot coming down to ground)
    # Require:
    # 1. Previous velocity was negative or zero (foot was going up or stationary)
    # 2. Current velocity is positive and significant (foot is descending)
    # 3. Ankle is in lower portion of frame (near ground)
    foot_strike = (previous_velocity is not None and 
                   previous_velocity <= 0.001 and  # Foot was going up or stationary
                   velocity > strike_velocity_threshold and  # Significant downward movement
                   ankle_y > ground_level_threshold)  # Ankle must be near ground
    
    # Toe-off: velocity changes from positive/zero to negative (foot lifting up from ground)
    # Require significant upward movement after being on ground
    toe_off = (previous_velocity is not None and 
               previous_velocity >= -0.001 and  # Foot was going down or stationary
               velocity < -contact_threshold)  # Significant upward movement
    
    # In contact: foot is at or near its lowest point
    # Use stricter criteria:
    # 1. Velocity is near zero (foot is stationary or moving slowly)
    # 2. Ankle is in lower portion of frame
    # 3. Previous velocity was positive (foot was descending, now stable)
    is_in_contact = (abs(velocity) < contact_threshold * 1.5 and  # Low velocity
                     ankle_y > ground_level_threshold and  # Near ground
                     (previous_velocity is None or previous_velocity >= -0.001))  # Was descending or stable
    
    return (is_in_contact, foot_strike, toe_off)

def generate_video_with_overlay(video_path: str, output_path: str, frame_data: List[Dict], 
                                left_pattern: str, right_pattern: str, avg_cadence: float, fps: float):
    """
    Generate a video with analysis overlays.
    
    Args:
        video_path: Path to original video
        output_path: Path to save processed video
        frame_data: List of dicts with frame analysis data
        left_pattern: Left leg gait pattern
        right_pattern: Right leg gait pattern
        avg_cadence: Average cadence
        fps: Video frame rate
    """
    cv2 = get_cv2()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer - use H.264 codec for web compatibility
    # Try different codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264/AVC'),  # Best for web
        ('H264', 'H.264'),      # Alternative H.264
        ('mp4v', 'MPEG-4'),     # Fallback
    ]
    
    out = None
    for codec_str, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                break
            else:
                out.release()
                out = None
        except:
            out = None
    
    if out is None or not out.isOpened():
        print(f"Warning: Could not create video writer with any codec. Video overlay may not be generated.")
        return False
    
    frame_idx = 0
    frame_data_dict = {data['frame']: data for data in frame_data}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Get frame data if available
        frame_info = frame_data_dict.get(frame_idx, {})
        left_angle = frame_info.get('left_angle')
        right_angle = frame_info.get('right_angle')
        left_in_contact = frame_info.get('left_in_contact', False)
        right_in_contact = frame_info.get('right_in_contact', False)
        
        # Draw overlay
        overlay_frame = draw_overlay(
            frame, frame_idx, left_angle, right_angle, 
            left_in_contact, right_in_contact,
            left_pattern, right_pattern, avg_cadence,
            fps, frame_width, frame_height
        )
        
        out.write(overlay_frame)
    
    cap.release()
    out.release()
    return True

def analyze_video(video_path: str, progress_callback: Optional[Callable[[int, int], None]] = None, 
                 generate_overlay_video: bool = True, overlay_output_path: Optional[str] = None) -> Dict:
    """
    Analyze a video file for gait patterns (REAR VIEW REQUIRED).
    
    The video should be recorded from behind the runner, showing both legs from the rear.
    This allows accurate measurement of pronation/supination (medial-lateral foot movement),
    ground contact time, and cadence.
    
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
    
    # Ground contact tracking
    left_ankle_positions = []  # Store (frame, ankle_y) for contact detection
    right_ankle_positions = []
    left_contact_events = []  # Store (start_frame, end_frame) tuples
    right_contact_events = []
    left_foot_strikes = []  # Frame numbers of foot strikes
    right_foot_strikes = []
    
    # Frame-by-frame data for overlay video
    frame_data_list = []
    
    # Previous frame data for velocity calculation
    prev_l_ankle_y = None
    prev_r_ankle_y = None
    prev_l_velocity = None
    prev_r_velocity = None
    in_contact_left = False
    in_contact_right = False
    contact_start_left = None
    contact_start_right = None
    
    # Current frame angle tracking
    current_left_angle = None
    current_right_angle = None
    
    # Track last timestamp for MediaPipe Tasks API (must be monotonically increasing)
    last_timestamp_ms = -1
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 fps if unavailable
    frame_time = 1.0 / fps  # Time per frame in seconds
    
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
        frame_height, frame_width, _ = frame.shape
        
        # Process with MediaPipe (handle both old and new API)
        # Check which API we're using based on the instance type
        try:
            from mediapipe.tasks.python.vision import PoseLandmarker
            is_tasks_api = isinstance(pose, PoseLandmarker)
        except ImportError:
            is_tasks_api = False
        
        if is_tasks_api:
            # Using new Tasks API
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            # Calculate timestamp in milliseconds - ensure it's strictly increasing
            # MediaPipe requires timestamps to be monotonically increasing
            timestamp_ms = int((frame_count - 1) * (1000.0 / fps))
            # Ensure timestamp is always strictly increasing
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            
            results = pose.detect_for_video(mp_image, timestamp_ms)
            # In Tasks API: results.pose_landmarks is a list of lists [pose_index][landmark_index]
            # Each landmark has .x, .y, .z, .visibility attributes directly
            landmarks = results.pose_landmarks[0] if results.pose_landmarks and len(results.pose_landmarks) > 0 else None
        else:
            # Using old solutions API
            results = pose.process(image_rgb)
            pose_landmarks = results.pose_landmarks if hasattr(results, 'pose_landmarks') else None
            landmarks = pose_landmarks.landmark if pose_landmarks else None
        
        if landmarks:
            
            # LEFT LEG
            try:
                l_knee = [landmarks[25].x, landmarks[25].y]
                l_ankle = [landmarks[27].x, landmarks[27].y]
                l_heel = [landmarks[29].x, landmarks[29].y]
                
                # Track ankle position for ground contact detection (if ankle is visible)
                if landmarks[27].visibility > 0.5:
                    left_ankle_positions.append((frame_count, l_ankle[1]))
                    
                    # Detect ground contact
                    is_contact, foot_strike, toe_off = detect_ground_contact(
                        l_ankle[1], prev_l_ankle_y, prev_l_velocity
                    )
                    
                    if foot_strike:
                        # Only count as foot strike if we're not already in contact
                        # This prevents counting multiple strikes during the same contact phase
                        if not in_contact_left:
                            left_foot_strikes.append(frame_count)
                            contact_start_left = frame_count
                            in_contact_left = True
                    
                    if toe_off and in_contact_left and contact_start_left is not None:
                        # End of contact - calculate contact time
                        contact_duration_frames = frame_count - contact_start_left
                        if contact_duration_frames > 0:
                            left_contact_events.append((contact_start_left, frame_count))
                        in_contact_left = False
                        contact_start_left = None
                    
                    # Update previous values
                    if prev_l_ankle_y is not None:
                        prev_l_velocity = l_ankle[1] - prev_l_ankle_y
                    prev_l_ankle_y = l_ankle[1]
                
                # Use higher visibility threshold for better accuracy (0.6 instead of 0.5)
                if landmarks[27].visibility > 0.6 and landmarks[29].visibility > 0.6:
                    # Calculate pronation/supination deviation angle (rear view)
                    deviation_angle_l = calculate_pronation_supination_angle(
                        l_knee, l_ankle, l_heel, frame_width, frame_height
                    )
                    # Only add valid angles (not None)
                    if deviation_angle_l is not None:
                        # For left leg in rear view: positive = pronation (heel moves right/inward), negative = supination (heel moves left/outward)
                        left_angles.append(deviation_angle_l)
                        current_left_angle = deviation_angle_l
                    else:
                        current_left_angle = None
                else:
                    current_left_angle = None
            except:
                pass
            
            # RIGHT LEG
            try:
                r_knee = [landmarks[26].x, landmarks[26].y]
                r_ankle = [landmarks[28].x, landmarks[28].y]
                r_heel = [landmarks[30].x, landmarks[30].y]
                
                # Track ankle position for ground contact detection (if ankle is visible)
                if landmarks[28].visibility > 0.5:
                    right_ankle_positions.append((frame_count, r_ankle[1]))
                    
                    # Detect ground contact
                    is_contact, foot_strike, toe_off = detect_ground_contact(
                        r_ankle[1], prev_r_ankle_y, prev_r_velocity
                    )
                    
                    if foot_strike:
                        # Only count as foot strike if we're not already in contact
                        # This prevents counting multiple strikes during the same contact phase
                        if not in_contact_right:
                            right_foot_strikes.append(frame_count)
                            contact_start_right = frame_count
                            in_contact_right = True
                    
                    if toe_off and in_contact_right and contact_start_right is not None:
                        # End of contact - calculate contact time
                        contact_duration_frames = frame_count - contact_start_right
                        if contact_duration_frames > 0:
                            right_contact_events.append((contact_start_right, frame_count))
                        in_contact_right = False
                        contact_start_right = None
                    
                    # Update previous values
                    if prev_r_ankle_y is not None:
                        prev_r_velocity = r_ankle[1] - prev_r_ankle_y
                    prev_r_ankle_y = r_ankle[1]
                
                # Use higher visibility threshold for better accuracy (0.6 instead of 0.5)
                if landmarks[28].visibility > 0.6 and landmarks[30].visibility > 0.6:
                    # Calculate pronation/supination deviation angle (rear view)
                    deviation_angle_r = calculate_pronation_supination_angle(
                        r_knee, r_ankle, r_heel, frame_width, frame_height
                    )
                    # Only add valid angles (not None)
                    if deviation_angle_r is not None:
                        # For right leg in rear view: negative horizontal (heel left of ankle) = pronation (inward)
                        # Flip sign so positive = pronation for both legs (consistent convention)
                        deviation_angle_r = -deviation_angle_r
                        right_angles.append(deviation_angle_r)
                        current_right_angle = deviation_angle_r
                    else:
                        current_right_angle = None
                else:
                    current_right_angle = None
            except:
                pass
        
        # Store frame data for overlay
        frame_data_list.append({
            'frame': frame_count,
            'left_angle': current_left_angle,
            'right_angle': current_right_angle,
            'left_in_contact': in_contact_left,
            'right_in_contact': in_contact_right
        })
    
    cap.release()
    
    # Calculate ground contact times
    left_gct_times = []
    right_gct_times = []
    
    # Filter and validate ground contact times for accuracy
    for start_frame, end_frame in left_contact_events:
        contact_time_ms = (end_frame - start_frame) * frame_time * 1000  # Convert to milliseconds
        # Filter out noise and unrealistic values
        # Normal GCT for running: 150-300ms, walking: 500-800ms
        # Accept range: 50ms to 1000ms (very conservative)
        if 50 <= contact_time_ms <= 1000:
            left_gct_times.append(contact_time_ms)
    
    for start_frame, end_frame in right_contact_events:
        contact_time_ms = (end_frame - start_frame) * frame_time * 1000  # Convert to milliseconds
        # Filter out noise and unrealistic values
        if 50 <= contact_time_ms <= 1000:
            right_gct_times.append(contact_time_ms)
    
    # Calculate cadence with improved accuracy
    video_duration_seconds = frame_count * frame_time
    total_steps = len(left_foot_strikes) + len(right_foot_strikes)
    cadence = 0
    
    # Only calculate cadence if we have enough data and reasonable duration
    if video_duration_seconds > 1.0 and total_steps >= 2:  # At least 1 second and 2 steps
        cadence = (total_steps / video_duration_seconds) * 60  # Steps per minute
        # Filter unrealistic cadence values (normal running: 160-180 spm, walking: 100-120 spm)
        # Accept range: 60-300 spm (very conservative)
        if cadence < 60 or cadence > 300:
            # If cadence is unrealistic, recalculate using only valid foot strikes
            # Filter strikes that are too close together (minimum 0.2 seconds apart)
            min_frames_between_strikes = int(0.2 * fps)  # 0.2 seconds minimum
            valid_left_strikes = []
            valid_right_strikes = []
            
            for i, strike in enumerate(left_foot_strikes):
                if i == 0 or strike - left_foot_strikes[i-1] >= min_frames_between_strikes:
                    valid_left_strikes.append(strike)
            
            for i, strike in enumerate(right_foot_strikes):
                if i == 0 or strike - right_foot_strikes[i-1] >= min_frames_between_strikes:
                    valid_right_strikes.append(strike)
            
            total_valid_steps = len(valid_left_strikes) + len(valid_right_strikes)
            if total_valid_steps >= 2:
                cadence = (total_valid_steps / video_duration_seconds) * 60
                # Update foot strikes lists for reporting
                left_foot_strikes = valid_left_strikes
                right_foot_strikes = valid_right_strikes
            else:
                cadence = 0  # Not enough valid data
    
    # Process results
    result = {
        "left_leg": None,
        "right_leg": None,
        "recommendation": None,
        "asymmetry_detected": False,
        "total_frames": frame_count,
        "frames_analyzed": len(left_angles) + len(right_angles),
        "cadence": {
            "steps_per_minute": round(cadence, 1),
            "left_steps": len(left_foot_strikes),
            "right_steps": len(right_foot_strikes),
            "total_steps": total_steps,
            "video_duration_seconds": round(video_duration_seconds, 2)
        },
        "ground_contact_time": {
            "left": None,
            "right": None,
            "asymmetry_ms": None
        }
    }
    
    # Analyze left leg with improved accuracy
    if left_angles and len(left_angles) >= 5:  # Require minimum 5 valid measurements
        # Remove outliers using IQR method for more accurate results
        sorted_angles = sorted(left_angles)
        q1_idx = len(sorted_angles) // 4
        q3_idx = (3 * len(sorted_angles)) // 4
        q1 = sorted_angles[q1_idx]
        q3 = sorted_angles[q3_idx]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out outliers
        filtered_angles = [a for a in left_angles if lower_bound <= a <= upper_bound]
        
        if len(filtered_angles) >= 3:  # Need at least 3 after filtering
            # Use median for more robust estimation (less affected by outliers)
            sorted_filtered = sorted(filtered_angles)
            median_angle_l = sorted_filtered[len(sorted_filtered) // 2]
            avg_angle_l = sum(filtered_angles) / len(filtered_angles)
            min_angle_l = min(filtered_angles)
            max_angle_l = max(filtered_angles)
            # Use median for classification (more accurate than mean)
            pattern_l, severity_l = classify_gait_pattern(median_angle_l)
        else:
            # Fallback to mean if not enough data after filtering
            avg_angle_l = sum(left_angles) / len(left_angles)
            min_angle_l = min(left_angles)
            max_angle_l = max(left_angles)
            pattern_l, severity_l = classify_gait_pattern(avg_angle_l)
            filtered_angles = left_angles
    elif left_angles:
        # Not enough data - use simple average
        avg_angle_l = sum(left_angles) / len(left_angles)
        min_angle_l = min(left_angles)
        max_angle_l = max(left_angles)
        pattern_l, severity_l = classify_gait_pattern(avg_angle_l)
        filtered_angles = left_angles
    else:
        avg_angle_l = None
        min_angle_l = None
        max_angle_l = None
        pattern_l = None
        severity_l = 0
        filtered_angles = []
    
    # Set left leg result if we have data
    if left_angles:
        result["left_leg"] = {
            "angles": {
                "min": float(min_angle_l),
                "max": float(max_angle_l),
                "avg": float(avg_angle_l),
                "median": float(sorted(filtered_angles)[len(filtered_angles) // 2]) if filtered_angles else float(avg_angle_l),
                "deviation": float(sorted(filtered_angles)[len(filtered_angles) // 2]) if filtered_angles else float(avg_angle_l),  # Use median for main deviation
                "all": [float(a) for a in filtered_angles] if filtered_angles else [float(a) for a in left_angles]
            },
            "pattern": pattern_l,
            "severity": severity_l,
            "warning": severity_l >= 3
        }
    
    # Analyze right leg with improved accuracy
    if right_angles and len(right_angles) >= 5:  # Require minimum 5 valid measurements
        # Remove outliers using IQR method for more accurate results
        sorted_angles = sorted(right_angles)
        q1_idx = len(sorted_angles) // 4
        q3_idx = (3 * len(sorted_angles)) // 4
        q1 = sorted_angles[q1_idx]
        q3 = sorted_angles[q3_idx]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out outliers
        filtered_angles_r = [a for a in right_angles if lower_bound <= a <= upper_bound]
        
        if len(filtered_angles_r) >= 3:  # Need at least 3 after filtering
            # Use median for more robust estimation (less affected by outliers)
            sorted_filtered = sorted(filtered_angles_r)
            median_angle_r = sorted_filtered[len(sorted_filtered) // 2]
            avg_angle_r = sum(filtered_angles_r) / len(filtered_angles_r)
            min_angle_r = min(filtered_angles_r)
            max_angle_r = max(filtered_angles_r)
            # Use median for classification (more accurate than mean)
            pattern_r, severity_r = classify_gait_pattern(median_angle_r)
        else:
            # Fallback to mean if not enough data after filtering
            avg_angle_r = sum(right_angles) / len(right_angles)
            min_angle_r = min(right_angles)
            max_angle_r = max(right_angles)
            pattern_r, severity_r = classify_gait_pattern(avg_angle_r)
            filtered_angles_r = right_angles
    elif right_angles:
        # Not enough data - use simple average
        avg_angle_r = sum(right_angles) / len(right_angles)
        min_angle_r = min(right_angles)
        max_angle_r = max(right_angles)
        pattern_r, severity_r = classify_gait_pattern(avg_angle_r)
        filtered_angles_r = right_angles
    else:
        avg_angle_r = None
        min_angle_r = None
        max_angle_r = None
        pattern_r = None
        severity_r = 0
        filtered_angles_r = []
    
    # Set right leg result if we have data
    if right_angles:
        result["right_leg"] = {
            "angles": {
                "min": float(min_angle_r),
                "max": float(max_angle_r),
                "avg": float(avg_angle_r),
                "median": float(sorted(filtered_angles_r)[len(filtered_angles_r) // 2]) if filtered_angles_r else float(avg_angle_r),
                "deviation": float(sorted(filtered_angles_r)[len(filtered_angles_r) // 2]) if filtered_angles_r else float(avg_angle_r),  # Use median for main deviation
                "all": [float(a) for a in filtered_angles_r] if filtered_angles_r else [float(a) for a in right_angles]
            },
            "pattern": pattern_r,
            "severity": severity_r,
            "warning": severity_r >= 3
        }
    
    # Process ground contact time data
    if left_gct_times:
        avg_gct_l = sum(left_gct_times) / len(left_gct_times)
        min_gct_l = min(left_gct_times)
        max_gct_l = max(left_gct_times)
        result["ground_contact_time"]["left"] = {
            "average_ms": round(avg_gct_l, 1),
            "min_ms": round(min_gct_l, 1),
            "max_ms": round(max_gct_l, 1),
            "contacts": len(left_gct_times)
        }
    
    if right_gct_times:
        avg_gct_r = sum(right_gct_times) / len(right_gct_times)
        min_gct_r = min(right_gct_times)
        max_gct_r = max(right_gct_times)
        result["ground_contact_time"]["right"] = {
            "average_ms": round(avg_gct_r, 1),
            "min_ms": round(min_gct_r, 1),
            "max_ms": round(max_gct_r, 1),
            "contacts": len(right_gct_times)
        }
    
    # Calculate GCT asymmetry
    if left_gct_times and right_gct_times:
        avg_gct_l = sum(left_gct_times) / len(left_gct_times)
        avg_gct_r = sum(right_gct_times) / len(right_gct_times)
        result["ground_contact_time"]["asymmetry_ms"] = round(abs(avg_gct_l - avg_gct_r), 1)
    
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
            "description": "Insufficient data collected. Please ensure the video is recorded from the rear view and clearly shows both legs while running."
        }
    
    # Generate overlay video if requested
    if generate_overlay_video and overlay_output_path and frame_data_list:
        try:
            left_pattern_str = pattern_l if left_angles else "N/A"
            right_pattern_str = pattern_r if right_angles else "N/A"
            success = generate_video_with_overlay(
                video_path, overlay_output_path, frame_data_list,
                left_pattern_str, right_pattern_str, cadence, fps
            )
            if success and os.path.exists(overlay_output_path):
                # Check file size to ensure video was actually written
                file_size = os.path.getsize(overlay_output_path)
                if file_size > 1000:  # At least 1KB
                    result["overlay_video_path"] = overlay_output_path
                else:
                    print(f"Warning: Overlay video file too small ({file_size} bytes), may be corrupted")
            else:
                print(f"Warning: Failed to generate overlay video or file not found")
        except Exception as e:
            print(f"Error generating overlay video: {str(e)}")
            # Don't fail the entire analysis if video generation fails
    
    return result

