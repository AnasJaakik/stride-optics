#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np

# --- SETUP ---
video_path = '../samples/test3.mp4'

# Initialize MediaPipe Pose
# CHANGED: Lowered detection_confidence to 0.3 to catch legs even when blurry
# CHANGED: model_complexity=1 is sometimes more stable for fast motion than 2
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)

# --- CONFIGURATION ---
BLUE_NEON = (255, 200, 0)
RED_NEON = (50, 50, 255)
WHITE = (255, 255, 255)
GREY_DARK = (30, 30, 30)
GREEN_OK = (50, 255, 50)
YELLOW_WARN = (0, 215, 255)
ORANGE_WARN = (0, 165, 255)
CYAN_INFO = (255, 255, 0)

# Gait pattern thresholds (in degrees)
SEVERE_OVERPRONATION_THRESHOLD = 150
OVERPRONATION_THRESHOLD = 170
NEUTRAL_MAX = 190
SUPINATION_THRESHOLD = 210

left_angles = []
right_angles = []
frame_count = 0

# --- NEW HELPER: CONTRAST BOOSTER ---
def enhance_contrast(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This helps the AI see 'black shoes on black treadmill'.
    """
    # Convert to LAB color space (Lightness, A, B)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel (Lightness) only
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def classify_gait_pattern(angle):
    """
    Classifies gait pattern based on ankle angle.
    Returns: (pattern_name, color, severity_level)
    """
    if angle < SEVERE_OVERPRONATION_THRESHOLD:
        return ("SEVERE OVERPRONATION", ORANGE_WARN, 4)
    elif angle < OVERPRONATION_THRESHOLD:
        return ("OVERPRONATION", YELLOW_WARN, 3)
    elif angle <= NEUTRAL_MAX:
        return ("NEUTRAL", GREEN_OK, 1)
    elif angle <= SUPINATION_THRESHOLD:
        return ("SUPINATION", YELLOW_WARN, 2)
    else:
        return ("SEVERE SUPINATION", ORANGE_WARN, 4)

def draw_slick_dashboard(img, l_angle, r_angle, f_num):
    h, w, c = img.shape
    overlay = img.copy()
    panel_width = int(w * 0.25)
    
    cv2.rectangle(overlay, (0, 0), (panel_width, h), GREY_DARK, -1)
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), GREY_DARK, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    font_main = cv2.FONT_HERSHEY_DUPLEX
    font_sub = cv2.FONT_HERSHEY_SIMPLEX
    
    cx_l = int(panel_width / 2)
    draw_centered_text(img, "LEFT", (cx_l, 50), font_main, 0.8, BLUE_NEON)
    
    if l_angle:
        draw_centered_text(img, f"{int(l_angle)}", (cx_l, 140), font_main, 2.5, BLUE_NEON, 3)
        draw_centered_text(img, "deg", (cx_l, 180), font_sub, 0.6, WHITE)
        status, s_color, _ = classify_gait_pattern(l_angle)
        # Split long status text into two lines if needed
        if len(status) > 12:
            words = status.split()
            if len(words) == 2:
                draw_centered_text(img, words[0], (cx_l, 230), font_sub, 0.5, s_color, 2)
                draw_centered_text(img, words[1], (cx_l, 255), font_sub, 0.5, s_color, 2)
            else:
                draw_centered_text(img, status, (cx_l, 252), font_sub, 0.5, s_color, 2)
        else:
            draw_centered_text(img, status, (cx_l, 252), font_sub, 0.6, s_color, 2)
        cv2.rectangle(img, (20, 220), (panel_width-20, 280), s_color, 2)

    cx_r = w - int(panel_width / 2)
    draw_centered_text(img, "RIGHT", (cx_r, 50), font_main, 0.8, RED_NEON)
    
    if r_angle:
        draw_centered_text(img, f"{int(r_angle)}", (cx_r, 140), font_main, 2.5, RED_NEON, 3)
        draw_centered_text(img, "deg", (cx_r, 180), font_sub, 0.6, WHITE)
        status, s_color, _ = classify_gait_pattern(r_angle)
        # Split long status text into two lines if needed
        if len(status) > 12:
            words = status.split()
            if len(words) == 2:
                draw_centered_text(img, words[0], (cx_r, 230), font_sub, 0.5, s_color, 2)
                draw_centered_text(img, words[1], (cx_r, 255), font_sub, 0.5, s_color, 2)
            else:
                draw_centered_text(img, status, (cx_r, 252), font_sub, 0.5, s_color, 2)
        else:
            draw_centered_text(img, status, (cx_r, 252), font_sub, 0.6, s_color, 2)
        cv2.rectangle(img, (w-panel_width+20, 220), (w-20, 280), s_color, 2)

    cv2.putText(img, f"FRAME: {f_num}", (w//2 - 50, 30), font_sub, 0.6, WHITE, 1)
    return img

def draw_centered_text(img, text, center_point, font, scale, color, thickness=1):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = center_point[0] - (text_size[0] // 2)
    text_y = center_point[1] + (text_size[1] // 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness)

# --- MAIN ENGINE ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
    exit()

cv2.namedWindow('Slick Gait Engine', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Slick Gait Engine', 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # 1. ENHANCE CONTRAST (The Fix)
    # We feed the 'enhanced' image to AI, but draw on the 'original' frame
    ai_frame = enhance_contrast(frame)
    image_rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)
    frame_height, frame_width, _ = frame.shape
    
    current_angle_l = None
    current_angle_r = None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # DRAW ALL LANDMARKS (DEBUG MODE)
        # Uncomment the line below if you want to see the full skeleton for debugging
        # mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # LEFT LEG
        try:
            l_knee = [landmarks[25].x, landmarks[25].y]
            l_ankle = [landmarks[27].x, landmarks[27].y]
            l_heel = [landmarks[29].x, landmarks[29].y]
            
            # Confidence Check: If AI is unsure where the foot is (visibility < 0.5), ignore it
            if landmarks[27].visibility > 0.5:
                raw_angle_l = calculate_angle(l_knee, l_ankle, l_heel)
                if raw_angle_l > 140:
                    current_angle_l = raw_angle_l
                    left_angles.append(raw_angle_l)
                    
                    lk_px = tuple(np.multiply(l_knee, [frame_width, frame_height]).astype(int))
                    la_px = tuple(np.multiply(l_ankle, [frame_width, frame_height]).astype(int))
                    lh_px = tuple(np.multiply(l_heel, [frame_width, frame_height]).astype(int))
                    cv2.line(frame, lk_px, la_px, BLUE_NEON, 3)
                    cv2.line(frame, la_px, lh_px, BLUE_NEON, 3)
                    cv2.circle(frame, la_px, 8, BLUE_NEON, -1)
        except: pass

        # RIGHT LEG
        try:
            r_knee = [landmarks[26].x, landmarks[26].y]
            r_ankle = [landmarks[28].x, landmarks[28].y]
            r_heel = [landmarks[30].x, landmarks[30].y]
            
            if landmarks[28].visibility > 0.5:
                raw_angle_r = calculate_angle(r_knee, r_ankle, r_heel)
                if raw_angle_r > 140:
                    current_angle_r = raw_angle_r
                    right_angles.append(raw_angle_r)
                    
                    rk_px = tuple(np.multiply(r_knee, [frame_width, frame_height]).astype(int))
                    ra_px = tuple(np.multiply(r_ankle, [frame_width, frame_height]).astype(int))
                    rh_px = tuple(np.multiply(r_heel, [frame_width, frame_height]).astype(int))
                    cv2.line(frame, rk_px, ra_px, RED_NEON, 3)
                    cv2.line(frame, ra_px, rh_px, RED_NEON, 3)
                    cv2.circle(frame, ra_px, 8, RED_NEON, -1)
        except: pass

    final_frame = draw_slick_dashboard(frame, current_angle_l, current_angle_r, frame_count)
    cv2.imshow('Slick Gait Engine', final_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# FINAL ANALYSIS
print("-" * 50)
print("ANALYSIS COMPLETE")
print("-" * 50)

# Analyze left leg
if left_angles:
    min_angle_l = min(left_angles)
    max_angle_l = max(left_angles)
    avg_angle_l = sum(left_angles) / len(left_angles)
    pattern_l, _, severity_l = classify_gait_pattern(min_angle_l)
    
    print(f"\nLEFT LEG ANALYSIS:")
    print(f"  Minimum Angle: {int(min_angle_l)}°")
    print(f"  Maximum Angle: {int(max_angle_l)}°")
    print(f"  Average Angle: {int(avg_angle_l)}°")
    print(f"  Detected Pattern: {pattern_l}")
    
    if severity_l >= 3:
        print(f"  ⚠️  WARNING: Abnormal gait pattern detected")
    else:
        print(f"\nLEFT LEG: No data collected")

# Analyze right leg
if right_angles:
    min_angle_r = min(right_angles)
    max_angle_r = max(right_angles)
    avg_angle_r = sum(right_angles) / len(right_angles)
    pattern_r, _, severity_r = classify_gait_pattern(min_angle_r)
    
    print(f"\nRIGHT LEG ANALYSIS:")
    print(f"  Minimum Angle: {int(min_angle_r)}°")
    print(f"  Maximum Angle: {int(max_angle_r)}°")
    print(f"  Average Angle: {int(avg_angle_r)}°")
    print(f"  Detected Pattern: {pattern_r}")
    
    if severity_r >= 3:
        print(f"  ⚠️  WARNING: Abnormal gait pattern detected")
    else:
        print(f"\nRIGHT LEG: No data collected")

# Overall recommendation
print(f"\n" + "-" * 50)
print("RECOMMENDATIONS:")
print("-" * 50)

def get_shoe_recommendation(pattern):
    """Returns shoe recommendation based on gait pattern"""
    if "SEVERE OVERPRONATION" in pattern:
        return ("Maximum Stability Shoes", 
                "Examples: Brooks Beast, Asics Gel-Kayano, New Balance 1340v3")
    elif "OVERPRONATION" in pattern:
        return ("Stability Shoes", 
                "Examples: Brooks Adrenaline, Asics GT-2000, Saucony Guide")
    elif "NEUTRAL" in pattern:
        return ("Neutral Cushioned Shoes", 
                "Examples: Brooks Ghost, Nike Pegasus, Asics Gel-Nimbus")
    elif "SUPINATION" in pattern:
        return ("Neutral Cushioned Shoes with Extra Padding", 
                "Examples: Brooks Glycerin, Hoka Clifton, Asics Gel-Cumulus")
    elif "SEVERE SUPINATION" in pattern:
        return ("Maximum Cushioning Shoes", 
                "Examples: Hoka Bondi, Brooks Glycerin, Asics Gel-Nimbus")
    return ("Consult a podiatrist", "Professional assessment recommended")

if left_angles and right_angles:
    # Use the more severe pattern for recommendation
    if severity_l >= severity_r:
        rec_type, rec_examples = get_shoe_recommendation(pattern_l)
    else:
        rec_type, rec_examples = get_shoe_recommendation(pattern_r)
    
    print(f"Primary Recommendation: {rec_type}")
    print(f"  {rec_examples}")
    
    # Check for asymmetry
    if abs(severity_l - severity_r) >= 2:
        print(f"\n⚠️  ASYMMETRY DETECTED:")
        print(f"  Left leg: {pattern_l}")
        print(f"  Right leg: {pattern_r}")
        print(f"  Consider consulting a podiatrist for custom orthotics")
        
elif left_angles:
    rec_type, rec_examples = get_shoe_recommendation(pattern_l)
    print(f"Recommendation: {rec_type}")
    print(f"  {rec_examples}")
elif right_angles:
    rec_type, rec_examples = get_shoe_recommendation(pattern_r)
    print(f"Recommendation: {rec_type}")
    print(f"  {rec_examples}")
else:
    print("No gait data available for recommendations")

print("-" * 50)