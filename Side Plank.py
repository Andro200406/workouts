import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Variables for side plank count
side_plank_detected = False
hold_time = 0
start_time = None

# Video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize the Mediapipe Pose model
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Pose Detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates of key points (shoulders, hips, ankles)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate body alignment angles
            left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # Detect Side Plank
            if 160 < left_body_angle < 180 or 160 < right_body_angle < 180:  # Check if body is aligned
                hip_height = left_hip[1] if left_body_angle > right_body_angle else right_hip[1]
                
                # Check if hips are elevated above a certain threshold
                if hip_height < 0.5:  # Assuming normalized coordinates
                    if not side_plank_detected:
                        side_plank_detected = True
                        start_time = cv2.getTickCount()  # Record start time
                    else:
                        # Calculate hold time
                        hold_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                else:
                    side_plank_detected = False
            else:
                side_plank_detected = False

            # Display Side Plank hold time
            cv2.putText(image, f'Side Plank Time: {hold_time:.2f} sec', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Side Plank Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
