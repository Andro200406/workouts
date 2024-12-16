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

# Variables for low plank detection
low_plank_detected = False
plank_time = 0
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
            
            # Get coordinates of key points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles for body alignment
            left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # Ensure elbows are directly under shoulders
            left_elbow_under_shoulder = abs(left_shoulder[0] - left_elbow[0]) < 0.05
            right_elbow_under_shoulder = abs(right_shoulder[0] - right_elbow[0]) < 0.05
            
            # Check if the body is aligned (angles close to 180 degrees)
            if 160 < left_body_angle < 180 and 160 < right_body_angle < 180 and left_elbow_under_shoulder and right_elbow_under_shoulder:
                if not low_plank_detected:
                    low_plank_detected = True
                    start_time = cv2.getTickCount()  # Start timing
                else:
                    # Calculate plank hold time
                    plank_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            else:
                low_plank_detected = False

            # Display Low Plank hold time
            cv2.putText(image, f'Low Plank Time: {plank_time:.2f} sec', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Low Plank Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
