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

# Variables for counting Fingertip push-ups
fingertip_push_up_count = 0
push_up_state = "down"  # Track whether the user is in the "down" or "up" position
fingertip_threshold = 0.1  # Threshold for finger placement (adjust based on hand position)

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
            
            # Get coordinates of key points (wrist, fingers, elbows)
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_index_finger = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            right_index_finger = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
            
            # Check if fingertips are touching the ground
            left_finger_contact = left_index_finger[1] > 0.8  # Adjust this threshold based on your camera positioning
            right_finger_contact = right_index_finger[1] > 0.8  # Same as above

            # Detect push-up phases
            body_angle = calculate_angle(left_wrist, left_index_finger, right_index_finger)  # Check wrist/finger angle for better push-up alignment

            # Count fingertip push-ups based on finger contact and body angle
            if left_finger_contact and right_finger_contact:
                if push_up_state == "down" and body_angle > 160:  # Modify body angle threshold as needed
                    fingertip_push_up_count += 1
                    push_up_state = "up"
            elif body_angle < 160:
                push_up_state = "down"
                
            # Display Fingertip Push-up count on the frame
            cv2.putText(image, f'Fingertip Push-ups: {fingertip_push_up_count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Fingertip Push-up Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
