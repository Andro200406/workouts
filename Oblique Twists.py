import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for tracking oblique twists
count = 0
in_twist_position = False

# Function to check if the person is performing an oblique twist
def is_in_oblique_twist(landmarks):
    # Get the positions of key points: Shoulders, Hips
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    # Calculate the angle between the shoulders and hips (used to determine torso twist)
    angle_left = np.arctan2(left_hip[1] - left_shoulder[1], left_hip[0] - left_shoulder[0])
    angle_right = np.arctan2(right_hip[1] - right_shoulder[1], right_hip[0] - right_shoulder[0])
    
    # Threshold for determining the angle of torso twist (you can adjust this)
    twist_threshold = 0.3  # Angle difference threshold for detecting a twist

    return abs(angle_left - angle_right) > twist_threshold

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
            
            # Check if the person is performing an oblique twist
            if is_in_oblique_twist(landmarks):
                if not in_twist_position:
                    in_twist_position = True
                    # This indicates an oblique twist movement
            elif in_twist_position:
                in_twist_position = False
                count += 1  # Count the oblique twist movement

            # Display the count on the frame
            cv2.putText(image, f'Oblique Twist Count: {count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Oblique Twist Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
