import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting jackknife sit-ups
count = 0
in_situp_position = False

# Function to check if the person is performing a jackknife sit-up
def is_in_jackknife_situp(landmarks):
    # Get the positions of key points: Shoulders, Hips, and Ankles
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate the angle between the shoulders and hips (used to determine torso lift)
    torso_angle = np.arctan2(left_hip[1] - left_shoulder[1], left_hip[0] - left_shoulder[0]) - np.arctan2(right_hip[1] - right_shoulder[1], right_hip[0] - right_shoulder[0])

    # Calculate the angle between the hips and ankles (used to determine leg lift)
    leg_angle = np.arctan2(left_ankle[1] - left_hip[1], left_ankle[0] - left_hip[0]) - np.arctan2(right_ankle[1] - right_hip[1], right_ankle[0] - right_hip[0])

    # Threshold for detecting torso and leg lift (can be adjusted based on performance)
    torso_threshold = 0.4
    leg_threshold = 0.4

    # If both torso and legs are lifted beyond a certain threshold, we detect a jackknife sit-up
    return abs(torso_angle) > torso_threshold and abs(leg_angle) > leg_threshold

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
            
            # Check if the person is performing a jackknife sit-up
            if is_in_jackknife_situp(landmarks):
                if not in_situp_position:
                    in_situp_position = True
                    # This indicates a jackknife sit-up movement
            elif in_situp_position:
                in_situp_position = False
                count += 1  # Count the jackknife sit-up movement

            # Display the count on the frame
            cv2.putText(image, f'Jackknife Sit-up Count: {count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Jackknife Sit-up Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
