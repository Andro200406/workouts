import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate vertical distance
def vertical_distance(y1, y2):
    return abs(y2 - y1)

# Variables for counting calf raises
calf_raise_count = 0
is_calf_raise = False

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

            # Get coordinates of ankle and heel points
            left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
            right_heel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y

            # Calculate vertical displacement between ankle and heel
            left_displacement = vertical_distance(left_ankle_y, left_heel_y)
            right_displacement = vertical_distance(right_ankle_y, right_heel_y)

            # Threshold to detect calf raise
            threshold = 0.02  # Adjust based on camera position and person height

            # Check if both heels are raised significantly
            if left_displacement > threshold and right_displacement > threshold:
                if not is_calf_raise:
                    is_calf_raise = True
                    calf_raise_count += 1
            else:
                is_calf_raise = False

            # Display calf raise count on the frame
            cv2.putText(image, f'Calf Raises: {calf_raise_count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Calf Raise Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
