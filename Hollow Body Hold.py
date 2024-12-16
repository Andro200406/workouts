import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for Hollow Body Hold tracking
hold_start_time = None
holding = False
max_hold_time = 0

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
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip_center = [
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            ]
            lower_back = [
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 + 0.1
            ]
            
            # Thresholds for hold detection
            leg_elevation_threshold = 0.1  # Adjust for leg elevation sensitivity
            arm_elevation_threshold = 0.1  # Adjust for arm elevation sensitivity
            back_flat_threshold = 0.05  # Ensure back remains flat
            
            # Check leg and arm elevation and back position
            legs_elevated = (left_ankle[1] < hip_center[1] + leg_elevation_threshold and
                             right_ankle[1] < hip_center[1] + leg_elevation_threshold)
            arms_elevated = (left_wrist[1] < hip_center[1] + arm_elevation_threshold and
                             right_wrist[1] < hip_center[1] + arm_elevation_threshold)
            back_flat = lower_back[1] > hip_center[1] - back_flat_threshold

            if legs_elevated and arms_elevated and back_flat:
                if not holding:
                    hold_start_time = time.time()
                    holding = True
                else:
                    current_hold_time = time.time() - hold_start_time
                    max_hold_time = max(max_hold_time, current_hold_time)
            else:
                holding = False
                hold_start_time = None

            # Display hold time on the frame
            hold_time_display = max_hold_time if holding else 0
            cv2.putText(image, f'Hollow Body Hold: {hold_time_display:.2f}s', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Hollow Body Hold Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
