import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Pistol Squats
count = 0
in_squat = False
last_squat_time = 0

# Function to check if the person is performing Pistol Squats
def is_in_pistol_squat(landmarks):
    # Get positions of key points: Hips, Knees, Ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Check if one leg is extended
    is_left_leg_extended = left_knee[1] > left_ankle[1]  # Check if the left leg is extended (no contact with ground)
    is_right_leg_extended = right_knee[1] > right_ankle[1]  # Check if the right leg is extended (no contact with ground)
    
    # Calculate squat depth (similar to a normal squat)
    squat_depth = 0.6  # Set a threshold for squat depth
    knee_angle_left = np.arctan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0])
    knee_angle_right = np.arctan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0])
    
    # Detect if one leg is in pistol squat position
    return (is_left_leg_extended or is_right_leg_extended) and (knee_angle_left < squat_depth or knee_angle_right < squat_depth)

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

            # Check if the person is performing Pistol Squats
            if is_in_pistol_squat(landmarks):
                if not in_squat:
                    in_squat = True
                    last_squat_time = cv2.getTickCount()
            elif in_squat:
                in_squat = False
                duration = (cv2.getTickCount() - last_squat_time) / cv2.getTickFrequency()
                if duration > 0.5:  # Ensure a significant hold position before counting
                    count += 1  # Count the Pistol Squat movement

            # Display the count on the frame
            cv2.putText(image, f'Pistol Squats Count: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Pistol Squat Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
