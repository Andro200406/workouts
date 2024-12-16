import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Jump Squats
count = 0
in_squat = False
last_squat_time = 0

# Function to check if the person is performing Jump Squats
def is_in_jump_squat(landmarks):
    # Get positions of key points: Hips, Knees, and Ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    # Calculate squat depth (thigh parallel to the ground) and landing
    squat_depth = 0.6  # Set a threshold where knees bend to at least 60% depth
    knee_angle = np.arctan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0])

    # Detect if the person is in the squat position
    return knee_angle < squat_depth

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

            # Check if the person is performing Jump Squats
            if is_in_jump_squat(landmarks):
                if not in_squat:
                    in_squat = True
                    last_squat_time = cv2.getTickCount()
            elif in_squat:
                in_squat = False
                duration = (cv2.getTickCount() - last_squat_time) / cv2.getTickFrequency()
                if duration > 0.5:  # Ensure a significant hold position before counting
                    count += 1  # Count the Jump Squat movement

            # Display the count on the frame
            cv2.putText(image, f'Jump Squats Count: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Jump Squat Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
