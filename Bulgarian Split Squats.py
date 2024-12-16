import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Bulgarian Split Squats
count = 0
in_squat = False
last_squat_time = 0

# Function to check if the person is performing Bulgarian Split Squats
def is_in_bulgarian_split_squat(landmarks):
    # Get positions of key points: Hips, Knees, Ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Check if the grounded leg knee is bent sufficiently (typically a 90-degree angle for good form)
    knee_angle_left = np.arctan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0])
    knee_angle_right = np.arctan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0])
    
    # Detect if the person is in a good squat depth (90-degree bend for grounded leg)
    squat_depth = 0.7  # Adjust this threshold as needed (90-degree knee bend is around 0.7)
    
    # Check that one leg is elevated (assuming the elevated leg knee is slightly bent but not touching the ground)
    is_left_leg_elevated = left_ankle[1] > left_knee[1]  # Left leg elevated
    is_right_leg_elevated = right_ankle[1] > right_knee[1]  # Right leg elevated
    
    return (is_left_leg_elevated or is_right_leg_elevated) and (knee_angle_left < squat_depth or knee_angle_right < squat_depth)

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

            # Check if the person is performing Bulgarian Split Squats
            if is_in_bulgarian_split_squat(landmarks):
                if not in_squat:
                    in_squat = True
                    last_squat_time = cv2.getTickCount()
            elif in_squat:
                in_squat = False
                duration = (cv2.getTickCount() - last_squat_time) / cv2.getTickFrequency()
                if duration > 0.5:  # Ensure a significant hold position before counting
                    count += 1  # Count the Bulgarian Split Squat movement

            # Display the count on the frame
            cv2.putText(image, f'Bulgarian Split Squats: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Bulgarian Split Squat Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
