import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Side Lunges
count = 0
in_lunge = False
last_lunge_time = 0

# Function to check if the person is performing a Side Lunge
def is_in_side_lunge(landmarks):
    # Get positions of key points: Hips, Knees, Ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Angle between the hips, knees, and ankles of the lunging leg
    left_knee_angle = np.arctan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0])
    right_knee_angle = np.arctan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0])

    # Side Lunge check: Knee should bend outward while other leg stays straight
    lunge_depth = 0.8  # Adjust the threshold for sufficient lunge depth
    is_left_leg_in_lunge = left_knee_angle > lunge_depth
    is_right_leg_in_lunge = right_knee_angle > lunge_depth

    # Detect side lunge based on one leg bending while the other leg stays straight
    return is_left_leg_in_lunge or is_right_leg_in_lunge

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

            # Check if the person is performing Side Lunges
            if is_in_side_lunge(landmarks):
                if not in_lunge:
                    in_lunge = True
                    last_lunge_time = cv2.getTickCount()
            elif in_lunge:
                in_lunge = False
                duration = (cv2.getTickCount() - last_lunge_time) / cv2.getTickFrequency()
                if duration > 0.5:  # Ensure a significant hold position before counting
                    count += 1  # Count the Side Lunge movement

            # Display the count on the frame
            cv2.putText(image, f'Side Lunges: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Side Lunge Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
