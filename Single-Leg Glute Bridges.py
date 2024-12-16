import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Single-leg Glute Bridges
count = 0
in_bridge = False
last_bridge_time = 0

# Function to check if the person is in the Single-leg Glute Bridge position
def is_in_single_leg_glute_bridge(landmarks):
    # Get positions of key points: Shoulders, Hips, Knees, Ankles
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Check if the body forms a straight line from shoulders to hips (like a normal glute bridge)
    angle_hips = np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0])
    angle_shoulders = np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0])

    # Detect if one leg is elevated and hips are raised
    is_hips_elevated = abs(left_hip[1] - right_hip[1]) > 0.2  # Adjust this threshold to detect when hips are raised
    is_body_in_line = abs(angle_hips - angle_shoulders) < 0.3  # Ensure the body is in a straight line

    # Check if one knee is bent and the other leg is extended (knee angle > 45 degrees for extended leg)
    leg_angle_left = np.arctan2(left_knee[1] - left_ankle[1], left_knee[0] - left_ankle[0])
    leg_angle_right = np.arctan2(right_knee[1] - right_ankle[1], right_knee[0] - right_ankle[0])

    # One leg should be bent and the other extended
    is_left_leg_bent = abs(leg_angle_left) < 0.8
    is_right_leg_bent = abs(leg_angle_right) < 0.8
    is_left_leg_extended = abs(leg_angle_left) > 0.8
    is_right_leg_extended = abs(leg_angle_right) > 0.8

    return (is_hips_elevated and is_body_in_line) and (is_left_leg_bent and is_right_leg_extended) or (is_left_leg_extended and is_right_leg_bent)

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

            # Check if the person is performing Single-leg Glute Bridge
            if is_in_single_leg_glute_bridge(landmarks):
                if not in_bridge:
                    in_bridge = True
                    last_bridge_time = cv2.getTickCount()
            elif in_bridge:
                in_bridge = False
                duration = (cv2.getTickCount() - last_bridge_time) / cv2.getTickFrequency()
                if duration > 0.5:  # Ensure a significant hold position before counting
                    count += 1  # Count the Single-leg Glute Bridge movement

            # Display the count on the frame
            cv2.putText(image, f'Single-leg Glute Bridges: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Single-leg Glute Bridge Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
