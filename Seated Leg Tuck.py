import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Seated Leg Tucks
count = 0
in_tuck_position = False

# Function to check if the person is performing a Seated Leg Tuck
def is_in_seated_leg_tuck(landmarks):
    # Get the positions of key points: Hips, Knees, and Ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate the angle of the knees relative to the hips and ankles
    knee_angle_left = np.arctan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0])
    knee_angle_right = np.arctan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0])

    # Threshold for detecting knees moving toward chest (sign of leg tuck)
    leg_tuck_threshold = 0.4

    # Detect if knees are moving towards the chest (bending position)
    return abs(knee_angle_left) > leg_tuck_threshold or abs(knee_angle_right) > leg_tuck_threshold

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
            
            # Check if the person is performing a Seated Leg Tuck
            if is_in_seated_leg_tuck(landmarks):
                if not in_tuck_position:
                    in_tuck_position = True
                    # This indicates a leg tuck movement
            elif in_tuck_position:
                in_tuck_position = False
                count += 1  # Count the Seated Leg Tuck movement

            # Display the count on the frame
            cv2.putText(image, f'Seated Leg Tucks Count: {count}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Seated Leg Tuck Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
