import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Duck Walks
count = 0
in_duck_walk = False
previous_left_ankle = None
previous_right_ankle = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to check if the person is performing a Duck Walk
def is_in_duck_walk(landmarks):
    global in_duck_walk, previous_left_ankle, previous_right_ankle
    
    # Get coordinates for hips, knees, and ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate the knee angles for squat detection
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Detect if the knees are bent enough for a deep squat (angle between 85 and 170 degrees)
    is_in_squat_position = 85 < left_knee_angle < 170 or 85 < right_knee_angle < 170

    # Check if the person is walking (movement of ankles)
    is_walking = False
    if previous_left_ankle and previous_right_ankle:
        # Calculate horizontal movement (change in x-coordinate of ankles)
        left_ankle_movement = abs(left_ankle[0] - previous_left_ankle[0])
        right_ankle_movement = abs(right_ankle[0] - previous_right_ankle[0])

        # If both ankles have moved horizontally, consider it walking
        if left_ankle_movement > 0.05 or right_ankle_movement > 0.05:
            is_walking = True

    # Update ankle positions for next iteration
    previous_left_ankle = left_ankle
    previous_right_ankle = right_ankle

    # If person is in squat position and walking, it is a duck walk
    if is_in_squat_position and is_walking and not in_duck_walk:
        in_duck_walk = True
        return True  # Duck walk detected

    # Reset if the person is no longer performing a duck walk
    if not is_in_squat_position and in_duck_walk:
        in_duck_walk = False

    return False

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

            # Check if the person is performing a Duck Walk
            if is_in_duck_walk(landmarks):
                count += 1  # Increment count when a duck walk is detected

            # Display the count on the frame
            cv2.putText(image, f'Duck Walks: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Duck Walk Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
