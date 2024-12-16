import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Side-to-Side Hops
count = 0
jump_detected = False

# Function to calculate the distance between two points
def calculate_distance(a, b):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point
    return np.linalg.norm(a - b)

# Function to detect the jump
def check_side_to_side_hop(landmarks):
    global jump_detected

    # Get coordinates for key points (hips, knees, and ankles)
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate the distance between the ankles
    ankle_distance = calculate_distance(left_ankle, right_ankle)

    # Check if the person is in a squat position (knee bend should be noticeable)
    knee_angle = calculate_distance(left_knee, left_hip) + calculate_distance(right_knee, right_hip)
    
    if knee_angle < 0.35:  # Tolerance for squat depth, adjust as needed
        # Detect side-to-side hops based on distance of the feet
        if ankle_distance > 0.2:  # Tolerance for feet distance in lateral motion
            if not jump_detected:
                jump_detected = True
                return True
        else:
            jump_detected = False
    
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

            # Check Side-to-Side Hops
            if check_side_to_side_hop(landmarks):
                count += 1  # Increment the count when a hop is detected
                cv2.putText(image, f'Side-to-Side Hops: {count}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Side-to-Side Hop Detected', (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Side-to-Side Hop Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
