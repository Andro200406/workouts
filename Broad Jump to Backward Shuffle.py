import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Broad Jumps and Backward Shuffles
count = 0
broad_jump_detected = False
backward_shuffle_detected = False

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to check if the person is in the broad jump position
def check_broad_jump(landmarks):
    global broad_jump_detected

    # Get coordinates of the hips, knees, and ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    # Calculate the angle of the knee (bend) to check if it is deep enough for a jump
    knee_angle = calculate_angle(left_hip, left_knee, right_knee)
    
    # Assuming jump threshold for knee bend is less than 140 degrees
    if knee_angle < 140:  
        if not broad_jump_detected:
            broad_jump_detected = True
            return True
    else:
        broad_jump_detected = False
    return False

# Function to check if the person is performing the backward shuffle
def check_backward_shuffle(landmarks, previous_position):
    global backward_shuffle_detected

    # Get the position of the hips (x-coordinate for movement tracking)
    current_position = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x]
    
    # Check if there has been backward movement (hip x-coordinate decreases)
    if current_position[0] < previous_position[0]:  
        if not backward_shuffle_detected:
            backward_shuffle_detected = True
            return True
    else:
        backward_shuffle_detected = False
    return False

# Video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize the Mediapipe Pose model
previous_position = [0, 0]  # Track previous position of the person
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

            # Check for Broad Jump
            if check_broad_jump(landmarks):
                cv2.putText(image, 'Broad Jump Detected', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check for Backward Shuffle
            if check_backward_shuffle(landmarks, previous_position):
                count += 1  # Increment the count for the combination
                cv2.putText(image, f'Broad Jump to Backward Shuffle Count: {count}', (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Update the previous position
            previous_position = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x]

        except Exception as e:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Broad Jump to Backward Shuffle Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
