import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Skater Jumps
count = 0
skater_jump_detected = False

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

# Function to check if the person is performing the skater jump
def check_skater_jump(landmarks, previous_position):
    global skater_jump_detected

    # Get the coordinates of the hips (for side-to-side movement tracking)
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    
    # Calculate the horizontal movement (side-to-side motion)
    movement = abs(left_hip[0] - right_hip[0])

    # If there is a noticeable lateral movement, it could indicate a skater jump
    if movement > 0.15:  # Threshold for lateral movement, can be adjusted
        if not skater_jump_detected:
            skater_jump_detected = True
            return True
    else:
        skater_jump_detected = False
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

            # Check for Skater Jump
            if check_skater_jump(landmarks, previous_position):
                count += 1  # Increment the count for the skater jump
                cv2.putText(image, f'Skater Jumps Count: {count}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Update the previous position for comparison
            previous_position = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x]

        except Exception as e:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Skater Jumps Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
