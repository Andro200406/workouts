import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Lizard Crawl Push-ups
count = 0
in_lizard_crawl = False

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

# Function to check if the person is performing Lizard Crawl Push-up
def is_lizard_crawl(landmarks):
    global in_lizard_crawl

    # Get coordinates for key points (hips, knees, wrists, shoulders, and ankles)
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    # Check if the person is in the low position for the crawl (hips close to the ground)
    if left_hip[1] > left_knee[1] and right_hip[1] > right_knee[1]:
        # Check for opposite arm-leg coordination
        if left_knee[1] > left_ankle[1] and right_knee[1] < right_ankle[1]:
            # The left knee should be lower than the left ankle while the right knee should be higher than the right ankle during the crawl
            if abs(left_wrist[0] - left_shoulder[0]) < abs(right_wrist[0] - right_shoulder[0]):
                return True
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

            # Detect Lizard Crawl Push-up position
            if is_lizard_crawl(landmarks):
                if not in_lizard_crawl:
                    in_lizard_crawl = True
                    count += 1  # Increment the count when Lizard Crawl Push-up is detected

                cv2.putText(image, f'Lizard Crawl Push-ups: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Lizard Crawl Push-up Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                in_lizard_crawl = False

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Lizard Crawl Push-up Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
