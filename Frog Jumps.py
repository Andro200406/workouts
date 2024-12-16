import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for counting Frog Jumps
count = 0
in_jump = False
in_squat = False

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

# Function to check if the person is performing a Frog Jump
def is_in_frog_jump(landmarks):
    global in_jump, in_squat
    
    # Get coordinates for hips, knees, and ankles
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate the angle of the knees and hips to check for squat
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Check if both knees are bent deeply (more than 90 degrees)
    is_in_squat_position = 85 < left_knee_angle < 170 and 85 < right_knee_angle < 170

    # Check if the person is airborne (i.e., in the air)
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    # Detect if person leaves the ground (airborne check)
    is_airborne = left_ankle_y < 0.5 and right_ankle_y < 0.5  # Adjust this threshold for better detection

    if is_in_squat_position and not in_jump:
        in_squat = True  # They are in a squat position, but not yet in a jump
    elif is_airborne and in_squat:
        in_jump = True  # They are in the air (jumping)
        in_squat = False  # Not in squat anymore
    elif not is_airborne and in_jump:
        in_jump = False
        return True  # Return True if jump is completed (i.e., frog jump is done)
    
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

            # Check if the person is performing a Frog Jump
            if is_in_frog_jump(landmarks):
                count += 1  # Increment count when a frog jump is completed

            # Display the count on the frame
            cv2.putText(image, f'Frog Jumps: {count}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Frog Jump Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
