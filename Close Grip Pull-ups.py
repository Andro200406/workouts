import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Variables for counting close grip pull-ups
close_pull_up_count = 0
close_pull_up_state = "down"  # Track whether the user is in the "down" or "up" position
head_initial_y = 0  # Initial Y position of the head for tracking vertical motion

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
            
            # Get coordinates of key points (shoulders, elbows, wrists, and head)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            
            # Calculate the angle of the elbows for flexion/extension
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Check for close grip by calculating the distance between the shoulders
            shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

            # Threshold for determining a close grip (shoulder width typically between 0.2 and 0.3 depending on individual)
            if shoulder_distance < 0.3:  # Threshold for close grip (adjust as needed)
                
                # Track vertical motion of the head to detect upward/downward movement
                head_y = head[1]

                # If head initial position is not set, set it to the current Y position
                if head_initial_y == 0:
                    head_initial_y = head_y

                # Check if the body is moving up (head rising)
                if head_y < head_initial_y - 0.1:  # Head moves up by a certain threshold
                    if close_pull_up_state == "down" and left_elbow_angle < 160 and right_elbow_angle < 160:  # Elbows are flexed
                        close_pull_up_count += 1
                        close_pull_up_state = "up"  # State changes to "up"

                # Check if the body is coming down (head lowering)
                if head_y > head_initial_y + 0.1:  # Head moves down by a certain threshold
                    close_pull_up_state = "down"  # State changes to "down" when lowering
                
                # Display Close Grip Pull-up count on the frame
                cv2.putText(image, f'Close Grip Pull-ups: {close_pull_up_count}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Close Grip Pull-up Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
