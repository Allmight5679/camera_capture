import cv2
import mediapipe as mp

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose detection
        results = pose.process(rgb_frame)

        # Convert back to BGR for OpenCV
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw landmarks and lines
        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Helper function to get pixel coordinates
            def get_point(index):
                h, w, _ = frame.shape
                landmark = landmarks[index]
                return int(landmark.x * w), int(landmark.y * h)

            # Draw circles on selected joints
            selected_points = [
                11, 12,  # shoulders
                13, 14,  # elbows
                15, 16,  # wrists
                23, 24,  # hips
            ]

            for point in selected_points:
                x, y = get_point(point)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # Define lines between joints
            connections = [
                (11, 13),  # left shoulder -> left elbow
                (13, 15),  # left elbow -> left wrist
                (12, 14),  # right shoulder -> right elbow
                (14, 16),  # right elbow -> right wrist
                (11, 12),  # shoulders connected
                (11, 23),  # left shoulder -> left hip
                (12, 24),  # right shoulder -> right hip
                (23, 24),  # hips connected
            ]

            # Draw lines
            for start, end in connections:
                x1, y1 = get_point(start)
                x2, y2 = get_point(end)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Show window
        cv2.imshow("Pose Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()