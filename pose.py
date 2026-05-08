import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker_lite.task"


def draw_point(frame, landmark, radius=8):
    h, w, _ = frame.shape
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
    return x, y


def get_point(frame, landmarks, index):
    h, w, _ = frame.shape
    landmark = landmarks[index]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    selected_points = [
        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists
        23, 24,  # hips
    ]

    connections = [
        (11, 13),  # left shoulder -> left elbow
        (13, 15),  # left elbow -> left wrist
        (12, 14),  # right shoulder -> right elbow
        (14, 16),  # right elbow -> right wrist
        (11, 12),  # shoulder line
        (11, 23),  # left shoulder -> left hip
        (12, 24),  # right shoulder -> right hip
        (23, 24),  # hip line
    ]

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            for point in selected_points:
                draw_point(frame, landmarks[point])

            for start, end in connections:
                x1, y1 = get_point(frame, landmarks, start)
                x2, y2 = get_point(frame, landmarks, end)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()