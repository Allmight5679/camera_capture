import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker_full.task"

BODY_COLOR = (255, 0, 0)
POINT_COLOR = (0, 0, 255)
LINE_THICKNESS = 3


def draw_point(frame, landmark, radius=8):
    h, w, _ = frame.shape
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    cv2.circle(frame, (x, y), radius, POINT_COLOR, -1)
    return x, y


def get_point(frame, landmarks, index):
    h, w, _ = frame.shape
    landmark = landmarks[index]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def draw_head_shape(frame, landmarks):
    nose = get_point(frame, landmarks, 0)
    left_eye = get_point(frame, landmarks, 2)
    right_eye = get_point(frame, landmarks, 5)
    left_ear = get_point(frame, landmarks, 7)
    right_ear = get_point(frame, landmarks, 8)

    center_x = (left_ear[0] + right_ear[0]) // 2
    center_y = (left_eye[1] + right_eye[1] + nose[1]) // 3

    head_width = abs(right_ear[0] - left_ear[0])

    if head_width > 0:
        head_height = int(head_width * 1.25)

        cv2.ellipse(
            frame,
            (center_x, center_y),
            (head_width // 2, head_height // 2),
            0,
            0,
            360,
            BODY_COLOR,
            LINE_THICKNESS
        )

        bottom_head = (center_x, center_y + head_height // 2)
        return bottom_head

    return None


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
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # face/head

        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists

        23, 24,  # hips
        25, 26,  # knees
        27, 28,  # ankles
        29, 30,  # heels
        31, 32,  # foot index
    ]

    connections = [
        # face
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 6),
        (9, 10),

        # arms
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),

        # torso
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),

        # legs
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),

        # feet
        (27, 29),
        (29, 31),
        (28, 30),
        (30, 32),
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

            bottom_head = draw_head_shape(frame, landmarks)

            left_shoulder = get_point(frame, landmarks, 11)
            right_shoulder = get_point(frame, landmarks, 12)

            shoulder_midpoint = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )

            if bottom_head is not None:
                cv2.line(
                    frame,
                    bottom_head,
                    shoulder_midpoint,
                    BODY_COLOR,
                    LINE_THICKNESS
                )

            for point in selected_points:
                draw_point(frame, landmarks[point])

            for start, end in connections:
                x1, y1 = get_point(frame, landmarks, start)
                x2, y2 = get_point(frame, landmarks, end)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    BODY_COLOR,
                    LINE_THICKNESS
                )

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()