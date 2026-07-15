import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"

BODY_COLOR = (255, 0, 0)
POINT_COLOR = (0, 0, 255)
LINE_THICKNESS = 3


CONNECTIONS = [
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
    # hands
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
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


def draw_point(frame, point, radius=8):
    x, y = point
    cv2.circle(
        frame,
        (int(x), int(y)),
        radius,
        POINT_COLOR,
        -1,
    )


def get_point(frame, landmarks, index):
    h, w = frame.shape[:2]

    landmark = landmarks[index]

    x = int(landmark.x * w)
    y = int(landmark.y * h)

    return x, y


def draw_head_shape(frame, points):
    nose = points[0]
    left_eye = points[2]
    right_eye = points[5]
    left_ear = points[7]
    right_ear = points[8]

    center_x = (left_ear[0] + right_ear[0]) // 2

    center_y = (left_eye[1] + right_eye[1] + nose[1]) // 3

    head_width = abs(right_ear[0] - left_ear[0])

    if head_width > 0:
        head_height = int(head_width * 1.25)

        cv2.ellipse(
            frame,
            (
                center_x,
                center_y,
            ),
            (
                head_width // 2,
                head_height // 2,
            ),
            0,
            0,
            360,
            BODY_COLOR,
            LINE_THICKNESS,
        )

        return (
            center_x,
            center_y + head_height // 2,
        )

    return None


def create_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(detector, frame):
    """
    Detect pose landmarks in the ORIGINAL, unflipped frame.

    Returns:
        List of 33 (x, y) pixel coordinates,
        or None if no pose was detected.
    """

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]

    points = []

    for landmark in landmarks:
        h, w = frame.shape[:2]

        x = float(landmark.x * w)

        y = float(landmark.y * h)

        points.append((x, y))

    return points


def draw_pose(frame, points):
    """
    Draw the detected pose using pixel coordinates.
    """

    if points is None:
        return frame

    output = frame.copy()

    bottom_head = draw_head_shape(
        output,
        points,
    )

    left_shoulder = points[11]
    right_shoulder = points[12]

    shoulder_midpoint = (
        int((left_shoulder[0] + right_shoulder[0]) / 2),
        int((left_shoulder[1] + right_shoulder[1]) / 2),
    )

    if bottom_head is not None:
        cv2.line(
            output,
            bottom_head,
            shoulder_midpoint,
            BODY_COLOR,
            LINE_THICKNESS,
        )

    for point in points:
        draw_point(
            output,
            point,
        )

    for start, end in CONNECTIONS:
        point1 = points[start]
        point2 = points[end]

        cv2.line(
            output,
            (
                int(point1[0]),
                int(point1[1]),
            ),
            (
                int(point2[0]),
                int(point2[1]),
            ),
            BODY_COLOR,
            LINE_THICKNESS,
        )

    return output


def process_frame(detector, frame):
    """
    Detect landmarks and return both:
        annotated frame
        raw 2D landmark coordinates
    """

    points = detect_pose(
        detector,
        frame,
    )

    output = draw_pose(
        frame,
        points,
    )

    return output, points


def triangulate_points(
    points1,
    points2,
    K1,
    dist1,
    K2,
    dist2,
    R,
    t,
):
    """
    Triangulate matching MediaPipe landmarks.

    The resulting 3D coordinates are expressed in
    Camera 0's coordinate system.

    Returns:
        numpy array with shape (33, 3)
        or None if landmarks are missing.
    """

    if points1 is None or points2 is None:
        return None

    if len(points1) != len(points2):
        return None

    points1 = np.array(
        points1,
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    points2 = np.array(
        points2,
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    # Remove lens distortion and convert
    # pixel coordinates to normalized camera coordinates.
    points1_undistorted = cv2.undistortPoints(
        points1,
        K1,
        dist1,
    )

    points2_undistorted = cv2.undistortPoints(
        points2,
        K2,
        dist2,
    )

    # Camera 0 is the reference camera.
    P1 = np.hstack(
        (
            np.eye(3),
            np.zeros((3, 1)),
        )
    )

    # Camera 1 pose relative to Camera 0.
    P2 = np.hstack(
        (
            R,
            t.reshape(3, 1),
        )
    )

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        points1_undistorted.reshape(-1, 2).T,
        points2_undistorted.reshape(-1, 2).T,
    )

    # Convert homogeneous coordinates:
    # (X, Y, Z, W) -> (X/W, Y/W, Z/W)
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d
