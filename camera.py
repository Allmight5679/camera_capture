import cv2
import json
import numpy as np
from pathlib import Path

import pose

CALIBRATION_ROOT = (
    Path(__file__).resolve().parent / "multi-camera-calibration" / "calibration_results"
)

CAMERA0_CALIBRATION = CALIBRATION_ROOT / "calibration" / "calibration_camera0.json"

CAMERA1_CALIBRATION = CALIBRATION_ROOT / "calibration" / "calibration_camera1.json"

EXTRINSIC_CALIBRATION = (
    CALIBRATION_ROOT / "extrinsic" / "initial_extrinsic_camera0_to_camera1.json"
)


def load_camera_calibration(path):
    """
    Load intrinsic camera calibration.

    Returns:
        K    : 3x3 camera matrix
        dist : distortion coefficients
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    K = np.array(
        data["camera_matrix"]["K"],
        dtype=np.float64,
    )

    dist = np.array(
        data["distortion_coefficients"],
        dtype=np.float64,
    )

    return K, dist


def load_extrinsic_calibration(path):
    """
    Load Camera 0 -> Camera 1 transformation.

    Returns:
        R : 3x3 rotation matrix
        t : 3-element translation vector
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    R = np.array(
        data["R"],
        dtype=np.float64,
    )

    t = np.array(
        data["t"],
        dtype=np.float64,
    )

    return R, t


def load_stereo_calibration():
    """
    Load all calibration data required
    for two-camera triangulation.
    """

    required_files = [
        CAMERA0_CALIBRATION,
        CAMERA1_CALIBRATION,
        EXTRINSIC_CALIBRATION,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found:\n{path}\n\n"
                "Run calibration steps 03 through 07 first."
            )

    K1, dist1 = load_camera_calibration(CAMERA0_CALIBRATION)

    K2, dist2 = load_camera_calibration(CAMERA1_CALIBRATION)

    R, t = load_extrinsic_calibration(EXTRINSIC_CALIBRATION)

    return (
        K1,
        dist1,
        K2,
        dist2,
        R,
        t,
    )


def main():
    print("1 or 2 Cameras?")
    num_of_camera = input().strip()

    while num_of_camera not in {
        "1",
        "2",
    }:
        print("Please enter 1 or 2.")

        num_of_camera = input().strip()

    detector = pose.create_detector()

    cap1 = cv2.VideoCapture(0)
    cap2 = None

    if not cap1.isOpened():
        print("Camera 0 could not be opened.")

        detector.close()
        return

    # Calibration values.
    K1 = None
    dist1 = None
    K2 = None
    dist2 = None
    R = None
    t = None

    if num_of_camera == "2":
        cap2 = cv2.VideoCapture(1)

        if not cap2.isOpened():
            print("Camera 1 could not be opened.")

            cap1.release()
            detector.close()
            return

        try:
            (
                K1,
                dist1,
                K2,
                dist2,
                R,
                t,
            ) = load_stereo_calibration()

            print("Stereo calibration loaded successfully.")

        except FileNotFoundError as error:
            print(error)

            cap1.release()
            cap2.release()
            detector.close()

            return

    while True:
        success1, frame1 = cap1.read()

        if not success1:
            print("Camera 0 not working.")
            break

        # IMPORTANT:
        # Landmark detection happens on the original,
        # unflipped calibrated camera image.
        frame1_display, points1 = pose.process_frame(
            detector,
            frame1,
        )

        if cap2 is not None:
            success2, frame2 = cap2.read()

            if success2:
                (
                    frame2_display,
                    points2,
                ) = pose.process_frame(
                    detector,
                    frame2,
                )

                # Triangulate only when both cameras
                # successfully detect a pose.
                points_3d = pose.triangulate_points(
                    points1,
                    points2,
                    K1,
                    dist1,
                    K2,
                    dist2,
                    R,
                    t,
                )

                if points_3d is not None:
                    # MediaPipe landmark indices:
                    # 0  = Nose
                    # 15 = Left wrist
                    # 16 = Right wrist
                    nose_3d = points_3d[0]
                    left_hand_3d = points_3d[15]
                    right_hand_3d = points_3d[16]

                    # Coordinate text settings.
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1.0
                    font_color = (0, 0, 0)
                    font_thickness = 1

                    # Nose coordinates.
                    cv2.putText(
                        frame1_display,
                        (
                            f"Nose | "
                            f"X: {nose_3d[0]:.2f}  "
                            f"Y: {nose_3d[1]:.2f}  "
                            f"Z: {nose_3d[2]:.2f} m"
                        ),
                        (15, 35),
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

                    # Left hand / wrist coordinates.
                    cv2.putText(
                        frame1_display,
                        (
                            f"L Hand | "
                            f"X: {left_hand_3d[0]:.2f}  "
                            f"Y: {left_hand_3d[1]:.2f}  "
                            f"Z: {left_hand_3d[2]:.2f} m"
                        ),
                        (15, 65),
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

                    # Right hand / wrist coordinates.
                    cv2.putText(
                        frame1_display,
                        (
                            f"R Hand | "
                            f"X: {right_hand_3d[0]:.2f}  "
                            f"Y: {right_hand_3d[1]:.2f}  "
                            f"Z: {right_hand_3d[2]:.2f} m"
                        ),
                        (15, 95),
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

                frame1_display = cv2.resize(
                    frame1_display,
                    (640, 480),
                )

                frame2_display = cv2.resize(
                    frame2_display,
                    (640, 480),
                )

                split_screen = np.hstack(
                    (
                        frame1_display,
                        frame2_display,
                    )
                )

                cv2.imshow(
                    "Two Camera Pose Detection",
                    split_screen,
                )

            else:
                cv2.imshow(
                    "Camera 0 Pose Detection",
                    frame1_display,
                )

        else:
            cv2.imshow(
                "Camera 0 Pose Detection",
                frame1_display,
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()

    if cap2 is not None:
        cap2.release()

    detector.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
