"""
03_live_capture.py

OpenCV multi-camera live viewer with image capture capability.

- Lets the user choose 1 or 2 cameras.
- Uses cv2.VideoCapture instead of TISGrabberWrapper.
- Press SPACE to capture images from all active cameras.
- Saves matching filenames for both cameras so later calibration scripts
  can identify synchronized image pairs.

Output:
    captured_images/
        camera0/
            20260715_123456_123.png
        camera1/
            20260715_123456_123.png

Controls:
    SPACE : Capture images
    q     : Quit
    ESC   : Quit
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """
    Open a camera using OpenCV.

    Args:
        camera_index: OpenCV camera index, such as 0 or 1.

    Returns:
        Opened VideoCapture object, or None if the camera could not be opened.
    """
    print(f"Opening Camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"✗ Failed to open Camera {camera_index}")
        cap.release()
        return None

    print(f"✓ Camera {camera_index} opened")
    return cap


def save_images(
    images: List[Optional[np.ndarray]],
    output_dir: Path,
    capture_count: int,
) -> bool:
    """
    Save one image from every active camera.

    The exact same timestamp filename is used for every camera.
    This is important because later calibration stages use the matching
    filenames to determine which frames were captured together.

    Args:
        images: Current image from each camera.
        output_dir: Base capture directory.
        capture_count: Number of captures taken so far.

    Returns:
        True if every camera image was successfully saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    saved_count = 0

    print(f"\n[Capture #{capture_count}]")

    for camera_index, image in enumerate(images):
        if image is None:
            print(f"✗ Camera {camera_index}: No image available")
            continue

        camera_dir = output_dir / f"camera{camera_index}"
        camera_dir.mkdir(parents=True, exist_ok=True)

        filename = camera_dir / f"{timestamp}.png"

        success = cv2.imwrite(str(filename), image)

        if success:
            saved_count += 1
            print(f"✓ Camera {camera_index}: {filename}")
        else:
            print(f"✗ Camera {camera_index}: Failed to save image")

    if saved_count == len(images):
        print(
            f"✓ Successfully saved synchronized capture "
            f"#{capture_count} from all cameras"
        )
        return True

    print(
        f"⚠ Saved {saved_count}/{len(images)} camera images "
        f"for capture #{capture_count}"
    )
    return False


def choose_number_of_cameras() -> int:
    """
    Ask the user whether to use one or two cameras.

    Returns:
        1 or 2.
    """
    while True:
        choice = input("One or Two Cameras? Enter 1 or 2: ").strip()

        if choice in {"1", "2"}:
            return int(choice)

        print("Please enter either 1 or 2.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenCV multi-camera calibration image capture"
    )

    parser.add_argument(
        "--output",
        default="captured_images",
        help="Output directory for captured images " "(default: captured_images)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Camera Calibration Image Capture")
    print("=" * 60)

    num_cameras = choose_number_of_cameras()

    cameras: List[cv2.VideoCapture] = []

    try:
        # Camera 0 is always used.
        camera0 = open_camera(0)

        if camera0 is None:
            return 1

        cameras.append(camera0)

        # Camera 1 is opened only when the user selects two cameras.
        if num_cameras == 2:
            camera1 = open_camera(1)

            if camera1 is None:
                return 2

            cameras.append(camera1)

        print()
        print(f"✓ {len(cameras)} camera(s) ready")
        print(f"✓ Images will be saved to: {output_dir.absolute()}")
        print()
        print("Controls:")
        print("  SPACE : Capture calibration images")
        print("  q     : Quit")
        print("  ESC   : Quit")
        print()

        capture_count = 0

        while True:
            current_images: List[Optional[np.ndarray]] = []

            # Read one frame from every camera.
            for camera_index, cap in enumerate(cameras):
                success, frame = cap.read()

                if not success or frame is None:
                    print(
                        f"Warning: Could not read frame " f"from Camera {camera_index}"
                    )
                    current_images.append(None)
                    continue

                # Save a copy of the original frame for image capture.
                current_images.append(frame.copy())

                # Create a display copy.
                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"Camera {camera_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Captured: {capture_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    f"Calibration - Camera {camera_index}",
                    display_frame,
                )

            key = cv2.waitKey(1) & 0xFF

            # Quit.
            if key == ord("q") or key == 27:
                break

            # Capture images.
            if key == ord(" "):
                # Only count the capture if every camera has a valid frame.
                if all(image is not None for image in current_images):
                    capture_count += 1

                    save_images(
                        current_images,
                        output_dir,
                        capture_count,
                    )
                else:
                    print(
                        "\n✗ Capture skipped because at least one "
                        "camera does not have a valid frame."
                    )

    finally:
        print("\nCleaning up...")

        for camera_index, cap in enumerate(cameras):
            cap.release()
            print(f"✓ Camera {camera_index} released")

        cv2.destroyAllWindows()

    print()
    print(f"✓ Total successful captures: {capture_count}")
    print(f"✓ Images saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
