"""
delete_previous_calibration.py

Deletes all previously generated calibration data so the
camera calibration process can be started from scratch.

Deletes contents from:
    captured_images/
    detection_cache/
    calibration_results/

Does NOT delete:
    - Python source files
    - calibration_config.yaml
    - The directories themselves
"""

from pathlib import Path
import shutil


DIRECTORIES_TO_CLEAR = [
    Path("captured_images"),
    Path("detection_cache"),
    Path("calibration_results"),
]


def clear_directory(directory: Path):
    """
    Delete all files and subdirectories inside a directory,
    while keeping the directory itself.
    """

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created missing directory: {directory}")
        return

    for item in directory.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        except Exception as error:
            print(f"Failed to delete {item}: {error}")


def main():
    print("=" * 60)
    print("Delete Previous Camera Calibration")
    print("=" * 60)

    print()
    print("This will permanently delete all generated data from:")

    for directory in DIRECTORIES_TO_CLEAR:
        print(f"  - {directory}/")

    print()
    confirmation = input(
        "Are you sure you want to delete the previous calibration? "
        "(y/n): "
    ).strip().lower()

    if confirmation not in {"y", "yes"}:
        print("Deletion cancelled.")
        return

    print()

    for directory in DIRECTORIES_TO_CLEAR:
        print(f"Clearing {directory}/...")
        clear_directory(directory)
        print(f"✓ Cleared {directory}/")

    print()
    print("✓ Previous calibration data deleted.")
    print("✓ Ready to start a new calibration.")
    print()
    print("Next step:")
    print("  python 03_live_capture.py")


if __name__ == "__main__":
    main()