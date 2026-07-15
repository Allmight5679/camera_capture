"""
Multi-camera live viewer using TISGrabberWrapper and calibration_config.yaml.
- Loads camera configurations from calibration_config.yaml and shows each stream in its own window.
- Supports dynamic number of cameras based on configuration.

Requirements:
  - Environment variable TISGRABBER_DLL_PATH must point to tisgrabber_x64.dll.
  - Python packages: numpy, opencv-python, pyyaml

Usage (Windows cmd):
  python 02_live_view.py
  python 02_live_view.py --config my_config.yaml
  python 02_live_view.py --select  # Interactive device selection (ignores calibration_config.yaml)

Keys:
  q or ESC: Quit

Notes:
  - Use 01_create-configs.py to generate camera configs and calibration_config.yaml.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List

import time

try:
    import cv2
    import numpy as np
except Exception as e:  # pragma: no cover
    print(
        "This script requires OpenCV (cv2) and numpy. Install with: pip install opencv-python numpy",
        file=sys.stderr,
    )
    raise

from tis_wrapper import TISGrabberWrapper


def open_from_config(config_path: Path) -> Optional[TISGrabberWrapper]:
    cam = TISGrabberWrapper()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        cam.release()
        return None
    if not cam.load_device_config(str(config_path)):
        print(f"Failed to load device from {config_path}")
        cam.release()
        return None
    if not cam.start_live():
        print(f"Failed to start live for {config_path}")
        cam.release()
        return None
    return cam


def open_by_selection(label: str) -> Optional[TISGrabberWrapper]:
    print(f"=== Select {label} camera ===")
    cam = TISGrabberWrapper()
    cam.select_device()
    if not cam.is_device_valid():
        print(f"No device selected for {label}")
        cam.release()
        return None
    if not cam.start_live():
        print(f"Failed to start live for {label}")
        cam.release()
        return None
    return cam


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-camera live viewer with calibration_config.yaml support"
    )
    parser.add_argument(
        "--config",
        default="calibration_config.yaml",
        help="Path to calibration_config.yaml (default: calibration_config.yaml)",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Open selection dialogs interactively (ignores calibration_config.yaml)",
    )
    parser.add_argument(
        "--num", type=int, help="Number of cameras when using --select (default: 2)"
    )
    args = parser.parse_args()

    cameras: List[TISGrabberWrapper] = []
    window_titles: List[str] = []
    display_scales: List[float] = []

    try:
        if args.select:
            # インタラクティブモード
            num_cameras = args.num if args.num else 2
            print(f"Interactive mode: Selecting {num_cameras} camera(s)...")

            for i in range(num_cameras):
                label = f"Camera {i}/{num_cameras - 1}"
                cam = open_by_selection(label)
                if cam is None:
                    print(f"Failed to select camera {i}")
                    return i + 1
                cameras.append(cam)
                window_titles.append(f"Camera {i}")
                display_scales.append(0.5)
        else:
            # calibration_config.yamlから読み込み
            try:
                from config_manager import ConfigManager

                config_path = Path(args.config)
                if not config_path.exists():
                    print(f"Error: {config_path} not found.")
                    print(
                        "Run 01_create-configs.py first to create camera configurations."
                    )
                    return 1

                manager = ConfigManager(config_path)
                if not manager.load():
                    print(f"Error: Failed to load {config_path}")
                    return 1

                num_cameras = manager.get_camera_count()
                if num_cameras == 0:
                    print(f"Error: No cameras configured in {config_path}")
                    return 1

                print(f"Loading {num_cameras} camera(s) from {config_path}...")

                # グローバルdisplay_scaleを取得
                global_display_scale = manager.get_display_scale()
                print(f"Global display_scale: {global_display_scale}")
                window_prefix = "Camera"

                # 各カメラを読み込み
                for i in range(num_cameras):
                    cam_cfg = manager.get_camera_config(i)
                    xml_path = Path(cam_cfg["xml_path"])  # type: ignore
                    # カメラ個別のdisplay_scaleがあればそれを使用、なければグローバルを使用
                    scale = cam_cfg.get("display_scale", global_display_scale)  # type: ignore

                    print(f"  [{i}] Loading {xml_path}...")
                    cam = open_from_config(xml_path)
                    if cam is None:
                        print(f"  ✗ Failed to load camera {i}")
                        return i + 1

                    cameras.append(cam)
                    window_titles.append(f"{window_prefix} {i}")
                    display_scales.append(scale)
                    print(f"  ✓ Camera {i} loaded")

            except ImportError:
                print(
                    "Error: config_manager.py not found. Cannot load calibration_config.yaml"
                )
                print(
                    "Please ensure config_manager.py exists in the current directory."
                )
                return 1
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return 1

        # ウィンドウを作成
        for title in window_titles:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        print(f"\n✓ {len(cameras)} camera(s) ready")
        print("Press 'q' or ESC to quit.\n")

        # メインループ
        while True:
            for i, (cam, title, scale) in enumerate(
                zip(cameras, window_titles, display_scales)
            ):
                img = cam.capture_image(timeout_ms=1000)

                if img is not None:
                    h, w = img.shape[:2]
                    scale = max(scale, 0.01)
                    resized = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow(title, resized)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        # クリーンアップ
        print("\nCleaning up...")
        for i, cam in enumerate(cameras):
            try:
                if cam is not None:
                    cam.stop_live()
                    cam.release()
                    print(f"  ✓ Camera {i} released")
            except Exception as e:
                print(f"  ✗ Error releasing camera {i}: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
