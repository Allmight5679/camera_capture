"""
Multi-camera live viewer with image capture capability.
- Loads camera configurations from calibration_config.yaml and shows each stream in its own window.
- Press SPACE to capture and save images from all cameras simultaneously.
- Supports dynamic number of cameras based on configuration.

Requirements:
  - Environment variable TISGRABBER_DLL_PATH must point to tisgrabber_x64.dll.
  - Python packages: numpy, opencv-python, pyyaml

Usage (Windows cmd):
  python 03_live_capture.py
  python 03_live_capture.py --config my_config.yaml
  python 03_live_capture.py --select  # Interactive device selection (ignores calibration_config.yaml)
  python 03_live_capture.py --output captured_images  # Custom output directory

Keys:
  SPACE: Capture and save images from all cameras
  q or ESC: Quit

Output:
  Images are saved to captured_images/ (or custom --output directory)
  Each camera has its own subdirectory: captured_images/camera0/, camera1/, etc.
  Filename format: YYYYMMDD_HHMMSS_mmm.png (with milliseconds)

Notes:
  - Use 01_create-configs.py to generate camera configs and calibration_config.yaml.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

try:
    import cv2
    import numpy as np
except Exception as e:  # pragma: no cover
    print("This script requires OpenCV (cv2) and numpy. Install with: pip install opencv-python numpy", file=sys.stderr)
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


def save_images(images: List[Optional[np.ndarray]], output_dir: Path, capture_count: int) -> bool:
    """
    Save images from all cameras with timestamp.
    
    Args:
        images: List of images (one per camera, can contain None)
        output_dir: Base output directory
        capture_count: Current capture number (for display)
    
    Returns:
        True if at least one image was saved successfully
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits to get milliseconds
    
    saved_count = 0
    failed_cameras = []
    
    for i, img in enumerate(images):
        if img is None:
            failed_cameras.append(i)
            continue
        
        # Create camera subdirectory
        camera_dir = output_dir / f"camera{i}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        filename = camera_dir / f"{timestamp}.png"
        try:
            success = cv2.imwrite(str(filename), img)
            if success:
                saved_count += 1
                print(f"  ✓ Camera {i}: {filename}")
            else:
                failed_cameras.append(i)
                print(f"  ✗ Camera {i}: Failed to write {filename}")
        except Exception as e:
            failed_cameras.append(i)
            print(f"  ✗ Camera {i}: Error - {e}")
    
    if saved_count > 0:
        print(f"✓ Saved {saved_count}/{len(images)} images (capture #{capture_count})")
        if failed_cameras:
            print(f"  ⚠ Failed cameras: {failed_cameras}")
        return True
    else:
        print(f"✗ Failed to save any images (capture #{capture_count})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-camera live viewer with image capture")
    parser.add_argument("--config", default="calibration_config.yaml", help="Path to calibration_config.yaml (default: calibration_config.yaml)")
    parser.add_argument("--select", action="store_true", 
                       help="Open selection dialogs interactively (ignores calibration_config.yaml)")
    parser.add_argument("--num", type=int, help="Number of cameras when using --select (default: 2)")
    parser.add_argument("--output", default="captured_images", 
                       help="Output directory for captured images (default: captured_images)")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")

    cameras: List[TISGrabberWrapper] = []
    window_titles: List[str] = []
    display_scales: List[float] = []
    capture_count = 0
    
    try:
        if args.select:
            # Interactive mode
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
            # Load from calibration_config.yaml
            try:
                from config_manager import ConfigManager
                
                config_path = Path(args.config)
                if not config_path.exists():
                    print(f"Error: {config_path} not found.")
                    print("Run 01_create-configs.py first to create camera configurations.")
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
                
                # Load each camera
                for i in range(num_cameras):
                    cam_cfg = manager.get_camera_config(i)
                    xml_path = Path(cam_cfg['xml_path']) # type: ignore
                    # カメラ個別のdisplay_scaleがあればそれを使用、なければグローバルを使用
                    scale = cam_cfg.get('display_scale', global_display_scale) # type: ignore
                    
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
                print("Error: config_manager.py not found. Cannot load calibration_config.yaml")
                print("Please ensure config_manager.py exists in the current directory.")
                return 1
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return 1
        
        # Create windows
        for title in window_titles:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        
        print(f"\n✓ {len(cameras)} camera(s) ready")
        print("─" * 60)
        print("Controls:")
        print("  SPACE: Capture and save images from all cameras")
        print("  q or ESC: Quit")
        print("─" * 60)
        print()

        # Main loop
        current_images: List[Optional[np.ndarray]] = [None] * len(cameras)
        
        while True:
            # Capture and display frames from all cameras
            for i, (cam, title, scale) in enumerate(zip(cameras, window_titles, display_scales)):
                img = cam.capture_image(timeout_ms=1000)
                
                if img is not None:
                    # Store original image for saving
                    current_images[i] = img.copy()
                    
                    # Display scaled image
                    h, w = img.shape[:2]
                    scale = max(scale, 0.01)
                    resized = cv2.resize(img, (int(w * scale), int(h * scale)), 
                                       interpolation=cv2.INTER_AREA)
                    
                    # Add capture count overlay
                    if capture_count > 0:
                        cv2.putText(resized, f"Captured: {capture_count}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(title, resized)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key in (27, ord('q')):  # ESC or 'q'
                break
            elif key == ord(' '):  # SPACE
                capture_count += 1
                print(f"\n[Capture #{capture_count}]")
                save_images(current_images, output_dir, capture_count)
                print()
                
    finally:
        # Cleanup
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
        
        print(f"\n✓ Total captures: {capture_count}")
        if capture_count > 0:
            print(f"✓ Images saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
