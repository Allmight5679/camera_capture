"""
Multi-camera real-time hand skeleton detection using MediaPipe with async processing.
- Loads camera configurations from calibration_config.yaml and shows each stream with hand landmarks.
- Detects hand skeleton (21 landmarks) in real-time for each camera.
- Supports dynamic number of cameras based on configuration.
- Uses async/await for concurrent capture, detection, and display.

Requirements:
  - Environment variable TISGRABBER_DLL_PATH must point to tisgrabber_x64.dll.
  - Python packages: numpy, opencv-python, pyyaml, mediapipe

Usage (Windows cmd):
  python 12_hand_detection.py
  python 12_hand_detection.py --config my_config.yaml
  python 12_hand_detection.py --select  # Interactive device selection (ignores calibration_config.yaml)

Keys:
  q or ESC: Quit
  s: Toggle statistics display (FPS, detection count)

Notes:
  - Use 01_create_configs.py to generate camera configs and calibration_config.yaml.
  - MediaPipe detects up to 2 hands per frame by default.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, List

try:
    import cv2
    import numpy as np
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    print("This script requires OpenCV, numpy, and mediapipe. Install with:", file=sys.stderr)
    print("  pip install opencv-python numpy mediapipe", file=sys.stderr)
    raise

from tis_wrapper import TISGrabberWrapper


class HandDetector:
    """MediaPipe Hands detector wrapper for multi-camera use."""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hands detector.
        
        Args:
            static_image_mode: If False, treats input as video stream for better performance
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.detection_count = 0
        self.frame_count = 0
        
    async def process_async(self, image: np.ndarray) -> tuple[np.ndarray, Optional[object]]:
        """
        Process image and detect hands asynchronously.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Run the synchronous processing in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_sync, image)
    
    def _process_sync(self, image: np.ndarray) -> tuple[np.ndarray, Optional[object]]:
        """Synchronous processing logic."""
        self.frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Convert back to BGR for OpenCV
        annotated_image = image.copy()
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            self.detection_count += len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_image, results
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'avg_detections_per_frame': self.detection_count / max(1, self.frame_count)
        }
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.detection_count = 0
        self.frame_count = 0
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


def open_from_config(config_path: Path) -> Optional[TISGrabberWrapper]:
    """Open camera from config file."""
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
    """Open camera through interactive selection dialog."""
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


def draw_stats(image: np.ndarray, camera_id: int, detector: HandDetector, 
               fps: float, num_hands: int) -> np.ndarray:
    """
    Draw statistics on image.
    
    Args:
        image: Input image
        camera_id: Camera identifier
        detector: HandDetector instance
        fps: Current FPS
        num_hands: Number of hands detected in current frame
        
    Returns:
        Image with stats overlay
    """
    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    
    # Text
    stats = detector.get_stats()
    y_pos = 35
    texts = [
        f"Camera {camera_id}",
        f"FPS: {fps:.1f}",
        f"Hands detected: {num_hands}",
        f"Total detections: {stats['detection_count']}",
    ]
    
    for text in texts:
        cv2.putText(image, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y_pos += 20
    
    return image


async def process_camera(
    camera_id: int,
    cam: TISGrabberWrapper,
    detector: HandDetector,
    title: str,
    scale: float,
    show_stats: bool
) -> tuple[Optional[np.ndarray], float, int]:
    """
    Process a single camera frame asynchronously.
    
    Args:
        camera_id: Camera identifier
        cam: Camera wrapper instance
        detector: HandDetector instance
        title: Window title
        scale: Display scale
        show_stats: Whether to show statistics
        
    Returns:
        Tuple of (display_image, fps, num_hands)
    """
    # Capture frame (synchronous, but fast)
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, cam.capture_image, 1000)
    
    if img is None:
        return None, 0.0, 0
    
    # Process hand detection asynchronously
    annotated_img, results = await detector.process_async(img)
    
    # Count detected hands
    num_hands = 0
    if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
    
    return annotated_img, 0.0, num_hands


async def async_main(
    cameras: List[TISGrabberWrapper],
    detectors: List[HandDetector],
    window_titles: List[str],
    display_scales: List[float]
) -> None:
    """
    Main async loop for processing multiple cameras concurrently.
    
    Args:
        cameras: List of camera instances
        detectors: List of detector instances
        window_titles: List of window titles
        display_scales: List of display scales
    """
    # FPS tracking
    fps_list: List[float] = [0.0] * len(cameras)
    last_time_list: List[float] = [time.time()] * len(cameras)
    
    # Display options
    show_stats = True
    
    print("\nStarting async processing loop...")
    print("Press 'q' or ESC to quit, 's' to toggle stats\n")
    
    try:
        while True:
            # Process all cameras concurrently
            tasks = [
                process_camera(
                    i, cam, detector, title, scale, show_stats
                )
                for i, (cam, detector, title, scale) in enumerate(
                    zip(cameras, detectors, window_titles, display_scales)
                )
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Display results
            for i, (annotated_img, _, num_hands) in enumerate(results):
                if annotated_img is not None:
                    # Calculate FPS
                    current_time = time.time()
                    elapsed = current_time - last_time_list[i]
                    if elapsed > 0:
                        fps_list[i] = 0.9 * fps_list[i] + 0.1 * (1.0 / elapsed)
                    last_time_list[i] = current_time
                    
                    # Draw stats if enabled
                    if show_stats:
                        annotated_img = draw_stats(
                            annotated_img, i, detectors[i], fps_list[i], num_hands
                        )
                    
                    # Resize and display
                    h, w = annotated_img.shape[:2]
                    scale = max(display_scales[i], 0.01)
                    resized = cv2.resize(
                        annotated_img, 
                        (int(w * scale), int(h * scale)), 
                        interpolation=cv2.INTER_AREA
                    )
                    cv2.imshow(window_titles[i], resized)
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break
            elif key == ord('s'):  # Toggle stats
                show_stats = not show_stats
                print(f"Statistics display: {'ON' if show_stats else 'OFF'}")
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-camera hand detection with MediaPipe")
    parser.add_argument("--config", default="calibration_config.yaml", help="Path to calibration_config.yaml (default: calibration_config.yaml)")
    parser.add_argument("--select", action="store_true", 
                       help="Open selection dialogs interactively (ignores calibration_config.yaml)")
    parser.add_argument("--num", type=int, help="Number of cameras when using --select (default: 2)")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect per frame")
    parser.add_argument("--min-detection-conf", type=float, default=0.5, 
                       help="Minimum detection confidence (0.0-1.0)")
    parser.add_argument("--min-tracking-conf", type=float, default=0.5,
                       help="Minimum tracking confidence (0.0-1.0)")
    args = parser.parse_args()

    cameras: List[TISGrabberWrapper] = []
    detectors: List[HandDetector] = []
    window_titles: List[str] = []
    display_scales: List[float] = []
    
    # FPS tracking
    fps_list: List[float] = []
    last_time_list: List[float] = []
    
    # Display options
    show_stats = True
    
    try:
        # Initialize cameras
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
                window_titles.append(f"Camera {i} - Hand Detection")
                display_scales.append(0.5)
        else:
            # Load from calibration_config.yaml
            try:
                from config_manager import ConfigManager
                
                config_path = Path(args.config)
                if not config_path.exists():
                    print(f"Error: {config_path} not found.")
                    print("Run 01_create_configs.py first to create camera configurations.")
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
                
                # Load each camera
                for i in range(num_cameras):
                    cam_cfg = manager.get_camera_config(i)
                    xml_path = Path(cam_cfg['xml_path'])  # type: ignore
                    # カメラ個別のdisplay_scaleがあればそれを使用、なければグローバルを使用
                    scale = cam_cfg.get('display_scale', global_display_scale)  # type: ignore
                    
                    print(f"  [{i}] Loading {xml_path}...")
                    cam = open_from_config(xml_path)
                    if cam is None:
                        print(f"  ✗ Failed to load camera {i}")
                        return i + 1
                    
                    cameras.append(cam)
                    window_titles.append(f"Camera {i} - Hand Detection")
                    display_scales.append(scale)
                    print(f"  ✓ Camera {i} loaded")
                
            except ImportError:
                print("Error: config_manager.py not found. Cannot load calibration_config.yaml")
                print("Please ensure config_manager.py exists in the current directory.")
                return 1
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return 1
        
        # Initialize hand detectors for each camera
        print("\nInitializing MediaPipe Hand detectors...")
        for i in range(len(cameras)):
            detector = HandDetector(
                static_image_mode=False,
                max_num_hands=args.max_hands,
                min_detection_confidence=args.min_detection_conf,
                min_tracking_confidence=args.min_tracking_conf
            )
            detectors.append(detector)
            fps_list.append(0.0)
            last_time_list.append(time.time())
            print(f"  ✓ Detector {i} initialized")
        
        # Create windows
        for title in window_titles:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        
        print(f"\n✓ {len(cameras)} camera(s) ready with hand detection")
        print("\nControls:")
        print("  'q' or ESC: Quit")
        print("  's': Toggle statistics display")
        print()

        # Run async main loop
        asyncio.run(async_main(cameras, detectors, window_titles, display_scales))
                
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        for i, detector in enumerate(detectors):
            stats = detector.get_stats()
            print(f"Camera {i}:")
            print(f"  Total frames: {stats['frame_count']}")
            print(f"  Total detections: {stats['detection_count']}")
            print(f"  Avg detections/frame: {stats['avg_detections_per_frame']:.2f}")
            print(f"  Final FPS: {fps_list[i]:.1f}")
        
        # Release detectors
        for i, detector in enumerate(detectors):
            try:
                detector.release()
                print(f"  ✓ Detector {i} released")
            except Exception as e:
                print(f"  ✗ Error releasing detector {i}: {e}")
        
        # Release cameras
        for i, cam in enumerate(cameras):
            try:
                if cam is not None:
                    cam.stop_live()
                    cam.release()
                    print(f"  ✓ Camera {i} released")
            except Exception as e:
                print(f"  ✗ Error releasing camera {i}: {e}")
        
        # Close windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
