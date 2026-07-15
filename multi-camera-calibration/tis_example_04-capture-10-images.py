"""
This sample demonstrates how to capture 10 consecutive images from a camera
and save them to files using tisgrabber.
"""

import os
import sys
import dotenv
from pathlib import Path
import time
from datetime import datetime
dotenv.load_dotenv()

TISGRABBER_DLL_PATH = Path(str(os.environ.get("TISGRABBER_DLL_PATH")))
sys.path.append(str(TISGRABBER_DLL_PATH.parent))

import ctypes
import tisgrabber as tis

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV (cv2) and NumPy are required.")
    print("Please install them with: pip install opencv-python numpy")
    sys.exit(1)

# Load the tisgrabber DLL
ic = ctypes.cdll.LoadLibrary(TISGRABBER_DLL_PATH.as_posix())
tis.declareFunctions(ic)  # type: ignore

# Initialize the library
ic.IC_InitLibrary(0)

# Create output directory for captured images
output_dir = Path("captured_images")
output_dir.mkdir(exist_ok=True)

# Open device using XML configuration or device selection dialog
hGrabber = ic.IC_CreateGrabber()

# Try to load device configuration from file
if Path("config/device.xml").exists():
    ic.IC_LoadDeviceStateFromFile(hGrabber, tis.T("config/device.xml"))  # type: ignore
    ic.IC_OpenVideoCaptureDevice(hGrabber)
else:
    # Show device selection dialog if no configuration file exists
    hGrabber = ic.IC_ShowDeviceSelectionDialog(None)

if ic.IC_IsDevValid(hGrabber):
    print("Camera opened successfully")
    
    # Start live video stream
    ic.IC_StartLive(hGrabber, 1)
    print("Live stream started")
    
    # Wait a bit for camera to stabilize
    time.sleep(1)
    
    # Capture 10 images
    num_images = 10
    successful_captures = 0
    
    for i in range(num_images):
        print(f"\nCapturing image {i + 1}/{num_images}...")
        
        # Snap an image with 2 second timeout
        if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS: # type: ignore
            # Declare variables for image description
            Width = ctypes.c_long()
            Height = ctypes.c_long()
            BitsPerPixel = ctypes.c_int()
            colorformat = ctypes.c_int()
            
            # Query the image description
            ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)
            
            # Calculate buffer size
            bpp = int(BitsPerPixel.value / 8.0)
            buffer_size = Width.value * Height.value * BitsPerPixel.value
            
            # Get the image data pointer
            imagePtr = ic.IC_GetImagePtr(hGrabber)
            
            # Cast to ctypes array
            imagedata = ctypes.cast(imagePtr, 
                                   ctypes.POINTER(ctypes.c_ubyte * buffer_size))
            
            # Create numpy array from image data
            image = np.ndarray(buffer=imagedata.contents,  # type: ignore
                             dtype=np.uint8,
                             shape=(Height.value, Width.value, bpp))
            
            # Flip the image (TIS cameras typically need vertical flip)
            image = cv2.flip(image, 0)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"image_{i+1:02d}_{timestamp}.png"
            
            # Save the image
            cv2.imwrite(str(filename), image)
            print(f"  ✓ Saved: {filename}")
            successful_captures += 1
            
            # Small delay between captures
            time.sleep(0.1)
        else:
            print(f"  ✗ Failed to capture image {i + 1}")
    
    # Stop live video stream
    ic.IC_StopLive(hGrabber)
    print(f"\n{'='*50}")
    print(f"Capture complete: {successful_captures}/{num_images} images saved")
    print(f"Images saved to: {output_dir.absolute()}")
    
else:
    print("Error: No device opened")

# Release the grabber
ic.IC_ReleaseGrabber(hGrabber)
