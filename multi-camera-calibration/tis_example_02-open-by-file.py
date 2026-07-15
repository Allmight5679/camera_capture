"""
This sample shows, how to use a XML configuration file 
for opening a video capture device.
"""

import os
import sys
import dotenv
from pathlib import Path
dotenv.load_dotenv()

TISGRABBER_DLL_PATH = Path(str(os.environ.get("TISGRABBER_DLL_PATH")))
sys.path.append(str(TISGRABBER_DLL_PATH.parent))

import ctypes
import tisgrabber as tis

ic = ctypes.cdll.LoadLibrary(TISGRABBER_DLL_PATH.as_posix())
tis.declareFunctions(ic) # type: ignore

ic.IC_InitLibrary(0)

hGrabber = ic.IC_CreateGrabber()

ic.IC_LoadDeviceStateFromFile(hGrabber, tis.T("config/device.xml")) # type: ignore
ic.IC_OpenVideoCaptureDevice(hGrabber)


if( ic.IC_IsDevValid(hGrabber)): 
    ic.IC_StartLive(hGrabber, 1)
    ic.IC_MsgBox( "Click OK to stop".encode("utf-8"),"Simple Live Video".encode("utf-8"))
    ic.IC_StopLive(hGrabber)
else:
    ic.IC_MsgBox("No device opened".encode("utf-8"), "Simple Live Video".encode("utf-8"),)

ic.IC_ReleaseGrabber(hGrabber)


