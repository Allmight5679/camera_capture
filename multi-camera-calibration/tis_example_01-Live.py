'''
This sample demonstrates, how to open a camera with the
built in device selection dialog and show a live video stream.
Needed DLLs for 64 bit environment are
- tisgrabber_x64.dll
- TIS_UDSHL11_x64.dll
- tisgrabber.py
'''
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

hGrabber = ic.IC_ShowDeviceSelectionDialog(None)

if(ic.IC_IsDevValid(hGrabber)):
    ic.IC_StartLive(hGrabber, 1)
    ic.IC_MsgBox(tis.T("Click OK to stop"), tis.T("Simple Live Video")) # type: ignore
    ic.IC_StopLive(hGrabber)
else:
    ic.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video")) # type: ignore

ic.IC_ReleaseGrabber(hGrabber)
