"""
This sample shows, how to create an XML configuration file 
for a video capture device.
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

hGrabber = ic.IC_ShowDeviceSelectionDialog(None)

if(ic.IC_IsDevValid(hGrabber)):
    # configディレクトリが存在しない場合は作成
    from pathlib import Path
    Path("config").mkdir(exist_ok=True)
    ic.IC_SaveDeviceStateToFile(hGrabber, tis.T("config/device.xml")) # type: ignore
else:
    ic.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video")) # type: ignore

ic.IC_ReleaseGrabber(hGrabber)
