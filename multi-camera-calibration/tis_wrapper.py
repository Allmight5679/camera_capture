"""
tisgrabberラッパー: 依存関係の自動読み込みとカメラ基本アクセス
"""
import os
import sys
from pathlib import Path
import ctypes
import threading

# .envファイルを自動読み込み
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass  # dotenvがなくても動作可能

"""
tisgrabberラッパー: 依存関係の自動読み込みとカメラ基本アクセス

変更点（Singleton対応）
- DLL のロードと IC_InitLibrary をモジュール単位で一度だけ行う。
- 以後の TISGrabberWrapper インスタンスは同じライブラリハンドル(self.ic)を共有する。
"""

# --- Module-level singleton state for TIS library ---
_IC_SINGLETON = None           # ctypes.CDLL for tisgrabber
_IC_INIT_DONE = False          # IC_InitLibrary(0) が呼ばれたか
_IC_LOCK = threading.Lock()    # thread-safety (最低限)


def _load_ic_singleton(dll_path: Path):
    """DLL と tisgrabber.py を読み込み、宣言と IC_InitLibrary を一度だけ実行して共有する。

    Args:
        dll_path: tisgrabber_x64.dll への絶対パス

    Returns:
        ic: 共有の ctypes.CDLL ハンドル

    Raises:
        FileNotFoundError: 必要ファイルが見つからない場合
    """
    global _IC_SINGLETON, _IC_INIT_DONE

    with _IC_LOCK:
        # すでに初期化済みなら、そのまま返す
        if _IC_SINGLETON is not None:
            return _IC_SINGLETON

        tisgrabber_dir = dll_path.parent
        tisgrabber_py = tisgrabber_dir / "tisgrabber.py"
        if not tisgrabber_py.exists():
            raise FileNotFoundError(f"tisgrabber.py not found next to DLL: {tisgrabber_py}")

        # import パス確保（重複追加は避ける）
        if str(tisgrabber_dir) not in sys.path:
            sys.path.insert(0, str(tisgrabber_dir))

        # tisgrabber.py を直接ロード
        import importlib.util
        spec = importlib.util.spec_from_file_location("tisgrabber", tisgrabber_py)
        if not spec or not spec.loader:
            raise ImportError(f"Failed to load tisgrabber from {tisgrabber_py}")
        tisgrabber = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tisgrabber)

        # DLL をロード
        ic = ctypes.cdll.LoadLibrary(dll_path.as_posix())

        # 関数宣言
        tisgrabber.declareFunctions(ic)

        # ライブラリ初期化は一度だけ
        if not _IC_INIT_DONE:
            ic.IC_InitLibrary(0)
            _IC_INIT_DONE = True

        # シングルトンに固定
        _IC_SINGLETON = ic
        return ic


class TISGrabberWrapper:
    def capture_image(self, timeout_ms=2000):
        """
        カメラから画像を1枚取得し、OpenCV(numpy)形式で返す
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpyが必要です。pip install numpy でインストールしてください。")
        # Snap an image
        if self.ic.IC_SnapImage(self.hGrabber, timeout_ms) != 1:
            return None
        # 画像情報取得
        Width = ctypes.c_long()
        Height = ctypes.c_long()
        BitsPerPixel = ctypes.c_int()
        colorformat = ctypes.c_int()
        self.ic.IC_GetImageDescription(self.hGrabber, Width, Height, BitsPerPixel, colorformat)
        bpp = int(BitsPerPixel.value // 8)
        buffer_size = Width.value * Height.value * bpp
        imagePtr = self.ic.IC_GetImagePtr(self.hGrabber)
        imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
        img_bytes = bytes(imagedata.contents)
        image = np.frombuffer(img_bytes, dtype=np.uint8)
        image = image.reshape((Height.value, Width.value, bpp))
        return image
    def __init__(self, dll_env_var="TISGRABBER_DLL_PATH", dll_name="tisgrabber_x64.dll", py_name="tisgrabber.py"):
        """コンストラクタ

        注意: ライブラリのロードと初期化はシングルトンに委譲。
        """
        # DLLパスは環境変数のみ参照
        dll_path = os.environ.get(dll_env_var)
        if not dll_path:
            raise FileNotFoundError(f"{dll_name} not found. Set {dll_env_var} environment variable.")
        dll_path = Path(dll_path)
        self.dll_path = dll_path

        # シングルトンから共有ハンドルを取得
        self.ic = _load_ic_singleton(dll_path)
        self.hGrabber = None

    # 内部ヘルパ: 文字列をライブラリ渡し用に変換（T が無ければ utf-8 の c_char_p を使用）
    def _t(self, s: str):
        # 可能ならロード済みの tisgrabber モジュールの T を使用、なければ c_char_p にフォールバック
        mod = sys.modules.get("tisgrabber")
        if mod is not None:
            T = getattr(mod, "T", None)
            if callable(T):
                return T(s)
        return ctypes.c_char_p(str(s).encode("utf-8"))

    def select_device(self):
        """デバイス選択ダイアログを表示してカメラを選択"""
        self.hGrabber = self.ic.IC_ShowDeviceSelectionDialog(None)
        return self.hGrabber
    
    def create_grabber(self):
        """新しいグラバーを作成"""
        self.hGrabber = self.ic.IC_CreateGrabber()
        return self.hGrabber
    
    def open_video_capture_device(self):
        """ビデオキャプチャデバイスを開く（load_device_config後に使用）"""
        if self.hGrabber:
            return self.ic.IC_OpenVideoCaptureDevice(self.hGrabber)
        return 0
    
    def save_device_config(self, config_file="config/device.xml"):
        """現在のデバイス設定をXMLファイルに保存"""
        if self.is_device_valid():
            # configディレクトリが存在しない場合は作成
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            result = self.ic.IC_SaveDeviceStateToFile(self.hGrabber, self._t(config_file))
            return result == 1
        return False
    
    def load_device_config(self, config_file="config/device.xml"):
        """XMLファイルからデバイス設定を読み込み"""
        if self.hGrabber is None:
            self.create_grabber()
        # IC_LoadDeviceStateFromFileは設定をグラバーにロードする
        self.ic.IC_LoadDeviceStateFromFile(self.hGrabber, self._t(config_file))
        # デバイスを開く
        result = self.ic.IC_OpenVideoCaptureDevice(self.hGrabber)
        # デバイスが有効かチェック
        if self.is_device_valid():
            return True
        return False

    def is_device_valid(self):
        return bool(self.ic.IC_IsDevValid(self.hGrabber))

    def start_live(self):
        if self.is_device_valid():
            self.ic.IC_StartLive(self.hGrabber, 0)
            return True
        return False

    def stop_live(self):
        if self.is_device_valid():
            self.ic.IC_StopLive(self.hGrabber)

    def show_message(self, msg, title="TIS Camera"):
        self.ic.IC_MsgBox(self._t(msg), self._t(title))

    def release(self):
        if self.hGrabber:
            self.ic.IC_ReleaseGrabber(self.hGrabber)
            self.hGrabber = None

# 利用例
if __name__ == "__main__":
    grabber = TISGrabberWrapper()
    grabber.select_device()
    if grabber.is_device_valid():
        grabber.start_live()
        grabber.show_message("Click OK to stop", "Simple Live Video")
        grabber.stop_live()
    else:
        grabber.show_message("No device opened", "Simple Live Video")
    grabber.release()
