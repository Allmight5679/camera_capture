"""
Create camera configuration XML files for a multi-camera setup.
- Shows the device selection dialog for each camera you want to add.
- Saves XML files and updates calibration_config.yaml with camera configurations.

Usage (Windows cmd):
  python 01_create-configs.py --num 2
  python 01_create-configs.py --num 3 --pattern config/cam{}.xml
"""

from __future__ import annotations
import argparse
from pathlib import Path


def prompt_and_save(label: str, out_path: Path) -> bool:
    """Open the TIS device selection dialog and save the chosen camera's config.
    If config file already exists, load it first to show as default in dialog.
    Returns True on success.
    """
    # Import lazily so --help works without needing the DLL/env ready
    from tis_wrapper import TISGrabberWrapper

    print(f"=== Select {label} camera ===")
    grabber = TISGrabberWrapper()

    # 既存の設定ファイルがあれば読み込む
    if out_path.exists():
        print(f"Loading existing config from {out_path}...")
        grabber.create_grabber()
        try:
            grabber.ic.IC_LoadDeviceStateFromFile(
                grabber.hGrabber, grabber._t(str(out_path))
            )
            print(f"✓ Loaded existing configuration as default")
        except Exception as e:
            print(f"⚠ Could not load existing config: {e}")
            print(f"  Continuing with blank configuration...")

    # デバイス選択ダイアログを表示（既存設定があればそれがデフォルトとして表示される）
    grabber.select_device()

    ok = False
    try:
        if grabber.is_device_valid():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if grabber.save_device_config(str(out_path)):
                print(f"✓ Saved {label} config to {out_path}")
                ok = True
            else:
                print(f"✗ Failed to save {label} config to {out_path}")
        else:
            print(f"✗ No camera selected for {label}")
    finally:
        grabber.release()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Create TIS camera configs and update calibration_config.yaml"
    )
    parser.add_argument(
        "--num", type=int, default=2, help="Number of cameras to configure (default: 2)"
    )
    parser.add_argument(
        "--pattern",
        default="config/camera{}.xml",
        help="Output path pattern (use {} for camera number, default: config/camera{}.xml)",
    )
    parser.add_argument(
        "--config",
        default="calibration_config.yaml",
        help="Path to calibration_config.yaml (default: calibration_config.yaml)",
    )
    args = parser.parse_args()

    if args.num < 1:
        print("Error: Number of cameras must be at least 1.")
        return 1

    print(f"This script will configure {args.num} camera(s).")
    print("The device selection dialog will open for each camera.\n")

    # カメラ設定を順次作成
    camera_configs = []
    for i in range(args.num):
        camera_num = i + 1
        xml_path = Path(args.pattern.format(camera_num))

        label = f"Camera {camera_num}/{args.num}"
        ok = prompt_and_save(label, xml_path)

        if not ok:
            print(f"\nAborting: Camera {camera_num} configuration failed.")
            return 1

        camera_configs.append(xml_path)
        print()  # 空行

    # calibration_config.yamlを更新
    try:
        from config_manager import ConfigManager

        config_path = Path(args.config)
        manager = ConfigManager(config_path)

        # 既存の設定があれば読み込み、なければ新規作成
        if config_path.exists():
            print(f"Loading existing {config_path}...")
            manager.load()
            # 既存のカメラ設定をクリア
            manager.camera_config["cameras"] = []
        else:
            print(f"Creating new {config_path}...")
            manager.create_default_config(num_cameras=0, display_scale=0.5)

        # 新しいカメラ設定を追加
        for xml_path in camera_configs:
            manager.add_camera(str(xml_path))

        # 保存
        if manager.save():
            print(f"✓ Updated {config_path}")
        else:
            print(f"✗ Failed to update {config_path}")
            return 2

    except ImportError:
        print(
            "\n警告: config_manager.py が見つかりません。calibration_config.yamlは更新されません。"
        )
        print("XMLファイルのみが作成されました。")
    except Exception as e:
        print(f"\n警告: calibration_config.yamlの更新中にエラーが発生しました: {e}")
        print("XMLファイルは正常に作成されました。")

    print("\n✓ All done! Configured cameras:")
    for i, xml_path in enumerate(camera_configs):
        print(f" [{i}] {xml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
