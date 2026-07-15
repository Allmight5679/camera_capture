"""
ConfigManager: マルチカメラシステムの設定管理

2層構造の設定管理:
- calibration_config.yaml: グローバル設定とキャリブレーション設定（display_scale, config_path, ボード設定など）
- config/multi_camera_config.yaml: 個別カメラ設定（自動生成）
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


class ConfigManager:
    """マルチカメラシステムの設定管理クラス（2層構造）"""
    
    DEFAULT_CONFIG_PATH = Path("calibration_config.yaml")
    DEFAULT_MULTI_CAMERA_CONFIG = Path("config/multi_camera_config.yaml")
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: グローバル設定ファイルのパス（デフォルト: calibration_config.yaml）
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.global_config: Dict[str, Any] = {}
        self.camera_config: Dict[str, Any] = {}
        self.multi_camera_config_path: Optional[Path] = None
        
        # PyYAMLの遅延インポート
        try:
            import yaml
            self.yaml = yaml
        except ImportError:
            print("警告: PyYAMLがインストールされていません。", file=sys.stderr)
            print("インストールするには: pip install pyyaml", file=sys.stderr)
            raise
    
    def load(self) -> bool:
        """
        calibration_config.yamlとmulti_camera_config.yamlを読み込む
        
        Returns:
            成功時True、失敗時False
        """
        # グローバル設定の読み込み
        if not self.config_path.exists():
            print(f"警告: {self.config_path} が見つかりません。", file=sys.stderr)
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.global_config = self.yaml.safe_load(f) or {}
        except Exception as e:
            print(f"エラー: {self.config_path}の読み込みに失敗しました: {e}", file=sys.stderr)
            return False
        
        # マルチカメラ設定のパスを取得
        config_path_str = self.global_config.get('config_path', str(self.DEFAULT_MULTI_CAMERA_CONFIG))
        self.multi_camera_config_path = Path(config_path_str)
        
        # マルチカメラ設定の読み込み
        if not self.multi_camera_config_path.exists():
            print(f"警告: {self.multi_camera_config_path} が見つかりません。", file=sys.stderr)
            return False
        
        try:
            with open(self.multi_camera_config_path, 'r', encoding='utf-8') as f:
                self.camera_config = self.yaml.safe_load(f) or {}
            return self.validate()
        except Exception as e:
            print(f"エラー: {self.multi_camera_config_path}の読み込みに失敗しました: {e}", file=sys.stderr)
            return False
    
    def save(self) -> bool:
        """
        グローバル設定とマルチカメラ設定を両方保存
        
        Returns:
            成功時True、失敗時False
        """
        # グローバル設定の保存
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(self.global_config, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"エラー: {self.config_path}の保存に失敗しました: {e}", file=sys.stderr)
            return False
        
        # マルチカメラ設定の保存
        if self.multi_camera_config_path is None:
            config_path_str = self.global_config.get('config_path', str(self.DEFAULT_MULTI_CAMERA_CONFIG))
            self.multi_camera_config_path = Path(config_path_str)
        
        try:
            self.multi_camera_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.multi_camera_config_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(self.camera_config, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"エラー: {self.multi_camera_config_path}の保存に失敗しました: {e}", file=sys.stderr)
            return False
    
    def validate(self) -> bool:
        """
        設定の妥当性を検証
        
        Returns:
            検証成功時True、失敗時False
        """
        # グローバル設定の検証
        if 'display_scale' not in self.global_config:
            print("警告: 'display_scale'が設定されていません。デフォルト値を使用します。", file=sys.stderr)
        
        if 'config_path' not in self.global_config:
            print("警告: 'config_path'が設定されていません。デフォルト値を使用します。", file=sys.stderr)
        
        # カメラ設定の検証
        if 'cameras' not in self.camera_config:
            print("エラー: 'cameras'キーが見つかりません。", file=sys.stderr)
            return False
        
        cameras = self.camera_config['cameras']
        if not isinstance(cameras, list):
            print("エラー: 'cameras'は配列である必要があります。", file=sys.stderr)
            return False
        
        if len(cameras) == 0:
            print("警告: カメラが1台も登録されていません。", file=sys.stderr)
            return False
        
        # 各カメラの設定を検証
        for i, cam in enumerate(cameras):
            if not isinstance(cam, dict):
                print(f"エラー: cameras[{i}]は辞書である必要があります。", file=sys.stderr)
                return False
            
            if 'xml_path' not in cam:
                print(f"エラー: cameras[{i}]に'xml_path'が設定されていません。", file=sys.stderr)
                return False
        
        return True
    
    def get_camera_count(self) -> int:
        """
        登録されているカメラの台数を取得
        
        Returns:
            カメラ台数
        """
        cameras = self.camera_config.get('cameras', [])
        return len(cameras)
    
    def get_camera_config(self, index: int) -> Optional[Dict[str, Any]]:
        """
        指定インデックスのカメラ設定を取得（グローバルdisplay_scaleを含む）
        
        Args:
            index: カメラのインデックス（0始まり）
        
        Returns:
            カメラ設定の辞書、存在しない場合はNone
        """
        cameras = self.camera_config.get('cameras', [])
        if 0 <= index < len(cameras):
            cam_cfg = cameras[index].copy()
            # グローバルdisplay_scaleを適用（個別設定がない場合）
            if 'display_scale' not in cam_cfg:
                cam_cfg['display_scale'] = self.get_display_scale()
            return cam_cfg
        return None
    
    def get_all_camera_configs(self) -> List[Dict[str, Any]]:
        """
        すべてのカメラ設定を取得（グローバルdisplay_scaleを適用）
        
        Returns:
            カメラ設定のリスト
        """
        cameras = self.camera_config.get('cameras', [])
        result = []
        for cam in cameras:
            cam_cfg = cam.copy()
            if 'display_scale' not in cam_cfg:
                cam_cfg['display_scale'] = self.get_display_scale()
            result.append(cam_cfg)
        return result
    
    def get_display_scale(self) -> float:
        """
        グローバルdisplay_scaleを取得
        
        Returns:
            display_scale値（デフォルト: 0.5）
        """
        return self.global_config.get('display_scale', 0.5)
    
    def set_display_scale(self, scale: float) -> None:
        """
        グローバルdisplay_scaleを設定
        
        Args:
            scale: 表示スケール
        """
        self.global_config['display_scale'] = scale
    
    def get_multi_camera_config_path(self) -> str:
        """
        マルチカメラ設定ファイルのパスを取得
        
        Returns:
            config_path値
        """
        return self.global_config.get('config_path', str(self.DEFAULT_MULTI_CAMERA_CONFIG))
    
    def add_camera(self, xml_path: str) -> bool:
        """
        カメラ設定を追加（display_scaleはグローバル設定を使用）
        
        Args:
            xml_path: カメラのXML設定ファイルパス
        
        Returns:
            成功時True
        """
        if 'cameras' not in self.camera_config:
            self.camera_config['cameras'] = []
        
        camera_config = {
            'xml_path': xml_path
        }
        
        self.camera_config['cameras'].append(camera_config)
        return True
    
    def remove_camera(self, index: int) -> bool:
        """
        指定インデックスのカメラ設定を削除
        
        Args:
            index: 削除するカメラのインデックス（0始まり）
        
        Returns:
            成功時True、失敗時False
        """
        cameras = self.camera_config.get('cameras', [])
        if 0 <= index < len(cameras):
            cameras.pop(index)
            return True
        return False
    
    def update_camera(self, index: int, xml_path: Optional[str] = None) -> bool:
        """
        指定インデックスのカメラ設定を更新
        
        Args:
            index: 更新するカメラのインデックス（0始まり）
            xml_path: 新しいXMLパス（Noneの場合は変更しない）
        
        Returns:
            成功時True、失敗時False
        """
        cameras = self.camera_config.get('cameras', [])
        if 0 <= index < len(cameras):
            if xml_path is not None:
                cameras[index]['xml_path'] = xml_path
            return True
        return False
    
    def create_default_config(self, num_cameras: int = 2, 
                            camera_xml_pattern: str = "config/camera{}.xml",
                            display_scale: float = 0.5,
                            multi_camera_config_path: Optional[str] = None) -> None:
        """
        デフォルトの設定を生成（2層構造）
        
        Args:
            num_cameras: カメラ台数（デフォルト: 2）
            camera_xml_pattern: XMLファイルパスのパターン（{}にインデックス+1が入る）
            display_scale: グローバル表示スケール（デフォルト: 0.5）
            multi_camera_config_path: マルチカメラ設定ファイルのパス
        """
        # グローバル設定
        if multi_camera_config_path is None:
            multi_camera_config_path = str(self.DEFAULT_MULTI_CAMERA_CONFIG)
        
        self.global_config = {
            'display_scale': display_scale,
            'config_path': multi_camera_config_path
        }
        
        self.multi_camera_config_path = Path(multi_camera_config_path)
        
        # カメラ設定
        self.camera_config = {
            'cameras': []
        }
        
        for i in range(num_cameras):
            xml_path = camera_xml_pattern.format(i + 1)
            self.add_camera(xml_path)


# 利用例
if __name__ == "__main__":
    # デフォルト設定の作成例
    manager = ConfigManager()
    manager.create_default_config(num_cameras=3, display_scale=0.5)
    
    print(f"グローバル display_scale: {manager.get_display_scale()}")
    print(f"マルチカメラ設定パス: {manager.get_multi_camera_config_path()}")
    print(f"カメラ台数: {manager.get_camera_count()}")
    
    for i in range(manager.get_camera_count()):
        cam_cfg = manager.get_camera_config(i)
        print(f"Camera {i}: {cam_cfg}")
    
    # 保存
    if manager.save():
        print(f"\n設定を保存しました:")
        print(f"  - {manager.config_path}")
        print(f"  - {manager.multi_camera_config_path}")
