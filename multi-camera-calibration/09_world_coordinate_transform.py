"""
10_world_coordinate_transform.py
ワールド座標系の設定と座標変換

実装内容:
1. 設定ファイルからワールド座標系の基準を読み込み
2. Bundle Adjustment結果（camera0基準）を読み込み
3. 指定されたワールド座標系への変換行列を計算
4. カメラ外部パラメータとボード姿勢を変換
5. 変換後のデータを保存
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.transform import Rotation
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorldCoordinateTransform:
    """ワールド座標系変換を行うクラス"""
    
    def __init__(self, 
                 config_path: str = "calibration_config.yaml",
                 calibration_config_path: str = "calibration_config.yaml"):
        """
        初期化
        
        Args:
            config_path: カメラ設定ファイルのパス
            calibration_config_path: キャリブレーション設定ファイルのパス
        """
        self.config_manager = ConfigManager(Path(config_path))
        if not self.config_manager.load():
            raise RuntimeError(f"設定ファイルの読み込みに失敗: {config_path}")
        
        self.config = self._load_config(calibration_config_path)
        self.ba_dir = Path(self.config['paths']['output_dir']) / "bundle_adjustment"
        self.output_dir = self.ba_dir / "world_coordinate"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保持
        # ConfigManagerから動的にカメラ名を生成
        num_cameras = self.config_manager.get_camera_count()
        self.camera_names = [f"camera{i}" for i in range(num_cameras)]
        self.camera_extrinsics = {}  # {camera_name: {R, t}}
        self.camera_poses = {}  # {camera_name: [{rvec, tvec, frame_id, image_name}]}
        
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_bundle_adjustment_results(self):
        """Bundle Adjustment結果を読み込む"""
        logger.info("Loading Bundle Adjustment results...")
        
        # 外部パラメータを読み込み（camera0は恒等変換）
        self.camera_extrinsics[self.camera_names[0]] = {
            'R': np.eye(3),
            't': np.zeros(3)
        }
        
        # camera1以降の外部パラメータを読み込み
        for i in range(1, len(self.camera_names)):
            cam_i = self.camera_names[0]
            cam_j = self.camera_names[i]
            extrinsic_file = self.ba_dir / f"extrinsic_{cam_i}_to_{cam_j}.json"
            
            if not extrinsic_file.exists():
                raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_file}")
            
            with open(extrinsic_file, 'r') as f:
                data = json.load(f)
            
            self.camera_extrinsics[cam_j] = {
                'R': np.array(data['rotation_matrix']),
                't': np.array(data['translation_vector'])
            }
            logger.info(f"  Loaded extrinsic {cam_i} -> {cam_j}")
        
        # 各カメラのボード姿勢を読み込み
        for cam_name in self.camera_names:
            poses_file = self.ba_dir / cam_name / "optimized_poses.json"
            
            if not poses_file.exists():
                raise FileNotFoundError(f"Poses file not found: {poses_file}")
            
            with open(poses_file, 'r') as f:
                data = json.load(f)
            
            self.camera_poses[cam_name] = data['poses']
            logger.info(f"  Loaded {len(data['poses'])} poses for {cam_name}")
    
    def compute_transform_to_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ワールド座標系への変換行列を計算
        
        Returns:
            R_cam0_to_world: camera0 からワールド座標系への回転行列
            t_cam0_to_world: camera0 からワールド座標系への並進ベクトル
        """
        ref_type = self.config['world_coordinate']['reference_type']
        logger.info(f"Computing transform to world coordinate: {ref_type}")
        
        # カメラ基準の場合
        if ref_type.startswith('camera'):
            camera_name = ref_type
            if camera_name not in self.camera_names:
                raise ValueError(f"Unknown camera: {camera_name}")
            
            if camera_name == self.camera_names[0]:
                # camera0の場合は恒等変換
                logger.info("  World coordinate = camera0 (no transformation)")
                return np.eye(3), np.zeros(3)
            else:
                # world = camera_i の場合
                # camera0 -> camera_i の変換
                R_0_to_i = self.camera_extrinsics[camera_name]['R']
                t_0_to_i = self.camera_extrinsics[camera_name]['t']
                
                # これがそのまま camera0 -> world の変換
                R_cam0_to_world = R_0_to_i
                t_cam0_to_world = t_0_to_i
                
                logger.info(f"  World coordinate = {camera_name}")
                return R_cam0_to_world, t_cam0_to_world
        
        # ボード基準の場合
        elif ref_type in ['first_board', 'last_board'] or ref_type.startswith('board'):
            # フレームIDを決定
            if ref_type == 'first_board':
                frame_id = 0
            elif ref_type == 'last_board':
                frame_id = len(self.camera_poses[self.camera_names[0]]) - 1
            elif ref_type.startswith('board'):
                # board0, board1, ... の形式
                frame_id = int(ref_type.replace('board', ''))
            else:
                frame_id = self.config['world_coordinate'].get('frame_id', 0)
            
            # camera0での指定フレームのボード姿勢を取得
            cam0_poses = self.camera_poses[self.camera_names[0]]
            if frame_id < 0 or frame_id >= len(cam0_poses):
                raise ValueError(f"Invalid frame_id: {frame_id} (available: 0-{len(cam0_poses)-1})")
            
            pose = cam0_poses[frame_id]
            rvec = np.array(pose['rvec'])
            tvec = np.array(pose['tvec'])
            
            # board -> camera0 の変換（元データ）
            R_board_to_cam0, _ = cv2.Rodrigues(rvec)
            t_board_to_cam0 = tvec
            
            # world = board なので、camera0 -> board(=world) の逆変換を求める
            # P_cam0 = R_board_to_cam0 @ P_board + t_board_to_cam0
            # P_board = R_board_to_cam0.T @ (P_cam0 - t_board_to_cam0)
            #         = R_board_to_cam0.T @ P_cam0 - R_board_to_cam0.T @ t_board_to_cam0
            R_cam0_to_board = R_board_to_cam0.T
            t_cam0_to_board = -R_board_to_cam0.T @ t_board_to_cam0
            
            # ボード座標系調整: Z軸を上向きにする（設定で有効な場合）
            board_z_axis_up = self.config['world_coordinate'].get('board_z_axis_up', True)
            if board_z_axis_up:
                # x軸とy軸を入れ替え、z軸を反転
                R_board_adjust = np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, -1]], dtype=np.float64)
                
                # 追加変換を適用
                # 最終的なワールド座標系 = 調整後のボード座標系
                R_cam0_to_world = R_board_adjust @ R_cam0_to_board
                t_cam0_to_world = R_board_adjust @ t_cam0_to_board
                
                logger.info(f"  World coordinate = board at frame {frame_id} ({pose['image_name']})")
                logger.info(f"  Board coordinate adjustment applied: Z-axis up")
            else:
                # 調整なし
                R_cam0_to_world = R_cam0_to_board
                t_cam0_to_world = t_cam0_to_board
                
                logger.info(f"  World coordinate = board at frame {frame_id} ({pose['image_name']})")
                logger.info(f"  Board coordinate adjustment: disabled")
            
            return R_cam0_to_world, t_cam0_to_world
        
        else:
            raise ValueError(f"Unknown reference_type: {ref_type}")
    
    def transform_camera_extrinsics(self, R_cam0_to_world: np.ndarray, 
                                    t_cam0_to_world: np.ndarray) -> Dict:
        """
        カメラ外部パラメータをワールド座標系に変換
        
        Args:
            R_cam0_to_world: camera0 -> world の回転行列
            t_cam0_to_world: camera0 -> world の並進ベクトル
        
        Returns:
            transformed_extrinsics: 変換後の外部パラメータ（world -> camera_i）
        """
        transformed = {}
        
        for cam_name in self.camera_names:
            # camera0 -> camera_i の変換
            R_0_to_i = self.camera_extrinsics[cam_name]['R']
            t_0_to_i = self.camera_extrinsics[cam_name]['t']
            
            # world -> camera_i の変換を計算
            # P_cam_i = R_0_to_i @ P_cam0 + t_0_to_i
            # P_cam0 = R_cam0_to_world.T @ (P_world - t_cam0_to_world)
            #        = R_cam0_to_world.T @ P_world - R_cam0_to_world.T @ t_cam0_to_world
            # P_cam_i = R_0_to_i @ (R_cam0_to_world.T @ P_world - R_cam0_to_world.T @ t_cam0_to_world) + t_0_to_i
            #         = (R_0_to_i @ R_cam0_to_world.T) @ P_world + (- R_0_to_i @ R_cam0_to_world.T @ t_cam0_to_world + t_0_to_i)
            
            R_world_to_i = R_0_to_i @ R_cam0_to_world.T
            t_world_to_i = -R_0_to_i @ R_cam0_to_world.T @ t_cam0_to_world + t_0_to_i
            
            transformed[cam_name] = {
                'R': R_world_to_i,
                't': t_world_to_i
            }
        
        return transformed
    
    def transform_board_poses(self, R_cam0_to_world: np.ndarray, 
                              t_cam0_to_world: np.ndarray) -> Dict:
        """
        ボード姿勢をワールド座標系に変換
        
        Args:
            R_cam0_to_world: camera0 -> world の回転行列
            t_cam0_to_world: camera0 -> world の並進ベクトル
        
        Returns:
            transformed_poses: 変換後のボード姿勢（board -> camera_i の変換、world座標系基準）
            
        Note:
            projectPointsで使用するための board -> camera の変換を計算
            ワールド座標系で表現されたboardの3D点を、各カメラの画像平面に投影する
        """
        transformed = {}
        
        # world -> camera_i の変換を先に計算
        camera_transforms = self.transform_camera_extrinsics(R_cam0_to_world, t_cam0_to_world)
        
        for cam_name in self.camera_names:
            poses = []
            
            R_world_to_i = camera_transforms[cam_name]['R']
            t_world_to_i = camera_transforms[cam_name]['t']
            
            for pose in self.camera_poses[cam_name]:
                rvec = np.array(pose['rvec'])
                tvec = np.array(pose['tvec'])
                
                # 元データ: board -> camera_i の変換（camera0基準の座標系）
                R_board_to_i, _ = cv2.Rodrigues(rvec)
                t_board_to_i = tvec
                
                # board -> camera_i の変換（元データ、camera0基準）
                # このboardは現在のフレームでのboard位置
                
                # まず、このboardのworld座標系での位置を求める
                # 元データ: P_cam_i = R_board_to_i @ P_board_local + t_board_to_i
                # 
                # board -> camera0 の変換を求める
                if cam_name == self.camera_names[0]:
                    R_board_to_cam0 = R_board_to_i
                    t_board_to_cam0 = t_board_to_i
                else:
                    R_0_to_i = self.camera_extrinsics[cam_name]['R']
                    t_0_to_i = self.camera_extrinsics[cam_name]['t']
                    R_board_to_cam0 = R_0_to_i.T @ R_board_to_i
                    t_board_to_cam0 = R_0_to_i.T @ (t_board_to_i - t_0_to_i)
                
                # board -> world の変換
                # board の world座標系での位置（この特定フレームでの）
                R_board_to_world = R_cam0_to_world @ R_board_to_cam0
                t_board_to_world = R_cam0_to_world @ t_board_to_cam0 + t_cam0_to_world
                
                # projectPointsで使う変換は board_local -> camera_i
                # board_local はボードローカル座標（ChArUcoコーナー）
                # これを world座標に変換してから camera_i に変換
                # 
                # P_world = R_board_to_world @ P_board_local + t_board_to_world
                # P_cam_i = R_world_to_i @ P_world + t_world_to_i
                R_board_to_i_new = R_world_to_i @ R_board_to_world
                t_board_to_i_new = R_world_to_i @ t_board_to_world + t_world_to_i
                
                # Rodrigues変換
                rvec_new, _ = cv2.Rodrigues(R_board_to_i_new)
                
                poses.append({
                    'frame_id': pose['frame_id'],
                    'image_name': pose['image_name'],
                    'rvec': rvec_new.flatten().tolist(),
                    'tvec': t_board_to_i_new.tolist()
                })
            
            transformed[cam_name] = poses
        
        return transformed
    
    def save_results(self, R_cam0_to_world: np.ndarray, t_cam0_to_world: np.ndarray,
                    transformed_extrinsics: Dict, transformed_poses: Dict):
        """結果を保存"""
        logger.info("Saving results...")
        
        # ボード座標系調整が適用されたかチェック
        ref_type = self.config['world_coordinate']['reference_type']
        is_board_based = ref_type in ['first_board', 'last_board'] or ref_type.startswith('board')
        board_z_axis_up = self.config['world_coordinate'].get('board_z_axis_up', True)
        adjustment_applied = is_board_based and board_z_axis_up
        
        # 設定情報を保存
        config_data = {
            'timestamp': datetime.now().isoformat(),
            'reference_type': self.config['world_coordinate']['reference_type'],
            'frame_id': self.config['world_coordinate'].get('frame_id', None),
            'transform_cam0_to_world': {
                'rotation_matrix': R_cam0_to_world.tolist(),
                'translation_vector': t_cam0_to_world.tolist()
            },
            'board_coordinate_adjustment': {
                'enabled': board_z_axis_up,
                'applied': adjustment_applied,
                'adjustment_matrix': [[0, 1, 0], [1, 0, 0], [0, 0, -1]] if adjustment_applied else None,
                'description': 'Z-axis adjusted to point upward (swapped X-Y, inverted Z)' if adjustment_applied else 
                              ('Adjustment disabled by configuration' if is_board_based else 'Not applicable for camera-based coordinate system')
            },
            'note': 'All coordinates are now in the specified world coordinate system'
        }
        
        config_file = self.output_dir / "world_coordinate_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"  Saved config: {config_file}")
        
        # カメラ外部パラメータを保存
        for i in range(len(self.camera_names)):
            cam_name = self.camera_names[i]
            R = transformed_extrinsics[cam_name]['R']
            t = transformed_extrinsics[cam_name]['t']
            
            # 回転ベクトルと角度を計算
            rvec, _ = cv2.Rodrigues(R)
            angle_rad = np.linalg.norm(rvec)
            angle_deg = np.degrees(angle_rad)
            
            extrinsic_data = {
                'timestamp': datetime.now().isoformat(),
                'method': 'world_coordinate_transform',
                'camera_name': cam_name,
                'reference_type': self.config['world_coordinate']['reference_type'],
                'rotation_matrix': R.tolist(),
                'translation_vector': t.tolist(),
                'rotation_vector': rvec.flatten().tolist(),
                'rotation_angle_deg': float(angle_deg),
                'translation_norm_m': float(np.linalg.norm(t))
            }
            
            extrinsic_file = self.output_dir / f"extrinsic_{cam_name}.json"
            with open(extrinsic_file, 'w') as f:
                json.dump(extrinsic_data, f, indent=2)
            logger.info(f"  Saved extrinsic for {cam_name}: {extrinsic_file}")
        
        # カメラ間の相対外部パラメータも保存（参考用）
        for i in range(len(self.camera_names) - 1):
            cam_i = self.camera_names[i]
            cam_j = self.camera_names[i + 1]
            
            R_i = transformed_extrinsics[cam_i]['R']
            t_i = transformed_extrinsics[cam_i]['t']
            R_j = transformed_extrinsics[cam_j]['R']
            t_j = transformed_extrinsics[cam_j]['t']
            
            # camera_i -> camera_j の変換
            R_i_to_j = R_j @ R_i.T
            t_i_to_j = t_j - R_j @ R_i.T @ t_i
            
            rvec, _ = cv2.Rodrigues(R_i_to_j)
            angle_rad = np.linalg.norm(rvec)
            angle_deg = np.degrees(angle_rad)
            
            relative_data = {
                'timestamp': datetime.now().isoformat(),
                'method': 'world_coordinate_transform',
                'camera_pair': {'camera_i': cam_i, 'camera_j': cam_j},
                'rotation_matrix': R_i_to_j.tolist(),
                'translation_vector': t_i_to_j.tolist(),
                'rotation_vector': rvec.flatten().tolist(),
                'rotation_angle_deg': float(angle_deg),
                'translation_norm_m': float(np.linalg.norm(t_i_to_j))
            }
            
            relative_file = self.output_dir / f"extrinsic_{cam_i}_to_{cam_j}.json"
            with open(relative_file, 'w') as f:
                json.dump(relative_data, f, indent=2)
            logger.info(f"  Saved relative extrinsic {cam_i} -> {cam_j}")
        
        # ボード姿勢を保存
        for cam_name in self.camera_names:
            cam_dir = self.output_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            
            poses_data = {
                'timestamp': datetime.now().isoformat(),
                'camera_name': cam_name,
                'method': 'world_coordinate_transform',
                'reference_type': self.config['world_coordinate']['reference_type'],
                'num_poses': len(transformed_poses[cam_name]),
                'poses': transformed_poses[cam_name]
            }
            
            poses_file = cam_dir / "optimized_poses.json"
            with open(poses_file, 'w') as f:
                json.dump(poses_data, f, indent=2)
            logger.info(f"  Saved {len(transformed_poses[cam_name])} poses for {cam_name}")
    
    def run(self):
        """座標変換を実行"""
        logger.info("="*80)
        logger.info("World Coordinate Transform")
        logger.info("="*80)
        
        # Bundle Adjustment結果を読み込み
        self.load_bundle_adjustment_results()
        
        # ワールド座標系への変換行列を計算
        R_cam0_to_world, t_cam0_to_world = self.compute_transform_to_world()
        
        # カメラ外部パラメータを変換
        transformed_extrinsics = self.transform_camera_extrinsics(R_cam0_to_world, t_cam0_to_world)
        
        # ボード姿勢を変換
        transformed_poses = self.transform_board_poses(R_cam0_to_world, t_cam0_to_world)
        
        # 結果を保存
        self.save_results(R_cam0_to_world, t_cam0_to_world, 
                         transformed_extrinsics, transformed_poses)
        
        logger.info("="*80)
        logger.info("World Coordinate Transform completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80)


def main():
    """メイン処理"""
    import sys
    
    config_path = "calibration_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        transformer = WorldCoordinateTransform(config_path)
        transformer.run()
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
