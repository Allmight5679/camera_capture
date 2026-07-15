"""
08_bundle_adjustment.py
ステレオ/マルチカメラの同時最適化（Bundle Adjustment）

実装内容:
1. 全カメラの検出データとキャリブレーション結果を読み込み
2. 初期外部パラメータと姿勢推定結果を読み込み
3. Bundle Adjustment: 全観測点の再投影誤差を最小化
4. 最適化されたカメラ外部パラメータとボード姿勢を保存
5. 統計情報の出力と可視化
"""

import cv2
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BundleAdjustment:
    """マルチカメラのバンドル調整を行うクラス"""
    
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
        
        # データ保持
        self.cameras = {}  # {camera_name: {K, dist, detections, poses}}
        self.camera_names = []
        self.board = None
        self.synchronized_frames = []
        
        # 最適化パラメータ
        self.camera_extrinsics = {}  # {camera_name: {'R': R, 't': t}}
        self.board_poses = {}  # {frame_id: {'R': R, 't': t}}
        
        # 最適化結果
        self.optimization_result = None
        self.initial_rms = None
        self.final_rms = None
        
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_board(self):
        """ChArUcoボードのセットアップ"""
        board_cfg = self.config['board']
        
        # ArUco辞書の取得
        dict_name = board_cfg['dictionary']
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        
        # ChArUcoボードの作成 (新しいAPI)
        self.board = cv2.aruco.CharucoBoard(
            (board_cfg['squares_x'], board_cfg['squares_y']),
            board_cfg['square_length'] / 1000.0,  # mm -> m
            board_cfg['marker_length'] / 1000.0,  # mm -> m
            dictionary
        )
        
        logger.info(f"ChArUcoボード作成完了: {board_cfg['squares_x']}x{board_cfg['squares_y']}")
    
    def load_camera_data(self) -> bool:
        """
        全カメラのデータを読み込み
        - キャリブレーション結果
        - 検出データ
        - 姿勢推定結果
        
        Returns:
            読み込み成功ならTrue
        """
        logger.info("カメラデータを読み込み中...")
        
        calib_dir = Path(self.config['paths']['output_dir']) / 'calibration'
        detection_dir = Path('detection_cache')
        pose_dir = Path(self.config['paths']['output_dir']) / 'pose_estimation'
        
        # ConfigManagerから動的にカメラ数を取得
        num_cameras = self.config_manager.get_camera_count()
        logger.info(f"カメラ数: {num_cameras}")
        
        for i in range(num_cameras):
            camera_name = f"camera{i}"
            self.camera_names.append(camera_name)
            
            # キャリブレーション結果の読み込み
            calib_path = calib_dir / f"calibration_{camera_name}.json"
            if not calib_path.exists():
                logger.error(f"キャリブレーション結果が見つかりません: {calib_path}")
                return False
            
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            
            # 検出データの読み込み
            detection_path = detection_dir / f"detections_{camera_name}.json"
            if not detection_path.exists():
                logger.error(f"検出データが見つかりません: {detection_path}")
                return False
            
            with open(detection_path, 'r') as f:
                detection_data = json.load(f)
            
            # 姿勢推定結果の読み込み
            pose_path = pose_dir / camera_name / 'pose_estimation.json'
            if not pose_path.exists():
                logger.error(f"姿勢推定結果が見つかりません: {pose_path}")
                return False
            
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            
            # カメラ行列と歪み係数
            K = np.array(calib_data['camera_matrix']['K'])
            dist = np.array(calib_data['distortion_coefficients'])
            
            self.cameras[camera_name] = {
                'K': K,
                'dist': dist,
                'detections': detection_data,
                'poses': pose_data,
                'img_size': tuple(calib_data['img_size'])
            }
            
            logger.info(f"  {camera_name}: {detection_data['num_frames']} フレーム")
        
        # 同期レポートの読み込み
        sync_path = pose_dir / 'synchronization_report.json'
        if not sync_path.exists():
            logger.error(f"同期レポートが見つかりません: {sync_path}")
            return False
        
        with open(sync_path, 'r') as f:
            sync_data = json.load(f)
        
        self.synchronized_frames = sync_data['synchronized_frames']
        logger.info(f"  同期フレーム数: {len(self.synchronized_frames)}")
        
        return True
    
    def load_initial_extrinsics(self) -> bool:
        """
        初期外部パラメータの読み込み
        
        Returns:
            読み込み成功ならTrue
        """
        logger.info("初期外部パラメータを読み込み中...")
        
        extrinsic_dir = Path(self.config['paths']['output_dir']) / 'extrinsic'
        
        # 最初のカメラを基準座標系とする
        reference_camera = self.camera_names[0]
        self.camera_extrinsics[reference_camera] = {
            'R': np.eye(3),
            't': np.zeros(3)
        }
        logger.info(f"  {reference_camera}: 基準座標系（恒等変換）")
        
        # 他のカメラの外部パラメータを読み込み
        for i in range(1, len(self.camera_names)):
            camera_name = self.camera_names[i]
            extrinsic_path = extrinsic_dir / f"initial_extrinsic_{reference_camera}_to_{camera_name}.json"
            
            if not extrinsic_path.exists():
                logger.error(f"外部パラメータが見つかりません: {extrinsic_path}")
                return False
            
            with open(extrinsic_path, 'r') as f:
                extrinsic_data = json.load(f)
            
            R = np.array(extrinsic_data['R'])
            t = np.array(extrinsic_data['t'])
            
            self.camera_extrinsics[camera_name] = {
                'R': R,
                't': t
            }
            
            logger.info(f"  {camera_name}: R={R.shape}, t={t.shape}")
        
        return True
    
    def initialize_board_poses(self):
        """
        各フレームのボード姿勢の初期値を設定
        基準カメラの姿勢推定結果を使用
        """
        logger.info("ボード姿勢の初期値を設定中...")
        
        reference_camera = self.camera_names[0]
        pose_data = self.cameras[reference_camera]['poses']
        
        # フレームIDをマッピング
        frame_name_to_id = {}
        for i, frame_name in enumerate(self.synchronized_frames):
            frame_name_to_id[frame_name] = i
        
        # 各フレームの姿勢を取得
        # poses は辞書形式 {frame_name: {success, rvec, tvec, ...}}
        for frame_name, pose_entry in pose_data['poses'].items():
            if frame_name not in frame_name_to_id:
                continue
            
            if not pose_entry.get('success', False):
                continue
            
            frame_id = frame_name_to_id[frame_name]
            
            rvec = np.array(pose_entry['rvec'])
            tvec = np.array(pose_entry['tvec'])
            
            R, _ = cv2.Rodrigues(rvec)
            
            self.board_poses[frame_id] = {
                'R': R,
                't': tvec,
                'frame_name': frame_name
            }
        
        logger.info(f"  {len(self.board_poses)} フレームのボード姿勢を初期化")
    
    def parameters_to_vector(self) -> np.ndarray:
        """
        最適化パラメータをベクトルに変換
        
        パラメータの順序:
        1. カメラ外部パラメータ (各カメラ6次元: rvec(3) + tvec(3))
           ※ 基準カメラは除く
        2. ボード姿勢 (各フレーム6次元: rvec(3) + tvec(3))
        
        Returns:
            パラメータベクトル
        """
        params = []
        
        # カメラ外部パラメータ（基準カメラを除く）
        for camera_name in self.camera_names[1:]:
            R = self.camera_extrinsics[camera_name]['R']
            t = self.camera_extrinsics[camera_name]['t']
            
            rvec, _ = cv2.Rodrigues(R)
            params.extend(rvec.flatten())
            params.extend(t.flatten())
        
        # ボード姿勢
        for frame_id in sorted(self.board_poses.keys()):
            R = self.board_poses[frame_id]['R']
            t = self.board_poses[frame_id]['t']
            
            rvec, _ = cv2.Rodrigues(R)
            params.extend(rvec.flatten())
            params.extend(t.flatten())
        
        return np.array(params)
    
    def vector_to_parameters(self, params: np.ndarray):
        """
        ベクトルから最適化パラメータを復元
        
        Args:
            params: パラメータベクトル
        """
        idx = 0
        
        # カメラ外部パラメータ（基準カメラを除く）
        for camera_name in self.camera_names[1:]:
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            idx += 6
            
            R, _ = cv2.Rodrigues(rvec)
            self.camera_extrinsics[camera_name]['R'] = R
            self.camera_extrinsics[camera_name]['t'] = tvec
        
        # ボード姿勢
        for frame_id in sorted(self.board_poses.keys()):
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            idx += 6
            
            R, _ = cv2.Rodrigues(rvec)
            self.board_poses[frame_id]['R'] = R
            self.board_poses[frame_id]['t'] = tvec
    
    def compute_residuals(self, params: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        再投影誤差を計算
        
        Args:
            params: パラメータベクトル
            verbose: 詳細情報を出力するか
            
        Returns:
            残差ベクトル (2 * 総観測点数,)
        """
        # パラメータを復元
        self.vector_to_parameters(params)
        
        residuals = []
        
        # 各カメラ、各フレームについて処理
        for camera_name in self.camera_names:
            K = self.cameras[camera_name]['K']
            dist = self.cameras[camera_name]['dist']
            
            # カメラの外部パラメータ
            R_cam = self.camera_extrinsics[camera_name]['R']
            t_cam = self.camera_extrinsics[camera_name]['t']
            
            # 検出データからフレーム情報を取得
            detection_data = self.cameras[camera_name]['detections']
            
            for frame_entry in detection_data['frames']:
                frame_name = frame_entry['image_name']
                
                if frame_name not in self.synchronized_frames:
                    continue
                
                # フレームIDを取得
                frame_id = None
                for fid, pose in self.board_poses.items():
                    if pose['frame_name'] == frame_name:
                        frame_id = fid
                        break
                
                if frame_id is None:
                    continue
                
                # ボードの姿勢
                R_board = self.board_poses[frame_id]['R']
                t_board = self.board_poses[frame_id]['t']
                
                # ボードからカメラへの変換
                # P_cam = R_cam * (R_board * P_board + t_board) + t_cam
                R = R_cam @ R_board
                t = R_cam @ t_board + t_cam
                
                rvec, _ = cv2.Rodrigues(R)
                
                # ChArUco検出点
                ch_corners = np.array(frame_entry['ch_corners']).reshape(-1, 1, 2)
                ch_ids = np.array(frame_entry['ch_ids']).flatten()
                
                # ボード上の3D座標を取得
                all_obj_points = self.board.getChessboardCorners()
                # 指定されたIDの点のみを抽出
                obj_points = np.array([all_obj_points[int(ch_id)] for ch_id in ch_ids], dtype=np.float32)
                
                # 再投影
                projected, _ = cv2.projectPoints(
                    obj_points, rvec, t, K, dist
                )
                projected = projected.reshape(-1, 2)
                observed = ch_corners.reshape(-1, 2)
                
                # 残差を計算
                error = (projected - observed).flatten()
                residuals.extend(error)
        
        residuals = np.array(residuals)
        
        if verbose:
            rms = np.sqrt(np.mean(residuals**2))
            logger.info(f"  RMS: {rms:.4f} px")
        
        return residuals
    
    def compute_rms(self, params: np.ndarray) -> float:
        """
        RMS再投影誤差を計算
        
        Args:
            params: パラメータベクトル
            
        Returns:
            RMS誤差 [px]
        """
        residuals = self.compute_residuals(params, verbose=False)
        return np.sqrt(np.mean(residuals**2))
    
    def optimize(self, 
                 loss: str = 'linear',
                 max_nfev: int = 1000,
                 ftol: float = 1e-8,
                 verbose: int = 2) -> bool:
        """
        バンドル調整を実行
        
        Args:
            loss: ロバストロス関数 ('linear', 'huber', 'soft_l1', 'cauchy')
            max_nfev: 最大関数評価回数
            ftol: 許容誤差
            verbose: 詳細レベル (0: なし, 1: 最小限, 2: 詳細)
            
        Returns:
            最適化成功ならTrue
        """
        logger.info("=" * 60)
        logger.info("バンドル調整を開始")
        logger.info("=" * 60)
        
        # 初期パラメータ
        params_init = self.parameters_to_vector()
        logger.info(f"最適化パラメータ数: {len(params_init)}")
        logger.info(f"  カメラ外部: {6 * (len(self.camera_names) - 1)} (基準カメラを除く)")
        logger.info(f"  ボード姿勢: {6 * len(self.board_poses)}")
        
        # 初期RMS
        self.initial_rms = self.compute_rms(params_init)
        logger.info(f"初期RMS: {self.initial_rms:.4f} px")
        
        # 最適化
        logger.info("最適化中...")
        result = least_squares(
            self.compute_residuals,
            params_init,
            method='trf',  # Trust Region Reflective
            loss=loss,
            ftol=ftol,
            max_nfev=max_nfev,
            verbose=verbose
        )
        
        self.optimization_result = result
        
        # 最適化後のパラメータを設定
        self.vector_to_parameters(result.x)
        
        # 最終RMS
        self.final_rms = self.compute_rms(result.x)
        
        logger.info("=" * 60)
        logger.info("バンドル調整完了")
        logger.info("=" * 60)
        logger.info(f"初期RMS: {self.initial_rms:.4f} px")
        logger.info(f"最終RMS: {self.final_rms:.4f} px")
        logger.info(f"改善率: {(1 - self.final_rms / self.initial_rms) * 100:.2f}%")
        logger.info(f"反復回数: {result.nfev}")
        logger.info(f"成功: {result.success}")
        logger.info(f"終了理由: {result.message}")
        
        return result.success
    
    def save_results(self, output_dir: Optional[str] = None):
        """
        最適化結果を保存
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合は設定ファイルから取得）
        """
        if output_dir is None:
            output_dir = Path(self.config['paths']['output_dir']) / 'bundle_adjustment'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"結果を保存中: {output_dir}")
        
        # カメラ外部パラメータの保存
        reference_camera = self.camera_names[0]
        
        for i in range(1, len(self.camera_names)):
            camera_name = self.camera_names[i]
            
            R = self.camera_extrinsics[camera_name]['R']
            t = self.camera_extrinsics[camera_name]['t']
            rvec, _ = cv2.Rodrigues(R)
            
            # 回転角度と並進ノルムを計算
            rot = Rotation.from_matrix(R)
            angle_deg = np.linalg.norm(rot.as_rotvec()) * 180 / np.pi
            translation_norm = np.linalg.norm(t)
            
            extrinsic_result = {
                'timestamp': datetime.now().isoformat(),
                'method': 'bundle_adjustment',
                'camera_pair': {
                    'camera_i': reference_camera,
                    'camera_j': camera_name
                },
                'rotation_matrix': R.tolist(),
                'translation_vector': t.tolist(),
                'rotation_vector': rvec.flatten().tolist(),
                'rotation_angle_deg': float(angle_deg),
                'translation_norm_m': float(translation_norm)
            }
            
            extrinsic_path = output_dir / f"extrinsic_{reference_camera}_to_{camera_name}.json"
            with open(extrinsic_path, 'w') as f:
                json.dump(extrinsic_result, f, indent=2)
            
            logger.info(f"  {extrinsic_path.name}")
        
        # ボード姿勢の保存（各カメラ座標系での姿勢）
        for camera_name in self.camera_names:
            poses_dir = output_dir / camera_name
            poses_dir.mkdir(exist_ok=True)
            
            R_cam = self.camera_extrinsics[camera_name]['R']
            t_cam = self.camera_extrinsics[camera_name]['t']
            
            poses_list = []
            
            for frame_id in sorted(self.board_poses.keys()):
                R_board = self.board_poses[frame_id]['R']
                t_board = self.board_poses[frame_id]['t']
                frame_name = self.board_poses[frame_id]['frame_name']
                
                # ボードからカメラへの変換
                R = R_cam @ R_board
                t = R_cam @ t_board + t_cam
                
                rvec, _ = cv2.Rodrigues(R)
                
                poses_list.append({
                    'frame_id': int(frame_id),
                    'image_name': frame_name,
                    'rvec': rvec.flatten().tolist(),
                    'tvec': t.tolist()
                })
            
            poses_data = {
                'timestamp': datetime.now().isoformat(),
                'camera_name': camera_name,
                'method': 'bundle_adjustment',
                'num_poses': len(poses_list),
                'poses': poses_list
            }
            
            poses_path = poses_dir / 'optimized_poses.json'
            with open(poses_path, 'w') as f:
                json.dump(poses_data, f, indent=2)
            
            logger.info(f"  {camera_name}/optimized_poses.json")
        
        # サマリーの保存
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cameras': self.camera_names,
            'num_frames': len(self.board_poses),
            'optimization': {
                'initial_rms': float(self.initial_rms),
                'final_rms': float(self.final_rms),
                'improvement_percent': float((1 - self.final_rms / self.initial_rms) * 100),
                'num_iterations': int(self.optimization_result.nfev),
                'success': bool(self.optimization_result.success),
                'message': str(self.optimization_result.message)
            }
        }
        
        summary_path = output_dir / 'bundle_adjustment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  bundle_adjustment_summary.json")
        logger.info("保存完了")


def main():
    """メイン処理"""
    try:
        # BundleAdjustmentインスタンスの作成
        ba = BundleAdjustment()
        
        # ボードのセットアップ
        ba.setup_board()
        
        # データの読み込み
        if not ba.load_camera_data():
            logger.error("カメラデータの読み込みに失敗しました")
            return
        
        if not ba.load_initial_extrinsics():
            logger.error("初期外部パラメータの読み込みに失敗しました")
            return
        
        # ボード姿勢の初期化
        ba.initialize_board_poses()
        
        # バンドル調整の実行
        success = ba.optimize(
            loss='huber',  # ロバストロス関数
            max_nfev=1000,
            ftol=1e-8,
            verbose=2
        )
        
        if not success:
            logger.warning("最適化は完全には収束しませんでしたが、結果を保存します")
        
        # 結果の保存
        ba.save_results()
        
        logger.info("=" * 60)
        logger.info("処理完了")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
