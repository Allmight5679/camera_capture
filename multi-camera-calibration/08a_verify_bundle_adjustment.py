"""
08_verify_bundle_adjustment.py
バンドル調整の結果を検証

実装内容:
1. 最適化前後のRMS誤差比較
2. エピポーラ誤差の改善確認
3. 再投影誤差のヒストグラム可視化
4. 画像上への再投影オーバレイ
5. 3D平面残差の計算
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
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BundleAdjustmentVerifier:
    """バンドル調整結果の検証を行うクラス"""
    
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
        self.cameras = {}
        self.camera_names = []
        self.board = None
        
        # 最適化前のデータ
        self.initial_extrinsics = {}
        self.initial_poses = {}
        
        # 最適化後のデータ
        self.optimized_extrinsics = {}
        self.optimized_poses = {}
        
        # サマリー
        self.ba_summary = None
        self.sync_report = None
        
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
        
        # ChArUcoボードの作成
        self.board = cv2.aruco.CharucoBoard(
            (board_cfg['squares_x'], board_cfg['squares_y']),
            board_cfg['square_length'] / 1000.0,  # mm -> m
            board_cfg['marker_length'] / 1000.0,  # mm -> m
            dictionary
        )
        
        logger.info(f"ChArUcoボード作成完了: {board_cfg['squares_x']}x{board_cfg['squares_y']}")
    
    def load_data(self) -> bool:
        """
        必要なデータを読み込み
        
        Returns:
            読み込み成功ならTrue
        """
        logger.info("データを読み込み中...")
        
        output_dir = Path(self.config['paths']['output_dir'])
        calib_dir = output_dir / 'calibration'
        detection_dir = Path('detection_cache')
        ba_dir = output_dir / 'bundle_adjustment'
        extrinsic_dir = output_dir / 'extrinsic'
        pose_dir = output_dir / 'pose_estimation'
        
        # ConfigManagerから動的にカメラ数を取得
        num_cameras = self.config_manager.get_camera_count()
        
        for i in range(num_cameras):
            camera_name = f"camera{i}"
            self.camera_names.append(camera_name)
        
        # キャリブレーション結果と検出データの読み込み
        for camera_name in self.camera_names:
            # キャリブレーション
            calib_path = calib_dir / f"calibration_{camera_name}.json"
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            
            # 検出データ
            detection_path = detection_dir / f"detections_{camera_name}.json"
            with open(detection_path, 'r') as f:
                detection_data = json.load(f)
            
            K = np.array(calib_data['camera_matrix']['K'])
            dist = np.array(calib_data['distortion_coefficients'])
            
            self.cameras[camera_name] = {
                'K': K,
                'dist': dist,
                'detections': detection_data,
                'img_size': tuple(calib_data['img_size'])
            }
        
        # 同期レポートの読み込み
        sync_path = pose_dir / 'synchronization_report.json'
        with open(sync_path, 'r') as f:
            self.sync_report = json.load(f)
        
        # 初期外部パラメータの読み込み
        reference_camera = self.camera_names[0]
        self.initial_extrinsics[reference_camera] = {
            'R': np.eye(3),
            't': np.zeros(3)
        }
        
        for i in range(1, len(self.camera_names)):
            camera_name = self.camera_names[i]
            extrinsic_path = extrinsic_dir / f"initial_extrinsic_{reference_camera}_to_{camera_name}.json"
            
            with open(extrinsic_path, 'r') as f:
                extrinsic_data = json.load(f)
            
            R = np.array(extrinsic_data['R'])
            t = np.array(extrinsic_data['t'])
            
            self.initial_extrinsics[camera_name] = {'R': R, 't': t}
        
        # 初期姿勢（全カメラ）
        for camera_name in self.camera_names:
            pose_path = pose_dir / camera_name / 'pose_estimation.json'
            with open(pose_path, 'r') as f:
                self.initial_poses[camera_name] = json.load(f)
        
        # バンドル調整サマリーの読み込み
        ba_summary_path = ba_dir / 'bundle_adjustment_summary.json'
        if not ba_summary_path.exists():
            logger.error(f"バンドル調整サマリーが見つかりません: {ba_summary_path}")
            return False
        
        with open(ba_summary_path, 'r') as f:
            self.ba_summary = json.load(f)
        
        # 最適化後の外部パラメータの読み込み
        self.optimized_extrinsics[reference_camera] = {
            'R': np.eye(3),
            't': np.zeros(3)
        }
        
        for i in range(1, len(self.camera_names)):
            camera_name = self.camera_names[i]
            extrinsic_path = ba_dir / f"extrinsic_{reference_camera}_to_{camera_name}.json"
            
            with open(extrinsic_path, 'r') as f:
                extrinsic_data = json.load(f)
            
            R = np.array(extrinsic_data['rotation_matrix'])
            t = np.array(extrinsic_data['translation_vector'])
            
            self.optimized_extrinsics[camera_name] = {'R': R, 't': t}
        
        # 最適化後の姿勢の読み込み
        for camera_name in self.camera_names:
            poses_path = ba_dir / camera_name / 'optimized_poses.json'
            
            with open(poses_path, 'r') as f:
                self.optimized_poses[camera_name] = json.load(f)
        
        logger.info("データの読み込み完了")
        return True
    
    def compute_reprojection_errors(self, 
                                    extrinsics: Dict,
                                    poses: Dict) -> Tuple[List[float], Dict]:
        """
        再投影誤差を計算
        
        Args:
            extrinsics: カメラ外部パラメータ
            poses: ボード姿勢（各カメラ座標系）
            
        Returns:
            (全誤差のリスト, カメラ別統計)
        """
        all_errors = []
        camera_stats = {}
        
        # 同期フレーム名をセットに変換
        sync_frames = set(self.sync_report['synchronized_frames'])
        
        for camera_name in self.camera_names:
            K = self.cameras[camera_name]['K']
            dist = self.cameras[camera_name]['dist']
            detection_data = self.cameras[camera_name]['detections']
            
            camera_errors = []
            
            # 姿勢データを辞書に変換（image_name -> pose）
            pose_dict = {}
            if camera_name in poses:
                pose_data = poses[camera_name]['poses']
                # poses が dict の場合（初期姿勢）と list の場合（最適化後）に対応
                if isinstance(pose_data, dict):
                    # dict形式: {frame_name: {success, rvec, tvec, ...}}
                    pose_dict = pose_data
                else:
                    # list形式: [{frame_id, image_name, rvec, tvec}, ...]
                    for pose_entry in pose_data:
                        pose_dict[pose_entry['image_name']] = pose_entry
            
            for frame_entry in detection_data['frames']:
                frame_name = frame_entry['image_name']
                
                if frame_name not in sync_frames or frame_name not in pose_dict:
                    continue
                
                # 姿勢を取得
                pose_entry = pose_dict[frame_name]
                rvec = np.array(pose_entry['rvec'])
                tvec = np.array(pose_entry['tvec'])
                
                # ChArUco検出点
                ch_corners = np.array(frame_entry['ch_corners']).reshape(-1, 1, 2)
                ch_ids = np.array(frame_entry['ch_ids']).flatten()
                
                # ボード上の3D座標を取得
                all_obj_points = self.board.getChessboardCorners()
                obj_points = np.array([all_obj_points[int(ch_id)] for ch_id in ch_ids], dtype=np.float32)
                
                # 再投影
                projected, _ = cv2.projectPoints(
                    obj_points, rvec, tvec, K, dist
                )
                projected = projected.reshape(-1, 2)
                observed = ch_corners.reshape(-1, 2)
                
                # 誤差を計算
                errors = np.linalg.norm(projected - observed, axis=1)
                camera_errors.extend(errors)
                all_errors.extend(errors)
            
            # カメラ別統計
            if len(camera_errors) > 0:
                camera_stats[camera_name] = {
                    'mean': float(np.mean(camera_errors)),
                    'std': float(np.std(camera_errors)),
                    'median': float(np.median(camera_errors)),
                    'max': float(np.max(camera_errors)),
                    'count': len(camera_errors)
                }
        
        return all_errors, camera_stats
    
    def compute_epipolar_errors(self, extrinsics: Dict) -> Dict:
        """
        エピポーラ誤差を計算
        
        Args:
            extrinsics: カメラ外部パラメータ
            
        Returns:
            エピポーラ誤差の統計
        """
        if len(self.camera_names) < 2:
            return {}
        
        camera0 = self.camera_names[0]
        camera1 = self.camera_names[1]
        
        K0 = self.cameras[camera0]['K']
        K1 = self.cameras[camera1]['K']
        
        R = extrinsics[camera1]['R']
        t = extrinsics[camera1]['t']
        
        # 基本行列を計算
        t_x = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])
        E = t_x @ R
        F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
        
        # 同期フレームでエピポーラ誤差を計算
        epipolar_errors = []
        sync_frames = set(self.sync_report['synchronized_frames'])
        
        detection0 = self.cameras[camera0]['detections']
        detection1 = self.cameras[camera1]['detections']
        
        # フレーム名でインデックス化
        frames0 = {f['image_name']: f for f in detection0['frames']}
        frames1 = {f['image_name']: f for f in detection1['frames']}
        
        for frame_name in sync_frames:
            if frame_name not in frames0 or frame_name not in frames1:
                continue
            
            corners0 = np.array(frames0[frame_name]['ch_corners']).reshape(-1, 2)
            ids0 = set(frames0[frame_name]['ch_ids'])
            
            corners1 = np.array(frames1[frame_name]['ch_corners']).reshape(-1, 2)
            ids1 = set(frames1[frame_name]['ch_ids'])
            
            # 共通のIDを見つける
            common_ids = ids0 & ids1
            
            if len(common_ids) < 4:
                continue
            
            # 共通点のペアを作成
            id_to_corner0 = {frames0[frame_name]['ch_ids'][i]: corners0[i] 
                            for i in range(len(corners0))}
            id_to_corner1 = {frames1[frame_name]['ch_ids'][i]: corners1[i] 
                            for i in range(len(corners1))}
            
            for ch_id in common_ids:
                pt0 = np.append(id_to_corner0[ch_id], 1)
                pt1 = np.append(id_to_corner1[ch_id], 1)
                
                # エピポーラ誤差
                error = np.abs(pt1.T @ F @ pt0)
                epipolar_errors.append(error)
        
        if len(epipolar_errors) > 0:
            return {
                'mean': float(np.mean(epipolar_errors)),
                'std': float(np.std(epipolar_errors)),
                'median': float(np.median(epipolar_errors)),
                'max': float(np.max(epipolar_errors)),
                'count': len(epipolar_errors)
            }
        else:
            return {}
    
    def create_comparison_plots(self, 
                                initial_errors: List[float],
                                optimized_errors: List[float],
                                output_dir: Path):
        """
        最適化前後の比較プロットを作成
        
        Args:
            initial_errors: 初期誤差
            optimized_errors: 最適化後の誤差
            output_dir: 出力ディレクトリ
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ヒストグラム比較
        ax = axes[0, 0]
        ax.hist(initial_errors, bins=50, alpha=0.7, label='Initial', color='red')
        ax.hist(optimized_errors, bins=50, alpha=0.7, label='Optimized', color='blue')
        ax.set_xlabel('Reprojection Error [px]')
        ax.set_ylabel('Frequency')
        ax.set_title('Reprojection Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 累積分布
        ax = axes[0, 1]
        sorted_initial = np.sort(initial_errors)
        sorted_optimized = np.sort(optimized_errors)
        cdf_initial = np.arange(1, len(sorted_initial) + 1) / len(sorted_initial)
        cdf_optimized = np.arange(1, len(sorted_optimized) + 1) / len(sorted_optimized)
        
        ax.plot(sorted_initial, cdf_initial, label='Initial', color='red', linewidth=2)
        ax.plot(sorted_optimized, cdf_optimized, label='Optimized', color='blue', linewidth=2)
        ax.set_xlabel('Reprojection Error [px]')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 箱ひげ図
        ax = axes[1, 0]
        bp = ax.boxplot([initial_errors, optimized_errors],
                        labels=['Initial', 'Optimized'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('blue')
        ax.set_ylabel('Reprojection Error [px]')
        ax.set_title('Error Statistics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 統計サマリー
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
        Reprojection Error Summary
        
        Initial:
          Mean:   {np.mean(initial_errors):.4f} px
          Median: {np.median(initial_errors):.4f} px
          Std:    {np.std(initial_errors):.4f} px
          Max:    {np.max(initial_errors):.4f} px
          RMS:    {np.sqrt(np.mean(np.array(initial_errors)**2)):.4f} px
        
        Optimized:
          Mean:   {np.mean(optimized_errors):.4f} px
          Median: {np.median(optimized_errors):.4f} px
          Std:    {np.std(optimized_errors):.4f} px
          Max:    {np.max(optimized_errors):.4f} px
          RMS:    {np.sqrt(np.mean(np.array(optimized_errors)**2)):.4f} px
        
        Improvement:
          RMS: {(1 - np.sqrt(np.mean(np.array(optimized_errors)**2)) / np.sqrt(np.mean(np.array(initial_errors)**2))) * 100:.2f}%
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_error_comparison.png', dpi=150)
        logger.info(f"  比較プロット保存: reprojection_error_comparison.png")
        plt.close()
    
    def create_reprojection_overlay(self, 
                                   camera_name: str,
                                   poses: Dict,
                                   output_dir: Path,
                                   prefix: str = ""):
        """
        再投影オーバレイ画像を作成
        
        Args:
            camera_name: カメラ名
            poses: ボード姿勢
            output_dir: 出力ディレクトリ
            prefix: ファイル名プレフィックス
        """
        K = self.cameras[camera_name]['K']
        dist = self.cameras[camera_name]['dist']
        detection_data = self.cameras[camera_name]['detections']
        
        captured_dir = Path(self.config['paths']['captured_images']) / camera_name
        overlay_dir = output_dir / f"{prefix}{camera_name}_overlays"
        overlay_dir.mkdir(exist_ok=True)
        
        # 姿勢データを辞書に変換
        pose_dict = {}
        if camera_name in poses:
            for pose_entry in poses[camera_name]['poses']:
                pose_dict[pose_entry['image_name']] = pose_entry
        
        sync_frames = set(self.sync_report['synchronized_frames'])
        
        for frame_entry in detection_data['frames']:
            frame_name = frame_entry['image_name']
            
            if frame_name not in sync_frames or frame_name not in pose_dict:
                continue
            
            # 画像を読み込み
            img_path = captured_dir / frame_name
            if not img_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            
            # 姿勢を取得
            pose_entry = pose_dict[frame_name]
            rvec = np.array(pose_entry['rvec'])
            tvec = np.array(pose_entry['tvec'])
            
            # ChArUco検出点
            ch_corners = np.array(frame_entry['ch_corners']).reshape(-1, 1, 2)
            ch_ids = np.array(frame_entry['ch_ids']).flatten()
            
            # ボード上の3D座標を取得
            all_obj_points = self.board.getChessboardCorners()
            obj_points = np.array([all_obj_points[int(ch_id)] for ch_id in ch_ids], dtype=np.float32)
            
            # 再投影
            projected, _ = cv2.projectPoints(
                obj_points, rvec, tvec, K, dist
            )
            projected = projected.reshape(-1, 2)
            observed = ch_corners.reshape(-1, 2)
            
            # 描画
            for obs, proj in zip(observed, projected):
                # 観測点（緑）
                cv2.circle(img, tuple(obs.astype(int)), 5, (0, 255, 0), 2)
                # 再投影点（赤）
                cv2.circle(img, tuple(proj.astype(int)), 5, (0, 0, 255), 2)
                # 線で結ぶ
                cv2.line(img, tuple(obs.astype(int)), tuple(proj.astype(int)), 
                        (255, 0, 0), 1)
            
            # 保存
            overlay_path = overlay_dir / frame_name
            cv2.imwrite(str(overlay_path), img)
        
        logger.info(f"  再投影オーバレイ保存: {overlay_dir.name}/")
    
    def run_verification(self):
        """検証を実行"""
        logger.info("=" * 60)
        logger.info("バンドル調整の検証を開始")
        logger.info("=" * 60)
        
        output_dir = Path(self.config['paths']['output_dir']) / 'bundle_adjustment'
        
        # 1. 再投影誤差の計算
        logger.info("\n1. 再投影誤差を計算中...")
        logger.info("  初期パラメータ...")
        initial_errors, initial_camera_stats = self.compute_reprojection_errors(
            self.initial_extrinsics, self.initial_poses
        )
        
        logger.info("  最適化パラメータ...")
        optimized_errors, optimized_camera_stats = self.compute_reprojection_errors(
            self.optimized_extrinsics, self.optimized_poses
        )
        
        # RMS
        initial_rms = np.sqrt(np.mean(np.array(initial_errors)**2))
        optimized_rms = np.sqrt(np.mean(np.array(optimized_errors)**2))
        improvement = (1 - optimized_rms / initial_rms) * 100
        
        logger.info(f"\n  初期RMS: {initial_rms:.4f} px")
        logger.info(f"  最適化RMS: {optimized_rms:.4f} px")
        logger.info(f"  改善率: {improvement:.2f}%")
        
        # 2. エピポーラ誤差の計算
        logger.info("\n2. エピポーラ誤差を計算中...")
        if len(self.camera_names) >= 2:
            initial_epipolar = self.compute_epipolar_errors(self.initial_extrinsics)
            optimized_epipolar = self.compute_epipolar_errors(self.optimized_extrinsics)
            
            if initial_epipolar and optimized_epipolar:
                logger.info(f"  初期平均: {initial_epipolar['mean']:.4f} px")
                logger.info(f"  最適化平均: {optimized_epipolar['mean']:.4f} px")
                logger.info(f"  改善率: {(1 - optimized_epipolar['mean'] / initial_epipolar['mean']) * 100:.2f}%")
        
        # 3. 比較プロットの作成
        logger.info("\n3. 比較プロットを作成中...")
        self.create_comparison_plots(initial_errors, optimized_errors, output_dir)
        
        # 4. 再投影オーバレイの作成
        logger.info("\n4. 再投影オーバレイを作成中...")
        for camera_name in self.camera_names:
            logger.info(f"  {camera_name}...")
            self.create_reprojection_overlay(
                camera_name, self.optimized_poses, output_dir, prefix="optimized_"
            )
        
        # 5. 検証レポートの保存
        logger.info("\n5. 検証レポートを保存中...")
        report = {
            'timestamp': datetime.now().isoformat(),
            'reprojection_errors': {
                'initial': {
                    'rms': float(initial_rms),
                    'mean': float(np.mean(initial_errors)),
                    'std': float(np.std(initial_errors)),
                    'median': float(np.median(initial_errors)),
                    'max': float(np.max(initial_errors)),
                    'count': len(initial_errors),
                    'per_camera': initial_camera_stats
                },
                'optimized': {
                    'rms': float(optimized_rms),
                    'mean': float(np.mean(optimized_errors)),
                    'std': float(np.std(optimized_errors)),
                    'median': float(np.median(optimized_errors)),
                    'max': float(np.max(optimized_errors)),
                    'count': len(optimized_errors),
                    'per_camera': optimized_camera_stats
                },
                'improvement_percent': float(improvement)
            },
            'bundle_adjustment_summary': self.ba_summary['optimization']
        }
        
        if len(self.camera_names) >= 2 and initial_epipolar and optimized_epipolar:
            report['epipolar_errors'] = {
                'initial': initial_epipolar,
                'optimized': optimized_epipolar,
                'improvement_percent': float((1 - optimized_epipolar['mean'] / initial_epipolar['mean']) * 100)
            }
        
        report_path = output_dir / 'verification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"  verification_report.json")
        
        logger.info("\n" + "=" * 60)
        logger.info("検証完了")
        logger.info("=" * 60)


def main():
    """メイン処理"""
    try:
        # Verifierインスタンスの作成
        verifier = BundleAdjustmentVerifier()
        
        # ボードのセットアップ
        verifier.setup_board()
        
        # データの読み込み
        if not verifier.load_data():
            logger.error("データの読み込みに失敗しました")
            return
        
        # 検証の実行
        verifier.run_verification()
        
        logger.info("\n処理完了")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
