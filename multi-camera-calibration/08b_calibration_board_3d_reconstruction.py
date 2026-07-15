"""
09_calibration_board_3d_reconstruction.py
キャリブレーションボード格子点の三次元再構成と可視化

キャリブレーション済みパラメータを用いて、全フレームのChArUco格子点を
ステレオ三角測量により三次元再構成し、カメラ姿勢と共にOpen3Dで可視化する。

実装内容:
1. キャリブレーションデータの読み込み
2. ChArUco検出キャッシュの読み込み
3. 対応点マッチングとステレオ三角測量
4. 3D点群の生成
5. Open3Dによる可視化（カメラ姿勢 + ボード格子点）
6. 統計情報の保存

実行方法:
    python 09_calibration_board_3d_reconstruction.py
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import open3d as o3d
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationBoard3DReconstruction:
    """キャリブレーションボード格子点の三次元再構成クラス"""
    
    def __init__(self, config_path="calibration_config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        
        # キャリブレーションデータ
        self.camera0_calib = None
        self.camera1_calib = None
        self.extrinsic = None
        
        # 検出データ
        self.detections_camera0 = {}
        self.detections_camera1 = {}
        
        # 再構成結果
        self.reconstructed_points = []  # 各点: {'frame_id', 'corner_id', 'point_3d', 'color'}
        self.frame_correspondences = []  # フレーム対応情報
        
        # 出力ディレクトリ
        self.output_dir = Path(self.config['paths']['output_dir']) / 'board_3d_reconstruction'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_calibration_data(self):
        """キャリブレーション結果の読み込み"""
        logger.info("="*60)
        logger.info("キャリブレーションデータの読み込み")
        logger.info("="*60)
        
        calib_dir = Path(self.config['paths']['output_dir']) / 'calibration'
        extrinsic_dir = Path(self.config['paths']['output_dir']) / 'extrinsic'
        
        # Camera0の内部パラメータ
        with open(calib_dir / 'calibration_camera0.json', 'r') as f:
            self.camera0_calib = json.load(f)
        logger.info(f"✓ Camera0: RMS={self.camera0_calib['rms_reprojection_error']:.4f} px")
        
        # Camera1の内部パラメータ
        with open(calib_dir / 'calibration_camera1.json', 'r') as f:
            self.camera1_calib = json.load(f)
        logger.info(f"✓ Camera1: RMS={self.camera1_calib['rms_reprojection_error']:.4f} px")
        
        # 外部パラメータ（Camera0 → Camera1）
        with open(extrinsic_dir / 'initial_extrinsic_camera0_to_camera1.json', 'r') as f:
            self.extrinsic = json.load(f)
        
        baseline = np.linalg.norm(self.extrinsic['t'])
        logger.info(f"✓ Extrinsic: Baseline={baseline:.4f} m, Angle={self.extrinsic['rotation_angle_deg']:.2f}°")
        logger.info("")
        
    def load_detection_caches(self):
        """ChArUco検出キャッシュの読み込み"""
        logger.info("="*60)
        logger.info("ChArUco検出キャッシュの読み込み")
        logger.info("="*60)
        
        cache_dir = Path(self.config['paths']['detection_cache'])
        
        # Camera0の検出結果
        with open(cache_dir / 'detections_camera0.json', 'r') as f:
            data0 = json.load(f)
        
        # フレームリストを辞書に変換（image_nameをキーに）
        self.detections_camera0 = {
            frame['image_name']: frame
            for frame in data0.get('frames', [])
        }
        logger.info(f"✓ Camera0: {len(self.detections_camera0)} フレーム")
        
        # Camera1の検出結果
        with open(cache_dir / 'detections_camera1.json', 'r') as f:
            data1 = json.load(f)
        
        self.detections_camera1 = {
            frame['image_name']: frame
            for frame in data1.get('frames', [])
        }
        logger.info(f"✓ Camera1: {len(self.detections_camera1)} フレーム")
        logger.info("")
        
    def find_corresponding_frames(self) -> List[Tuple[str, str]]:
        """
        対応するフレームペアを検索
        
        Returns:
            対応フレームペアのリスト [(image_name0, image_name1), ...]
        """
        # タイムスタンプから対応を検索（同じ画像名を持つペア）
        corresponding_frames = []
        
        for img_name0 in self.detections_camera0.keys():
            if img_name0 in self.detections_camera1:
                corresponding_frames.append((img_name0, img_name0))
        
        logger.info(f"対応フレームペア数: {len(corresponding_frames)}")
        return corresponding_frames
    
    def triangulate_points(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray
    ) -> np.ndarray:
        """
        ステレオ三角測量で3D点を計算
        
        Args:
            points1: Camera0の2D点 (N, 2)
            points2: Camera1の2D点 (N, 2)
            P1: Camera0の投影行列 (3, 4)
            P2: Camera1の投影行列 (3, 4)
            
        Returns:
            3D点 (N, 3)
        """
        # OpenCVのtriangulatePointsは (2, N) の形状を要求
        points1_t = points1.T.astype(np.float64)
        points2_t = points2.T.astype(np.float64)
        
        # 三角測量（結果は同次座標 (4, N)）
        points_4d_hom = cv2.triangulatePoints(P1, P2, points1_t, points2_t)
        
        # 同次座標から3D座標に変換
        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        
        return points_3d.T  # (N, 3)
    
    def reconstruct_frame_pair(
        self,
        img_name0: str,
        img_name1: str,
        frame_idx: int
    ) -> Dict:
        """
        1つのフレームペアについて三次元再構成
        
        Args:
            img_name0: Camera0の画像名
            img_name1: Camera1の画像名
            frame_idx: フレームインデックス
            
        Returns:
            再構成結果の辞書
        """
        det0 = self.detections_camera0[img_name0]
        det1 = self.detections_camera1[img_name1]
        
        # ChArUcoコーナーIDと座標を取得
        ids0 = np.array(det0['ch_ids'], dtype=np.int32).flatten()
        corners0 = np.array(det0['ch_corners'], dtype=np.float32).reshape(-1, 2)
        
        ids1 = np.array(det1['ch_ids'], dtype=np.int32).flatten()
        corners1 = np.array(det1['ch_corners'], dtype=np.float32).reshape(-1, 2)
        
        # 共通のコーナーIDを探索
        common_ids = np.intersect1d(ids0, ids1)
        
        if len(common_ids) == 0:
            return {
                'success': False,
                'num_points': 0
            }
        
        # 共通IDに対応する2D点を抽出
        mask0 = np.isin(ids0, common_ids)
        mask1 = np.isin(ids1, common_ids)
        
        # IDでソートして対応を確実にする
        sorted_indices0 = np.argsort(ids0[mask0])
        sorted_indices1 = np.argsort(ids1[mask1])
        
        matched_ids = ids0[mask0][sorted_indices0]
        matched_corners0 = corners0[mask0][sorted_indices0]
        matched_corners1 = corners1[mask1][sorted_indices1]
        
        # カメラ行列と外部パラメータを取得
        K0 = np.array(self.camera0_calib['camera_matrix']['K'], dtype=np.float64)
        dist0 = np.array(self.camera0_calib['distortion_coefficients'], dtype=np.float64)
        
        K1 = np.array(self.camera1_calib['camera_matrix']['K'], dtype=np.float64)
        dist1 = np.array(self.camera1_calib['distortion_coefficients'], dtype=np.float64)
        
        R = np.array(self.extrinsic['R'], dtype=np.float64)
        t = np.array(self.extrinsic['t'], dtype=np.float64).reshape(3, 1)
        
        # 歪み補正
        matched_corners0_undist = cv2.undistortPoints(
            matched_corners0.reshape(-1, 1, 2), K0, dist0, P=K0
        ).reshape(-1, 2)
        
        matched_corners1_undist = cv2.undistortPoints(
            matched_corners1.reshape(-1, 1, 2), K1, dist1, P=K1
        ).reshape(-1, 2)
        
        # 投影行列の構築
        # Camera0: [K | 0]
        P0 = np.hstack([K0, np.zeros((3, 1))])
        
        # Camera1: K[R | t]
        Rt = np.hstack([R, t])
        P1 = K1 @ Rt
        
        # 三角測量
        points_3d = self.triangulate_points(
            matched_corners0_undist,
            matched_corners1_undist,
            P0,
            P1
        )
        
        # フレームごとに色を割り当て（HSV色空間）
        hue = frame_idx / max(len(self.find_corresponding_frames()) - 1, 1)
        color = self._hsv_to_rgb(hue, 0.8, 0.9)
        
        # 再構成点を記録
        for i, (corner_id, pt_3d) in enumerate(zip(matched_ids, points_3d)):
            self.reconstructed_points.append({
                'frame_idx': frame_idx,
                'image_name': img_name0,
                'corner_id': int(corner_id),
                'point_3d': pt_3d.tolist(),
                'color': color.tolist(),
                'point_2d_camera0': matched_corners0[i].tolist(),
                'point_2d_camera1': matched_corners1[i].tolist()
            })
        
        return {
            'success': True,
            'num_points': len(matched_ids),
            'matched_ids': matched_ids.tolist(),
            'points_3d': points_3d.tolist()
        }
    
    def reconstruct_all_frames(self):
        """全フレームペアの三次元再構成"""
        logger.info("="*60)
        logger.info("三次元再構成の実行")
        logger.info("="*60)
        
        corresponding_frames = self.find_corresponding_frames()
        
        if len(corresponding_frames) == 0:
            logger.error("対応フレームが見つかりません")
            return
        
        successful_frames = 0
        total_points = 0
        
        for idx, (img_name0, img_name1) in enumerate(corresponding_frames):
            result = self.reconstruct_frame_pair(img_name0, img_name1, idx)
            
            if result['success']:
                successful_frames += 1
                total_points += result['num_points']
                logger.info(f"✓ Frame {idx:2d} ({img_name0}): {result['num_points']} points")
                
                self.frame_correspondences.append({
                    'frame_idx': idx,
                    'image_name0': img_name0,
                    'image_name1': img_name1,
                    'num_points': result['num_points'],
                    'matched_ids': result['matched_ids']
                })
            else:
                logger.warning(f"✗ Frame {idx:2d} ({img_name0}): 共通点なし")
        
        logger.info("")
        logger.info(f"再構成完了: {successful_frames}/{len(corresponding_frames)} フレーム成功")
        logger.info(f"総3D点数: {total_points}")
        logger.info("")
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """HSV色空間からRGB色空間に変換"""
        import colorsys
        return np.array(colorsys.hsv_to_rgb(h, s, v))
    
    def create_camera_frustum(
        self,
        K: np.ndarray,
        img_size: Tuple[int, int],
        scale: float = 0.3,
        color: np.ndarray = np.array([0.0, 0.0, 1.0])
    ) -> o3d.geometry.LineSet:
        """
        カメラ視錐台の作成
        
        Args:
            K: カメラ内部パラメータ行列
            img_size: 画像サイズ (width, height)
            scale: 視錐台のスケール
            color: 線の色
            
        Returns:
            視錐台のLineSet
        """
        width, height = img_size
        
        # 画像の4隅
        corners_2d = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # カメラ座標系での3D点を計算
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        corners_3d = []
        for x, y in corners_2d:
            # 正規化座標
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy
            # 奥行きscaleでスケーリング
            corners_3d.append([x_norm * scale, y_norm * scale, scale])
        
        corners_3d = np.array(corners_3d)
        
        # 頂点（カメラ中心 + 4隅）
        vertices = np.vstack([
            np.array([[0, 0, 0]]),  # カメラ中心
            corners_3d
        ])
        
        # エッジ
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # カメラ中心から4隅へ
            [1, 2], [2, 3], [3, 4], [4, 1]   # 4隅の矩形
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def visualize_3d_scene(self):
        """Open3Dで3Dシーンを可視化"""
        logger.info("="*60)
        logger.info("3D可視化の準備")
        logger.info("="*60)
        
        if len(self.reconstructed_points) == 0:
            logger.error("再構成点が存在しません")
            return
        
        geometries = []
        
        # ===== Camera0（原点） =====
        # 座標系
        camera0_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0, 0, 0]
        )
        geometries.append(camera0_frame)
        
        # 視錐台
        K0 = np.array(self.camera0_calib['camera_matrix']['K'])
        img_size0 = tuple(self.camera0_calib['img_size'])
        camera0_frustum = self.create_camera_frustum(
            K0, img_size0, scale=0.3, color=np.array([0.0, 0.5, 1.0])
        )
        geometries.append(camera0_frustum)
        
        # ===== Camera1 =====
        # 外部パラメータで変換
        R = np.array(self.extrinsic['R'])
        t = np.array(self.extrinsic['t']).reshape(3, 1)
        
        # Camera1の座標系
        camera1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0, 0, 0]
        )
        T1 = np.eye(4)
        T1[:3, :3] = R
        T1[:3, 3] = t.flatten()
        camera1_frame.transform(T1)
        geometries.append(camera1_frame)
        
        # Camera1の視錐台
        K1 = np.array(self.camera1_calib['camera_matrix']['K'])
        img_size1 = tuple(self.camera1_calib['img_size'])
        camera1_frustum = self.create_camera_frustum(
            K1, img_size1, scale=0.3, color=np.array([1.0, 0.5, 0.0])
        )
        camera1_frustum.transform(T1)
        geometries.append(camera1_frustum)
        
        # ===== ボード格子点（点群） =====
        points = np.array([pt['point_3d'] for pt in self.reconstructed_points])
        colors = np.array([pt['color'] for pt in self.reconstructed_points])
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)
        
        logger.info(f"✓ Camera0: 原点に配置（青色視錐台）")
        logger.info(f"✓ Camera1: 変換後位置に配置（橙色視錐台）")
        logger.info(f"✓ ボード格子点: {len(points)} 点（フレームごとに色分け）")
        logger.info("")
        logger.info("3D可視化ウィンドウを表示中...")
        logger.info("（マウス操作: 左ドラッグ=回転、右ドラッグ=移動、スクロール=ズーム）")
        logger.info("")
        
        # 可視化
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Calibration Board 3D Reconstruction",
            width=1280,
            height=960,
            left=50,
            top=50
        )
        
        logger.info("可視化完了")
        logger.info("")
    
    def compute_statistics(self) -> Dict:
        """統計情報の計算"""
        logger.info("="*60)
        logger.info("統計情報の計算")
        logger.info("="*60)
        
        if len(self.reconstructed_points) == 0:
            return {}
        
        points_3d = np.array([pt['point_3d'] for pt in self.reconstructed_points])
        num_points_per_frame = [f['num_points'] for f in self.frame_correspondences]
        
        # 基本統計
        stats = {
            'total_points': int(len(points_3d)),
            'num_frames': int(len(self.frame_correspondences)),
            'points_per_frame': {
                'mean': float(np.mean(num_points_per_frame)),
                'std': float(np.std(num_points_per_frame)),
                'min': int(np.min(num_points_per_frame)),
                'max': int(np.max(num_points_per_frame))
            },
            'spatial_extent': {
                'x_range': [float(points_3d[:, 0].min()), float(points_3d[:, 0].max())],
                'y_range': [float(points_3d[:, 1].min()), float(points_3d[:, 1].max())],
                'z_range': [float(points_3d[:, 2].min()), float(points_3d[:, 2].max())],
                'x_span_m': float(points_3d[:, 0].max() - points_3d[:, 0].min()),
                'y_span_m': float(points_3d[:, 1].max() - points_3d[:, 1].min()),
                'z_span_m': float(points_3d[:, 2].max() - points_3d[:, 2].min())
            },
            'centroid': {
                'x': float(points_3d[:, 0].mean()),
                'y': float(points_3d[:, 1].mean()),
                'z': float(points_3d[:, 2].mean())
            },
            'distance_from_camera0': {
                'mean': float(np.linalg.norm(points_3d, axis=1).mean()),
                'std': float(np.linalg.norm(points_3d, axis=1).std()),
                'min': float(np.linalg.norm(points_3d, axis=1).min()),
                'max': float(np.linalg.norm(points_3d, axis=1).max())
            }
        }
        
        logger.info(f"総3D点数: {stats['total_points']}")
        logger.info(f"フレーム数: {stats['num_frames']}")
        logger.info(f"平均点数/フレーム: {stats['points_per_frame']['mean']:.1f}")
        logger.info(f"空間範囲:")
        logger.info(f"  X: [{stats['spatial_extent']['x_range'][0]:.3f}, {stats['spatial_extent']['x_range'][1]:.3f}] m (幅 {stats['spatial_extent']['x_span_m']:.3f} m)")
        logger.info(f"  Y: [{stats['spatial_extent']['y_range'][0]:.3f}, {stats['spatial_extent']['y_range'][1]:.3f}] m (幅 {stats['spatial_extent']['y_span_m']:.3f} m)")
        logger.info(f"  Z: [{stats['spatial_extent']['z_range'][0]:.3f}, {stats['spatial_extent']['z_range'][1]:.3f}] m (幅 {stats['spatial_extent']['z_span_m']:.3f} m)")
        logger.info(f"重心: ({stats['centroid']['x']:.3f}, {stats['centroid']['y']:.3f}, {stats['centroid']['z']:.3f}) m")
        logger.info(f"Camera0からの距離: {stats['distance_from_camera0']['mean']:.3f} ± {stats['distance_from_camera0']['std']:.3f} m")
        logger.info("")
        
        return stats
    
    def _convert_to_serializable(self, obj):
        """numpy型をJSON serializable型に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_results(self):
        """再構成結果と統計情報の保存"""
        logger.info("="*60)
        logger.info("結果の保存")
        logger.info("="*60)
        
        timestamp = datetime.now().isoformat()
        
        # 統計情報を計算
        stats = self.compute_statistics()
        
        # 保存データの構築
        output_data = {
            'timestamp': timestamp,
            'config': self.config,
            'statistics': stats,
            'frame_correspondences': self._convert_to_serializable(self.frame_correspondences),
            'reconstructed_points': self._convert_to_serializable(self.reconstructed_points),
            'camera0_calibration': {
                'K': self.camera0_calib['camera_matrix']['K'],
                'dist': self.camera0_calib['distortion_coefficients'],
                'rms': self.camera0_calib['rms_reprojection_error']
            },
            'camera1_calibration': {
                'K': self.camera1_calib['camera_matrix']['K'],
                'dist': self.camera1_calib['distortion_coefficients'],
                'rms': self.camera1_calib['rms_reprojection_error']
            },
            'extrinsic': self.extrinsic
        }
        
        # JSON保存
        output_path = self.output_dir / 'reconstruction_results.json'
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ 再構成結果を保存: {output_path}")
        
        # 点群をPLY形式で保存（Open3D形式）
        if len(self.reconstructed_points) > 0:
            points = np.array([pt['point_3d'] for pt in self.reconstructed_points])
            colors = np.array([pt['color'] for pt in self.reconstructed_points])
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            ply_path = self.output_dir / 'reconstructed_board_points.ply'
            o3d.io.write_point_cloud(str(ply_path), pcd)
            logger.info(f"✓ 点群を保存: {ply_path}")
        
        logger.info("")
    
    def run(self):
        """メイン実行フロー"""
        logger.info("")
        logger.info("="*60)
        logger.info("キャリブレーションボード格子点 三次元再構成")
        logger.info("="*60)
        logger.info("")
        
        # データ読み込み
        self.load_calibration_data()
        self.load_detection_caches()
        
        # 三次元再構成
        self.reconstruct_all_frames()
        
        # 結果保存
        self.save_results()
        
        # 3D可視化
        self.visualize_3d_scene()
        
        logger.info("="*60)
        logger.info("処理完了")
        logger.info("="*60)


def main():
    """メイン関数"""
    reconstructor = CalibrationBoard3DReconstruction()
    reconstructor.run()


if __name__ == "__main__":
    main()
