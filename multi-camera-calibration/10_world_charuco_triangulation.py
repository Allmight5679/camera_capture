"""
11_world_charuco_triangulation.py
ワールド座標系でのChArUcoボード格子点の三角測量

実装内容:
1. ワールド座標系パラメータの読み込み
2. 各カメラ画像からChArUcoボードを検出
3. 対応点マッチングとステレオ三角測量（world座標系）
4. 再投影誤差による精度検証
5. Open3Dによる3D可視化
6. 結果の保存

実行方法:
    python 11_world_charuco_triangulation.py
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
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorldCharucoTriangulation:
    """ワールド座標系でのChArUco三角測量クラス"""
    
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
        
        # パス設定
        self.output_dir = Path(self.config['paths']['output_dir']) / 'world_charuco_triangulation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_details_dir = self.output_dir / 'frame_details'
        self.frame_details_dir.mkdir(parents=True, exist_ok=True)
        
        # カメラ名（ConfigManagerから動的に生成）
        num_cameras = self.config_manager.get_camera_count()
        self.camera_names = [f"camera{i}" for i in range(num_cameras)]
        
        # キャリブレーションデータ
        self.camera_calibrations = {}  # {camera_name: {K, dist, img_size}}
        self.camera_extrinsics = {}    # {camera_name: {R, t}} (world -> camera)
        self.world_config = None
        
        # ChArUco検出結果
        self.detections = {}  # {camera_name: [{frame_id, image_name, ids, corners}]}
        
        # 三角測量結果
        self.triangulated_points = []  # [{frame_id, corner_id, point_3d_world, color, ...}]
        self.frame_results = []  # フレームごとの結果
        
        # ChArUcoボード設定
        self.board = self._create_charuco_board()
        
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_charuco_board(self):
        """ChArUcoボードを作成"""
        board_config = self.config['board']
        
        # ArUcoディクショナリ
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, board_config['dictionary'])
        )
        
        # ChArUcoボード
        board = cv2.aruco.CharucoBoard(
            (board_config['squares_x'], board_config['squares_y']),
            board_config['square_length'],
            board_config['marker_length'],
            aruco_dict
        )
        
        return board
    
    def load_calibration_data(self):
        """キャリブレーションデータの読み込み"""
        logger.info("="*80)
        logger.info("キャリブレーションデータの読み込み")
        logger.info("="*80)
        
        calib_dir = Path(self.config['paths']['output_dir']) / 'calibration'
        world_dir = Path(self.config['paths']['output_dir']) / 'bundle_adjustment' / 'world_coordinate'
        
        # ワールド座標系設定の読み込み
        world_config_file = world_dir / 'world_coordinate_config.json'
        if not world_config_file.exists():
            raise FileNotFoundError(f"World coordinate config not found: {world_config_file}")
        
        with open(world_config_file, 'r') as f:
            self.world_config = json.load(f)
        
        logger.info(f"✓ World coordinate system: {self.world_config['reference_type']}")
        
        # 各カメラの内部パラメータと外部パラメータを読み込み
        for cam_name in self.camera_names:
            # 内部パラメータ
            calib_file = calib_dir / f'calibration_{cam_name}.json'
            if not calib_file.exists():
                raise FileNotFoundError(f"Calibration file not found: {calib_file}")
            
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            
            self.camera_calibrations[cam_name] = {
                'K': np.array(calib_data['camera_matrix']['K'], dtype=np.float64),
                'dist': np.array(calib_data['distortion_coefficients'], dtype=np.float64),
                'img_size': tuple(calib_data['img_size']),
                'rms': calib_data['rms_reprojection_error']
            }
            
            logger.info(f"✓ {cam_name}: K loaded, RMS={self.camera_calibrations[cam_name]['rms']:.4f} px")
            
            # 外部パラメータ (world -> camera)
            extrinsic_file = world_dir / f'extrinsic_{cam_name}.json'
            if not extrinsic_file.exists():
                raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_file}")
            
            with open(extrinsic_file, 'r') as f:
                extrinsic_data = json.load(f)
            
            self.camera_extrinsics[cam_name] = {
                'R': np.array(extrinsic_data['rotation_matrix'], dtype=np.float64),
                't': np.array(extrinsic_data['translation_vector'], dtype=np.float64)
            }
            
            logger.info(f"✓ {cam_name}: Extrinsic (world -> camera) loaded")
        
        logger.info("")
    
    def detect_charuco_in_images(self):
        """全画像からChArUcoボードを検出"""
        logger.info("="*80)
        logger.info("ChArUcoボード検出")
        logger.info("="*80)
        
        for cam_name in self.camera_names:
            logger.info(f"Camera: {cam_name}")
            
            # 画像ディレクトリ
            img_dir = Path(self.config['paths']['captured_images']) / cam_name
            if not img_dir.exists():
                logger.warning(f"  Image directory not found: {img_dir}")
                continue
            
            # 画像ファイルを取得
            image_files = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
            
            detections = []
            
            for frame_id, img_path in enumerate(image_files):
                # 画像読み込み
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"  Failed to load: {img_path.name}")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # ArUcoマーカー検出
                detector_params = cv2.aruco.DetectorParameters()
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray,
                    self.board.getDictionary(),
                    parameters=detector_params
                )
                
                if ids is None or len(ids) == 0:
                    continue
                
                # ChArUcoコーナー補間
                num_corners, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                
                if num_corners < 4:
                    continue
                
                # 検出結果を保存
                detections.append({
                    'frame_id': frame_id,
                    'image_name': img_path.name,
                    'image_path': str(img_path),
                    'num_corners': int(num_corners),
                    'ch_ids': ch_ids.flatten().tolist(),
                    'ch_corners': ch_corners.reshape(-1, 2).tolist()
                })
            
            self.detections[cam_name] = detections
            logger.info(f"  Detected: {len(detections)} frames")
        
        logger.info("")
    
    def find_corresponding_frames(self) -> List[Tuple[int, int]]:
        """
        対応するフレームペアを検索
        
        Returns:
            対応フレームインデックスのリスト [(idx0, idx1), ...]
        """
        cam0_name = self.camera_names[0]
        cam1_name = self.camera_names[1]
        
        detections0 = self.detections[cam0_name]
        detections1 = self.detections[cam1_name]
        
        # 画像名でマッチング
        name_to_idx0 = {det['image_name']: i for i, det in enumerate(detections0)}
        name_to_idx1 = {det['image_name']: i for i, det in enumerate(detections1)}
        
        corresponding = []
        for name in name_to_idx0.keys():
            if name in name_to_idx1:
                corresponding.append((name_to_idx0[name], name_to_idx1[name]))
        
        logger.info(f"対応フレームペア数: {len(corresponding)}")
        return corresponding
    
    def triangulate_frame_pair(
        self,
        idx0: int,
        idx1: int,
        pair_id: int
    ) -> Dict:
        """
        1つのフレームペアについて三角測量（ワールド座標系）
        
        Args:
            idx0: Camera0の検出インデックス
            idx1: Camera1の検出インデックス
            pair_id: ペアID（連番）
            
        Returns:
            三角測量結果
        """
        cam0_name = self.camera_names[0]
        cam1_name = self.camera_names[1]
        
        det0 = self.detections[cam0_name][idx0]
        det1 = self.detections[cam1_name][idx1]
        
        # ChArUcoコーナー取得
        ids0 = np.array(det0['ch_ids'], dtype=np.int32)
        corners0 = np.array(det0['ch_corners'], dtype=np.float32)
        
        ids1 = np.array(det1['ch_ids'], dtype=np.int32)
        corners1 = np.array(det1['ch_corners'], dtype=np.float32)
        
        # 共通コーナーIDを検索
        common_ids = np.intersect1d(ids0, ids1)
        
        if len(common_ids) < 4:
            return {
                'success': False,
                'num_points': 0,
                'error': 'Insufficient common corners'
            }
        
        # 共通IDに対応する2D点を抽出（ソート済み）
        mask0 = np.isin(ids0, common_ids)
        mask1 = np.isin(ids1, common_ids)
        
        sort_idx0 = np.argsort(ids0[mask0])
        sort_idx1 = np.argsort(ids1[mask1])
        
        matched_ids = ids0[mask0][sort_idx0]
        matched_corners0 = corners0[mask0][sort_idx0]
        matched_corners1 = corners1[mask1][sort_idx1]
        
        # カメラパラメータ取得
        K0 = self.camera_calibrations[cam0_name]['K']
        dist0 = self.camera_calibrations[cam0_name]['dist']
        R0 = self.camera_extrinsics[cam0_name]['R']
        t0 = self.camera_extrinsics[cam0_name]['t']
        
        K1 = self.camera_calibrations[cam1_name]['K']
        dist1 = self.camera_calibrations[cam1_name]['dist']
        R1 = self.camera_extrinsics[cam1_name]['R']
        t1 = self.camera_extrinsics[cam1_name]['t']
        
        # 歪み補正
        corners0_undist = cv2.undistortPoints(
            matched_corners0.reshape(-1, 1, 2), K0, dist0, P=K0
        ).reshape(-1, 2)
        
        corners1_undist = cv2.undistortPoints(
            matched_corners1.reshape(-1, 1, 2), K1, dist1, P=K1
        ).reshape(-1, 2)
        
        # 投影行列の構築（ワールド座標系基準）
        # P = K @ [R_world_to_cam | t_world_to_cam]
        Rt0 = np.hstack([R0, t0.reshape(3, 1)])
        P0 = K0 @ Rt0
        
        Rt1 = np.hstack([R1, t1.reshape(3, 1)])
        P1 = K1 @ Rt1
        
        # 三角測量（結果はワールド座標系）
        points_4d_hom = cv2.triangulatePoints(
            P0, P1,
            corners0_undist.T, corners1_undist.T
        )
        
        # 同次座標から3D座標に変換
        points_3d_world = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d_world = points_3d_world.T
        
        # 再投影誤差の計算
        reprojection_errors = []
        
        for i, pt_3d in enumerate(points_3d_world):
            # World座標をカメラ座標に変換して再投影
            pt_3d_hom = np.append(pt_3d, 1.0)
            
            # Camera0への再投影
            pt_cam0 = Rt0 @ pt_3d_hom
            pt_img0 = K0 @ pt_cam0
            pt_img0 = pt_img0[:2] / pt_img0[2]
            error0 = np.linalg.norm(pt_img0 - matched_corners0[i])
            
            # Camera1への再投影
            pt_cam1 = Rt1 @ pt_3d_hom
            pt_img1 = K1 @ pt_cam1
            pt_img1 = pt_img1[:2] / pt_img1[2]
            error1 = np.linalg.norm(pt_img1 - matched_corners1[i])
            
            reprojection_errors.append({
                'corner_id': int(matched_ids[i]),
                'error_camera0': float(error0),
                'error_camera1': float(error1),
                'error_mean': float((error0 + error1) / 2)
            })
        
        # フレームごとに色を割り当て
        hue = pair_id / max(len(self.find_corresponding_frames()) - 1, 1) if len(self.find_corresponding_frames()) > 1 else 0.0
        color = self._hsv_to_rgb(hue, 0.8, 0.9)
        
        # 結果を記録
        for i, (corner_id, pt_3d) in enumerate(zip(matched_ids, points_3d_world)):
            self.triangulated_points.append({
                'pair_id': pair_id,
                'image_name': det0['image_name'],
                'corner_id': int(corner_id),
                'point_3d_world': pt_3d.tolist(),
                'color': color.tolist(),
                'point_2d_camera0': matched_corners0[i].tolist(),
                'point_2d_camera1': matched_corners1[i].tolist(),
                'reprojection_error_camera0': reprojection_errors[i]['error_camera0'],
                'reprojection_error_camera1': reprojection_errors[i]['error_camera1'],
                'reprojection_error_mean': reprojection_errors[i]['error_mean']
            })
        
        # 統計計算
        errors_cam0 = [e['error_camera0'] for e in reprojection_errors]
        errors_cam1 = [e['error_camera1'] for e in reprojection_errors]
        errors_mean = [e['error_mean'] for e in reprojection_errors]
        
        result = {
            'success': True,
            'pair_id': pair_id,
            'image_name': det0['image_name'],
            'num_points': len(matched_ids),
            'matched_corner_ids': matched_ids.tolist(),
            'points_3d_world': points_3d_world.tolist(),
            'reprojection_errors': reprojection_errors,
            'statistics': {
                'reprojection_error_camera0': {
                    'mean': float(np.mean(errors_cam0)),
                    'std': float(np.std(errors_cam0)),
                    'min': float(np.min(errors_cam0)),
                    'max': float(np.max(errors_cam0)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_cam0)**2)))
                },
                'reprojection_error_camera1': {
                    'mean': float(np.mean(errors_cam1)),
                    'std': float(np.std(errors_cam1)),
                    'min': float(np.min(errors_cam1)),
                    'max': float(np.max(errors_cam1)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_cam1)**2)))
                },
                'reprojection_error_mean': {
                    'mean': float(np.mean(errors_mean)),
                    'std': float(np.std(errors_mean)),
                    'min': float(np.min(errors_mean)),
                    'max': float(np.max(errors_mean)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_mean)**2)))
                }
            }
        }
        
        return result
    
    def triangulate_all_frames(self):
        """全フレームペアの三角測量"""
        logger.info("="*80)
        logger.info("三角測量の実行（ワールド座標系）")
        logger.info("="*80)
        
        corresponding_frames = self.find_corresponding_frames()
        
        if len(corresponding_frames) == 0:
            logger.error("対応フレームが見つかりません")
            return
        
        successful_frames = 0
        total_points = 0
        
        for pair_id, (idx0, idx1) in enumerate(corresponding_frames):
            result = self.triangulate_frame_pair(idx0, idx1, pair_id)
            
            if result['success']:
                successful_frames += 1
                total_points += result['num_points']
                
                self.frame_results.append(result)
                
                logger.info(
                    f"✓ Pair {pair_id:2d} ({result['image_name']}): "
                    f"{result['num_points']} points, "
                    f"RMS error = {result['statistics']['reprojection_error_mean']['rms']:.3f} px"
                )
                
                # フレーム詳細を保存
                detail_file = self.frame_details_dir / f"frame_{pair_id:03d}_{result['image_name']}.json"
                with open(detail_file, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                logger.warning(f"✗ Pair {pair_id:2d}: {result.get('error', 'Unknown error')}")
        
        logger.info("")
        logger.info(f"三角測量完了: {successful_frames}/{len(corresponding_frames)} フレーム成功")
        logger.info(f"総3D点数: {total_points}")
        logger.info("")
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """HSV色空間からRGB色空間に変換"""
        import colorsys
        return np.array(colorsys.hsv_to_rgb(h, s, v))
    
    def compute_statistics(self) -> Dict:
        """統計情報の計算"""
        logger.info("="*80)
        logger.info("統計情報の計算")
        logger.info("="*80)
        
        if len(self.triangulated_points) == 0:
            return {}
        
        points_3d = np.array([pt['point_3d_world'] for pt in self.triangulated_points])
        errors_cam0 = [pt['reprojection_error_camera0'] for pt in self.triangulated_points]
        errors_cam1 = [pt['reprojection_error_camera1'] for pt in self.triangulated_points]
        errors_mean = [pt['reprojection_error_mean'] for pt in self.triangulated_points]
        
        num_points_per_frame = [r['num_points'] for r in self.frame_results]
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'world_coordinate_system': self.world_config['reference_type'],
            'total_points': int(len(points_3d)),
            'num_frames': int(len(self.frame_results)),
            'points_per_frame': {
                'mean': float(np.mean(num_points_per_frame)),
                'std': float(np.std(num_points_per_frame)),
                'min': int(np.min(num_points_per_frame)),
                'max': int(np.max(num_points_per_frame))
            },
            'spatial_extent_world': {
                'x_range_m': [float(points_3d[:, 0].min()), float(points_3d[:, 0].max())],
                'y_range_m': [float(points_3d[:, 1].min()), float(points_3d[:, 1].max())],
                'z_range_m': [float(points_3d[:, 2].min()), float(points_3d[:, 2].max())],
                'x_span_m': float(points_3d[:, 0].max() - points_3d[:, 0].min()),
                'y_span_m': float(points_3d[:, 1].max() - points_3d[:, 1].min()),
                'z_span_m': float(points_3d[:, 2].max() - points_3d[:, 2].min())
            },
            'centroid_world': {
                'x': float(points_3d[:, 0].mean()),
                'y': float(points_3d[:, 1].mean()),
                'z': float(points_3d[:, 2].mean())
            },
            'reprojection_errors_px': {
                'camera0': {
                    'mean': float(np.mean(errors_cam0)),
                    'std': float(np.std(errors_cam0)),
                    'min': float(np.min(errors_cam0)),
                    'max': float(np.max(errors_cam0)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_cam0)**2)))
                },
                'camera1': {
                    'mean': float(np.mean(errors_cam1)),
                    'std': float(np.std(errors_cam1)),
                    'min': float(np.min(errors_cam1)),
                    'max': float(np.max(errors_cam1)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_cam1)**2)))
                },
                'overall': {
                    'mean': float(np.mean(errors_mean)),
                    'std': float(np.std(errors_mean)),
                    'min': float(np.min(errors_mean)),
                    'max': float(np.max(errors_mean)),
                    'rms': float(np.sqrt(np.mean(np.array(errors_mean)**2)))
                }
            }
        }
        
        logger.info(f"総3D点数: {stats['total_points']}")
        logger.info(f"フレーム数: {stats['num_frames']}")
        logger.info(f"平均点数/フレーム: {stats['points_per_frame']['mean']:.1f}")
        logger.info("")
        logger.info(f"空間範囲（ワールド座標系）:")
        logger.info(f"  X: [{stats['spatial_extent_world']['x_range_m'][0]:.3f}, {stats['spatial_extent_world']['x_range_m'][1]:.3f}] m (幅 {stats['spatial_extent_world']['x_span_m']:.3f} m)")
        logger.info(f"  Y: [{stats['spatial_extent_world']['y_range_m'][0]:.3f}, {stats['spatial_extent_world']['y_range_m'][1]:.3f}] m (幅 {stats['spatial_extent_world']['y_span_m']:.3f} m)")
        logger.info(f"  Z: [{stats['spatial_extent_world']['z_range_m'][0]:.3f}, {stats['spatial_extent_world']['z_range_m'][1]:.3f}] m (幅 {stats['spatial_extent_world']['z_span_m']:.3f} m)")
        logger.info("")
        logger.info(f"重心（ワールド座標系）: ({stats['centroid_world']['x']:.3f}, {stats['centroid_world']['y']:.3f}, {stats['centroid_world']['z']:.3f}) m")
        logger.info("")
        logger.info(f"再投影誤差:")
        logger.info(f"  Camera0: RMS = {stats['reprojection_errors_px']['camera0']['rms']:.3f} px")
        logger.info(f"  Camera1: RMS = {stats['reprojection_errors_px']['camera1']['rms']:.3f} px")
        logger.info(f"  Overall: RMS = {stats['reprojection_errors_px']['overall']['rms']:.3f} px")
        logger.info("")
        
        return stats
    
    def create_camera_frustum(
        self,
        K: np.ndarray,
        img_size: Tuple[int, int],
        scale: float = 0.3,
        color: np.ndarray = np.array([0.0, 0.0, 1.0])
    ) -> o3d.geometry.LineSet:
        """カメラ視錐台の作成"""
        width, height = img_size
        
        corners_2d = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        corners_3d = []
        for x, y in corners_2d:
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy
            corners_3d.append([x_norm * scale, y_norm * scale, scale])
        
        corners_3d = np.array(corners_3d)
        
        vertices = np.vstack([
            np.array([[0, 0, 0]]),
            corners_3d
        ])
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def visualize_3d_scene(self):
        """Open3Dで3Dシーンを可視化"""
        logger.info("="*80)
        logger.info("3D可視化（ワールド座標系）")
        logger.info("="*80)
        
        if len(self.triangulated_points) == 0:
            logger.error("三角測量点が存在しません")
            return
        
        geometries = []
        
        # ワールド座標系の軸
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(world_frame)
        logger.info("✓ ワールド座標系の軸を配置")
        
        # 各カメラの位置と視錐台（ワールド座標系での位置）
        for cam_name in self.camera_names:
            R_world_to_cam = self.camera_extrinsics[cam_name]['R']
            t_world_to_cam = self.camera_extrinsics[cam_name]['t']
            
            # camera -> world の変換（逆変換）
            R_cam_to_world = R_world_to_cam.T
            t_cam_to_world = -R_world_to_cam.T @ t_world_to_cam
            
            # カメラの座標系
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
            )
            
            T_cam = np.eye(4)
            T_cam[:3, :3] = R_cam_to_world
            T_cam[:3, 3] = t_cam_to_world
            cam_frame.transform(T_cam)
            geometries.append(cam_frame)
            
            # 視錐台
            K = self.camera_calibrations[cam_name]['K']
            img_size = self.camera_calibrations[cam_name]['img_size']
            
            if cam_name == self.camera_names[0]:
                color = np.array([0.0, 0.5, 1.0])  # 青
            else:
                color = np.array([1.0, 0.5, 0.0])  # 橙
            
            frustum = self.create_camera_frustum(K, img_size, scale=0.3, color=color)
            frustum.transform(T_cam)
            geometries.append(frustum)
            
            logger.info(f"✓ {cam_name}: ワールド座標 ({t_cam_to_world[0]:.3f}, {t_cam_to_world[1]:.3f}, {t_cam_to_world[2]:.3f}) m")
        
        # 三角測量された点群
        points = np.array([pt['point_3d_world'] for pt in self.triangulated_points])
        colors = np.array([pt['color'] for pt in self.triangulated_points])
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)
        
        logger.info(f"✓ ChArUco格子点: {len(points)} 点（フレームごとに色分け）")
        logger.info("")
        logger.info("3D可視化ウィンドウを表示中...")
        logger.info("（マウス操作: 左ドラッグ=回転、右ドラッグ=移動、スクロール=ズーム）")
        logger.info("")
        
        # 可視化
        o3d.visualization.draw_geometries(
            geometries,
            window_name="World ChArUco Triangulation",
            width=1280,
            height=960,
            left=50,
            top=50
        )
        
        logger.info("可視化完了")
        logger.info("")
    
    def save_results(self):
        """結果の保存"""
        logger.info("="*80)
        logger.info("結果の保存")
        logger.info("="*80)
        
        # 統計情報を計算
        stats = self.compute_statistics()
        
        # 統計情報を保存
        stats_file = self.output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ 統計情報を保存: {stats_file}")
        
        # 全結果を保存
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'world_coordinate_config': self.world_config,
            'camera_calibrations': {
                cam_name: {
                    'K': self.camera_calibrations[cam_name]['K'].tolist(),
                    'dist': self.camera_calibrations[cam_name]['dist'].tolist(),
                    'img_size': self.camera_calibrations[cam_name]['img_size'],
                    'rms': self.camera_calibrations[cam_name]['rms']
                }
                for cam_name in self.camera_names
            },
            'camera_extrinsics_world_to_camera': {
                cam_name: {
                    'R': self.camera_extrinsics[cam_name]['R'].tolist(),
                    't': self.camera_extrinsics[cam_name]['t'].tolist()
                }
                for cam_name in self.camera_names
            },
            'statistics': stats,
            'frame_results_summary': [
                {
                    'pair_id': r['pair_id'],
                    'image_name': r['image_name'],
                    'num_points': r['num_points'],
                    'reprojection_error_rms': r['statistics']['reprojection_error_mean']['rms']
                }
                for r in self.frame_results
            ],
            'triangulated_points': self.triangulated_points
        }
        
        results_file = self.output_dir / 'triangulation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"✓ 三角測量結果を保存: {results_file}")
        
        # 点群をPLY形式で保存
        if len(self.triangulated_points) > 0:
            points = np.array([pt['point_3d_world'] for pt in self.triangulated_points])
            colors = np.array([pt['color'] for pt in self.triangulated_points])
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            ply_file = self.output_dir / 'triangulated_points.ply'
            o3d.io.write_point_cloud(str(ply_file), pcd)
            logger.info(f"✓ 点群を保存: {ply_file}")
        
        logger.info("")
    
    def run(self):
        """メイン実行フロー"""
        logger.info("")
        logger.info("="*80)
        logger.info("ワールド座標系でのChArUco三角測量")
        logger.info("="*80)
        logger.info("")
        
        # キャリブレーションデータ読み込み
        self.load_calibration_data()
        
        # ChArUco検出
        self.detect_charuco_in_images()
        
        # 三角測量
        self.triangulate_all_frames()
        
        # 結果保存
        self.save_results()
        
        # 3D可視化
        self.visualize_3d_scene()
        
        logger.info("="*80)
        logger.info("処理完了")
        logger.info("="*80)


def main():
    """メイン関数"""
    import sys
    
    config_path = "calibration_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        triangulator = WorldCharucoTriangulation(config_path)
        triangulator.run()
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
