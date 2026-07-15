#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrinsic Calibration 3D Visualization

推定された外部パラメータを使用してマルチカメラセットアップを3D可視化するスクリプト
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse

import numpy as np
import cv2
import open3d as o3d
import yaml
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtrinsicVisualizer:
    """外部パラメータを使用したマルチカメラセットアップの3D可視化クラス"""
    
    def __init__(self, config_path: Path, calibration_config_path: Path):
        """
        初期化
        
        Args:
            config_path: グローバル設定ファイルのパス（カメラ設定用）
            calibration_config_path: キャリブレーション設定ファイルのパス
        """
        # カメラ設定の読み込み（calibration_config.yaml経由）
        self.config_manager = ConfigManager(Path(config_path))
        if not self.config_manager.load():
            raise ValueError(f"Failed to load camera configuration from {config_path}")
        
        # キャリブレーション設定ファイルの読み込み
        self.calibration_config_path = calibration_config_path
        self.config = self._load_config()
        self.camera_data = {}
        self.extrinsic_data = {}
        
    def _load_config(self) -> dict:
        """設定ファイルを読み込む"""
        with open(self.calibration_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_camera_calibration(self, camera_name: str):
        """
        カメラキャリブレーション結果を読み込む
        
        Args:
            camera_name: カメラ名
        """
        calib_path = Path("calibration_results/calibration") / f"calibration_{camera_name}.json"
        
        if not calib_path.exists():
            logger.error(f"キャリブレーションファイルが見つかりません: {calib_path}")
            return False
        
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        
        self.camera_data[camera_name] = {
            'K': np.array(calib['camera_matrix']['K']),
            'dist': np.array(calib['distortion_coefficients']),
            'img_size': tuple(calib['img_size'])
        }
        
        logger.info(f"{camera_name}: キャリブレーションデータ読み込み完了")
        return True
    
    def load_extrinsic_calibration(self):
        """外部パラメータを読み込む"""
        extrinsic_dir = Path("calibration_results/extrinsic")
        
        # サマリーファイルを読み込み
        summary_path = extrinsic_dir / "extrinsic_summary.json"
        if not summary_path.exists():
            logger.error(f"外部パラメータサマリーが見つかりません: {summary_path}")
            return False
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # 各ペアの詳細データを読み込み
        for pair in summary['camera_pairs']:
            camera_i = pair['camera_i']
            camera_j = pair['camera_j']
            
            detail_path = extrinsic_dir / f"initial_extrinsic_{camera_i}_to_{camera_j}.json"
            if detail_path.exists():
                with open(detail_path, 'r') as f:
                    data = json.load(f)
                
                pair_key = f"{camera_i}_to_{camera_j}"
                self.extrinsic_data[pair_key] = {
                    'camera_i': camera_i,
                    'camera_j': camera_j,
                    'R': np.array(data['R']),
                    't': np.array(data['t'])
                }
        
        logger.info(f"外部パラメータ読み込み完了: {len(self.extrinsic_data)} ペア")
        return True
    
    def load_pose_estimation(self, camera_name: str, frame_name: str) -> Optional[Dict]:
        """
        特定フレームの姿勢推定結果を読み込む
        
        Args:
            camera_name: カメラ名
            frame_name: フレーム名
            
        Returns:
            姿勢データ（成功時）またはNone
        """
        pose_path = Path("calibration_results/pose_estimation") / camera_name / "pose_estimation.json"
        
        if not pose_path.exists():
            logger.warning(f"姿勢推定結果が見つかりません: {pose_path}")
            return None
        
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        poses = pose_data.get('poses', {})
        pose = poses.get(frame_name)
        
        if pose and pose.get('success', False):
            return {
                'rvec': np.array(pose['rvec']),
                'tvec': np.array(pose['tvec'])
            }
        
        return None
    
    def create_coordinate_frame(
        self,
        size: float = 0.1,
        origin: np.ndarray = np.array([0.0, 0.0, 0.0])
    ) -> o3d.geometry.TriangleMesh:
        """座標系を表すフレームを作成"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
            origin=origin
        )
    
    def create_camera_frustum(
        self,
        K: np.ndarray,
        img_size: Tuple[int, int],
        scale: float = 0.3,
        color: np.ndarray = np.array([0.0, 0.0, 1.0])
    ) -> o3d.geometry.LineSet:
        """カメラの視錐台を作成"""
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
            X = (x - cx) / fx * scale
            Y = (y - cy) / fy * scale
            Z = scale
            corners_3d.append([X, Y, Z])
        
        corners_3d = np.array(corners_3d)
        
        # 原点（カメラ中心）
        origin = np.array([[0, 0, 0]])
        
        # すべての点
        points = np.vstack([origin, corners_3d])
        
        # エッジを定義
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 原点から4隅へ
            [1, 2], [2, 3], [3, 4], [4, 1]   # 4隅を結ぶ
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def create_camera_body(
        self,
        size: float = 0.05,
        color: np.ndarray = np.array([0.3, 0.3, 0.3])
    ) -> o3d.geometry.TriangleMesh:
        """カメラ本体を表す箱を作成"""
        box = o3d.geometry.TriangleMesh.create_box(
            width=size * 2,
            height=size,
            depth=size
        )
        
        # 中心をカメラ位置に移動
        box.translate([-size, -size/2, -size/2])
        box.paint_uniform_color(color)
        box.compute_vertex_normals()
        
        return box
    
    def create_board_mesh(
        self,
        board_size: Tuple[float, float] = (0.29, 0.203),
        color: np.ndarray = np.array([0.9, 0.9, 0.9]),
        back_color: Optional[np.ndarray] = None
    ) -> o3d.geometry.TriangleMesh:
        """ChArUcoボードのメッシュを作成（両面表示）"""
        if back_color is None:
            back_color = color
        
        width, height = board_size
        
        # 表面の頂点（Z=0）
        front_vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        # 裏面の頂点
        back_vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        # 全頂点
        vertices = np.vstack([front_vertices, back_vertices])
        
        # 三角形
        triangles = np.array([
            # 表面（反時計回り）
            [0, 1, 2],
            [0, 2, 3],
            # 裏面（時計回り）
            [4, 6, 5],
            [4, 7, 6]
        ])
        
        # 頂点色
        vertex_colors = np.vstack([
            np.tile(color, (4, 1)),
            np.tile(back_color, (4, 1))
        ])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh.compute_vertex_normals()
        
        return mesh
    
    def create_board_edges(
        self,
        board_size: Tuple[float, float] = (0.29, 0.203),
        color: np.ndarray = np.array([0.2, 0.2, 0.2])
    ) -> o3d.geometry.LineSet:
        """ボードの輪郭線を作成"""
        width, height = board_size
        
        vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        lines = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def rodrigues_to_matrix(self, rvec: np.ndarray) -> np.ndarray:
        """Rodrigues回転ベクトルを回転行列に変換"""
        R, _ = cv2.Rodrigues(rvec)
        return R
    
    def transform_geometry(
        self,
        geometry: o3d.geometry.Geometry3D,
        R: np.ndarray,
        t: np.ndarray
    ) -> o3d.geometry.Geometry3D:
        """ジオメトリに回転・平行移動を適用"""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        geometry_transformed = geometry.__copy__()
        geometry_transformed.transform(T)
        
        return geometry_transformed
    
    def create_camera_label(
        self,
        text: str,
        position: np.ndarray,
        size: float = 0.03
    ) -> o3d.geometry.TriangleMesh:
        """カメラ名のラベルを作成（簡易版：球で代用）"""
        # Open3Dには直接テキストを3D空間に配置する機能がないため、
        # 代わりに位置を示す小さな球を作成
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.translate(position)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
        return sphere
    
    def compute_camera_positions(self) -> Dict[str, Dict]:
        """
        外部パラメータから各カメラの位置と姿勢を計算
        
        Returns:
            カメラ名をキーとした位置・姿勢の辞書
        """
        # カメラ0を原点に配置
        cameras = {}
        num_cameras = self.config_manager.get_camera_count()
        camera_names = [f"camera{i}" for i in range(num_cameras)]
        
        # 基準カメラ（camera0）
        base_camera = camera_names[0]
        cameras[base_camera] = {
            'position': np.zeros(3),
            'rotation': np.eye(3)
        }
        
        # 外部パラメータからカメラ位置を計算
        for pair_key, extrinsic in self.extrinsic_data.items():
            camera_i = extrinsic['camera_i']
            camera_j = extrinsic['camera_j']
            R = extrinsic['R']
            t = extrinsic['t']
            
            if camera_i in cameras:
                # カメラjの位置を計算
                # カメラi座標系でのカメラjの位置がt
                cameras[camera_j] = {
                    'position': cameras[camera_i]['position'] + t,
                    'rotation': R @ cameras[camera_i]['rotation']
                }
        
        return cameras
    
    def visualize_multiple_frames(self, frame_names: Optional[List[str]] = None):
        """
        複数フレームでのボード位置を同時に可視化
        
        Args:
            frame_names: フレーム名のリスト（Noneの場合は全フレーム）
        """
        logger.info("="*60)
        logger.info("マルチフレーム可視化")
        logger.info("="*60)
        
        # カメラ位置を計算
        cameras = self.compute_camera_positions()
        
        # フレーム名が指定されていない場合、全フレームを取得
        if frame_names is None:
            # 同期レポートから全フレーム名を取得
            sync_report_path = Path("calibration_results/pose_estimation/synchronization_report.json")
            if sync_report_path.exists():
                with open(sync_report_path, 'r') as f:
                    sync_report = json.load(f)
                    frame_names = sync_report.get('synchronized_frames', [])
                    logger.info(f"全同期フレームを表示: {len(frame_names)} フレーム")
            else:
                logger.error("同期レポートが見つかりません")
                return
        
        if not frame_names:
            logger.error("表示するフレームがありません")
            return
        
        # ボード設定
        board_config = self.config['board']
        board_width = board_config['squares_x'] * board_config['square_length'] / 1000.0
        board_height = board_config['squares_y'] * board_config['square_length'] / 1000.0
        
        # ジオメトリを作成
        geometries = []
        
        # ワールド座標系
        world_frame = self.create_coordinate_frame(size=0.2)
        geometries.append(world_frame)
        
        # カメラを配置
        camera_colors = {
            'camera0': np.array([1.0, 0.0, 0.0]),
            'camera1': np.array([0.0, 0.0, 1.0]),
        }
        
        for camera_name, cam_info in cameras.items():
            position = cam_info['position']
            rotation = cam_info['rotation']
            color = camera_colors.get(camera_name, np.array([0.5, 0.5, 0.5]))
            
            camera_frame = self.create_coordinate_frame(size=0.1)
            camera_frame = self.transform_geometry(camera_frame, rotation, position)
            geometries.append(camera_frame)
            
            if camera_name in self.camera_data:
                K = self.camera_data[camera_name]['K']
                img_size = self.camera_data[camera_name]['img_size']
                
                frustum = self.create_camera_frustum(K, img_size, scale=0.15, color=color)
                frustum = self.transform_geometry(frustum, rotation, position)
                geometries.append(frustum)
        
        # 各フレームのボードを配置
        for idx, frame_name in enumerate(frame_names):
            # ボード色を変える
            hue = idx / max(len(frame_names) - 1, 1)
            import colorsys
            color = np.array(colorsys.hsv_to_rgb(hue, 0.7, 0.9))
            
            # いずれかのカメラでボード姿勢を取得
            for camera_name in cameras.keys():
                pose = self.load_pose_estimation(camera_name, frame_name)
                if pose:
                    cam_info = cameras[camera_name]
                    R_cam = cam_info['rotation']
                    t_cam = cam_info['position']
                    
                    R_board_in_cam = self.rodrigues_to_matrix(pose['rvec'])
                    t_board_in_cam = pose['tvec']
                    
                    R_board_in_world = R_cam @ R_board_in_cam
                    t_board_in_world = t_cam + R_cam @ t_board_in_cam
                    
                    board_mesh = self.create_board_mesh(
                        board_size=(board_width, board_height),
                        color=color
                    )
                    board_edges = self.create_board_edges(
                        board_size=(board_width, board_height)
                    )
                    
                    board_mesh = self.transform_geometry(board_mesh, R_board_in_world, t_board_in_world)
                    board_edges = self.transform_geometry(board_edges, R_board_in_world, t_board_in_world)
                    
                    geometries.append(board_mesh)
                    geometries.append(board_edges)
                    
                    break
        
        logger.info(f"{len(frame_names)} フレームのボードを表示")
        
        # 可視化
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Multi-Frame Visualization ({len(frame_names)} frames)",
            width=1280,
            height=960
        )
        logger.info("可視化完了")


def main():
    parser = argparse.ArgumentParser(
        description="Extrinsic Calibration 3D Visualization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='calibration_config.yaml',
        help='グローバル設定ファイルのパス（カメラ設定用）'
    )
    parser.add_argument(
        '--calibration-config',
        type=str,
        default='calibration_config.yaml',
        help='キャリブレーション設定ファイルのパス（ボード設定用）'
    )
    parser.add_argument(
        '--frame-names',
        type=str,
        nargs='+',
        help='表示するフレーム名（指定しない場合は全同期フレーム）'
    )
    
    args = parser.parse_args()
    
    # Visualizerを初期化
    visualizer = ExtrinsicVisualizer(
        Path(args.config),
        Path(args.calibration_config)
    )
    
    # カメラキャリブレーションを読み込み
    num_cameras = visualizer.config_manager.get_camera_count()
    for i in range(num_cameras):
        camera_name = f"camera{i}"
        visualizer.load_camera_calibration(camera_name)
    
    # 外部パラメータを読み込み
    if not visualizer.load_extrinsic_calibration():
        logger.error("外部パラメータの読み込みに失敗しました")
        return
    
    # マルチフレーム可視化
    visualizer.visualize_multiple_frames(args.frame_names)


if __name__ == "__main__":
    main()
