#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChArUco Board Pose 3D Visualization

Open3Dを使用してカメラとボードの3D位置・姿勢を可視化するスクリプト
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse

import numpy as np
import cv2
import open3d as o3d

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pose3DVisualizer:
    """カメラとボードの3D姿勢を可視化するクラス"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """設定ファイルを読み込む"""
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_coordinate_frame(
        self,
        size: float = 0.1,
        origin: np.ndarray = np.array([0.0, 0.0, 0.0])
    ) -> o3d.geometry.TriangleMesh:
        """
        座標系を表すフレームを作成
        
        Args:
            size: 軸の長さ
            origin: 原点の位置
            
        Returns:
            座標系のメッシュ
        """
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
        """
        カメラの視錐台を作成
        
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
    
    def create_board_mesh(
        self,
        board_size: Tuple[float, float] = (0.29, 0.203),
        color: np.ndarray = np.array([0.9, 0.9, 0.9]),
        back_color: Optional[np.ndarray] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        ChArUcoボードのメッシュを作成（両面表示）
        
        Args:
            board_size: ボードサイズ (width, height) [m]
            color: 表面の色
            back_color: 裏面の色（Noneの場合は表面と同じ）
            
        Returns:
            ボードのTriangleMesh
        """
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
        
        # 裏面の頂点（Z=0、表面と同じ位置だが法線が逆）
        back_vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        # 全頂点（8頂点：表4 + 裏4）
        vertices = np.vstack([front_vertices, back_vertices])
        
        # 三角形（表面2つ + 裏面2つ）
        triangles = np.array([
            # 表面（反時計回り、カメラから見て）
            [0, 1, 2],
            [0, 2, 3],
            # 裏面（時計回り、裏から見て反時計回り）
            [4, 6, 5],
            [4, 7, 6]
        ])
        
        # 頂点色（表面4つ + 裏面4つ）
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
        """
        ボードの輪郭線を作成
        
        Args:
            board_size: ボードサイズ (width, height) [m]
            color: 線の色
            
        Returns:
            輪郭線のLineSet
        """
        width, height = board_size
        
        # 頂点（4隅）
        vertices = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, height, 0],
            [0, height, 0]
        ])
        
        # エッジ（4辺）
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
        """
        Rodrigues回転ベクトルを回転行列に変換
        
        Args:
            rvec: 回転ベクトル (3,)
            
        Returns:
            回転行列 (3, 3)
        """
        R, _ = cv2.Rodrigues(rvec)
        return R
    
    def transform_geometry(
        self,
        geometry: o3d.geometry.Geometry3D,
        R: np.ndarray,
        t: np.ndarray
    ) -> o3d.geometry.Geometry3D:
        """
        ジオメトリに回転・平行移動を適用
        
        Args:
            geometry: 変換するジオメトリ
            R: 回転行列 (3, 3)
            t: 平行移動ベクトル (3,)
            
        Returns:
            変換されたジオメトリ
        """
        # 4x4変換行列を作成
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        # ジオメトリをコピーして変換
        geometry_transformed = geometry.__copy__()
        geometry_transformed.transform(T)
        
        return geometry_transformed
    
    def hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """
        HSV色空間からRGB色空間に変換
        
        Args:
            h: 色相 [0, 1]
            s: 彩度 [0, 1]
            v: 明度 [0, 1]
            
        Returns:
            RGB値 (3,) [0, 1]
        """
        import colorsys
        return np.array(colorsys.hsv_to_rgb(h, s, v))
    
    def visualize_camera_and_boards(
        self,
        camera_name: str,
        board_indices: Optional[List[int]] = None
    ):
        """
        単一カメラとその検出したボードを3D表示
        
        Args:
            camera_name: カメラ名
            board_indices: 表示するボードのインデックスリスト（Noneなら全て）
        """
        logger.info("="*60)
        logger.info(f"Single Camera 3D Visualization: {camera_name}")
        logger.info("="*60)
        
        # カメラのキャリブレーション結果を読み込み
        calib_path = Path("calibration_results/calibration") / f"calibration_{camera_name}.json"
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['camera_matrix']['K'])
        img_size = tuple(calib_data['img_size'])
        
        # 姿勢推定結果を読み込み
        pose_path = Path("calibration_results/pose_estimation") / camera_name / "pose_estimation.json"
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        # posesは辞書形式（キー: 画像名, 値: 姿勢データ）
        poses = pose_data.get('poses', {})
        
        successful_poses = [
            (img_name, pose_info) for img_name, pose_info in poses.items()
            if pose_info.get('success', False)
        ]
        
        if not successful_poses:
            logger.error("姿勢推定成功フレームが見つかりません")
            return
        
        logger.info(f"{camera_name}: 姿勢データ読み込み完了")
        logger.info(f"総フレーム数: {len(poses)}")
        logger.info(f"成功フレーム数: {len(successful_poses)}")
        
        # 表示するボードを選択
        if board_indices is not None:
            selected_poses = [successful_poses[i] for i in board_indices if i < len(successful_poses)]
        else:
            selected_poses = successful_poses
        
        logger.info(f"表示するボード数: {len(selected_poses)}")
        
        # ボード設定を取得
        board_config = self.config['board']
        board_width = board_config['squares_x'] * board_config['square_length'] / 1000.0  # mm to m
        board_height = board_config['squares_y'] * board_config['square_length'] / 1000.0
        
        # ジオメトリを作成
        geometries = []
        
        # カメラ座標系（原点）
        camera_frame = self.create_coordinate_frame(size=0.15)
        geometries.append(camera_frame)
        
        # カメラ視錐台
        frustum = self.create_camera_frustum(K, img_size, scale=0.3)
        geometries.append(frustum)
        
        # 各ボードを配置
        for idx, (img_name, pose) in enumerate(selected_poses):
            rvec = np.array(pose['rvec'])
            tvec = np.array(pose['tvec'])
            R = self.rodrigues_to_matrix(rvec)
            
            # ボードごとに色を変える
            hue = idx / max(len(selected_poses) - 1, 1)
            color = self.hsv_to_rgb(hue, 0.7, 0.9)
            
            board_mesh = self.create_board_mesh(
                board_size=(board_width, board_height),
                color=color
            )
            
            # ボード輪郭線
            board_edges = self.create_board_edges(
                board_size=(board_width, board_height)
            )
            
            # ボード座標系
            board_frame = self.create_coordinate_frame(size=0.05)
            
            # 変換を適用
            board_mesh = self.transform_geometry(board_mesh, R, tvec)
            board_edges = self.transform_geometry(board_edges, R, tvec)
            board_frame = self.transform_geometry(board_frame, R, tvec)
            
            geometries.append(board_mesh)
            geometries.append(board_edges)
            geometries.append(board_frame)
        
        # 可視化
        logger.info("3D可視化ウィンドウを表示中...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"{camera_name} - Camera and Boards",
            width=1024,
            height=768
        )
        logger.info("可視化完了")
    
    def create_animation(
        self,
        camera_name: str,
        output_dir: Path,
        max_frames: int = 30
    ):
        """
        カメラとボードの動きをアニメーション化
        
        Args:
            camera_name: カメラ名
            output_dir: 出力ディレクトリ
            max_frames: 最大フレーム数
        """
        logger.info("="*60)
        logger.info(f"Animation Generation: {camera_name}")
        logger.info("="*60)
        
        # カメラのキャリブレーション結果を読み込み
        calib_path = Path("calibration_results/calibration") / f"calibration_{camera_name}.json"
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['camera_matrix']['K'])
        img_size = tuple(calib_data['img_size'])
        
        # 姿勢推定結果を読み込み
        pose_path = Path("calibration_results/pose_estimation") / camera_name / "pose_estimation.json"
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        # posesは辞書形式（キー: 画像名, 値: 姿勢データ）
        poses = pose_data.get('poses', {})
        
        successful_poses = [
            (img_name, pose_info) for img_name, pose_info in poses.items()
            if pose_info.get('success', False)
        ][:max_frames]
        
        if not successful_poses:
            logger.error("姿勢推定成功フレームが見つかりません")
            return
        
        logger.info(f"アニメーションフレーム数: {len(successful_poses)}")
        
        # 出力ディレクトリを作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ボード設定
        board_config = self.config['board']
        board_width = board_config['squares_x'] * board_config['square_length'] / 1000.0  # mm to m
        board_height = board_config['squares_y'] * board_config['square_length'] / 1000.0
        
        # Visualizerを作成（オフスクリーンレンダリング）
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=768)
        
        for idx, (img_name, pose) in enumerate(successful_poses):
            vis.clear_geometries()
            
            # カメラ座標系
            camera_frame = self.create_coordinate_frame(size=0.2)
            vis.add_geometry(camera_frame, reset_bounding_box=(idx == 0))
            
            # カメラ視錐台
            frustum = self.create_camera_frustum(K, img_size, scale=0.3)
            vis.add_geometry(frustum, reset_bounding_box=False)
            
            # ボード
            rvec = np.array(pose['rvec'])
            tvec = np.array(pose['tvec'])
            R = self.rodrigues_to_matrix(rvec)
            
            board_mesh = self.create_board_mesh(
                board_size=(board_width, board_height),
                color=np.array([0.9, 0.9, 0.9])
            )
            board_edges = self.create_board_edges(
                board_size=(board_width, board_height)
            )
            board_frame = self.create_coordinate_frame(size=0.08)
            
            board_mesh = self.transform_geometry(board_mesh, R, tvec)
            board_edges = self.transform_geometry(board_edges, R, tvec)
            board_frame = self.transform_geometry(board_frame, R, tvec)
            
            vis.add_geometry(board_mesh, reset_bounding_box=False)
            vis.add_geometry(board_edges, reset_bounding_box=False)
            vis.add_geometry(board_frame, reset_bounding_box=False)
            
            # レンダリングと保存
            vis.poll_events()
            vis.update_renderer()
            
            output_path = output_dir / f"frame_{idx:04d}.png"
            vis.capture_screen_image(str(output_path), do_render=True)
            
            logger.info(f"Frame {idx+1}/{len(successful_poses)}: {output_path.name}")
        
        vis.destroy_window()
        logger.info(f"アニメーションフレームを保存: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ChArUco Board Pose 3D Visualization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='calibration_config.yaml',
        help='キャリブレーション設定ファイルのパス（ボード設定用）'
    )
    parser.add_argument(
        '--camera',
        type=str,
        default='camera0',
        help='カメラ名'
    )
    parser.add_argument(
        '--board-indices',
        type=int,
        nargs='+',
        help='表示するボードのインデックス（指定しない場合は全て表示）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='calibration_results/animation',
        help='出力ディレクトリ（animationモード用）'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=30,
        help='最大フレーム数（animationモード用）'
    )
    parser.add_argument(
        '--animation',
        action='store_true',
        help='アニメーションを生成する'
    )
    
    args = parser.parse_args()
    
    # Visualizerを初期化
    visualizer = Pose3DVisualizer(Path(args.config))
    
    # アニメーションまたは単一カメラ可視化
    if args.animation:
        visualizer.create_animation(
            camera_name=args.camera,
            output_dir=Path(args.output_dir),
            max_frames=args.max_frames
        )
    else:
        visualizer.visualize_camera_and_boards(
            camera_name=args.camera,
            board_indices=args.board_indices
        )


if __name__ == "__main__":
    main()
