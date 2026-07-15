"""
11_visualize_world_coordinate.py
ワールド座標系での3D可視化

可視化内容:
1. ワールド座標系の軸（X, Y, Z）
2. 各カメラの位置と向き
3. 各フレームでのボード位置と向き
4. インタラクティブな視点操作
"""

import cv2
import numpy as np
import json
import yaml
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from config_manager import ConfigManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorldCoordinateVisualizer:
    """ワールド座標系の3D可視化を行うクラス"""
    
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
        self.world_dir = Path(self.config['paths']['output_dir']) / "bundle_adjustment" / "world_coordinate"
        
        # ChArUcoボードの設定
        self.board_config = self.config['board']
        
        # データ保持
        # ConfigManagerから動的にカメラ名を生成
        num_cameras = self.config_manager.get_camera_count()
        self.camera_names = [f"camera{i}" for i in range(num_cameras)]
        self.camera_extrinsics = {}
        self.board_poses = {}
        self.world_config = {}
        
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_world_coordinate_results(self):
        """World Coordinate結果を読み込む"""
        logger.info("Loading World Coordinate results...")
        
        # 設定を読み込み
        config_file = self.world_dir / "world_coordinate_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"World coordinate config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.world_config = json.load(f)
        
        logger.info(f"  Reference type: {self.world_config['reference_type']}")
        
        # カメラ外部パラメータを読み込み
        for cam_name in self.camera_names:
            extrinsic_file = self.world_dir / f"extrinsic_{cam_name}.json"
            
            if not extrinsic_file.exists():
                raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_file}")
            
            with open(extrinsic_file, 'r') as f:
                data = json.load(f)
            
            self.camera_extrinsics[cam_name] = {
                'R': np.array(data['rotation_matrix']),
                't': np.array(data['translation_vector'])
            }
            logger.info(f"  Loaded extrinsic for {cam_name}")
        
        # ボード姿勢を読み込み
        for cam_name in self.camera_names:
            poses_file = self.world_dir / cam_name / "optimized_poses.json"
            
            if not poses_file.exists():
                raise FileNotFoundError(f"Poses file not found: {poses_file}")
            
            with open(poses_file, 'r') as f:
                data = json.load(f)
            
            self.board_poses[cam_name] = data['poses']
            logger.info(f"  Loaded {len(data['poses'])} poses for {cam_name}")
    
    def create_coordinate_frame(self, size: float = 0.1) -> o3d.geometry.TriangleMesh:
        """座標系の軸を作成"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def create_camera_frustum(self, K: np.ndarray, width: int, height: int, 
                             scale: float = 0.1) -> o3d.geometry.LineSet:
        """カメラの視錐台を作成"""
        # カメラの4隅の画像座標
        corners_2d = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 正規化画像座標に変換
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        corners_3d = []
        for x, y in corners_2d:
            X = (x - cx) / fx * scale
            Y = (y - cy) / fy * scale
            Z = scale
            corners_3d.append([X, Y, Z])
        
        corners_3d = np.array(corners_3d)
        
        # 原点を追加
        points = np.vstack([[0, 0, 0], corners_3d])
        
        # 線のインデックス
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # 原点から各隅へ
            [1, 2], [2, 3], [3, 4], [4, 1]   # 隅を結ぶ
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        return line_set
    
    def create_board_mesh(self) -> o3d.geometry.TriangleMesh:
        """ChArUcoボードのメッシュを作成（両面表示）"""
        # ボードのサイズ（メートル）
        board_width = self.board_config['squares_x'] * self.board_config['square_length'] / 1000.0
        board_height = self.board_config['squares_y'] * self.board_config['square_length'] / 1000.0
        
        # 平面メッシュを作成（XY平面、Z=0）
        vertices = np.array([
            [0, 0, 0],
            [board_width, 0, 0],
            [board_width, board_height, 0],
            [0, board_height, 0]
        ])
        
        # 表面と裏面の三角形（両面表示）
        triangles = np.array([
            # 表面（法線がZ+方向）
            [0, 1, 2],
            [0, 2, 3],
            # 裏面（法線がZ-方向）
            [0, 2, 1],
            [0, 3, 2]
        ])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # グレー
        
        return mesh
    
    def transform_geometry(self, geometry, R: np.ndarray, t: np.ndarray):
        """ジオメトリを変換"""
        # 4x4変換行列を作成
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        geometry.transform(T)
        return geometry
    
    def create_camera_geometry(self, cam_name: str, K: Optional[np.ndarray] = None) -> List:
        """カメラのジオメトリを作成"""
        R = self.camera_extrinsics[cam_name]['R']
        t = self.camera_extrinsics[cam_name]['t']
        
        # world -> camera の変換なので、カメラのワールド座標での位置は
        # camera座標系の原点をworld座標系に変換する
        # P_cam = R @ P_world + t
        # P_cam = 0 のとき、P_world = -R.T @ t
        camera_pos_world = -R.T @ t
        
        # カメラ座標系をワールド座標系に変換
        # camera座標系の軸 -> world座標系の軸
        R_cam_to_world = R.T
        
        geometries = []
        
        # 座標軸
        axes = self.create_coordinate_frame(size=0.05)
        axes = self.transform_geometry(axes, R_cam_to_world, camera_pos_world)
        geometries.append(axes)
        
        # 視錐台（カメラパラメータがある場合）
        if K is not None:
            frustum = self.create_camera_frustum(K, 1920, 1200, scale=0.1)
            frustum = self.transform_geometry(frustum, R_cam_to_world, camera_pos_world)
            
            # カメラごとに色を変える
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
            cam_idx = self.camera_names.index(cam_name)
            color = colors[cam_idx % len(colors)]
            frustum.paint_uniform_color(color)
            
            geometries.append(frustum)
        
        # カメラ位置に球を追加
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(camera_pos_world)
        sphere.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(sphere)
        
        return geometries
    
    def create_board_geometry(self, frame_id: int, cam_name: Optional[str] = None) -> List:
        """ボードのジオメトリを作成"""
        # camera0のボード姿勢を使用（すべてのカメラで同じボード位置）
        if cam_name is None:
            cam_name = self.camera_names[0]
        
        pose = self.board_poses[cam_name][frame_id]
        rvec = np.array(pose['rvec'])
        tvec = np.array(pose['tvec'])
        
        # board -> camera の変換
        R_board_to_cam, _ = cv2.Rodrigues(rvec)
        t_board_to_cam = tvec
        
        # camera -> world の変換
        R_cam = self.camera_extrinsics[cam_name]['R']
        t_cam = self.camera_extrinsics[cam_name]['t']
        
        # board -> world の変換
        # P_cam = R_board_to_cam @ P_board + t_board_to_cam
        # P_world = R_cam.T @ (P_cam - t_cam)
        # P_world = R_cam.T @ (R_board_to_cam @ P_board + t_board_to_cam - t_cam)
        #         = R_cam.T @ R_board_to_cam @ P_board + R_cam.T @ (t_board_to_cam - t_cam)
        
        R_board_to_world = R_cam.T @ R_board_to_cam
        t_board_to_world = R_cam.T @ (t_board_to_cam - t_cam)
        
        geometries = []
        
        # ボードメッシュ
        board_mesh = self.create_board_mesh()
        board_mesh = self.transform_geometry(board_mesh, R_board_to_world, t_board_to_world)
        geometries.append(board_mesh)
        
        # ボード座標系の軸
        board_axes = self.create_coordinate_frame(size=0.05)
        board_axes = self.transform_geometry(board_axes, R_board_to_world, t_board_to_world)
        geometries.append(board_axes)
        
        return geometries
    
    def visualize(self, show_all_boards: bool = False, max_boards: int = 5):
        """3D可視化を実行"""
        logger.info("Creating 3D visualization...")
        
        geometries = []
        
        # ワールド座標系の軸
        world_axes = self.create_coordinate_frame(size=0.2)
        geometries.append(world_axes)
        
        # 各カメラを表示
        for cam_name in self.camera_names:
            logger.info(f"  Adding camera: {cam_name}")
            cam_geoms = self.create_camera_geometry(cam_name)
            geometries.extend(cam_geoms)
        
        # ボードを表示
        if show_all_boards:
            num_boards = len(self.board_poses[self.camera_names[0]])
            logger.info(f"  Adding all {num_boards} boards...")
            for frame_id in range(num_boards):
                board_geoms = self.create_board_geometry(frame_id)
                # 半透明にする
                # for geom in board_geoms:
                #     if isinstance(geom, o3d.geometry.TriangleMesh):
                #         colors = np.asarray(geom.vertex_colors)
                #         colors[:, 3] = 0.3  # Alpha値を設定（注：Open3Dは完全には対応していない）
                geometries.extend(board_geoms)
        else:
            # 最初と最後のいくつかを表示
            num_boards = len(self.board_poses[self.camera_names[0]])
            show_frames = list(range(min(max_boards, num_boards)))
            
            # 最後のフレームも追加
            if num_boards > max_boards:
                show_frames.append(num_boards - 1)
            
            logger.info(f"  Adding boards at frames: {show_frames}")
            for frame_id in show_frames:
                board_geoms = self.create_board_geometry(frame_id)
                geometries.extend(board_geoms)
        
        # 可視化
        logger.info("Displaying visualization...")
        logger.info("  - Use mouse to rotate/pan/zoom")
        logger.info("  - Press 'q' or 'ESC' to close")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"World Coordinate System - {self.world_config['reference_type']}",
            width=1280,
            height=720,
            left=100,
            top=100
        )
    
    def save_visualization(self, output_file: Optional[str] = None):
        """可視化を画像として保存"""
        if output_file is None:
            output_dir = Path(self.config['paths']['output_dir']) / "world_coordinate"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / "world_coordinate_visualization.png")
        
        logger.info(f"Saving visualization to: {output_file}")
        
        geometries = []
        
        # ワールド座標系の軸
        world_axes = self.create_coordinate_frame(size=0.2)
        geometries.append(world_axes)
        
        # 各カメラを表示
        for cam_name in self.camera_names:
            cam_geoms = self.create_camera_geometry(cam_name)
            geometries.extend(cam_geoms)
        
        # いくつかのボードを表示
        num_boards = len(self.board_poses[self.camera_names[0]])
        show_frames = [0, num_boards // 2, num_boards - 1] if num_boards > 1 else [0]
        
        for frame_id in show_frames:
            board_geoms = self.create_board_geometry(frame_id)
            geometries.extend(board_geoms)
        
        # 可視化ウィンドウを作成
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        vis.poll_events()
        vis.update_renderer()
        
        # 画像をキャプチャ
        vis.capture_screen_image(str(output_file))
        vis.destroy_window()
        
        logger.info(f"Visualization saved to: {output_file}")
    
    def run(self, show_all_boards: bool = False, max_boards: int = 5, 
           save_image: bool = True):
        """可視化を実行"""
        logger.info("="*80)
        logger.info("World Coordinate Visualization")
        logger.info("="*80)
        
        # データを読み込み
        self.load_world_coordinate_results()
        
        # 画像として保存
        if save_image:
            self.save_visualization()
        
        # インタラクティブ表示
        self.visualize(show_all_boards=show_all_boards, max_boards=max_boards)
        
        logger.info("="*80)
        logger.info("Visualization completed!")
        logger.info("="*80)


def main():
    """メイン処理"""
    import sys
    
    config_path = "calibration_config.yaml"
    show_all_boards = True
    max_boards = 5
    save_image = True
    
    # コマンドライン引数の解析
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--config='):
                config_path = arg.split('=')[1]
            elif arg == '--all-boards':
                show_all_boards = True
            elif arg.startswith('--max-boards='):
                max_boards = int(arg.split('=')[1])
            elif arg == '--no-save':
                save_image = False
    
    try:
        visualizer = WorldCoordinateVisualizer(config_path)
        visualizer.run(show_all_boards=show_all_boards, 
                      max_boards=max_boards,
                      save_image=save_image)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
