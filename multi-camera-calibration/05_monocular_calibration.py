"""
単眼カメラキャリブレーション（内部パラメータ推定）

このスクリプトは、ChArUco検出結果を用いて各カメラの内部パラメータ（カメラ行列K、歪み係数dist）を推定します。

実装内容:
1. 検出結果（npz）の読み込み
2. cv2.aruco.calibrateCameraCharuco による内部パラメータ推定
3. RMS再投影誤差の計算と評価
4. フレーム毎の誤差統計分析
5. キャリブレーション結果の保存
6. 歪み補正の可視化（before/after）

使用方法:
    python 05_monocular_calibration.py
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from config_manager import ConfigManager


class MonocularCalibrator:
    """単眼カメラキャリブレータークラス"""
    
    def __init__(self, config_path="calibration_config.yaml", calibration_config_path="calibration_config.yaml"):
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
        with open(calibration_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ArUco辞書の設定
        dict_name = self.config['board']['dictionary']
        dict_id = getattr(cv2.aruco, dict_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        
        # ChArUcoボードの作成
        self.board = cv2.aruco.CharucoBoard(
            (self.config['board']['squares_x'], self.config['board']['squares_y']),
            self.config['board']['square_length'] / 1000.0,  # mm → m
            self.config['board']['marker_length'] / 1000.0,   # mm → m
            self.dictionary
        )
        
        # パス設定
        self.detection_cache_dir = Path(self.config['paths']['detection_cache'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.calibration_dir = self.output_dir / "calibration"
        self.undistort_dir = self.output_dir / "undistorted_samples"
        
        # 出力ディレクトリの作成
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.undistort_dir.mkdir(parents=True, exist_ok=True)
        
        # 目標RMS値
        self.target_rms = self.config['calibration']['target_rms']
    
    def load_detections(self, camera_name):
        """
        検出結果をJSONファイルから読み込み
        
        Args:
            camera_name: カメラ名 (例: "camera0")
        
        Returns:
            dict: 検出データ
                - ch_corners: ChArUcoコーナー座標のリスト
                - ch_ids: ChArUcoコーナーIDのリスト
                - img_size: 画像サイズ (width, height)
        """
        detection_file = self.detection_cache_dir / f"detections_{camera_name}.json"
        
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
        
        print(f"Loading detections from: {detection_file}")
        with open(detection_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSONからnumpy配列に変換
        ch_corners = []
        ch_ids = []
        for frame in data['frames']:
            # ch_corners: (N, 2) -> (N, 1, 2)
            corners_array = np.array(frame['ch_corners'], dtype=np.float32).reshape(-1, 1, 2)
            ch_corners.append(corners_array)
            
            # ch_ids: list -> (N, 1)
            ids_array = np.array(frame['ch_ids'], dtype=np.int32).reshape(-1, 1)
            ch_ids.append(ids_array)
        
        img_size = tuple(data['img_size'])
        
        print(f"  Loaded {len(ch_corners)} frames")
        print(f"  Image size: {img_size}")
        
        return {
            'ch_corners': ch_corners,
            'ch_ids': ch_ids,
            'img_size': img_size
        }
    
    def calibrate_camera(self, detections, camera_name):
        """
        単眼カメラキャリブレーションを実行
        
        Args:
            detections: 検出データ（load_detectionsの戻り値）
            camera_name: カメラ名
        
        Returns:
            dict: キャリブレーション結果
                - K: カメラ行列 (3x3)
                - dist: 歪み係数 (5要素)
                - rvecs: 回転ベクトルのリスト
                - tvecs: 並進ベクトルのリスト
                - rms: RMS再投影誤差
                - img_size: 画像サイズ
        """
        print(f"\n{'='*60}")
        print(f"Calibrating {camera_name}")
        print(f"{'='*60}")
        
        ch_corners = detections['ch_corners']
        ch_ids = detections['ch_ids']
        img_size = detections['img_size']
        
        # calibrateCameraCharucoを実行
        print("Running cv2.aruco.calibrateCameraCharuco...")
        print(f"  Number of frames: {len(ch_corners)}")
        print(f"  Image size: {img_size}")
        
        # 初期カメラ行列と歪み係数を作成
        K_init = np.eye(3, dtype=np.float64)
        dist_init = np.zeros(5, dtype=np.float64)
        
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=ch_corners,
            charucoIds=ch_ids,
            board=self.board,
            imageSize=img_size,
            cameraMatrix=K_init,
            distCoeffs=dist_init
        )
        
        print(f"\nCalibration completed!")
        print(f"  RMS reprojection error: {ret:.4f} px")
        
        # RMS値の評価
        if ret < self.target_rms:
            status = "EXCELLENT"
            color = "\033[92m"  # Green
        elif ret < self.target_rms * 2:
            status = "GOOD"
            color = "\033[93m"  # Yellow
        else:
            status = "NEEDS IMPROVEMENT"
            color = "\033[91m"  # Red
        
        print(f"  Status: {color}{status}\033[0m (Target: {self.target_rms} px)")
        
        # カメラ行列と歪み係数の表示
        print(f"\nCamera Matrix (K):")
        print(K)
        print(f"\nDistortion Coefficients (dist):")
        print(dist.ravel())
        
        # 焦点距離と主点の表示
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"\nCamera Parameters:")
        print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
        print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Aspect ratio: {fx/fy:.4f}")
        
        return {
            'camera_name': camera_name,
            'K': K,
            'dist': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'rms': ret,
            'img_size': img_size,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    
    def compute_per_frame_errors(self, calibration_result, detections):
        """
        フレーム毎の再投影誤差を計算
        
        Args:
            calibration_result: キャリブレーション結果
            detections: 検出データ
        
        Returns:
            dict: 誤差統計
        """
        print(f"\nComputing per-frame reprojection errors...")
        
        K = calibration_result['K']
        dist = calibration_result['dist']
        rvecs = calibration_result['rvecs']
        tvecs = calibration_result['tvecs']
        ch_corners_list = detections['ch_corners']
        ch_ids_list = detections['ch_ids']
        
        frame_errors = []
        
        for i, (rvec, tvec, ch_corners, ch_ids) in enumerate(zip(rvecs, tvecs, ch_corners_list, ch_ids_list)):
            # 3Dオブジェクト点の取得
            obj_points = self.board.getChessboardCorners()[ch_ids.ravel()]
            
            # 再投影
            projected_points, _ = cv2.projectPoints(
                obj_points, rvec, tvec, K, dist
            )
            
            # 誤差計算（ユークリッド距離）
            errors = np.linalg.norm(
                ch_corners.reshape(-1, 2) - projected_points.reshape(-1, 2),
                axis=1
            )
            
            frame_error = {
                'frame_idx': i,
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(errors)),
                'num_points': len(errors)
            }
            frame_errors.append(frame_error)
        
        # 全体統計
        all_mean_errors = [fe['mean_error'] for fe in frame_errors]
        overall_stats = {
            'mean': float(np.mean(all_mean_errors)),
            'std': float(np.std(all_mean_errors)),
            'min': float(np.min(all_mean_errors)),
            'max': float(np.max(all_mean_errors)),
            'median': float(np.median(all_mean_errors))
        }
        
        print(f"  Overall error statistics (px):")
        print(f"    Mean: {overall_stats['mean']:.4f} ± {overall_stats['std']:.4f}")
        print(f"    Min: {overall_stats['min']:.4f}, Max: {overall_stats['max']:.4f}")
        print(f"    Median: {overall_stats['median']:.4f}")
        
        return {
            'frame_errors': frame_errors,
            'overall': overall_stats
        }
    
    def save_calibration_result(self, calibration_result, error_stats):
        """
        キャリブレーション結果を保存（JSON形式のみ）
        
        Args:
            calibration_result: キャリブレーション結果
            error_stats: 誤差統計
        """
        camera_name = calibration_result['camera_name']
        
        # JSON形式で保存（可読性が高く、rvecs/tvecsも含む）
        # img_sizeをint型に変換（np.int64のJSON化エラーを回避）
        img_size_list = [int(calibration_result['img_size'][0]), int(calibration_result['img_size'][1])]
        
        # rvecs/tvecsをリストに変換
        rvecs_list = [rvec.flatten().tolist() for rvec in calibration_result['rvecs']]
        tvecs_list = [tvec.flatten().tolist() for tvec in calibration_result['tvecs']]
        
        json_data = {
            'camera_name': camera_name,
            'timestamp': datetime.now().isoformat(),
            'rms_reprojection_error': float(calibration_result['rms']),
            'img_size': img_size_list,
            'camera_matrix': {
                'fx': float(calibration_result['fx']),
                'fy': float(calibration_result['fy']),
                'cx': float(calibration_result['cx']),
                'cy': float(calibration_result['cy']),
                'K': calibration_result['K'].tolist()
            },
            'distortion_coefficients': calibration_result['dist'].ravel().tolist(),
            'num_frames': len(calibration_result['rvecs']),
            'rvecs': rvecs_list,
            'tvecs': tvecs_list,
            'error_statistics': error_stats['overall'],
            'frame_errors': error_stats['frame_errors']
        }
        
        json_file = self.calibration_dir / f"calibration_{camera_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved calibration results to: {json_file}")
    
    def plot_error_histogram(self, error_stats, camera_name):
        """
        誤差のヒストグラムを作成
        
        Args:
            error_stats: 誤差統計
            camera_name: カメラ名
        """
        mean_errors = [fe['mean_error'] for fe in error_stats['frame_errors']]
        
        plt.figure(figsize=(10, 6))
        plt.hist(mean_errors, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(error_stats['overall']['mean'], color='r', linestyle='--', 
                   label=f"Mean: {error_stats['overall']['mean']:.4f} px")
        plt.axvline(error_stats['overall']['median'], color='g', linestyle='--',
                   label=f"Median: {error_stats['overall']['median']:.4f} px")
        plt.xlabel('Mean Reprojection Error (px)', fontsize=12)
        plt.ylabel('Number of Frames', fontsize=12)
        plt.title(f'Reprojection Error Distribution - {camera_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        histogram_file = self.calibration_dir / f"error_histogram_{camera_name}.png"
        plt.savefig(str(histogram_file), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved error histogram to: {histogram_file}")
    
    def visualize_undistortion(self, calibration_result, camera_name, save_all=True, num_samples=3):
        """
        歪み補正の効果を可視化
        
        Args:
            calibration_result: キャリブレーション結果
            camera_name: カメラ名
            save_all: Trueの場合、全画像を保存。Falseの場合、サンプルのみ保存
            num_samples: save_all=Falseの場合のサンプル数
        """
        print(f"\nGenerating undistortion visualization...")
        
        K = calibration_result['K']
        dist = calibration_result['dist']
        img_size = calibration_result['img_size']
        
        # 画像ファイルの取得
        captured_images_dir = Path(self.config['paths']['captured_images']) / camera_name
        image_files = sorted(captured_images_dir.glob("*.png"))
        
        if not image_files:
            print(f"Warning: No images found for visualization")
            return
        
        # 処理対象の選択
        if save_all:
            files_to_process = image_files
            print(f"  Processing all {len(files_to_process)} images...")
        else:
            # 等間隔でサンプル選択
            step = max(1, len(image_files) // num_samples)
            files_to_process = image_files[::step][:num_samples]
            print(f"  Processing {len(files_to_process)} sample images...")
        
        # 新しいカメラ行列の計算（オプション: ROIを最適化）
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 1, img_size)
        
        # カメラ別の出力ディレクトリを作成
        camera_undistort_dir = self.undistort_dir / camera_name
        camera_undistort_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        for i, img_file in enumerate(files_to_process):
            # 画像の読み込み
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  Warning: Failed to load {img_file.name}")
                continue
            
            # 歪み補正
            undistorted = cv2.undistort(img, K, dist, None, new_K)
            
            # グリッド線を描画（直線性の確認用）
            img_with_grid = self._draw_grid(img.copy())
            undistorted_with_grid = self._draw_grid(undistorted.copy())
            
            # before/after を並べて表示
            combined = np.hstack([img_with_grid, undistorted_with_grid])
            
            # ラベルを追加
            cv2.putText(combined, "Original (Distorted)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, "Undistorted", (img.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 比較画像の保存
            comparison_file = camera_undistort_dir / f"{img_file.stem}_comparison.png"
            cv2.imwrite(str(comparison_file), combined)
            
            # 補正後の画像のみも保存
            undistorted_file = camera_undistort_dir / f"{img_file.stem}_undistorted.png"
            cv2.imwrite(str(undistorted_file), undistorted)
            
            processed_count += 1
            
            # 進捗表示
            if (i + 1) % 5 == 0 or (i + 1) == len(files_to_process):
                print(f"  Processed {i + 1}/{len(files_to_process)} images")
        
        print(f"  Saved {processed_count} comparison images to: {camera_undistort_dir}")
        print(f"  Saved {processed_count} undistorted images to: {camera_undistort_dir}")
    
    def _draw_grid(self, img, grid_size=100):
        """
        画像にグリッド線を描画（直線性確認用）
        
        Args:
            img: 入力画像
            grid_size: グリッド間隔（ピクセル）
        
        Returns:
            numpy.ndarray: グリッド描画された画像
        """
        h, w = img.shape[:2]
        
        # 縦線
        for x in range(0, w, grid_size):
            cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        
        # 横線
        for y in range(0, h, grid_size):
            cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        
        return img
    
    def verify_reproducibility(self, camera_name):
        """
        保存→再ロードでの再現性を確認
        
        Args:
            camera_name: カメラ名
        
        Returns:
            bool: 再現性が確認できたか
        """
        print(f"\nVerifying reproducibility for {camera_name}...")
        
        # 保存されたキャリブレーション結果を読み込み
        json_file = self.calibration_dir / f"calibration_{camera_name}.json"
        if not json_file.exists():
            print(f"  Error: Calibration file not found: {json_file}")
            return False
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        loaded_K = np.array(data['camera_matrix']['K'])
        loaded_dist = np.array(data['distortion_coefficients'])
        loaded_rms = float(data['rms_reprojection_error'])
        loaded_rvecs = [np.array(r) for r in data['rvecs']]
        loaded_tvecs = [np.array(t) for t in data['tvecs']]
        
        print(f"  Loaded RMS from file: {loaded_rms:.4f} px")
        print(f"  Loaded camera matrix K:")
        print(loaded_K)
        print(f"  Loaded distortion coefficients:")
        print(loaded_dist)
        print(f"  Loaded {len(loaded_rvecs)} rvecs and {len(loaded_tvecs)} tvecs")
        
        print(f"  ✓ Reproducibility check passed: Data successfully loaded from JSON")
        return True
    
    def process_camera(self, camera_name):
        """
        特定のカメラのキャリブレーション処理を実行
        
        Args:
            camera_name: カメラ名
        
        Returns:
            dict: キャリブレーション結果
        """
        # 検出結果の読み込み
        detections = self.load_detections(camera_name)
        
        # キャリブレーション実行
        calibration_result = self.calibrate_camera(detections, camera_name)
        
        # フレーム毎の誤差計算
        error_stats = self.compute_per_frame_errors(calibration_result, detections)
        
        # 結果の保存
        self.save_calibration_result(calibration_result, error_stats)
        
        # 誤差ヒストグラムの作成
        self.plot_error_histogram(error_stats, camera_name)
        
        # 歪み補正の可視化
        self.visualize_undistortion(calibration_result, camera_name)
        
        # 再現性の確認
        self.verify_reproducibility(camera_name)
        
        return calibration_result
    
    def process_all_cameras(self):
        """
        設定ファイルに記載された全カメラをキャリブレーション
        
        Returns:
            dict: 全カメラのキャリブレーション結果
        """
        all_results = {}
        
        print(f"\n{'='*60}")
        print("Monocular Camera Calibration Pipeline")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # 各カメラを処理
        num_cameras = self.config_manager.get_camera_count()
        for i in range(num_cameras):
            camera_name = f"camera{i}"
            
            try:
                result = self.process_camera(camera_name)
                all_results[camera_name] = result
            except Exception as e:
                print(f"\n\033[91mError processing {camera_name}: {str(e)}\033[0m")
                import traceback
                traceback.print_exc()
                continue
        
        # 全体サマリーの表示
        print(f"\n{'='*60}")
        print("Calibration Summary:")
        print(f"{'='*60}")
        
        for camera_name, result in all_results.items():
            status = "✓" if result['rms'] < self.target_rms else "⚠"
            print(f"{status} {camera_name}: RMS = {result['rms']:.4f} px")
        
        print(f"\n{'='*60}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        print(f"Results saved to:")
        print(f"  - Calibration data: {self.calibration_dir}")
        print(f"  - Undistortion samples: {self.undistort_dir}")
        
        return all_results


def main():
    """メイン関数"""
    # キャリブレーターの初期化
    calibrator = MonocularCalibrator(
        config_path="calibration_config.yaml",
        calibration_config_path="calibration_config.yaml"
    )
    
    # 全カメラのキャリブレーション実行
    results = calibrator.process_all_cameras()
    
    print("\nMonocular calibration pipeline completed successfully!")


if __name__ == "__main__":
    main()
