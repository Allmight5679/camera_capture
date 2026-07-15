"""
ChArUco検出パイプライン（ArUco→ChArUco）

このスクリプトは、キャプチャされた画像からChArUcoボードを検出し、
検出結果を可視化・保存します。

実装内容:
1. ArUcoマーカーの検出
2. コーナーのサブピクセル精度への洗練
3. ChArUcoコーナーの補間
4. 検出結果のオーバーレイ可視化
5. 検出統計の記録
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import json


class CharucoDetector:
    """ChArUcoボード検出器クラス"""
    
    def __init__(self, calibration_config_path="calibration_config.yaml"):
        """
        初期化
        
        Args:
            calibration_config_path: キャリブレーション設定ファイルのパス
        """
        
        # キャリブレーション設定ファイルの読み込み
        with open(calibration_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ArUco辞書の設定
        dict_name = self.config['board']['dictionary']
        dict_id = getattr(cv2.aruco, dict_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        
        # ChArUcoボードの作成
        self.board = cv2.aruco.CharucoBoard(
            (
                self.config['board']['squares_x'],
                self.config['board']['squares_y']
            ),
            self.config['board']['square_length'] / 1000.0,  # mm → m
            self.config['board']['marker_length'] / 1000.0,  # mm → m
            self.dictionary
        )
        
        # 検出パラメータの設定
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # コーナー洗練パラメータ
        self.corner_win_size = self.config['detection']['corner_refinement_winsize']
        self.corner_max_iter = self.config['detection']['corner_refinement_max_iter']
        self.corner_epsilon = self.config['detection']['corner_refinement_epsilon']
        
        # 最小マーカー数
        self.min_markers = self.config['detection']['min_markers']
        
        # パス設定
        self.captured_images_dir = Path(self.config['paths']['captured_images'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.detection_cache_dir = Path(self.config['paths']['detection_cache'])
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(exist_ok=True)
        self.detection_cache_dir.mkdir(exist_ok=True)
    
    def detect_charuco(self, gray):
        """
        グレースケール画像からChArUcoコーナーを検出
        
        Args:
            gray: グレースケール画像 (numpy.ndarray)
        
        Returns:
            dict or None: 検出結果の辞書、または検出失敗時はNone
                - ch_corners: ChArUcoコーナー座標
                - ch_ids: ChArUcoコーナーID
                - corners: ArUcoマーカーコーナー座標
                - ids: ArUcoマーカーID
        """
        
        # ArUcoマーカーの検出
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_params
        )
        
        # マーカーが検出されなかった場合
        if ids is None or len(ids) == 0:
            return None
        
        # サブピクセル精度でコーナーを洗練
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.corner_max_iter,
            self.corner_epsilon
        )
        
        for corner in corners:
            cv2.cornerSubPix(
                gray,
                corner,
                (self.corner_win_size, self.corner_win_size),
                (-1, -1),
                criteria
            )
        
        # ChArUcoコーナーの補間
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            self.board
        )
        
        # 補間に失敗した場合、または検出数が少ない場合
        if not ret or ch_ids is None or len(ch_ids) < self.min_markers:
            return None
        
        return {
            'ch_corners': ch_corners,
            'ch_ids': ch_ids,
            'corners': corners,
            'ids': ids
        }
    
    def draw_overlay(self, img, det):
        """
        検出結果を画像にオーバーレイ描画
        
        Args:
            img: 入力画像 (BGR)
            det: 検出結果の辞書
        
        Returns:
            numpy.ndarray: オーバーレイ描画された画像
        """
        
        out = img.copy()
        
        # ArUcoマーカーの描画
        cv2.aruco.drawDetectedMarkers(
            out,
            det['corners'],
            det['ids']
        )
        
        # ChArUcoコーナーの描画（緑）
        cv2.aruco.drawDetectedCornersCharuco(
            out,
            det['ch_corners'],
            det['ch_ids'],
            (0, 255, 0)
        )
        
        # 検出情報のテキスト表示
        text = (
            f"Markers: {len(det['ids'])} | "
            f"ChArUco Corners: {len(det['ch_ids'])}"
        )
        
        cv2.putText(
            out,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return out
    
    def process_camera(self, camera_name):
        """
        特定のカメラの全画像を処理
        
        Args:
            camera_name: カメラ名 (例: "camera0")
        
        Returns:
            dict: 処理結果の統計情報
        """
        
        print(f"\n{'=' * 60}")
        print(f"Processing {camera_name}")
        print(f"{'=' * 60}")
        
        # 入力・出力ディレクトリの設定
        input_dir = self.captured_images_dir / camera_name
        overlay_output_dir = self.output_dir / f"{camera_name}_overlays"
        overlay_output_dir.mkdir(exist_ok=True)
        
        # 画像ファイルの取得
        image_files = sorted(input_dir.glob("*.png"))
        
        if not image_files:
            print(f"Warning: No images found in {input_dir}")
            return None
        
        print(f"Found {len(image_files)} images")
        
        # 検出結果の保存用リスト
        successful_detections = []
        success_count = 0
        failed_frames = []
        img_size = None
        
        # 各画像を処理
        for i, img_path in enumerate(image_files):
            
            # 画像の読み込み
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                failed_frames.append(img_path.name)
                continue
            
            # 画像サイズの記録（最初の画像のみ）
            if img_size is None:
                img_size = (
                    img.shape[1],
                    img.shape[0]
                )
            
            # グレースケール変換
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )
            
            # ChArUco検出
            det = self.detect_charuco(gray)
            
            if det is not None:
                
                # 検出成功
                success_count += 1
                
                # 成功した画像名と検出データを一緒に保存
                successful_detections.append({
                    'image_name': img_path.name,
                    'ch_corners': det['ch_corners'],
                    'ch_ids': det['ch_ids'],
                    'corners': det['corners'],
                    'ids': det['ids']
                })
                
                # オーバーレイ画像の作成と保存
                overlay_img = self.draw_overlay(
                    img,
                    det
                )
                
                overlay_path = (
                    overlay_output_dir
                    / img_path.name
                )
                
                cv2.imwrite(
                    str(overlay_path),
                    overlay_img
                )
                
                # 進捗表示
                if (
                    (i + 1) % 5 == 0
                    or (i + 1) == len(image_files)
                ):
                    print(
                        f"Processed {i + 1}/{len(image_files)} "
                        f"- Success: {success_count}"
                    )
            
            else:
                
                # 検出失敗
                failed_frames.append(
                    img_path.name
                )
                
                print(
                    f"Failed to detect: "
                    f"{img_path.name}"
                )
        
        # 統計情報の作成
        total_frames = len(image_files)
        
        success_rate = (
            success_count
            / total_frames
            * 100
            if total_frames > 0
            else 0
        )
        
        stats = {
            'camera_name': camera_name,
            'total_frames': total_frames,
            'success_count': success_count,
            'failed_count': total_frames - success_count,
            'success_rate': success_rate,
            'failed_frames': failed_frames,
            'img_size': img_size
        }
        
        # 統計情報の表示
        print(f"\n{'-' * 60}")
        print(f"Detection Statistics for {camera_name}:")
        print(f"  Total frames: {total_frames}")
        print(
            f"  Success: {success_count} "
            f"({success_rate:.1f}%)"
        )
        print(
            f"  Failed: {total_frames - success_count} "
            f"({100 - success_rate:.1f}%)"
        )
        
        if failed_frames:
            print(
                f"  Failed frames: "
                f"{', '.join(failed_frames[:5])}"
                + (
                    f" ... and {len(failed_frames) - 5} more"
                    if len(failed_frames) > 5
                    else ""
                )
            )
        
        print(f"{'-' * 60}")
        
        # 検出結果の保存（JSON形式）
        if success_count > 0:
            
            frames_data = []
            
            for frame_id, detection in enumerate(
                successful_detections
            ):
                
                frame_data = {
                    'frame_id': frame_id,
                    'image_name': detection['image_name'],
                    
                    'ch_corners':
                        detection['ch_corners']
                        .reshape(-1, 2)
                        .tolist(),
                    
                    'ch_ids':
                        detection['ch_ids']
                        .flatten()
                        .tolist(),
                    
                    'aruco_corners': [
                        corner
                        .reshape(-1, 2)
                        .tolist()
                        
                        for corner in detection['corners']
                    ],
                    
                    'aruco_ids':
                        detection['ids']
                        .flatten()
                        .tolist()
                }
                
                frames_data.append(
                    frame_data
                )
            
            # 検出結果を統合したJSONを作成
            detection_data = {
                'camera_name': camera_name,
                'img_size': (
                    list(img_size)
                    if img_size
                    else [0, 0]
                ),
                'num_frames': success_count,
                'frames': frames_data
            }
            
            detection_file = (
                self.detection_cache_dir
                / f"detections_{camera_name}.json"
            )
            
            with open(
                detection_file,
                'w',
                encoding='utf-8'
            ) as f:
                
                json.dump(
                    detection_data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            print(
                f"Saved detection results to: "
                f"{detection_file}"
            )
            
            # 統計情報の保存
            stats_file = (
                self.detection_cache_dir
                / f"stats_{camera_name}.json"
            )
            
            with open(
                stats_file,
                'w',
                encoding='utf-8'
            ) as f:
                
                json.dump(
                    stats,
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            print(
                f"Saved statistics to: "
                f"{stats_file}"
            )
        
        return stats
    
    def process_all_cameras(self):
        """
        captured_images内の全カメラフォルダを処理
        
        Returns:
            dict: 全カメラの統計情報
        """
        
        all_stats = {}
        
        print(f"\n{'=' * 60}")
        print("ChArUco Detection Pipeline")
        print(
            f"Started at: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'=' * 60}")
        
        # captured_images内のcameraフォルダを自動検出
        camera_dirs = sorted(
            path
            for path in self.captured_images_dir.iterdir()
            if (
                path.is_dir()
                and path.name.startswith("camera")
            )
        )
        
        if not camera_dirs:
            print(
                f"No camera folders found in "
                f"{self.captured_images_dir}"
            )
            return all_stats
        
        print(
            f"Found {len(camera_dirs)} camera(s):"
        )
        
        for camera_dir in camera_dirs:
            print(
                f"  - {camera_dir.name}"
            )
        
        # 各カメラを処理
        for camera_dir in camera_dirs:
            
            camera_name = camera_dir.name
            
            stats = self.process_camera(
                camera_name
            )
            
            if stats:
                all_stats[
                    camera_name
                ] = stats
        
        # 全体の統計情報を表示
        print(f"\n{'=' * 60}")
        print("Overall Statistics:")
        print(f"{'=' * 60}")
        
        total_success = sum(
            s['success_count']
            for s in all_stats.values()
        )
        
        total_frames = sum(
            s['total_frames']
            for s in all_stats.values()
        )
        
        overall_rate = (
            total_success
            / total_frames
            * 100
            if total_frames > 0
            else 0
        )
        
        print(
            f"Cameras processed: "
            f"{len(all_stats)}"
        )
        
        print(
            f"Total frames: "
            f"{total_frames}"
        )
        
        print(
            f"Total success: "
            f"{total_success} "
            f"({overall_rate:.1f}%)"
        )
        
        print(
            f"Total failed: "
            f"{total_frames - total_success} "
            f"({100 - overall_rate:.1f}%)"
        )
        
        # 全体統計の保存
        overall_stats = {
            'timestamp':
                datetime.now().isoformat(),
            
            'cameras':
                all_stats,
            
            'overall': {
                'total_frames':
                    total_frames,
                
                'total_success':
                    total_success,
                
                'total_failed':
                    total_frames - total_success,
                
                'overall_success_rate':
                    overall_rate
            }
        }
        
        overall_stats_file = (
            self.detection_cache_dir
            / "overall_stats.json"
        )
        
        with open(
            overall_stats_file,
            'w',
            encoding='utf-8'
        ) as f:
            
            json.dump(
                overall_stats,
                f,
                indent=2,
                ensure_ascii=False
            )
        
        print(
            f"\nSaved overall statistics to: "
            f"{overall_stats_file}"
        )
        
        print(f"\n{'=' * 60}")
        print(
            f"Finished at: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'=' * 60}\n")
        
        return all_stats


def main():
    """メイン関数"""
    
    # 検出器の初期化
    detector = CharucoDetector(
        calibration_config_path="calibration_config.yaml"
    )
    
    # 全カメラの画像を処理
    stats = detector.process_all_cameras()
    
    print(
        "\nDetection pipeline "
        "completed successfully!"
    )
    
    print("Results saved to:")
    
    print(
        f"  - Detection cache: "
        f"{detector.detection_cache_dir}"
    )
    
    print(
        f"  - Overlay images: "
        f"{detector.output_dir}"
    )


if __name__ == "__main__":
    main()