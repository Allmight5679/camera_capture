"""
06_pose_estimation.py
マルチカメラ同期と姿勢推定

実装内容:
1. タイムスタンプベースのフレーム対応
2. 各カメラ・各フレームでのボード姿勢推定
3. 姿勢可視化（座標軸オーバレイ）
4. 統計情報の記録と保存
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class PoseEstimator:
    """ChArUcoボードの姿勢推定を行うクラス"""

    def __init__(self, calibration_config_path: str = "calibration_config.yaml"):
        """
        初期化

        Args:
            calibration_config_path:
                キャリブレーション設定ファイルのパス
        """

        # キャリブレーション設定ファイルの読み込み
        self.config = self._load_config(calibration_config_path)

        self.board = self._create_board()

        self.detector_params = cv2.aruco.DetectorParameters()

        self.cameras_data = {}
        self.pose_results = {}

    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""

        with open(config_path, "r", encoding="utf-8") as f:

            return yaml.safe_load(f)

    def _create_board(self) -> cv2.aruco.CharucoBoard:
        """ChArUcoボードの作成"""

        board_cfg = self.config["board"]

        # ArUco辞書の取得
        dict_name = board_cfg["dictionary"]

        dict_id = getattr(cv2.aruco, dict_name)

        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # ChArUcoボードの作成
        board = cv2.aruco.CharucoBoard(
            (board_cfg["squares_x"], board_cfg["squares_y"]),
            board_cfg["square_length"] / 1000.0,
            board_cfg["marker_length"] / 1000.0,
            dictionary,
        )

        logger.info(
            f"ChArUcoボード作成: "
            f"{board_cfg['squares_x']}x"
            f"{board_cfg['squares_y']}, "
            f"square="
            f"{board_cfg['square_length']}mm, "
            f"marker="
            f"{board_cfg['marker_length']}mm"
        )

        return board

    def find_camera_names(self) -> List[str]:
        """
        calibration_results/calibration 内の
        calibration_camera*.json ファイルから
        カメラ名を自動検出する

        Returns:
            例:
            ["camera0", "camera1"]
        """

        calibration_dir = Path(self.config["paths"]["output_dir"]) / "calibration"

        calibration_files = sorted(calibration_dir.glob("calibration_camera*.json"))

        camera_names = [
            calibration_file.stem.replace("calibration_", "")
            for calibration_file in calibration_files
        ]

        return camera_names

    def load_camera_calibration(self, camera_name: str) -> bool:
        """
        カメラキャリブレーション結果の読み込み

        Args:
            camera_name: カメラ名

        Returns:
            読み込み成功ならTrue
        """

        calib_path = (
            Path(self.config["paths"]["output_dir"])
            / "calibration"
            / f"calibration_{camera_name}.json"
        )

        if not calib_path.exists():

            logger.error(
                f"キャリブレーションファイルが" f"見つかりません: " f"{calib_path}"
            )

            return False

        with open(calib_path, "r") as f:

            calib_data = json.load(f)

        # カメラ行列と歪み係数を抽出
        K = np.array(calib_data["camera_matrix"]["K"], dtype=np.float64)

        dist = np.array(calib_data["distortion_coefficients"], dtype=np.float64)

        self.cameras_data[camera_name] = {
            "K": K,
            "dist": dist,
            "img_size": tuple(calib_data["img_size"]),
            "rms": calib_data["rms_reprojection_error"],
        }

        logger.info(
            f"{camera_name}: "
            f"キャリブレーションデータ"
            f"読み込み完了 "
            f"(RMS: "
            f"{calib_data['rms_reprojection_error']:.4f}px)"
        )

        return True

    def load_detection_cache(self, camera_name: str) -> Dict:
        """
        検出キャッシュの読み込み

        Args:
            camera_name: カメラ名

        Returns:
            検出結果の辞書
            {image_name: detection_data}
        """

        cache_path = (
            Path(self.config["paths"]["detection_cache"])
            / f"detections_{camera_name}.json"
        )

        if not cache_path.exists():

            logger.error(f"検出キャッシュが" f"見つかりません: " f"{cache_path}")

            return {}

        with open(cache_path, "r") as f:

            data = json.load(f)

        # image_name をキーに変換
        detections = {}

        for frame in data.get("frames", []):

            image_name = frame.get("image_name", "")

            if image_name:

                detections[image_name] = frame

        logger.info(
            f"{camera_name}: "
            f"{len(detections)} フレームの"
            f"検出データを読み込みました"
        )

        return detections

    def estimate_pose_from_detection(
        self, detection: Dict, K: np.ndarray, dist: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        検出結果からボード姿勢を推定

        Returns:
            (rvec, tvec)
            または None
        """

        if not detection or "ch_corners" not in detection or "ch_ids" not in detection:

            return None

        # ChArUco corners
        ch_corners = np.array(detection["ch_corners"], dtype=np.float32).reshape(
            -1, 1, 2
        )

        # ChArUco IDs
        ch_ids = np.array(detection["ch_ids"], dtype=np.int32).reshape(-1, 1)

        if len(ch_corners) < self.config["detection"]["min_markers"]:

            return None

        rvec = np.zeros((3, 1), dtype=np.float64)

        tvec = np.zeros((3, 1), dtype=np.float64)

        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.board, K, dist, rvec, tvec
        )

        if not ret:
            return None

        return (rvec, tvec)

    def draw_pose_overlay(
        self,
        img: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """
        姿勢を画像上に描画
        """

        out = img.copy()

        cv2.drawFrameAxes(out, K, dist, rvec, tvec, axis_length)

        rvec_deg = np.linalg.norm(rvec) * 180 / np.pi

        tvec_norm = np.linalg.norm(tvec)

        cv2.putText(
            out,
            f"Rotation: " f"{rvec_deg:.2f} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            out,
            f"Distance: " f"{tvec_norm:.3f} m",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return out

    def process_camera(self, camera_name: str) -> Dict:
        """
        1つのカメラの全フレームを処理
        """

        logger.info(f"\n{'=' * 60}")

        logger.info(f"{camera_name} " f"の姿勢推定を開始")

        logger.info(f"{'=' * 60}")

        # キャリブレーションデータの読み込み
        if not self.load_camera_calibration(camera_name):

            return {}

        # 検出キャッシュの読み込み
        detections = self.load_detection_cache(camera_name)

        if not detections:
            return {}

        K = self.cameras_data[camera_name]["K"]

        dist = self.cameras_data[camera_name]["dist"]

        # 画像ディレクトリ
        img_dir = Path(self.config["paths"]["captured_images"]) / camera_name

        # 出力ディレクトリ
        output_dir = (
            Path(self.config["paths"]["output_dir"]) / "pose_estimation" / camera_name
        )

        overlay_dir = output_dir / "overlays"

        overlay_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        rvecs_list = []
        tvecs_list = []
        success_count = 0

        # 各フレームを処理
        for image_name, detection in detections.items():

            img_path = img_dir / image_name

            if not img_path.exists():

                logger.warning(f"画像ファイルが" f"見つかりません: " f"{img_path}")

                continue

            pose = self.estimate_pose_from_detection(detection, K, dist)

            if pose is None:

                results[image_name] = {"success": False}

                continue

            rvec, tvec = pose

            success_count += 1

            results[image_name] = {
                "success": True,
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),
                "rotation_angle_deg": float(np.linalg.norm(rvec) * 180 / np.pi),
                "distance_m": float(np.linalg.norm(tvec)),
            }

            rvecs_list.append(rvec.flatten())

            tvecs_list.append(tvec.flatten())

            # オーバーレイ画像
            img = cv2.imread(str(img_path))

            if img is not None:

                overlay = self.draw_pose_overlay(img, K, dist, rvec, tvec)

                timestamp = Path(image_name).stem

                overlay_path = overlay_dir / f"{timestamp}_pose.png"

                cv2.imwrite(str(overlay_path), overlay)

        # 統計情報
        if rvecs_list:

            rvecs_array = np.array(rvecs_list)

            tvecs_array = np.array(tvecs_list)

            rotation_angles = np.linalg.norm(rvecs_array, axis=1) * 180 / np.pi

            distances = np.linalg.norm(tvecs_array, axis=1)

            statistics = {
                "total_frames": len(detections),
                "successful_poses": success_count,
                "success_rate": success_count / len(detections),
                "rotation_angle_deg": {
                    "mean": float(np.mean(rotation_angles)),
                    "std": float(np.std(rotation_angles)),
                    "min": float(np.min(rotation_angles)),
                    "max": float(np.max(rotation_angles)),
                },
                "distance_m": {
                    "mean": float(np.mean(distances)),
                    "std": float(np.std(distances)),
                    "min": float(np.min(distances)),
                    "max": float(np.max(distances)),
                },
                "rvec_components": {
                    "mean": np.mean(rvecs_array, axis=0).tolist(),
                    "std": np.std(rvecs_array, axis=0).tolist(),
                },
                "tvec_components": {
                    "mean": np.mean(tvecs_array, axis=0).tolist(),
                    "std": np.std(tvecs_array, axis=0).tolist(),
                },
            }

        else:

            statistics = {
                "total_frames": len(detections),
                "successful_poses": 0,
                "success_rate": 0.0,
            }

        output_data = {
            "camera_name": camera_name,
            "timestamp": datetime.now().isoformat(),
            "statistics": statistics,
            "poses": results,
        }

        output_path = output_dir / "pose_estimation.json"

        with open(output_path, "w") as f:

            json.dump(output_data, f, indent=2)

        logger.info(f"\n{camera_name} " f"姿勢推定結果:")

        logger.info(f"  総フレーム数: " f"{statistics['total_frames']}")

        logger.info(f"  成功フレーム数: " f"{statistics['successful_poses']}")

        logger.info(f"  成功率: " f"{statistics['success_rate'] * 100:.1f}%")

        if success_count > 0:

            logger.info(
                f"  回転角度: "
                f"{statistics['rotation_angle_deg']['mean']:.2f} "
                f"± "
                f"{statistics['rotation_angle_deg']['std']:.2f} "
                f"deg"
            )

            logger.info(
                f"  距離: "
                f"{statistics['distance_m']['mean']:.3f} "
                f"± "
                f"{statistics['distance_m']['std']:.3f} "
                f"m"
            )

        logger.info(f"  結果保存先: " f"{output_path}")

        logger.info(f"  オーバレイ画像: " f"{overlay_dir}")

        return output_data

    def find_synchronized_frames(self) -> List[str]:
        """
        全カメラで共通の画像名を持つ
        成功フレームを検出
        """

        if len(self.pose_results) < 2:

            return []

        image_name_sets = []

        for camera_name, data in self.pose_results.items():

            successful_images = {
                img_name
                for (img_name, pose) in data["poses"].items()
                if pose.get("success", False)
            }

            image_name_sets.append(successful_images)

        common_images = set.intersection(*image_name_sets)

        return sorted(list(common_images))

    def generate_sync_report(self):
        """同期フレームのレポートを生成"""

        logger.info(f"\n{'=' * 60}")

        logger.info("マルチカメラ同期解析")

        logger.info(f"{'=' * 60}")

        common_frames = self.find_synchronized_frames()

        report = {
            "timestamp": datetime.now().isoformat(),
            "cameras": list(self.pose_results.keys()),
            "total_synchronized_frames": len(common_frames),
            "synchronized_frames": common_frames,
            "per_camera_stats": {},
        }

        for camera_name, data in self.pose_results.items():

            report["per_camera_stats"][camera_name] = data["statistics"]

        output_dir = Path(self.config["paths"]["output_dir"]) / "pose_estimation"

        report_path = output_dir / "synchronization_report.json"

        with open(report_path, "w") as f:

            json.dump(report, f, indent=2)

        logger.info(f"同期フレーム数: " f"{len(common_frames)}")

        logger.info(f"レポート保存先: " f"{report_path}")

        for camera_name in report["cameras"]:

            stats = report["per_camera_stats"][camera_name]

            logger.info(f"\n{camera_name}:")

            logger.info(f"  成功率: " f"{stats['success_rate'] * 100:.1f}%")

            if stats["successful_poses"] > 0:

                logger.info(f"  平均距離: " f"{stats['distance_m']['mean']:.3f} m")

        return report

    def run(self):
        """全カメラの姿勢推定を実行"""

        logger.info("=" * 60)

        logger.info("マルチカメラ姿勢推定を開始")

        logger.info("=" * 60)

        # calibration filesからカメラを自動検出
        camera_names = self.find_camera_names()

        if not camera_names:

            logger.error("キャリブレーション済みカメラが" "見つかりません")

            return

        logger.info(f"Found " f"{len(camera_names)} " f"camera(s):")

        for camera_name in camera_names:

            logger.info(f"  - " f"{camera_name}")

        # 各カメラを処理
        for camera_name in camera_names:

            result = self.process_camera(camera_name)

            if result:

                self.pose_results[camera_name] = result

        # 2台以上の場合のみ同期解析
        if len(self.pose_results) >= 2:

            self.generate_sync_report()

        logger.info("\n" + "=" * 60)

        logger.info("姿勢推定完了")

        logger.info("=" * 60)


def main():
    """メイン関数"""

    estimator = PoseEstimator(calibration_config_path=("calibration_config.yaml"))

    estimator.run()


if __name__ == "__main__":
    main()
