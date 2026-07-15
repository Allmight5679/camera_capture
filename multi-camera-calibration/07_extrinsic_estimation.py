"""
07_extrinsic_estimation.py
カメラ間外部パラメータの初期推定

実装内容:
1. 各カメラの姿勢推定結果から共通フレームを抽出
2. カメラペア間の剛体変換（回転R、並進t）を推定
3. 外れ値除去（統計的手法）
4. 平均化による初期外部パラメータ推定
5. 可視化と統計情報の出力
"""

import cv2
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ExtrinsicEstimator:
    """カメラ間外部パラメータの初期推定を行うクラス"""

    def __init__(
        self,
        calibration_config_path: str = "calibration_config.yaml"
    ):
        """
        初期化

        Args:
            calibration_config_path:
                キャリブレーション設定ファイルのパス
        """

        # キャリブレーション設定ファイルの読み込み
        self.config = self._load_config(
            calibration_config_path
        )

        self.pose_data = {}
        self.sync_report = None
        self.extrinsic_results = {}

    def _load_config(
        self,
        config_path: str
    ) -> dict:
        """設定ファイルの読み込み"""

        with open(
            config_path,
            'r',
            encoding='utf-8'
        ) as f:

            return yaml.safe_load(f)

    def load_pose_estimation_results(
        self
    ) -> bool:
        """
        各カメラの姿勢推定結果を読み込み

        Returns:
            読み込み成功ならTrue
        """

        logger.info(
            "姿勢推定結果を読み込み中..."
        )

        pose_dir = (
            Path(
                self.config[
                    'paths'
                ][
                    'output_dir'
                ]
            )
            / 'pose_estimation'
        )

        # 同期レポートを先に読み込む
        sync_path = (
            pose_dir
            / 'synchronization_report.json'
        )

        if not sync_path.exists():

            logger.error(
                f"同期レポートが見つかりません: "
                f"{sync_path}"
            )

            return False

        with open(
            sync_path,
            'r'
        ) as f:

            self.sync_report = json.load(f)

        # 06で実際に処理されたカメラを取得
        camera_names = (
            self.sync_report.get(
                'cameras',
                []
            )
        )

        if not camera_names:

            logger.error(
                "同期レポートに"
                "カメラ情報がありません"
            )

            return False

        logger.info(
            f"検出されたカメラ数: "
            f"{len(camera_names)}"
        )

        # 各カメラの姿勢推定結果を読み込む
        for camera_name in camera_names:

            pose_path = (
                pose_dir
                / camera_name
                / 'pose_estimation.json'
            )

            if not pose_path.exists():

                logger.error(
                    f"姿勢推定結果が"
                    f"見つかりません: "
                    f"{pose_path}"
                )

                return False

            with open(
                pose_path,
                'r'
            ) as f:

                self.pose_data[
                    camera_name
                ] = json.load(f)

            logger.info(
                f"  {camera_name}: "
                f"{self.pose_data[camera_name]['statistics']['successful_poses']} "
                f"姿勢"
            )

        logger.info(
            f"  同期フレーム数: "
            f"{self.sync_report['total_synchronized_frames']}"
        )

        return True

    def rvec_to_rotation_matrix(
        self,
        rvec: np.ndarray
    ) -> np.ndarray:
        """
        回転ベクトルを回転行列に変換

        Args:
            rvec: 回転ベクトル (3,) or (3,1)

        Returns:
            回転行列 (3,3)
        """

        rvec = np.array(
            rvec
        ).flatten()

        R, _ = cv2.Rodrigues(
            rvec
        )

        return R

    def compute_relative_transform(
        self,
        rvec_i: np.ndarray,
        tvec_i: np.ndarray,
        rvec_j: np.ndarray,
        tvec_j: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:
        """
        2つのカメラ姿勢からカメラ間の剛体変換を計算

        カメラiとjが同じボードを観測した時、
        カメラi座標系→カメラj座標系への変換を計算

        Args:
            rvec_i:
                カメラiの回転ベクトル
                （ボード→カメラi）

            tvec_i:
                カメラiの並進ベクトル
                （ボード→カメラi）

            rvec_j:
                カメラjの回転ベクトル
                （ボード→カメラj）

            tvec_j:
                カメラjの並進ベクトル
                （ボード→カメラj）

        Returns:
            (R_i_to_j, t_i_to_j):
                カメラi→jの回転行列と
                並進ベクトル
        """

        # rvecを回転行列に変換
        R_i = (
            self.rvec_to_rotation_matrix(
                rvec_i
            )
        )

        R_j = (
            self.rvec_to_rotation_matrix(
                rvec_j
            )
        )

        tvec_i = np.array(
            tvec_i
        ).flatten()

        tvec_j = np.array(
            tvec_j
        ).flatten()

        # カメラi→jの変換
        # R_{i→j} = R_j @ R_i^T
        R_i_to_j = (
            R_j
            @ R_i.T
        )

        # t_{i→j}
        # = t_j - R_j @ R_i^T @ t_i
        t_i_to_j = (
            tvec_j
            - R_j
            @ R_i.T
            @ tvec_i
        )

        return (
            R_i_to_j,
            t_i_to_j
        )

    def estimate_pairwise_extrinsic(
        self,
        camera_i: str,
        camera_j: str
    ) -> Dict:
        """
        カメラペア間の外部パラメータを推定

        Args:
            camera_i: カメラi名
            camera_j: カメラj名

        Returns:
            推定結果の辞書
        """

        logger.info(
            f"\n{'=' * 60}"
        )

        logger.info(
            f"{camera_i} → "
            f"{camera_j} "
            f"の外部パラメータ推定"
        )

        logger.info(
            f"{'=' * 60}"
        )

        # 共通フレームを取得
        if self.sync_report is None:

            logger.error(
                "同期レポートが"
                "読み込まれていません"
            )

            return {}

        sync_frames = (
            self.sync_report[
                'synchronized_frames'
            ]
        )

        logger.info(
            f"共通フレーム数: "
            f"{len(sync_frames)}"
        )

        poses_i = (
            self.pose_data[
                camera_i
            ][
                'poses'
            ]
        )

        poses_j = (
            self.pose_data[
                camera_j
            ][
                'poses'
            ]
        )

        # 各フレームで変換を計算
        transforms = []
        frame_names = []

        for frame_name in sync_frames:

            pose_i = poses_i.get(
                frame_name
            )

            pose_j = poses_j.get(
                frame_name
            )

            if (
                not pose_i
                or not pose_j
            ):

                continue

            if (
                not pose_i.get(
                    'success'
                )
                or not pose_j.get(
                    'success'
                )
            ):

                continue

            rvec_i = np.array(
                pose_i[
                    'rvec'
                ]
            )

            tvec_i = np.array(
                pose_i[
                    'tvec'
                ]
            )

            rvec_j = np.array(
                pose_j[
                    'rvec'
                ]
            )

            tvec_j = np.array(
                pose_j[
                    'tvec'
                ]
            )

            (
                R_i_to_j,
                t_i_to_j
            ) = (
                self.compute_relative_transform(
                    rvec_i,
                    tvec_i,
                    rvec_j,
                    tvec_j
                )
            )

            transforms.append({
                'R':
                    R_i_to_j,

                't':
                    t_i_to_j,

                'rotation_angle':
                    self._rotation_angle(
                        R_i_to_j
                    ),

                'translation_norm':
                    np.linalg.norm(
                        t_i_to_j
                    )
                    * 1000
            })

            frame_names.append(
                frame_name
            )

        logger.info(
            f"有効な変換数: "
            f"{len(transforms)}"
        )

        if len(
            transforms
        ) == 0:

            logger.error(
                "有効な変換が"
                "見つかりませんでした"
            )

            return {}

        # 外れ値除去
        (
            inlier_transforms,
            inlier_frames
        ) = (
            self._remove_outliers(
                transforms,
                frame_names
            )
        )

        logger.info(
            f"外れ値除去後: "
            f"{len(inlier_transforms)} "
            f"/ "
            f"{len(transforms)} "
            f"(インライア率: "
            f"{len(inlier_transforms) / len(transforms) * 100:.1f}%)"
        )

        if len(
            inlier_transforms
        ) == 0:

            logger.error(
                "外れ値除去後に"
                "有効な変換がありません"
            )

            return {}

        # 平均化
        (
            R_mean,
            t_mean
        ) = (
            self._average_transforms(
                inlier_transforms
            )
        )

        # 統計情報
        statistics = (
            self._compute_statistics(
                transforms,
                inlier_transforms
            )
        )

        # 生データも保存
        raw_data = {
            'all_rotation_angles': [
                t[
                    'rotation_angle'
                ]

                for t in transforms
            ],

            'all_translation_norms': [
                t[
                    'translation_norm'
                ]

                for t in transforms
            ],

            'inlier_rotation_angles': [
                t[
                    'rotation_angle'
                ]

                for t in inlier_transforms
            ],

            'inlier_translation_norms': [
                t[
                    'translation_norm'
                ]

                for t in inlier_transforms
            ]
        }

        result = {
            'camera_i':
                camera_i,

            'camera_j':
                camera_j,

            'total_frames':
                len(
                    transforms
                ),

            'inlier_frames':
                len(
                    inlier_transforms
                ),

            'inlier_ratio':
                len(
                    inlier_transforms
                )
                / len(
                    transforms
                ),

            'R':
                R_mean.tolist(),

            't':
                t_mean.tolist(),

            'rotation_angle_deg':
                float(
                    self._rotation_angle(
                        R_mean
                    )
                ),

            'translation_norm_mm':
                float(
                    np.linalg.norm(
                        t_mean
                    )
                    * 1000
                ),

            'statistics':
                statistics,

            'inlier_frame_names':
                inlier_frames,

            'raw_data':
                raw_data
        }

        logger.info(
            "\n推定結果:"
        )

        logger.info(
            f"  回転角度: "
            f"{result['rotation_angle_deg']:.2f} deg"
        )

        logger.info(
            f"  並進距離: "
            f"{result['translation_norm_mm']:.2f} mm"
        )

        logger.info(
            f"  並進ベクトル: "
            f"["
            f"{t_mean[0] * 1000:.2f}, "
            f"{t_mean[1] * 1000:.2f}, "
            f"{t_mean[2] * 1000:.2f}"
            f"] mm"
        )

        return result

    def _rotation_angle(
        self,
        R: np.ndarray
    ) -> float:
        """
        回転行列から回転角度（度）を計算
        """

        trace = np.trace(
            R
        )

        angle_rad = np.arccos(
            np.clip(
                (
                    trace
                    - 1
                )
                / 2,
                -1,
                1
            )
        )

        return np.rad2deg(
            angle_rad
        )

    def _remove_outliers(
        self,
        transforms: List[Dict],
        frame_names: List[str],
        sigma: float = 3.0
    ) -> Tuple[
        List[Dict],
        List[str]
    ]:
        """
        統計的外れ値除去
        （平均±σ範囲外を除去）
        """

        rotation_angles = np.array([
            t[
                'rotation_angle'
            ]

            for t in transforms
        ])

        translation_norms = np.array([
            t[
                'translation_norm'
            ]

            for t in transforms
        ])

        # 回転角度の統計
        rot_mean = np.mean(
            rotation_angles
        )

        rot_std = np.std(
            rotation_angles
        )

        rot_min = (
            rot_mean
            - sigma
            * rot_std
        )

        rot_max = (
            rot_mean
            + sigma
            * rot_std
        )

        # 並進ノルムの統計
        trans_mean = np.mean(
            translation_norms
        )

        trans_std = np.std(
            translation_norms
        )

        trans_min = (
            trans_mean
            - sigma
            * trans_std
        )

        trans_max = (
            trans_mean
            + sigma
            * trans_std
        )

        logger.info(
            "\n外れ値除去閾値:"
        )

        logger.info(
            f"  回転角度: "
            f"{rot_mean:.2f} "
            f"± {sigma}σ "
            f"({rot_std:.2f}) "
            f"= "
            f"[{rot_min:.2f}, "
            f"{rot_max:.2f}] deg"
        )

        logger.info(
            f"  並進距離: "
            f"{trans_mean:.2f} "
            f"± {sigma}σ "
            f"({trans_std:.2f}) "
            f"= "
            f"[{trans_min:.2f}, "
            f"{trans_max:.2f}] mm"
        )

        # インライアを抽出
        inlier_transforms = []
        inlier_frames = []

        for (
            t,
            frame
        ) in zip(
            transforms,
            frame_names
        ):

            rot_ok = (
                rot_min
                <= t[
                    'rotation_angle'
                ]
                <= rot_max
            )

            trans_ok = (
                trans_min
                <= t[
                    'translation_norm'
                ]
                <= trans_max
            )

            if (
                rot_ok
                and trans_ok
            ):

                inlier_transforms.append(
                    t
                )

                inlier_frames.append(
                    frame
                )

            else:

                logger.debug(
                    f"  外れ値除去: "
                    f"{frame} "
                    f"(rot="
                    f"{t['rotation_angle']:.2f}, "
                    f"trans="
                    f"{t['translation_norm']:.4f})"
                )

        return (
            inlier_transforms,
            inlier_frames
        )

    def _average_transforms(
        self,
        transforms: List[Dict]
    ) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:
        """
        複数の剛体変換を平均化

        回転行列はSVDによる最近傍SO(3)投影で平均化
        並進ベクトルは算術平均
        """

        # 並進ベクトルの平均
        t_list = [
            t[
                't'
            ]

            for t in transforms
        ]

        t_mean = np.mean(
            t_list,
            axis=0
        )

        # 回転行列の平均
        R_list = [
            t[
                'R'
            ]

            for t in transforms
        ]

        R_sum = np.sum(
            R_list,
            axis=0
        )

        # SVD分解して最近傍の回転行列を求める
        (
            U,
            _,
            Vt
        ) = np.linalg.svd(
            R_sum
        )

        R_mean = (
            U
            @ Vt
        )

        # det(R) = 1 を保証
        if np.linalg.det(
            R_mean
        ) < 0:

            U[
                :,
                -1
            ] *= -1

            R_mean = (
                U
                @ Vt
            )

        return (
            R_mean,
            t_mean
        )

    def _compute_statistics(
        self,
        all_transforms: List[Dict],
        inlier_transforms: List[Dict]
    ) -> Dict:
        """
        変換の統計情報を計算
        """

        def stats_dict(
            values
        ):

            return {
                'mean':
                    float(
                        np.mean(
                            values
                        )
                    ),

                'std':
                    float(
                        np.std(
                            values
                        )
                    ),

                'min':
                    float(
                        np.min(
                            values
                        )
                    ),

                'max':
                    float(
                        np.max(
                            values
                        )
                    ),

                'median':
                    float(
                        np.median(
                            values
                        )
                    )
            }

        # 全データ
        all_rot = [
            t[
                'rotation_angle'
            ]

            for t in all_transforms
        ]

        all_trans = [
            t[
                'translation_norm'
            ]

            for t in all_transforms
        ]

        # インライア
        inlier_rot = [
            t[
                'rotation_angle'
            ]

            for t in inlier_transforms
        ]

        inlier_trans = [
            t[
                'translation_norm'
            ]

            for t in inlier_transforms
        ]

        return {
            'all': {
                'rotation_angle_deg':
                    stats_dict(
                        all_rot
                    ),

                'translation_norm_mm':
                    stats_dict(
                        all_trans
                    )
            },

            'inliers': {
                'rotation_angle_deg':
                    stats_dict(
                        inlier_rot
                    ),

                'translation_norm_mm':
                    stats_dict(
                        inlier_trans
                    )
            }
        }

    def save_results(
        self,
        camera_i: str,
        camera_j: str,
        result: Dict
    ):
        """
        推定結果を保存
        """

        output_dir = (
            Path(
                self.config[
                    'paths'
                ][
                    'output_dir'
                ]
            )
            / 'extrinsic'
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / (
                f'initial_extrinsic_'
                f'{camera_i}_to_{camera_j}.json'
            )
        )

        output_data = {
            'timestamp':
                datetime.now().isoformat(),

            'method':
                'pairwise_board_pose_averaging',

            'description':
                (
                    f'Initial extrinsic calibration '
                    f'from {camera_i} '
                    f'to {camera_j}'
                ),

            **result
        }

        with open(
            output_path,
            'w'
        ) as f:

            json.dump(
                output_data,
                f,
                indent=2
            )

        logger.info(
            f"結果保存: "
            f"{output_path}"
        )

    def visualize_statistics(
        self,
        camera_i: str,
        camera_j: str,
        result: Dict
    ):
        """
        統計情報のヒストグラムを生成
        """

        output_dir = (
            Path(
                self.config[
                    'paths'
                ][
                    'output_dir'
                ]
            )
            / 'extrinsic'
        )

        # データ準備
        all_stats = (
            result[
                'statistics'
            ][
                'all'
            ]
        )

        inlier_stats = (
            result[
                'statistics'
            ][
                'inliers'
            ]
        )

        raw_data = (
            result[
                'raw_data'
            ]
        )

        # 図の作成
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                12,
                10
            )
        )

        fig.suptitle(
            (
                f'Extrinsic Estimation Statistics: '
                f'{camera_i} → {camera_j}'
            ),
            fontsize=14,
            fontweight='bold'
        )

        # 回転角度ヒストグラム（全データ）
        ax = axes[
            0,
            0
        ]

        rot_all = (
            all_stats[
                'rotation_angle_deg'
            ]
        )

        ax.hist(
            raw_data[
                'all_rotation_angles'
            ],
            bins=20,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )

        ax.axvline(
            rot_all[
                'mean'
            ],
            color='r',
            linestyle='--',
            linewidth=2,
            label='Mean'
        )

        ax.axvline(
            rot_all[
                'mean'
            ]
            - rot_all[
                'std'
            ],
            color='orange',
            linestyle=':',
            label='±1σ'
        )

        ax.axvline(
            rot_all[
                'mean'
            ]
            + rot_all[
                'std'
            ],
            color='orange',
            linestyle=':'
        )

        ax.set_xlabel(
            'Rotation Angle (deg)'
        )

        ax.set_ylabel(
            'Frequency'
        )

        ax.set_title(
            'Rotation Angle Distribution (All)'
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        ax.text(
            0.02,
            0.98,
            (
                f"Mean: "
                f"{rot_all['mean']:.2f}°\n"
                f"Std: "
                f"{rot_all['std']:.2f}°\n"
                f"Range: "
                f"[{rot_all['min']:.2f}, "
                f"{rot_all['max']:.2f}]°"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.5
            )
        )

        # 回転角度ヒストグラム（インライア）
        ax = axes[
            0,
            1
        ]

        rot_inlier = (
            inlier_stats[
                'rotation_angle_deg'
            ]
        )

        ax.hist(
            raw_data[
                'inlier_rotation_angles'
            ],
            bins=20,
            alpha=0.7,
            color='lightgreen',
            edgecolor='black'
        )

        ax.axvline(
            rot_inlier[
                'mean'
            ],
            color='r',
            linestyle='--',
            linewidth=2,
            label='Mean'
        )

        ax.axvline(
            rot_inlier[
                'mean'
            ]
            - rot_inlier[
                'std'
            ],
            color='orange',
            linestyle=':',
            label='±1σ'
        )

        ax.axvline(
            rot_inlier[
                'mean'
            ]
            + rot_inlier[
                'std'
            ],
            color='orange',
            linestyle=':'
        )

        ax.set_xlabel(
            'Rotation Angle (deg)'
        )

        ax.set_ylabel(
            'Frequency'
        )

        ax.set_title(
            'Rotation Angle Distribution (Inliers)'
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        ax.text(
            0.02,
            0.98,
            (
                f"Mean: "
                f"{rot_inlier['mean']:.2f}°\n"
                f"Std: "
                f"{rot_inlier['std']:.2f}°\n"
                f"Range: "
                f"[{rot_inlier['min']:.2f}, "
                f"{rot_inlier['max']:.2f}]°"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='lightgreen',
                alpha=0.5
            )
        )

        # 並進距離ヒストグラム（全データ）
        ax = axes[
            1,
            0
        ]

        trans_all = (
            all_stats[
                'translation_norm_mm'
            ]
        )

        ax.hist(
            raw_data[
                'all_translation_norms'
            ],
            bins=20,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )

        ax.axvline(
            trans_all[
                'mean'
            ],
            color='r',
            linestyle='--',
            linewidth=2,
            label='Mean'
        )

        ax.axvline(
            trans_all[
                'mean'
            ]
            - trans_all[
                'std'
            ],
            color='orange',
            linestyle=':',
            label='±1σ'
        )

        ax.axvline(
            trans_all[
                'mean'
            ]
            + trans_all[
                'std'
            ],
            color='orange',
            linestyle=':'
        )

        ax.set_xlabel(
            'Translation Distance (mm)'
        )

        ax.set_ylabel(
            'Frequency'
        )

        ax.set_title(
            'Translation Distance Distribution (All)'
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        ax.text(
            0.02,
            0.98,
            (
                f"Mean: "
                f"{trans_all['mean']:.2f} mm\n"
                f"Std: "
                f"{trans_all['std']:.2f} mm\n"
                f"Range: "
                f"[{trans_all['min']:.2f}, "
                f"{trans_all['max']:.2f}] mm"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.5
            )
        )

        # 並進距離ヒストグラム（インライア）
        ax = axes[
            1,
            1
        ]

        trans_inlier = (
            inlier_stats[
                'translation_norm_mm'
            ]
        )

        ax.hist(
            raw_data[
                'inlier_translation_norms'
            ],
            bins=20,
            alpha=0.7,
            color='lightgreen',
            edgecolor='black'
        )

        ax.axvline(
            trans_inlier[
                'mean'
            ],
            color='r',
            linestyle='--',
            linewidth=2,
            label='Mean'
        )

        ax.axvline(
            trans_inlier[
                'mean'
            ]
            - trans_inlier[
                'std'
            ],
            color='orange',
            linestyle=':',
            label='±1σ'
        )

        ax.axvline(
            trans_inlier[
                'mean'
            ]
            + trans_inlier[
                'std'
            ],
            color='orange',
            linestyle=':'
        )

        ax.set_xlabel(
            'Translation Distance (mm)'
        )

        ax.set_ylabel(
            'Frequency'
        )

        ax.set_title(
            'Translation Distance Distribution (Inliers)'
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        ax.text(
            0.02,
            0.98,
            (
                f"Mean: "
                f"{trans_inlier['mean']:.2f} mm\n"
                f"Std: "
                f"{trans_inlier['std']:.2f} mm\n"
                f"Range: "
                f"[{trans_inlier['min']:.2f}, "
                f"{trans_inlier['max']:.2f}] mm"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='lightgreen',
                alpha=0.5
            )
        )

        plt.tight_layout()

        # 保存
        output_path = (
            output_dir
            / (
                f'statistics_'
                f'{camera_i}_to_{camera_j}.png'
            )
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches='tight'
        )

        plt.close()

        logger.info(
            f"統計図保存: "
            f"{output_path}"
        )

    def generate_summary_report(
        self
    ):
        """全体のサマリーレポートを生成"""

        output_dir = (
            Path(
                self.config[
                    'paths'
                ][
                    'output_dir'
                ]
            )
            / 'extrinsic'
        )

        report = {
            'timestamp':
                datetime.now().isoformat(),

            'method':
                'pairwise_board_pose_averaging',

            'camera_pairs':
                []
        }

        for (
            pair_key,
            result
        ) in self.extrinsic_results.items():

            report[
                'camera_pairs'
            ].append({
                'camera_i':
                    result[
                        'camera_i'
                    ],

                'camera_j':
                    result[
                        'camera_j'
                    ],

                'rotation_angle_deg':
                    result[
                        'rotation_angle_deg'
                    ],

                'translation_norm_mm':
                    result[
                        'translation_norm_mm'
                    ],

                'inlier_ratio':
                    result[
                        'inlier_ratio'
                    ],

                'inlier_frames':
                    result[
                        'inlier_frames'
                    ]
            })

        output_path = (
            output_dir
            / 'extrinsic_summary.json'
        )

        with open(
            output_path,
            'w'
        ) as f:

            json.dump(
                report,
                f,
                indent=2
            )

        logger.info(
            f"\nサマリーレポート保存: "
            f"{output_path}"
        )

    def run(
        self
    ):
        """外部パラメータ推定を実行"""

        logger.info(
            "=" * 60
        )

        logger.info(
            "カメラ間外部パラメータの初期推定"
        )

        logger.info(
            "=" * 60
        )

        # 姿勢推定結果の読み込み
        if not self.load_pose_estimation_results():

            logger.error(
                "姿勢推定結果の"
                "読み込みに失敗しました"
            )

            return

        # 06の結果からカメラ一覧を取得
        cameras = list(
            self.pose_data.keys()
        )

        logger.info(
            f"使用するカメラ数: "
            f"{len(cameras)}"
        )

        for camera_name in cameras:

            logger.info(
                f"  - "
                f"{camera_name}"
            )

        if len(
            cameras
        ) < 2:

            logger.error(
                "外部パラメータ推定には"
                "少なくとも2台のカメラが必要です"
            )

            return

        # 全ペアについて推定
        # i < j のみ
        for i in range(
            len(cameras)
        ):

            for j in range(
                i + 1,
                len(cameras)
            ):

                camera_i = (
                    cameras[
                        i
                    ]
                )

                camera_j = (
                    cameras[
                        j
                    ]
                )

                result = (
                    self.estimate_pairwise_extrinsic(
                        camera_i,
                        camera_j
                    )
                )

                if result:

                    pair_key = (
                        f"{camera_i}"
                        f"_to_"
                        f"{camera_j}"
                    )

                    self.extrinsic_results[
                        pair_key
                    ] = result

                    # 結果保存
                    self.save_results(
                        camera_i,
                        camera_j,
                        result
                    )

                    # 可視化
                    self.visualize_statistics(
                        camera_i,
                        camera_j,
                        result
                    )

        # サマリーレポート
        if self.extrinsic_results:

            self.generate_summary_report()

        logger.info(
            "\n"
            + "=" * 60
        )

        logger.info(
            "外部パラメータ推定完了"
        )

        logger.info(
            "=" * 60
        )


def main():
    """メイン関数"""

    estimator = ExtrinsicEstimator(
        calibration_config_path=(
            "calibration_config.yaml"
        )
    )

    estimator.run()


if __name__ == '__main__':
    main()