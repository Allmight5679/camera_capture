"""
12_hand_3d_reconstruction.py
Multi-camera hand 3D reconstruction using epipolar geometry and triangulation.

Features:
- Loads world coordinate system calibration parameters
- Detects hands in multiple synchronized camera views (MediaPipe)
- Matches hand landmarks across views using epipolar geometry and Hungarian algorithm
- Triangulates 3D hand skeleton positions in world coordinates
- Supports multiple hands simultaneously
- Validates reconstructions using ray distance, reprojection error, and cheirality
- Real-time 3D visualization with PyVista

Method:
1. Epipolar Constraint: x_w^T F x_v = 0, where F = K_w^{-T}[t_hat]R K_v^{-1}
2. Sampson Distance: d_samp(x_v, x_w) = (x_w^T F x_v)^2 / ((Fx_v)_1^2 + (Fx_v)_2^2 + (F^T x_w)_1^2 + (F^T x_w)_2^2)
3. Confidence Penalty: c_conf = -log(max(ε, s_v * s_w))
4. Cost Matrix: C = α * d_samp + γ * c_conf
5. Hungarian Matching: Optimal 1-to-1 assignment per joint
6. Ray Triangulation: Find closest point between rays in world coordinates
7. Validation: Check ray distance, reprojection error, and cheirality

Usage:
    python 12_hand_3d_reconstruction.py
    python 12_hand_3d_reconstruction.py --config calibration_config.yaml
    python 12_hand_3d_reconstruction.py --hand3d-config hand_3d_reconstruction.yaml
    python 12_hand_3d_reconstruction.py --visualize  # Show 3D visualization
    python 12_hand_3d_reconstruction.py --visualize --move-camera  # Auto-rotating camera

Keys:
    q/ESC: Quit
    s: Save current frame
    v: Toggle 3D visualization window
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import cv2
import numpy as np
import yaml

try:
    import mediapipe as mp
    import pyvista as pv
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    print("Required packages not installed. Please run:")
    print("  pip install mediapipe pyvista scipy")
    sys.exit(1)

from tis_wrapper import TISGrabberWrapper
from config_manager import ConfigManager

# Logging configuration (will be set by command line argument)
logger = logging.getLogger(__name__)


# MediaPipe Hand landmark indices (21 landmarks)
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Hand connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


@dataclass
class PairHypothesis:
    cam_i: int
    hand_p: int
    cam_j: int
    hand_q: int
    joint_k: int
    point: np.ndarray
    reproj_mean: float
    cross_dist: float
    parallax: float
    confidence: float
    reproj_errors: Tuple[float, float]
    cheirality_ok: bool


@dataclass
class HandGroup:
    id: int
    members: List[Tuple[int, int]]  # (camera_index, hand_index)
    average_score: float
    parallax_mean: float
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class JointFusionResult:
    joint_k: int
    point: np.ndarray
    used_views: List[Dict[str, object]]
    reproj_mean: float
    reproj_median: float
    inlier_count: int
    cheirality: bool


def compute_parallax_angle(dir_a: np.ndarray, dir_b: np.ndarray) -> float:
    """Compute parallax (angle between rays) in degrees."""
    denom = np.linalg.norm(dir_a) * np.linalg.norm(dir_b)
    if denom < 1e-12:
        return 0.0
    cos_theta = np.clip(np.dot(dir_a, dir_b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def triangulate_two_view(
    cam_a: 'CameraCalibration',
    cam_b: 'CameraCalibration',
    x_a: np.ndarray,
    x_b: np.ndarray
) -> Optional[Dict[str, object]]:
    """Triangulate a 3D point from a pair of calibrated views."""

    C_a = cam_a.C_world.flatten()
    C_b = cam_b.C_world.flatten()

    d_a = cam_a.get_ray_direction(x_a)
    d_b = cam_b.get_ray_direction(x_b)

    A = np.array([
        [np.dot(d_a, d_a), -np.dot(d_a, d_b)],
        [np.dot(d_a, d_b), -np.dot(d_b, d_b)]
    ])
    b = np.array([
        np.dot(d_a, C_b - C_a),
        np.dot(d_b, C_b - C_a)
    ])

    try:
        params = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    lambda_a, mu_b = params
    P_a = C_a + lambda_a * d_a
    P_b = C_b + mu_b * d_b

    X_world = (P_a + P_b) / 2.0
    ray_distance = float(np.linalg.norm(P_a - P_b))

    x_a_proj = cam_a.project_3d_to_2d(X_world)
    x_b_proj = cam_b.project_3d_to_2d(X_world)
    error_a = float(np.linalg.norm(x_a - x_a_proj))
    error_b = float(np.linalg.norm(x_b - x_b_proj))
    reproj_mean = 0.5 * (error_a + error_b)

    parallax = compute_parallax_angle(d_a, d_b)

    X_cam_a = cam_a.R_world_to_cam @ X_world.reshape(3, 1) + cam_a.t_world_to_cam
    X_cam_b = cam_b.R_world_to_cam @ X_world.reshape(3, 1) + cam_b.t_world_to_cam
    cheirality_ok = bool(X_cam_a[2, 0] > 0 and X_cam_b[2, 0] > 0)

    return {
        'point': X_world,
        'ray_distance': ray_distance,
        'reproj_errors': (error_a, error_b),
        'reproj_mean': reproj_mean,
        'parallax': parallax,
        'cheirality_ok': cheirality_ok,
        'directions': (d_a, d_b)
    }


def compute_huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Compute Huber robust weights."""
    weights = np.ones_like(residuals)
    mask = np.abs(residuals) > delta
    weights[mask] = delta / np.maximum(np.abs(residuals[mask]), 1e-8)
    return weights


def compute_tukey_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Compute Tukey's biweight robust weights."""
    weights = np.zeros_like(residuals)
    mask = np.abs(residuals) <= delta
    r = residuals[mask] / delta
    weights[mask] = (1 - r ** 2) ** 2
    return weights


def compute_robust_weights(residuals: np.ndarray, delta: float, method: str) -> np.ndarray:
    """Dispatch to selected robust weighting scheme."""
    if method == 'tukey':
        return compute_tukey_weights(residuals, delta)
    return compute_huber_weights(residuals, delta)


class CameraCalibration:
    """Camera calibration data structure."""
    
    def __init__(self, camera_name: str, calib_data: Dict, extrinsic_data: Dict):
        """
        Initialize camera calibration.
        
        Args:
            camera_name: Camera identifier
            calib_data: Intrinsic calibration data (K, dist, img_size)
            extrinsic_data: Extrinsic data (rotation_matrix, translation_vector)
        """
        self.name = camera_name
        
        # Handle both flat and nested calibration data formats
        if 'K' in calib_data:
            self.K = np.array(calib_data['K'], dtype=np.float64)
            self.dist = np.array(calib_data['dist'], dtype=np.float64)
        else:
            self.K = np.array(calib_data['camera_matrix']['K'], dtype=np.float64)
            if isinstance(calib_data['distortion_coefficients'], dict):
                self.dist = np.array(calib_data['distortion_coefficients']['coefficients'], dtype=np.float64)
            else:
                self.dist = np.array(calib_data['distortion_coefficients'], dtype=np.float64)
        
        self.img_size = tuple(calib_data['img_size'])
        
        # Handle both key naming conventions
        if 'R' in extrinsic_data:
            self.R_world_to_cam = np.array(extrinsic_data['R'], dtype=np.float64)
            self.t_world_to_cam = np.array(extrinsic_data['t'], dtype=np.float64).reshape(3, 1)
        else:
            self.R_world_to_cam = np.array(extrinsic_data['rotation_matrix'], dtype=np.float64)
            self.t_world_to_cam = np.array(extrinsic_data['translation_vector'], dtype=np.float64).reshape(3, 1)
        
        # Compute camera center in world coordinates: C = -R^T @ t
        self.C_world = -self.R_world_to_cam.T @ self.t_world_to_cam
        
        # Projection matrix: P = K @ [R | t]
        Rt = np.hstack([self.R_world_to_cam, self.t_world_to_cam])
        self.P = self.K @ Rt
        
    def project_3d_to_2d(self, X_world: np.ndarray) -> np.ndarray:
        """
        Project 3D point in world coordinates to 2D image coordinates.
        
        Args:
            X_world: 3D point in world coordinates (3,) or (N, 3)
            
        Returns:
            2D image coordinates (2,) or (N, 2)
        """
        X_world = np.atleast_2d(X_world)
        X_hom = np.hstack([X_world, np.ones((X_world.shape[0], 1))])
        x_hom = (self.P @ X_hom.T).T
        x = x_hom[:, :2] / x_hom[:, 2:3]
        return x.squeeze()
    
    def get_ray_direction(self, x_img: np.ndarray) -> np.ndarray:
        """
        Get ray direction in world coordinates for image point.
        
        Args:
            x_img: Image point [u, v]
            
        Returns:
            Ray direction in world coordinates (normalized)
        """
        # Normalized image coordinates
        x_hom = np.array([x_img[0], x_img[1], 1.0])
        x_normalized = np.linalg.inv(self.K) @ x_hom
        
        # Ray direction in world frame: d = R^T @ x_normalized
        d_world = self.R_world_to_cam.T @ x_normalized
        d_world = d_world / np.linalg.norm(d_world)
        
        return d_world


class HandDetection:
    """Single hand detection result."""
    
    def __init__(self, landmarks: np.ndarray, scores: np.ndarray, handedness: str):
        """
        Initialize hand detection.
        
        Args:
            landmarks: (21, 2) array of 2D landmark coordinates
            scores: (21,) array of landmark confidence scores
            handedness: "Left" or "Right"
        """
        self.landmarks = landmarks  # (21, 2)
        self.scores = scores  # (21,)
        self.handedness = handedness
        self.scale = self._compute_scale()
        self.shape_vector = self._compute_shape_vector()

    def _compute_scale(self) -> float:
        """Compute median bone length as hand scale."""
        lengths = []
        for i, j in HAND_CONNECTIONS:
            diff = self.landmarks[i] - self.landmarks[j]
            length = float(np.linalg.norm(diff))
            if length > 0:
                lengths.append(length)
        if not lengths:
            return 1.0
        return float(np.median(lengths))

    def _compute_shape_vector(self) -> np.ndarray:
        """Compute normalized bone length vector for shape comparison."""
        scale = max(self.scale, 1e-6)
        normalized_lengths = []
        for i, j in HAND_CONNECTIONS:
            diff = self.landmarks[i] - self.landmarks[j]
            length = float(np.linalg.norm(diff))
            normalized_lengths.append(length / scale)
        return np.array(normalized_lengths, dtype=np.float32)


class MultiCameraHandDetector:
    """Multi-camera hand detection system."""
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.3, 
                 min_tracking_confidence: float = 0.3):
        """Initialize MediaPipe hand detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info(f"Initialized MediaPipe hand detector (max_hands={max_num_hands}, "
                   f"min_detection_conf={min_detection_confidence}, "
                   f"min_tracking_conf={min_tracking_confidence})")
        
    def detect(self, image: np.ndarray) -> List[HandDetection]:
        """
        Detect hands in image.
        
        Args:
            image: BGR image
            
        Returns:
            List of HandDetection objects
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        detections = []
        
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmarks
                landmarks = np.zeros((21, 2), dtype=np.float32)
                scores = np.zeros(21, dtype=np.float32)
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[i] = [landmark.x * w, landmark.y * h]
                    # MediaPipe Hands landmarks have: x, y, z (no visibility/presence in v0.10+)
                    # Use a constant confidence score since MediaPipe doesn't provide per-landmark confidence
                    scores[i] = 1.0  # All landmarks have equal confidence
                
                # Debug: log first detection
                if hand_idx == 0:
                    logger.debug(f"    Detection scores: min={np.min(scores):.3f}, max={np.max(scores):.3f}, "
                               f"sum={np.sum(scores):.3f}")
                
                # Get handedness
                handedness = "Unknown"
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                
                detection = HandDetection(landmarks, scores, handedness)
                detections.append(detection)
                
                # Debug: verify scores are valid
                if np.sum(detection.scores) == 0:
                    logger.warning(f"Hand detection {hand_idx} has zero scores!")
        
        return detections
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


class EpipolarMatcher:
    """Epipolar geometry-based hand correspondence matcher."""
    
    def __init__(
        self,
        cam_v: CameraCalibration,
        cam_w: CameraCalibration,
        alpha: float = 1.0,
        gamma: float = 0.1,
        epsilon: float = 1e-6,
        epipolar_threshold: float = 200.0  # Increased from 100.0 to handle calibration errors better
    ):
        """
        Initialize epipolar matcher.
        
        Args:
            cam_v: Source camera calibration
            cam_w: Target camera calibration
            alpha: Weight for Sampson distance
            gamma: Weight for confidence penalty
            epsilon: Small constant for log stability
            epipolar_threshold: Maximum epipolar distance (pixels)
        """
        self.cam_v = cam_v
        self.cam_w = cam_w
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epipolar_threshold = epipolar_threshold
        logger.debug(f"EpipolarMatcher initialized: {cam_v.name} <-> {cam_w.name}, "
                    f"epipolar_threshold={epipolar_threshold}px")
        
        # Compute fundamental matrix F (v -> w)
        # F = K_w^{-T} [t_hat] R K_v^{-1}
        # where R, t are relative pose from cam_v to cam_w
        self.F = self._compute_fundamental_matrix()
        
    def _compute_fundamental_matrix(self) -> np.ndarray:
        """
        Compute fundamental matrix from v to w.
        
        F = K_w^{-T} [t_hat] R K_v^{-1}
        where R, t transform from cam_v to cam_w
        """
        # Relative transformation from cam_v to cam_w
        # X_w = R_vw @ X_v + t_vw
        # R_vw = R_w @ R_v^T
        # t_vw = R_w @ (C_v - C_w)
        
        R_vw = self.cam_w.R_world_to_cam @ self.cam_v.R_world_to_cam.T
        t_vw = self.cam_w.R_world_to_cam @ (self.cam_v.C_world - self.cam_w.C_world)
        
        # Skew-symmetric matrix of t
        t_hat = np.array([
            [0, -t_vw[2, 0], t_vw[1, 0]],
            [t_vw[2, 0], 0, -t_vw[0, 0]],
            [-t_vw[1, 0], t_vw[0, 0], 0]
        ])
        
        # Essential matrix E = [t]_× R
        E = t_hat @ R_vw
        
        # Fundamental matrix F = K_w^{-T} E K_v^{-1}
        K_v_inv = np.linalg.inv(self.cam_v.K)
        K_w_inv_T = np.linalg.inv(self.cam_w.K).T
        F = K_w_inv_T @ E @ K_v_inv
        
        return F
    
    def compute_sampson_distance(self, x_v: np.ndarray, x_w: np.ndarray) -> float:
        """
        Compute Sampson distance for epipolar constraint.
        
        d_samp = (x_w^T F x_v)^2 / ((Fx_v)_1^2 + (Fx_v)_2^2 + (F^T x_w)_1^2 + (F^T x_w)_2^2)
        
        Args:
            x_v: Point in camera v [u, v]
            x_w: Point in camera w [u, v]
            
        Returns:
            Sampson distance
        """
        x_v_hom = np.array([x_v[0], x_v[1], 1.0])
        x_w_hom = np.array([x_w[0], x_w[1], 1.0])
        
        # Epipolar constraint residual
        numerator = (x_w_hom.T @ self.F @ x_v_hom) ** 2
        
        # Denominator terms
        Fx_v = self.F @ x_v_hom
        FT_x_w = self.F.T @ x_w_hom
        
        denominator = Fx_v[0]**2 + Fx_v[1]**2 + FT_x_w[0]**2 + FT_x_w[1]**2
        
        if denominator < 1e-10:
            return float('inf')
        
        return numerator / denominator
    
    def compute_epipolar_distance(self, x_v: np.ndarray, x_w: np.ndarray) -> float:
        """
        Compute point-to-epipolar-line distance.
        
        Args:
            x_v: Point in camera v [u, v]
            x_w: Point in camera w [u, v]
            
        Returns:
            Distance in pixels
        """
        x_v_hom = np.array([x_v[0], x_v[1], 1.0])
        x_w_hom = np.array([x_w[0], x_w[1], 1.0])
        
        # Epipolar line in w: l = F @ x_v
        l = self.F @ x_v_hom
        
        # Distance from x_w to line l
        distance = abs(l.T @ x_w_hom) / np.sqrt(l[0]**2 + l[1]**2)
        
        return distance

    def evaluate_hand_pair(
        self,
        det_v: 'HandDetection',
        det_w: 'HandDetection',
        tau_epi: float
    ) -> Dict[str, np.ndarray]:
        """Evaluate epipolar compatibility for a pair of detected hands."""

        sampson_distances = np.zeros(len(HAND_LANDMARKS), dtype=np.float32)
        inlier_mask = np.zeros(len(HAND_LANDMARKS), dtype=bool)
        weights = np.zeros(len(HAND_LANDMARKS), dtype=np.float32)

        for k in range(len(HAND_LANDMARKS)):
            x_v = det_v.landmarks[k]
            x_w = det_w.landmarks[k]
            d_samp = self.compute_sampson_distance(x_v, x_w)
            sampson_distances[k] = d_samp
            weights[k] = 0.5 * (det_v.scores[k] + det_w.scores[k])
            if d_samp < tau_epi:
                inlier_mask[k] = True

        # Debug: Check if scores are valid
        score_sum_v = float(np.sum(det_v.scores))
        score_sum_w = float(np.sum(det_w.scores))
        weight_sum = float(np.sum(weights))
        inlier_count = int(np.sum(inlier_mask))
        
        logger.debug(f"    Evaluate: scores_v_sum={score_sum_v:.3f}, scores_w_sum={score_sum_w:.3f}, "
                   f"weight_sum={weight_sum:.3f}, inliers={inlier_count}/21")

        return {
            'sampson': sampson_distances,
            'inliers': inlier_mask,
            'weights': weights
        }
    
    def match_hands(
        self,
        detections_v: List[HandDetection],
        detections_w: List[HandDetection]
    ) -> List[Tuple[int, int, float]]:
        """
        Match hands between two cameras using Hungarian algorithm.
        
        Args:
            detections_v: Hand detections in camera v
            detections_w: Hand detections in camera w
            
        Returns:
            List of (idx_v, idx_w, cost) tuples for matched hands
        """
        if len(detections_v) == 0 or len(detections_w) == 0:
            return []
        
        # Build cost matrix for all hand pairs
        n_v = len(detections_v)
        n_w = len(detections_w)
        
        cost_matrix = np.full((n_v, n_w), np.inf)
        
        for i, det_v in enumerate(detections_v):
            for j, det_w in enumerate(detections_w):
                # Compute average cost across all landmarks
                landmark_costs = []
                epipolar_violations = 0
                
                for k in range(21):  # 21 landmarks per hand
                    x_v = det_v.landmarks[k]
                    x_w = det_w.landmarks[k]
                    s_v = det_v.scores[k]
                    s_w = det_w.scores[k]
                    
                    # Check epipolar constraint
                    epi_dist = self.compute_epipolar_distance(x_v, x_w)
                    if epi_dist > self.epipolar_threshold:
                        landmark_costs.append(np.inf)
                        epipolar_violations += 1
                        continue
                    
                    # Sampson distance
                    d_samp = self.compute_sampson_distance(x_v, x_w)
                    
                    # Confidence penalty
                    c_conf = -np.log(max(self.epsilon, s_v * s_w))
                    
                    # Total cost
                    cost = self.alpha * d_samp + self.gamma * c_conf
                    landmark_costs.append(cost)
                
                # Average cost (ignore inf)
                valid_costs = [c for c in landmark_costs if not np.isinf(c)]
                if len(valid_costs) > 0:
                    cost_matrix[i, j] = np.mean(valid_costs)
                    logger.debug(f"Hand pair ({i},{j}): {len(valid_costs)}/21 valid landmarks, "
                               f"{epipolar_violations} epipolar violations, cost={cost_matrix[i, j]:.2f}")
                else:
                    logger.debug(f"Hand pair ({i},{j}): All landmarks invalid "
                               f"({epipolar_violations}/21 epipolar violations)")
        
        # Check if cost matrix has any valid assignments
        if np.all(np.isinf(cost_matrix)):
            logger.warning(f"No valid hand matches found (all costs are infinite)")
            logger.warning(f"  - Detections in camera v: {n_v}")
            logger.warning(f"  - Detections in camera w: {n_w}")
            logger.warning(f"  - Epipolar threshold: {self.epipolar_threshold} pixels")
            logger.warning(f"  - Consider increasing epipolar_threshold or checking camera calibration")
            return []
        
        # Hungarian algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.warning(f"Hungarian algorithm failed: {e}")
            return []
        
        # Filter out invalid matches
        matches = []
        for i, j in zip(row_ind, col_ind):
            if not np.isinf(cost_matrix[i, j]):
                matches.append((i, j, cost_matrix[i, j]))
        
        return matches


class RayTriangulator:
    """Ray-based triangulation in world coordinates."""
    
    def __init__(
        self,
        ray_distance_threshold: float = 50.0,  # millimeters - increased from 20.0
        reprojection_threshold: float = 100.0,  # pixels - increased from 50.0
    ):
        """
        Initialize triangulator.
        
        Args:
            ray_distance_threshold: Maximum distance between rays (millimeters)
            reprojection_threshold: Maximum reprojection error (pixels)
        """
        self.ray_distance_threshold = ray_distance_threshold
        self.reprojection_threshold = reprojection_threshold
        logger.debug(f"RayTriangulator initialized: ray_dist_thresh={ray_distance_threshold}mm, "
                    f"reproj_thresh={reprojection_threshold}px")
    
    def triangulate_point(
        self,
        cam_v: CameraCalibration,
        cam_w: CameraCalibration,
        x_v: np.ndarray,
        x_w: np.ndarray
    ) -> Optional[Dict]:
        """
        Triangulate 3D point from two views using ray intersection.
        
        Args:
            cam_v: Camera v calibration
            cam_w: Camera w calibration
            x_v: 2D point in camera v
            x_w: 2D point in camera w
            
        Returns:
            Dictionary with 3D point and validation metrics, or None if failed
        """
        # Camera centers in world coordinates
        C_v = cam_v.C_world.flatten()
        C_w = cam_w.C_world.flatten()
        
        # Ray directions in world coordinates
        d_v = cam_v.get_ray_direction(x_v)
        d_w = cam_w.get_ray_direction(x_w)
        
        # Find closest point between two rays
        # Ray v: P_v(λ) = C_v + λ * d_v
        # Ray w: P_w(μ) = C_w + μ * d_w
        # Minimize: |P_v(λ) - P_w(μ)|^2
        
        # Solve: [d_v, -d_w]^T @ [d_v, -d_w] @ [λ, μ]^T = [d_v, -d_w]^T @ (C_w - C_v)
        A = np.array([
            [d_v @ d_v, -d_v @ d_w],
            [d_v @ d_w, -d_w @ d_w]
        ])
        b = np.array([
            d_v @ (C_w - C_v),
            d_w @ (C_w - C_v)
        ])
        
        try:
            params = np.linalg.solve(A, b)
            lambda_v = params[0]
            mu_w = params[1]
        except np.linalg.LinAlgError:
            return None
        
        # 3D points on each ray
        P_v = C_v + lambda_v * d_v
        P_w = C_w + mu_w * d_w
        
        # Midpoint as final 3D point
        X_world = (P_v + P_w) / 2.0
        
        # Ray distance
        ray_distance = np.linalg.norm(P_v - P_w)
        
        # Check ray distance threshold
        if ray_distance > self.ray_distance_threshold:
            return None
        
        # Cheirality check (point must be in front of both cameras)
        X_cam_v = cam_v.R_world_to_cam @ X_world.reshape(3, 1) + cam_v.t_world_to_cam
        X_cam_w = cam_w.R_world_to_cam @ X_world.reshape(3, 1) + cam_w.t_world_to_cam
        
        if X_cam_v[2, 0] <= 0 or X_cam_w[2, 0] <= 0:
            return None
        
        # Reprojection error
        x_v_proj = cam_v.project_3d_to_2d(X_world)
        x_w_proj = cam_w.project_3d_to_2d(X_world)
        
        error_v = np.linalg.norm(x_v - x_v_proj)
        error_w = np.linalg.norm(x_w - x_w_proj)
        error_mean = (error_v + error_w) / 2.0
        
        # Check reprojection threshold
        if error_mean > self.reprojection_threshold:
            return None
        
        return {
            'point_3d_world': X_world,
            'ray_distance': ray_distance,
            'reprojection_error_v': error_v,
            'reprojection_error_w': error_w,
            'reprojection_error_mean': error_mean,
            'views': [cam_v.name, cam_w.name]
        }


class Hand3DReconstruction:
    """3D reconstructed hand skeleton."""
    
    def __init__(self, hand_id: int):
        """
        Initialize hand reconstruction.
        
        Args:
            hand_id: Unique hand identifier
        """
        self.hand_id = hand_id
        self.landmarks_3d = {}  # {landmark_idx: {'point_3d_world', 'confidence', 'views'}}
        self.handedness = "Unknown"
        
    def add_landmark(self, landmark_idx: int, triangulation_result: Dict, confidence: float):
        """Add reconstructed landmark."""
        # Allow both legacy pairwise results and new multi-view fusion outputs
        point = triangulation_result.get('point_3d_world')
        if point is None:
            point = triangulation_result.get('point')
        
        reproj_mean = triangulation_result.get('reprojection_error_mean')
        if reproj_mean is None:
            reproj_mean = triangulation_result.get('reproj_mean')
        
        reproj_med = triangulation_result.get('reprojection_error_median')
        if reproj_med is None:
            reproj_med = triangulation_result.get('reproj_median')
        
        views = triangulation_result.get('views')
        if views is None:
            views = triangulation_result.get('used_views', [])
        
        ray_distance = triangulation_result.get('ray_distance')
        residual_mean = triangulation_result.get('residual_mean')
        residual_median = triangulation_result.get('residual_median')

        self.landmarks_3d[landmark_idx] = {
            'point_3d_world': point,
            'ray_distance': ray_distance,
            'reprojection_error': reproj_mean,
            'reprojection_error_median': reproj_med,
            'residual_mean': residual_mean,
            'residual_median': residual_median,
            'confidence': confidence,
            'views': views,
            'metadata': triangulation_result
        }
    
    def get_landmarks_array(self) -> np.ndarray:
        """Get landmarks as (N, 3) array."""
        indices = sorted(self.landmarks_3d.keys())
        return np.array([self.landmarks_3d[i]['point_3d_world'] for i in indices])
    
    def get_centroid(self) -> Optional[np.ndarray]:
        """Get centroid (mean position) of all landmarks."""
        if len(self.landmarks_3d) == 0:
            return None
        landmarks = self.get_landmarks_array()
        return np.mean(landmarks, axis=0)
    
    def is_complete(self, min_landmarks: int = 15) -> bool:
        """Check if hand has enough reconstructed landmarks."""
        return len(self.landmarks_3d) >= min_landmarks


class MultiCameraHand3DSystem:
    """Complete multi-camera hand 3D reconstruction system."""
    
    def __init__(self, 
                 config_path: str = "calibration_config.yaml",
                 calibration_config_path: str = "calibration_config.yaml",
                 hand3d_config_path: str = "hand_3d_reconstruction.yaml",
                 move_camera: bool = False,
                 debug: bool = False):
        """
        Initialize system.
        
        Args:
            config_path: Path to camera config file
            calibration_config_path: Path to calibration config file
            hand3d_config_path: Path to hand 3D reconstruction parameters
            move_camera: Enable automatic camera rotation around detected hands
            debug: Enable detailed debug output
        """
        # Set logging level based on debug flag
        self.debug = debug
        # Load ConfigManager for camera configuration
        self.config_manager = ConfigManager(Path(config_path))
        if not self.config_manager.load():
            raise RuntimeError(f"設定ファイルの読み込みに失敗: {config_path}")
        
        self.config = self._load_config(calibration_config_path)
        
        # Load hand 3D reconstruction parameters
        self.hand3d_params = self._load_hand3d_config(hand3d_config_path)
        
        # Load display scale from calibration_config.yaml
        self.display_scale = self.config_manager.global_config.get('display_scale', 1.0)
        
        # Cameras
        self.cameras = []
        self.camera_calibrations = {}
        
        # Hand detector (using parameters from config)
        hand_det_cfg = self.hand3d_params['hand_detection']
        self.hand_detector = MultiCameraHandDetector(
            max_num_hands=hand_det_cfg['max_num_hands'],
            min_detection_confidence=hand_det_cfg['min_detection_confidence'],
            min_tracking_confidence=hand_det_cfg['min_tracking_confidence']
        )
        
        # Output directory
        self.output_dir = Path(self.config['paths']['output_dir']) / 'hand_3d_reconstruction'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.frame_count = 0
        self.total_reconstructions = 0
        
        # Camera movement settings (using parameters from config)
        vis3d_cfg = self.hand3d_params['visualization_3d']
        self.move_camera = move_camera
        self.camera_angle = 0.0  # Current rotation angle (degrees)
        self.camera_rotation_speed = vis3d_cfg['camera_rotation_speed']
        self.camera_distance = vis3d_cfg['camera_distance']
        self.camera_height_offset = vis3d_cfg['camera_height_offset']
        self.show_ghost_hands = vis3d_cfg.get('show_ghost_hands', True)
        self.ghost_opacity = vis3d_cfg.get('ghost_opacity', 0.3)
        
        # PyVista mesh management for efficient updates
        self.hand_point_actors = {}  # {hand_id: actor}
        self.hand_line_actors = {}   # {hand_id: actor}
        self.ghost_point_actors = {}  # {hand_id: actor} for ghost hands
        self.ghost_line_actors = {}   # {hand_id: actor} for ghost hands
        self.previous_hand_count = 0
        self.static_geometries_initialized = False
        
        # Hand tracking state for consistent ID assignment (using parameters from config)
        tracking_cfg = self.hand3d_params['hand_tracking']
        self.tracked_hands_state = {}  # {hand_id: {'centroid': np.ndarray, 'last_seen_frame': int, 'status': 'active'|'lost', 'hand_3d': Hand3DReconstruction}}
        self.next_available_hand_id = 0
        self.max_centroid_distance = tracking_cfg['max_centroid_distance']
        self.max_lost_frames = tracking_cfg.get('max_lost_frames', 30)
        self.reappear_distance_multiplier = tracking_cfg.get('reappear_distance_multiplier', 1.5)
        
        # Algorithm parameters (loaded from config)
        self.params = {
            'pair_hypothesis': self.hand3d_params['pair_hypothesis'],
            'epipolar': self.hand3d_params['epipolar'],
            'robust': self.hand3d_params['robust'],
            'weights': self.hand3d_params['weights']
        }

        logger.info("="*80)
        logger.info("Multi-Camera Hand 3D Reconstruction System")
        logger.info("="*80)
        logger.info("Configuration Parameters:")
        logger.info(f"  Hand Detection: max_hands={hand_det_cfg['max_num_hands']}, "
                   f"det_conf={hand_det_cfg['min_detection_confidence']}, "
                   f"track_conf={hand_det_cfg['min_tracking_confidence']}")
        logger.info(f"  Hand Tracking: max_distance={self.max_centroid_distance}m, "
                   f"max_lost_frames={self.max_lost_frames}, "
                   f"reappear_multiplier={self.reappear_distance_multiplier}x")
        logger.info(f"  Epipolar Matcher: threshold={self.hand3d_params['epipolar_matcher']['epipolar_threshold']}px")
        logger.info(f"  Ray Triangulator: ray_dist={self.hand3d_params['ray_triangulator']['ray_distance_threshold']}mm, "
                   f"reproj={self.hand3d_params['ray_triangulator']['reprojection_threshold']}px")
        logger.info(f"  Reconstruction Quality: min_partial={self.hand3d_params['reconstruction']['min_landmarks_partial']}, "
                   f"min_complete={self.hand3d_params['reconstruction']['min_landmarks_complete']}")
        logger.info(f"  Ghost Visualization: {'ENABLED' if self.show_ghost_hands else 'DISABLED'} "
                   f"(opacity={self.ghost_opacity})")
        if move_camera:
            logger.info("3D Camera Auto-Rotation: ENABLED")
            logger.info(f"  - Rotation Speed: {self.camera_rotation_speed}°/frame")
            logger.info(f"  - Distance: {self.camera_distance}m")
            logger.info(f"  - Height Offset: {self.camera_height_offset}m")
        
    def _load_config(self, config_path: str) -> dict:
        """Load calibration configuration."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_hand3d_config(self, config_path: str) -> dict:
        """Load hand 3D reconstruction parameters."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Hand 3D reconstruction config file not found: {config_path}\n"
                f"Please create '{config_path}' with the required parameters.\n"
                f"You can use 'hand_3d_reconstruction.yaml' as a template."
            )
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = [
                'hand_detection', 'epipolar_matcher', 'ray_triangulator',
                'pair_hypothesis', 'epipolar', 'robust', 'weights',
                'hand_tracking', 'visualization_3d', 'reconstruction'
            ]
            
            missing_sections = [section for section in required_sections if section not in params]
            if missing_sections:
                raise ValueError(
                    f"Missing required sections in config file: {', '.join(missing_sections)}"
                )
            
            logger.info(f"✓ Loaded hand 3D reconstruction config from: {config_path}")
            return params
            
        except Exception as e:
            logger.error(f"Error loading hand 3D config: {e}")
            raise
    
    def load_calibrations(self):
        """Load camera calibrations from world coordinate system."""
        logger.info("Loading camera calibrations...")
        
        calib_dir = Path(self.config['paths']['output_dir']) / 'calibration'
        world_dir = Path(self.config['paths']['output_dir']) / 'bundle_adjustment' / 'world_coordinate'
        
        # ConfigManagerから動的にカメラ名を生成
        num_cameras = self.config_manager.get_camera_count()
        camera_names = [f"camera{i}" for i in range(num_cameras)]
        
        for cam_name in camera_names:
            # Load intrinsics
            calib_file = calib_dir / f'calibration_{cam_name}.json'
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            
            # Load extrinsics (world to camera)
            extrinsic_file = world_dir / f'extrinsic_{cam_name}.json'
            with open(extrinsic_file, 'r') as f:
                extrinsic_data = json.load(f)
            
            # Create calibration object
            cam_calib = CameraCalibration(cam_name, calib_data, extrinsic_data)
            self.camera_calibrations[cam_name] = cam_calib
            
            logger.info(f"  ✓ Loaded {cam_name}")
        
        logger.info(f"✓ Loaded {len(self.camera_calibrations)} camera calibrations")
        logger.info("")
    
    def open_cameras(self):
        """Open camera devices from config."""
        logger.info("Opening camera devices...")
        
        num_cameras = self.config_manager.get_camera_count()
        
        for i in range(num_cameras):
            cam_cfg = self.config_manager.get_camera_config(i)
            xml_path = Path(cam_cfg['xml_path'])
            
            cam = TISGrabberWrapper()
            if not cam.load_device_config(str(xml_path)):
                raise RuntimeError(f"Failed to load camera from {xml_path}")
            
            if not cam.start_live():
                raise RuntimeError(f"Failed to start camera {i}")
            
            self.cameras.append(cam)
            logger.info(f"  ✓ Camera {i} opened")
        
        logger.info(f"✓ {len(self.cameras)} cameras ready")
        logger.info("")
    
    def capture_frames(self) -> List[np.ndarray]:
        """Capture synchronized frames from all cameras."""
        frames = []
        for cam in self.cameras:
            img = cam.capture_image(1000)
            if img is None:
                raise RuntimeError("Failed to capture frame")
            frames.append(img)
        return frames

    @staticmethod
    def _handedness_compatible(label_a: str, label_b: str) -> bool:
        """Check whether two handedness labels are compatible."""
        if label_a == "Unknown" or label_b == "Unknown":
            return True
        return label_a.lower() == label_b.lower()

    @staticmethod
    def _make_pair_key(node_a: Tuple[int, int], node_b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Create an ordered key for a pair of nodes."""
        return tuple(sorted([node_a, node_b]))  # type: ignore

    def _generate_pairwise_hypotheses(
        self,
        camera_names: List[str],
        detections_per_camera: List[List[HandDetection]]
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[int, PairHypothesis]]:
        """Generate pairwise joint hypotheses across all camera pairs."""

        params = self.params['pair_hypothesis']
        pair_hypotheses: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[int, PairHypothesis]] = defaultdict(dict)

        for (cam_i_idx, cam_i_name), (cam_j_idx, cam_j_name) in combinations(enumerate(camera_names), 2):
            cam_i_calib = self.camera_calibrations[cam_i_name]
            cam_j_calib = self.camera_calibrations[cam_j_name]

            detections_i = detections_per_camera[cam_i_idx]
            detections_j = detections_per_camera[cam_j_idx]

            for hand_i_idx, det_i in enumerate(detections_i):
                for hand_j_idx, det_j in enumerate(detections_j):
                    if not self._handedness_compatible(det_i.handedness, det_j.handedness):
                        continue

                    node_i = (cam_i_idx, hand_i_idx)
                    node_j = (cam_j_idx, hand_j_idx)
                    pair_key = self._make_pair_key(node_i, node_j)

                    for joint_k in range(len(HAND_LANDMARKS)):
                        tri_result = triangulate_two_view(
                            cam_i_calib, cam_j_calib,
                            det_i.landmarks[joint_k], det_j.landmarks[joint_k]
                        )
                        if tri_result is None:
                            continue

                        if not tri_result['cheirality_ok']:
                            continue

                        if tri_result['reproj_mean'] > params['reproj_threshold']:
                            continue

                        if tri_result['ray_distance'] > params['cross_dist_threshold']:
                            continue

                        if tri_result['parallax'] < params['min_parallax_deg']:
                            continue

                        confidence = 0.5 * (
                            float(det_i.scores[joint_k]) + float(det_j.scores[joint_k])
                        )

                        tri_point = np.asarray(cast(np.ndarray, tri_result['point']), dtype=np.float64)
                        reproj_mean = float(cast(float, tri_result['reproj_mean']))
                        cross_dist = float(cast(float, tri_result['ray_distance']))
                        parallax = float(cast(float, tri_result['parallax']))
                        reproj_errors_raw = cast(Tuple[Any, Any], tri_result['reproj_errors'])
                        reproj_errors = (
                            float(reproj_errors_raw[0]),
                            float(reproj_errors_raw[1])
                        )

                        pair_hypotheses[pair_key][joint_k] = PairHypothesis(
                            cam_i=cam_i_idx,
                            hand_p=hand_i_idx,
                            cam_j=cam_j_idx,
                            hand_q=hand_j_idx,
                            joint_k=joint_k,
                            point=tri_point,
                            reproj_mean=reproj_mean,
                            cross_dist=cross_dist,
                            parallax=parallax,
                            confidence=confidence,
                            reproj_errors=reproj_errors,
                            cheirality_ok=True
                        )

        return pair_hypotheses

    def _compute_pair_scores(
        self,
        camera_names: List[str],
        detections_per_camera: List[List[HandDetection]],
        pair_hypotheses: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[int, PairHypothesis]]
    ) -> Tuple[Dict[Tuple[Tuple[int, int], Tuple[int, int]], float], Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]]]:
        """Compute skeleton similarity scores for every hand pair."""

        params_epi = self.params['epipolar']
        params_pair = self.params['pair_hypothesis']

        pair_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        pair_details: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]] = {}

        # Use debug flag from system settings
        verbose_logging = self.debug
        
        # Counters for rejection reasons
        rejection_reasons = {
            'handedness_mismatch': 0,
            'zero_weight': 0,
            'negative_score': 0,
            'valid': 0
        }

        for (cam_i_idx, cam_i_name), (cam_j_idx, cam_j_name) in combinations(enumerate(camera_names), 2):
            cam_i_calib = self.camera_calibrations[cam_i_name]
            cam_j_calib = self.camera_calibrations[cam_j_name]
            
            # Get epipolar matcher config
            epi_matcher_cfg = self.hand3d_params['epipolar_matcher']
            matcher = EpipolarMatcher(
                cam_i_calib, cam_j_calib,
                alpha=epi_matcher_cfg['alpha'],
                gamma=epi_matcher_cfg['gamma'],
                epsilon=epi_matcher_cfg['epsilon'],
                epipolar_threshold=epi_matcher_cfg['epipolar_threshold']
            )

            for hand_i_idx, det_i in enumerate(detections_per_camera[cam_i_idx]):
                for hand_j_idx, det_j in enumerate(detections_per_camera[cam_j_idx]):
                    node_i = (cam_i_idx, hand_i_idx)
                    node_j = (cam_j_idx, hand_j_idx)
                    pair_key = self._make_pair_key(node_i, node_j)

                    if not self._handedness_compatible(det_i.handedness, det_j.handedness):
                        pair_scores[pair_key] = float('-inf')
                        rejection_reasons['handedness_mismatch'] += 1
                        if verbose_logging:
                            logger.debug(f"  Pair {node_i}-{node_j}: Handedness mismatch "
                                       f"({det_i.handedness} vs {det_j.handedness})")
                        continue

                    eval_result = matcher.evaluate_hand_pair(det_i, det_j, params_epi['tau_epi'])

                    weight_sum = float(np.sum(eval_result['weights']))
                    inlier_count = int(np.sum(eval_result['inliers']))
                    
                    if weight_sum < 1e-6:
                        pair_scores[pair_key] = float('-inf')
                        rejection_reasons['zero_weight'] += 1
                        if verbose_logging:
                            logger.debug(f"  Pair {node_i}-{node_j}: Zero weight sum")
                        continue

                    inlier_ratio = float(
                        np.sum(eval_result['weights'] * eval_result['inliers']) / weight_sum
                    )
                    
                    if verbose_logging:
                        logger.debug(f"  Pair {node_i}-{node_j}: Inliers={inlier_count}/21, "
                                   f"InlierRatio={inlier_ratio:.3f}")

                    scale_penalty = abs(
                        np.log(max(det_i.scale, 1e-6) / max(det_j.scale, 1e-6))
                    )
                    scale_penalty = max(0.0, scale_penalty - params_epi['tau_scale'])

                    shape_penalty = float(
                        np.mean(np.abs(det_i.shape_vector - det_j.shape_vector))
                    )

                    pair_data = pair_hypotheses.get(pair_key, {})
                    parallax_values: List[float] = [float(hyp.parallax) for hyp in pair_data.values()]
                    parallax_mean = (
                        float(sum(parallax_values) / len(parallax_values))
                        if parallax_values else 0.0
                    )
                    parallax_penalty = 0.0
                    if parallax_mean < params_pair['min_parallax_deg']:
                        parallax_penalty = (
                            params_pair['min_parallax_deg'] - parallax_mean
                        ) / max(params_pair['min_parallax_deg'], 1e-3)

                    score = (
                        params_epi['w1'] * inlier_ratio
                        - params_epi['w2'] * scale_penalty
                        - params_epi['w3'] * shape_penalty
                        - params_epi['w_parallax'] * parallax_penalty
                    )

                    pair_scores[pair_key] = score
                    pair_details[pair_key] = {
                        'inlier_ratio': inlier_ratio,
                        'scale_penalty': scale_penalty,
                        'shape_penalty': shape_penalty,
                        'parallax_mean': parallax_mean,
                        'parallax_penalty': parallax_penalty
                    }
                    
                    if score > float('-inf'):
                        rejection_reasons['valid'] += 1
                    else:
                        rejection_reasons['negative_score'] += 1
                    
                    if verbose_logging:
                        logger.debug(f"  Pair {node_i}-{node_j}: Score={score:.3f} "
                                   f"(inlier={inlier_ratio:.3f}, scale_pen={scale_penalty:.3f}, "
                                   f"shape_pen={shape_penalty:.3f}, parallax_pen={parallax_penalty:.3f}, "
                                   f"parallax={parallax_mean:.1f}°)")

        # Log rejection summary
        logger.info(f"Pair score rejections: handedness={rejection_reasons['handedness_mismatch']}, "
                   f"zero_weight={rejection_reasons['zero_weight']}, "
                   f"negative_score={rejection_reasons['negative_score']}, "
                   f"valid={rejection_reasons['valid']}")

        return pair_scores, pair_details

    def _cluster_hand_groups(
        self,
        pair_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
        pair_details: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]]
    ) -> List[HandGroup]:
        """Cluster hand detections across cameras using greedy max-density grouping."""

        theta_pair = self.params['epipolar']['theta_pair']
        theta_consensus = self.params['epipolar']['theta_consensus']

        edges = [
            (max(score, 0.0), pair_key)
            for pair_key, score in pair_scores.items()
            if max(score, 0.0) >= theta_pair
        ]
        edges.sort(key=lambda x: x[0], reverse=True)

        all_nodes: Set[Tuple[int, int]] = set()
        for _, pair_key in edges:
            all_nodes.update(pair_key)

        clusters: List[HandGroup] = []
        processed_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for weight, pair_key in edges:
            if pair_key in processed_pairs:
                continue

            cluster_nodes: Set[Tuple[int, int]] = set(pair_key)
            cameras_in_cluster: Set[int] = {node[0] for node in cluster_nodes}
            cluster_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = {pair_key}
            cluster_score_sum = pair_scores[pair_key]

            expanded = True
            while expanded:
                expanded = False
                candidates = [node for node in all_nodes if node not in cluster_nodes and node[0] not in cameras_in_cluster]
                for candidate in candidates:
                    candidate_pairs = []
                    candidate_scores = []
                    valid = True
                    for existing in cluster_nodes:
                        key = self._make_pair_key(candidate, existing)
                        score = pair_scores.get(key)
                        if score is None or score < theta_consensus:
                            valid = False
                            break
                        candidate_pairs.append(key)
                        candidate_scores.append(score)

                    if not valid:
                        continue

                    new_score_sum = cluster_score_sum + sum(candidate_scores)
                    new_pair_count = len(cluster_pairs) + len(candidate_pairs)
                    current_mean = cluster_score_sum / max(len(cluster_pairs), 1)
                    new_mean = new_score_sum / max(new_pair_count, 1)
                    if new_mean + 1e-6 < current_mean:
                        continue

                    cluster_nodes.add(candidate)
                    cameras_in_cluster.add(candidate[0])
                    cluster_pairs.update(candidate_pairs)
                    cluster_score_sum = new_score_sum
                    expanded = True

            cluster_avg = cluster_score_sum / max(len(cluster_pairs), 1)
            parallax_vals: List[float] = []
            for key in cluster_pairs:
                details = pair_details.get(key)
                if details:
                    parallax_vals.append(float(details.get('parallax_mean', 0.0)))
            parallax_mean = (
                float(sum(parallax_vals) / len(parallax_vals)) if parallax_vals else 0.0
            )

            cluster = HandGroup(
                id=len(clusters),
                members=sorted(cluster_nodes),
                average_score=cluster_avg,
                parallax_mean=parallax_mean,
                stats={'pair_count': len(cluster_pairs)}
            )
            clusters.append(cluster)
            processed_pairs.add(pair_key)

        if not clusters:
            return []

        node_to_groups: Dict[Tuple[int, int], List[HandGroup]] = defaultdict(list)
        for group in clusters:
            for node in group.members:
                node_to_groups[node].append(group)

        keep_flags = {group.id: True for group in clusters}
        for node, groups in node_to_groups.items():
            if len(groups) <= 1:
                continue
            groups_sorted = sorted(
                groups,
                key=lambda g: (g.average_score, len(g.members), g.parallax_mean),
                reverse=True
            )
            best_id = groups_sorted[0].id
            for g in groups_sorted[1:]:
                keep_flags[g.id] = False

        resolved_groups = [group for group in clusters if keep_flags.get(group.id, False)]

        assigned_nodes = {node for group in resolved_groups for node in group.members}
        unassigned_nodes = all_nodes - assigned_nodes

        if unassigned_nodes:
            for weight, pair_key in edges:
                if not unassigned_nodes.issuperset(pair_key):
                    continue
                cluster_nodes: Set[Tuple[int, int]] = set(pair_key)
                cameras_in_cluster: Set[int] = {node[0] for node in cluster_nodes}
                cluster_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = {pair_key}
                cluster_score_sum = pair_scores[pair_key]

                expanded = True
                while expanded:
                    expanded = False
                    candidates = [
                        node for node in unassigned_nodes
                        if node not in cluster_nodes and node[0] not in cameras_in_cluster
                    ]
                    for candidate in candidates:
                        candidate_pairs = []
                        candidate_scores = []
                        valid = True
                        for existing in cluster_nodes:
                            key = self._make_pair_key(candidate, existing)
                            score = pair_scores.get(key)
                            if score is None or score < theta_consensus:
                                valid = False
                                break
                            candidate_pairs.append(key)
                            candidate_scores.append(score)
                        if not valid:
                            continue

                        new_score_sum = cluster_score_sum + sum(candidate_scores)
                        new_pair_count = len(cluster_pairs) + len(candidate_pairs)
                        current_mean = cluster_score_sum / max(len(cluster_pairs), 1)
                        new_mean = new_score_sum / max(new_pair_count, 1)
                        if new_mean + 1e-6 < current_mean:
                            continue

                        cluster_nodes.add(candidate)
                        cameras_in_cluster.add(candidate[0])
                        cluster_pairs.update(candidate_pairs)
                        cluster_score_sum = new_score_sum
                        expanded = True

                cluster_avg = cluster_score_sum / max(len(cluster_pairs), 1)
                parallax_vals: List[float] = []
                for key in cluster_pairs:
                    details = pair_details.get(key)
                    if details:
                        parallax_vals.append(float(details.get('parallax_mean', 0.0)))
                parallax_mean = (
                    float(sum(parallax_vals) / len(parallax_vals)) if parallax_vals else 0.0
                )

                cluster = HandGroup(
                    id=len(resolved_groups) + len(clusters),
                    members=sorted(cluster_nodes),
                    average_score=cluster_avg,
                    parallax_mean=parallax_mean,
                    stats={'pair_count': len(cluster_pairs)}
                )
                resolved_groups.append(cluster)
                unassigned_nodes -= cluster_nodes
                if not unassigned_nodes:
                    break

        for new_id, group in enumerate(resolved_groups):
            group.id = new_id

        return resolved_groups

    def _extract_joint_observations(
        self,
        group: HandGroup,
        camera_names: List[str],
        detections_per_camera: List[List[HandDetection]],
        pair_hypotheses: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[int, PairHypothesis]]
    ) -> Dict[int, List[Dict[str, object]]]:
        """Extract reliable joint observations for a hand group."""

        joint_observations: Dict[int, List[Dict[str, object]]] = {}

        for joint_k in range(len(HAND_LANDMARKS)):
            support_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            parallax_map: Dict[Tuple[int, int], List[float]] = defaultdict(list)

            for node_a, node_b in combinations(group.members, 2):
                pair_key = self._make_pair_key(node_a, node_b)
                hypothesis = pair_hypotheses.get(pair_key, {}).get(joint_k)
                if not hypothesis:
                    continue
                support_counts[node_a] += 1
                support_counts[node_b] += 1
                parallax_map[node_a].append(hypothesis.parallax)
                parallax_map[node_b].append(hypothesis.parallax)

            views = [node for node, count in support_counts.items() if count > 0]
            if len(views) < 2:
                continue

            obs_list: List[Dict[str, object]] = []
            for node in views:
                cam_idx, hand_idx = node
                detection = detections_per_camera[cam_idx][hand_idx]
                obs_list.append({
                    'node': node,
                    'camera_idx': cam_idx,
                    'camera_name': camera_names[cam_idx],
                    'hand_idx': hand_idx,
                    'point2d': detection.landmarks[joint_k],
                    'score': float(detection.scores[joint_k]),
                    'handedness': detection.handedness,
                    'parallax_values': [float(v) for v in parallax_map.get(node, [])]
                })

            joint_observations[joint_k] = obs_list

        return joint_observations

    def _fuse_joint_multi_view(
        self,
        joint_k: int,
        observations: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Robustly fuse multi-view joint observations via IRLS."""

        if len(observations) < 2:
            return None

        base_weights: List[float] = []
        centers: List[np.ndarray] = []
        directions: List[np.ndarray] = []
        camera_refs: List[Tuple[int, str, int]] = []
        scores: List[float] = []

        for obs in observations:
            cam_idx = int(cast(int, obs['camera_idx']))
            cam_name = str(obs['camera_name'])
            hand_idx = int(cast(int, obs['hand_idx']))
            cam_calib = self.camera_calibrations[cam_name]
            point2d = obs['point2d']

            centers.append(cam_calib.C_world.flatten())
            directions.append(cam_calib.get_ray_direction(point2d))
            camera_refs.append((cam_idx, cam_name, hand_idx))
            score_val = float(cast(float, obs['score']))
            scores.append(score_val)

            parallax_values = cast(List[float], obs.get('parallax_values', []))
            parallax_mean = float(sum(parallax_values) / len(parallax_values)) if parallax_values else 0.0

            weight = (
                self.params['weights']['alpha_score'] * score_val
                + self.params['weights']['beta_baseline'] * (parallax_mean / max(5.0, 1e-3))
            )
            base_weights.append(max(weight, 1e-3))

        centers = [np.asarray(c, dtype=np.float64) for c in centers]
        directions = [np.asarray(d, dtype=np.float64) for d in directions]
        base_weights_array = np.asarray(base_weights, dtype=np.float64)

        method = self.params['robust']['method']
        delta = self.params['robust']['delta']
        max_iters = self.params['robust']['max_iters']

        weights = base_weights_array.copy()
        X = None

        for _ in range(max_iters):
            sqrt_weights = np.sqrt(weights)
            A_blocks = []
            b_blocks = []
            for w, C, d in zip(sqrt_weights, centers, directions):
                proj = np.eye(3) - np.outer(d, d)
                A_blocks.append(w * proj)
                b_blocks.append(w * proj @ C)
            A = np.vstack(A_blocks)
            b = np.concatenate(b_blocks)
            try:
                X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None

            residuals = []
            for C, d in zip(centers, directions):
                proj = np.eye(3) - np.outer(d, d)
                residual_vec = proj @ (X - C)
                residuals.append(np.linalg.norm(residual_vec))
            residuals = np.asarray(residuals, dtype=np.float64)

            robust = compute_robust_weights(residuals, delta, method)
            new_weights = base_weights_array * np.maximum(robust, 1e-6)

            if np.allclose(new_weights, weights, atol=1e-3):
                break
            weights = new_weights

        if X is None:
            return None

        residuals = []
        used_views = []
        reproj_errors = []
        cheirality = True

        for idx, (C, d, ref, score) in enumerate(zip(centers, directions, camera_refs, scores)):
            proj = np.eye(3) - np.outer(d, d)
            residual_vec = proj @ (X - C)
            residual_norm = float(np.linalg.norm(residual_vec))
            residuals.append(residual_norm)

            cam_idx, cam_name, hand_idx = ref
            cam_calib = self.camera_calibrations[cam_name]
            reproj = cam_calib.project_3d_to_2d(X)
            reproj_error = float(np.linalg.norm(observations[idx]['point2d'] - reproj))
            reproj_errors.append(reproj_error)

            X_cam = cam_calib.R_world_to_cam @ X.reshape(3, 1) + cam_calib.t_world_to_cam
            if X_cam[2, 0] <= 0:
                cheirality = False

            used_views.append({
                'camera_idx': cam_idx,
                'camera_name': cam_name,
                'hand_idx': hand_idx,
                'residual': residual_norm,
                'reprojection_error': reproj_error,
                'score': score,
                'weight': float(weights[idx])
            })

        residuals = np.asarray(residuals, dtype=np.float64)
        residual_threshold = self.params['robust']['residual_outlier']
        inlier_mask = residuals <= residual_threshold

        if np.sum(inlier_mask) < 2:
            return None

        used_views = [view for idx, view in enumerate(used_views) if inlier_mask[idx]]
        reproj_errors = [err for idx, err in enumerate(reproj_errors) if inlier_mask[idx]]
        residuals = residuals[inlier_mask]

        return {
            'point_3d_world': X,
            'used_views': used_views,
            'reprojection_error_mean': float(np.mean(reproj_errors)),
            'reprojection_error_median': float(np.median(reproj_errors)),
            'residual_mean': float(np.mean(residuals)),
            'residual_median': float(np.median(residuals)),
            'inlier_count': len(used_views),
            'cheirality': cheirality
        }

    def _fuse_hand_group(
        self,
        group: HandGroup,
        camera_names: List[str],
        detections_per_camera: List[List[HandDetection]],
        pair_hypotheses: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[int, PairHypothesis]]
    ) -> Optional[Hand3DReconstruction]:
        """Perform multi-view fusion for a grouped hand."""

        joint_observations = self._extract_joint_observations(
            group, camera_names, detections_per_camera, pair_hypotheses
        )

        if not joint_observations:
            return None

        hand_recon = Hand3DReconstruction(group.id)

        handedness_votes = []
        for cam_idx, hand_idx in group.members:
            detection = detections_per_camera[cam_idx][hand_idx]
            if detection.handedness != "Unknown":
                handedness_votes.append(detection.handedness)
        if handedness_votes:
            values, counts = np.unique(handedness_votes, return_counts=True)
            hand_recon.handedness = str(values[np.argmax(counts)])

        for joint_k, observations in joint_observations.items():
            fusion_result = self._fuse_joint_multi_view(joint_k, observations)
            if fusion_result is None:
                continue

            scores = [float(cast(float, view['score'])) for view in observations]
            confidence = float(sum(scores) / len(scores)) if scores else 0.0
            hand_recon.add_landmark(joint_k, fusion_result, confidence)

        if len(hand_recon.landmarks_3d) == 0:
            return None

        return hand_recon
    
    def reconstruct_hands(self, frames: List[np.ndarray]) -> List[Hand3DReconstruction]:
        """Reconstruct hands via hypothesis voting, grouping, and robust fusion."""
        # Detect hands in each frame
        detections_per_camera: List[List[HandDetection]] = []
        camera_names = list(self.camera_calibrations.keys())

        for cam_idx, (frame, cam_name) in enumerate(zip(frames, camera_names)):
            detections = self.hand_detector.detect(frame)
            detections_per_camera.append(detections)
            logger.debug(f"Camera {cam_idx} ({cam_name}): Detected {len(detections)} hands")

        total_detections = sum(len(dets) for dets in detections_per_camera)
        if total_detections == 0:
            logger.debug("No hands detected in any camera")
            return []

        if len(camera_names) < 2:
            logger.warning("Need at least 2 cameras for reconstruction")
            return []

        pair_hypotheses = self._generate_pairwise_hypotheses(camera_names, detections_per_camera)
        total_hypotheses = sum(len(hyps) for hyps in pair_hypotheses.values())
        num_hand_pairs = len(pair_hypotheses)
        logger.debug(f"Generated {total_hypotheses} pairwise joint hypotheses across {num_hand_pairs} hand pairs")

        if total_hypotheses == 0:
            logger.warning("No valid pairwise hypotheses generated - possible calibration or detection issues")
            return []

        pair_scores, pair_details = self._compute_pair_scores(
            camera_names, detections_per_camera, pair_hypotheses
        )

        valid_pairs = sum(1 for score in pair_scores.values() if score > float('-inf'))
        logger.debug(f"Computed pair scores: {valid_pairs} valid pairs out of {len(pair_scores)}")
        
        if valid_pairs == 0:
            logger.warning("No valid pair scores computed - all hands rejected by epipolar constraints")
            logger.warning("This may indicate calibration issues or incompatible hand detections")
            # Log score distribution for debugging
            scores_list = [s for s in pair_scores.values() if s > float('-inf')]
            if len(scores_list) == 0:
                logger.warning("All pair scores are -inf (handedness mismatch or zero weight)")
        else:
            scores_list = [s for s in pair_scores.values() if s > float('-inf')]
            logger.debug(f"Valid score range: [{min(scores_list):.3f}, {max(scores_list):.3f}]")

        hand_groups = self._cluster_hand_groups(pair_scores, pair_details)
        logger.debug(f"Formed {len(hand_groups)} hand groups from detections")

        reconstructed_hands: List[Hand3DReconstruction] = []
        recon_cfg = self.hand3d_params['reconstruction']
        
        for group in hand_groups:
            hand_recon = self._fuse_hand_group(
                group, camera_names, detections_per_camera, pair_hypotheses
            )
            if hand_recon is None:
                logger.debug(f"Hand group {group.id} failed fusion")
                continue
            # Accept partial reconstructions with minimum landmarks
            if len(hand_recon.landmarks_3d) < recon_cfg['min_landmarks_partial']:
                logger.debug(f"Hand group {group.id} has too few landmarks ({len(hand_recon.landmarks_3d)})")
                continue
            if not hand_recon.is_complete(min_landmarks=recon_cfg['min_landmarks_complete']):
                # Keep partial reconstructions but log for awareness
                logger.debug(
                    f"Hand group {group.id} partial reconstruction ({len(hand_recon.landmarks_3d)}/21 landmarks)"
                )
            else:
                logger.debug(f"Hand group {group.id} complete reconstruction ({len(hand_recon.landmarks_3d)}/21 landmarks)")
            reconstructed_hands.append(hand_recon)

        # Apply hand tracking to maintain consistent IDs across frames
        reconstructed_hands = self.match_hands_by_proximity(reconstructed_hands)

        return reconstructed_hands
    
    def match_hands_by_proximity(self, reconstructed_hands: List[Hand3DReconstruction]) -> List[Hand3DReconstruction]:
        """
        Match current hands with previous frame hands using greedy proximity matching with ghost tracking.
        Keeps tracking lost hands for max_lost_frames to enable re-matching when they reappear.
        
        Args:
            reconstructed_hands: List of reconstructed hands (with temporary IDs)
            
        Returns:
            List of hands with updated IDs for tracking consistency
        """
        current_frame = self.frame_count
        
        # Calculate centroids for current hands
        current_centroids = []
        for hand in reconstructed_hands:
            centroid = hand.get_centroid()
            if centroid is not None:
                current_centroids.append(centroid)
            else:
                current_centroids.append(None)
        
        # If no hands detected, mark all active hands as lost
        if len(reconstructed_hands) == 0:
            for hand_id in list(self.tracked_hands_state.keys()):
                state = self.tracked_hands_state[hand_id]
                if state['status'] == 'active':
                    state['status'] = 'lost'
                    state['last_seen_frame'] = current_frame - 1
                    logger.debug(f"Hand {hand_id} marked as lost at frame {current_frame}")
            
            # Remove old ghosts
            self._cleanup_old_ghosts(current_frame)
            return reconstructed_hands
        
        # If no previous hands, assign new IDs
        if len(self.tracked_hands_state) == 0:
            for i, hand in enumerate(reconstructed_hands):
                hand.hand_id = self.next_available_hand_id
                self.next_available_hand_id += 1
                if current_centroids[i] is not None:
                    self.tracked_hands_state[hand.hand_id] = {
                        'centroid': current_centroids[i],
                        'last_seen_frame': current_frame,
                        'status': 'active',
                        'hand_3d': hand  # Save the full 3D reconstruction
                    }
                    logger.debug(f"New hand {hand.hand_id} initialized at frame {current_frame}")
            return reconstructed_hands
        
        # Separate active and lost hands
        active_hands = {hid: state for hid, state in self.tracked_hands_state.items() if state['status'] == 'active'}
        lost_hands = {hid: state for hid, state in self.tracked_hands_state.items() if state['status'] == 'lost'}
        
        matched_current_indices = set()
        matched_prev_ids = set()
        id_mapping = {}  # {current_idx: tracked_hand_id}
        
        # Build distance matrix for matching
        matches = []  # List of (priority, distance, current_idx, hand_id, is_active)
        
        for curr_idx, curr_centroid in enumerate(current_centroids):
            if curr_centroid is None:
                continue
            
            # Match with active hands (higher priority, normal distance threshold)
            for hand_id, state in active_hands.items():
                prev_centroid = state['centroid']
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                if distance < self.max_centroid_distance:
                    matches.append((0, distance, curr_idx, hand_id, True))  # priority=0 for active
            
            # Match with lost hands (lower priority, relaxed distance threshold)
            relaxed_threshold = self.max_centroid_distance * self.reappear_distance_multiplier
            for hand_id, state in lost_hands.items():
                prev_centroid = state['centroid']
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                if distance < relaxed_threshold:
                    matches.append((1, distance, curr_idx, hand_id, False))  # priority=1 for lost
        
        # Sort by priority first, then by distance (closest first)
        matches.sort(key=lambda x: (x[0], x[1]))
        
        # Assign matches greedily
        for priority, distance, curr_idx, hand_id, is_active in matches:
            if curr_idx not in matched_current_indices and hand_id not in matched_prev_ids:
                id_mapping[curr_idx] = hand_id
                matched_current_indices.add(curr_idx)
                matched_prev_ids.add(hand_id)
                
                status_str = "active" if is_active else "lost->reappeared"
                logger.debug(f"Matched hand {hand_id} ({status_str}) with current detection {curr_idx} "
                           f"at distance {distance:.3f}m (frame {current_frame})")
        
        # Assign new IDs to unmatched hands
        for curr_idx in range(len(reconstructed_hands)):
            if curr_idx not in matched_current_indices:
                new_id = self.next_available_hand_id
                self.next_available_hand_id += 1
                id_mapping[curr_idx] = new_id
                logger.debug(f"New hand {new_id} created for unmatched detection {curr_idx} (frame {current_frame})")
        
        # Update hand IDs and tracking state
        new_tracked_state = {}
        for curr_idx, hand in enumerate(reconstructed_hands):
            hand_id = id_mapping[curr_idx]
            hand.hand_id = hand_id
            
            if current_centroids[curr_idx] is not None:
                new_tracked_state[hand_id] = {
                    'centroid': current_centroids[curr_idx],
                    'last_seen_frame': current_frame,
                    'status': 'active',
                    'hand_3d': hand  # Save the full 3D reconstruction
                }
        
        # Mark unmatched previous hands as lost
        for hand_id, state in self.tracked_hands_state.items():
            if hand_id not in matched_prev_ids:
                if state['status'] == 'active':
                    logger.debug(f"Hand {hand_id} lost at frame {current_frame}")
                new_tracked_state[hand_id] = {
                    'centroid': state['centroid'],
                    'last_seen_frame': state['last_seen_frame'],
                    'status': 'lost',
                    'hand_3d': state.get('hand_3d')  # Keep the last known 3D reconstruction
                }
        
        self.tracked_hands_state = new_tracked_state
        
        # Remove old ghosts
        self._cleanup_old_ghosts(current_frame)
        
        return reconstructed_hands
    
    def _cleanup_old_ghosts(self, current_frame: int):
        """Remove ghost hands that have been lost for too long."""
        hands_to_remove = []
        
        for hand_id, state in self.tracked_hands_state.items():
            if state['status'] == 'lost':
                frames_lost = current_frame - state['last_seen_frame']
                if frames_lost > self.max_lost_frames:
                    hands_to_remove.append(hand_id)
                    logger.debug(f"Removing ghost hand {hand_id} (lost for {frames_lost} frames)")
        
        for hand_id in hands_to_remove:
            del self.tracked_hands_state[hand_id]
    
    def visualize_2d(self, frames: List[np.ndarray], reconstructed_hands: List[Hand3DReconstruction]):
        """Visualize detections on 2D frames."""
        camera_names = list(self.camera_calibrations.keys())
        
        # First, detect and visualize raw MediaPipe detections
        for idx, (frame, cam_name) in enumerate(zip(frames, camera_names)):
            vis_frame = frame.copy()
            cam_calib = self.camera_calibrations[cam_name]
            
            # Draw raw MediaPipe detections for debugging
            raw_detections = self.hand_detector.detect(frame)
            for det_idx, detection in enumerate(raw_detections):
                for lm_idx, (landmark, score) in enumerate(zip(detection.landmarks, detection.scores)):
                    # Draw raw detection landmarks in red
                    cv2.circle(vis_frame, tuple(landmark.astype(int)), 3, (0, 0, 255), -1)
                    if lm_idx == 0:  # Wrist
                        cv2.putText(vis_frame, f"Raw {det_idx} ({detection.handedness})",
                                  tuple(landmark.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 1)
            
            # Project and draw reconstructed hands
            for hand_3d in reconstructed_hands:
                # Project landmarks
                landmarks_3d = hand_3d.get_landmarks_array()
                if len(landmarks_3d) == 0:
                    continue
                
                landmarks_2d = cam_calib.project_3d_to_2d(landmarks_3d)
                
                # Draw landmarks
                for pt in landmarks_2d:
                    cv2.circle(vis_frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
                
                # Draw connections
                landmark_dict = {i: landmarks_2d[list(hand_3d.landmarks_3d.keys()).index(i)] 
                               for i in hand_3d.landmarks_3d.keys()}
                
                for conn in HAND_CONNECTIONS:
                    if conn[0] in landmark_dict and conn[1] in landmark_dict:
                        pt1 = tuple(landmark_dict[conn[0]].astype(int))
                        pt2 = tuple(landmark_dict[conn[1]].astype(int))
                        cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 2)
                
                # Draw hand ID
                if len(landmarks_2d) > 0:
                    wrist = landmarks_2d[0].astype(int)
                    cv2.putText(vis_frame, f"Hand {hand_3d.hand_id} ({hand_3d.handedness})",
                              tuple(wrist), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Resize frame according to display_scale
            if self.display_scale != 1.0:
                h, w = vis_frame.shape[:2]
                new_w = int(w * self.display_scale)
                new_h = int(h * self.display_scale)
                vis_frame = cv2.resize(vis_frame, (new_w, new_h))
            
            # Show frame
            cv2.imshow(f"Camera {idx} - {cam_name}", vis_frame)
    
    def calculate_hands_center(self, reconstructed_hands: List[Hand3DReconstruction]) -> np.ndarray:
        """
        Calculate the center position of all detected hands.
        
        Args:
            reconstructed_hands: List of reconstructed hands
            
        Returns:
            3D center position (3,) or [0, 0, 0] if no hands
        """
        all_points = []
        for hand_3d in reconstructed_hands:
            landmarks = hand_3d.get_landmarks_array()
            if len(landmarks) > 0:
                all_points.extend(landmarks)
        
        if len(all_points) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        return np.mean(all_points, axis=0)
    
    def update_camera_view(self, plotter: pv.Plotter, target_center: np.ndarray):
        """
        Update 3D visualization camera view to rotate around target.
        
        Args:
            plotter: PyVista Plotter instance
            target_center: Target center position to look at (3,)
        """
        if not self.move_camera:
            return
        
        # Calculate camera position on circular orbit
        theta = np.radians(self.camera_angle)
        
        # Camera position in cylindrical coordinates
        cam_x = target_center[0] + self.camera_distance * np.cos(theta)
        cam_y = target_center[1] + self.camera_distance * np.sin(theta)
        cam_z = target_center[2] + self.camera_height_offset
        
        # Set camera position and focal point
        camera_position = np.array([cam_x, cam_y, cam_z])
        plotter.camera.position = camera_position
        plotter.camera.focal_point = target_center
        
        # Set up vector (Z-axis)
        plotter.camera.up = (0, 0, 1)
        
        # Increment angle for next frame
        self.camera_angle = (self.camera_angle + self.camera_rotation_speed) % 360.0
    
    def initialize_static_geometries(self, plotter: pv.Plotter):
        """Initialize static geometries (coordinate axes and camera positions) once."""
        if self.static_geometries_initialized:
            return
        
        # World coordinate frame (arrows for X, Y, Z axes)
        # X-axis (red)
        arrow_x = pv.Arrow(start=(0, 0, 0), direction=(0.2, 0, 0), scale='auto')
        plotter.add_mesh(arrow_x, color='red', label='X')
        
        # Y-axis (green)
        arrow_y = pv.Arrow(start=(0, 0, 0), direction=(0, 0.2, 0), scale='auto')
        plotter.add_mesh(arrow_y, color='green', label='Y')
        
        # Z-axis (blue)
        arrow_z = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 0.2), scale='auto')
        plotter.add_mesh(arrow_z, color='blue', label='Z')
        
        # Camera positions
        camera_colors = ['red', 'blue', 'green']
        for idx, (cam_name, cam_calib) in enumerate(self.camera_calibrations.items()):
            # Camera center as sphere
            C = cam_calib.C_world.flatten()
            sphere = pv.Sphere(radius=0.02, center=C)
            plotter.add_mesh(sphere, color=camera_colors[idx % len(camera_colors)], 
                           label=f'{cam_name}')
            
            # Camera coordinate frame (smaller arrows)
            R_cam_to_world = cam_calib.R_world_to_cam.T
            
            # Camera X-axis (transformed)
            cam_x_dir = R_cam_to_world @ np.array([0.1, 0, 0])
            cam_arrow_x = pv.Arrow(start=C, direction=cam_x_dir, scale='auto')
            plotter.add_mesh(cam_arrow_x, color='salmon', opacity=0.6)
            
            # Camera Y-axis (transformed)
            cam_y_dir = R_cam_to_world @ np.array([0, 0.1, 0])
            cam_arrow_y = pv.Arrow(start=C, direction=cam_y_dir, scale='auto')
            plotter.add_mesh(cam_arrow_y, color='lightgreen', opacity=0.6)
            
            # Camera Z-axis (transformed)
            cam_z_dir = R_cam_to_world @ np.array([0, 0, 0.1])
            cam_arrow_z = pv.Arrow(start=C, direction=cam_z_dir, scale='auto')
            plotter.add_mesh(cam_arrow_z, color='lightblue', opacity=0.6)
        
        self.static_geometries_initialized = True
    
    def update_hand_geometries(self, plotter: pv.Plotter, reconstructed_hands: List[Hand3DReconstruction]):
        """Update dynamic hand geometries efficiently by reusing actors."""
        # Hand colors
        hand_colors_rgb = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 0, 255]     # Magenta
        ]
        
        # Current hand IDs (active hands only)
        current_hand_ids = {hand.hand_id for hand in reconstructed_hands}
        existing_hand_ids = set(self.hand_point_actors.keys())
        
        # Remove actors for hands that no longer exist
        hands_to_remove = existing_hand_ids - current_hand_ids
        for hand_id in hands_to_remove:
            if hand_id in self.hand_point_actors:
                plotter.remove_actor(self.hand_point_actors[hand_id])
                del self.hand_point_actors[hand_id]
            if hand_id in self.hand_line_actors:
                plotter.remove_actor(self.hand_line_actors[hand_id])
                del self.hand_line_actors[hand_id]
        
        # Update or create actors for each hand
        for hand_3d in reconstructed_hands:
            hand_id = hand_3d.hand_id
            color = hand_colors_rgb[hand_id % len(hand_colors_rgb)]
            
            # Get all landmarks
            landmark_indices = sorted(hand_3d.landmarks_3d.keys())
            if len(landmark_indices) == 0:
                continue
                
            landmark_points = np.array([hand_3d.landmarks_3d[i]['point_3d_world'] 
                                       for i in landmark_indices])
            
            # Update or create point cloud actor
            if hand_id in self.hand_point_actors:
                # Update existing actor's points
                actor = self.hand_point_actors[hand_id]
                mesh = actor.GetMapper().GetInput()
                mesh.points = landmark_points
                mesh.Modified()
            else:
                # Create new point cloud
                point_cloud = pv.PolyData(landmark_points)
                actor = plotter.add_mesh(point_cloud, color=color, point_size=10, 
                                       render_points_as_spheres=True,
                                       label=f'Hand {hand_id} ({hand_3d.handedness})')
                self.hand_point_actors[hand_id] = actor
            
            # Create lines for connections
            lines = []
            for conn in HAND_CONNECTIONS:
                if conn[0] in hand_3d.landmarks_3d and conn[1] in hand_3d.landmarks_3d:
                    idx0 = landmark_indices.index(conn[0])
                    idx1 = landmark_indices.index(conn[1])
                    lines.append(2)  # Number of points in this line
                    lines.append(idx0)
                    lines.append(idx1)
            
            # Update or create line actor
            if len(lines) > 0:
                line_cells = np.array(lines)
                
                if hand_id in self.hand_line_actors:
                    # Update existing actor's points and lines
                    actor = self.hand_line_actors[hand_id]
                    mesh = actor.GetMapper().GetInput()
                    mesh.points = landmark_points
                    mesh.lines = line_cells
                    mesh.Modified()
                else:
                    # Create new line mesh
                    line_mesh = pv.PolyData(landmark_points)
                    line_mesh.lines = line_cells
                    actor = plotter.add_mesh(line_mesh, color=color, line_width=3)
                    self.hand_line_actors[hand_id] = actor
    
    def update_ghost_hand_geometries(self, plotter: pv.Plotter):
        """Update ghost hand geometries (lost hands) with semi-transparent display using original colors."""
        if not self.show_ghost_hands:
            # Remove all ghost actors if ghost display is disabled
            for hand_id in list(self.ghost_point_actors.keys()):
                plotter.remove_actor(self.ghost_point_actors[hand_id])
                del self.ghost_point_actors[hand_id]
            for hand_id in list(self.ghost_line_actors.keys()):
                plotter.remove_actor(self.ghost_line_actors[hand_id])
                del self.ghost_line_actors[hand_id]
            return
        
        # Hand colors (same as active hands)
        hand_colors_rgb = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 0, 255]     # Magenta
        ]
        
        # Ghost opacity (from config)
        ghost_opacity = self.ghost_opacity
        
        # Get all lost/ghost hands from tracking state
        ghost_hand_ids = {hid for hid, state in self.tracked_hands_state.items() 
                         if state['status'] == 'lost'}
        existing_ghost_ids = set(self.ghost_point_actors.keys())
        
        # Remove ghost actors for hands that are no longer lost or don't exist
        ghosts_to_remove = existing_ghost_ids - ghost_hand_ids
        for hand_id in ghosts_to_remove:
            if hand_id in self.ghost_point_actors:
                plotter.remove_actor(self.ghost_point_actors[hand_id])
                del self.ghost_point_actors[hand_id]
            if hand_id in self.ghost_line_actors:
                plotter.remove_actor(self.ghost_line_actors[hand_id])
                del self.ghost_line_actors[hand_id]
        
        # Update or create actors for each ghost hand
        for hand_id in ghost_hand_ids:
            state = self.tracked_hands_state[hand_id]
            hand_3d = state.get('hand_3d')
            
            # If no 3D data available, skip this ghost
            if hand_3d is None:
                continue
            
            # Use the original color for this hand ID
            color = hand_colors_rgb[hand_id % len(hand_colors_rgb)]
            
            # Get all landmarks
            landmark_indices = sorted(hand_3d.landmarks_3d.keys())
            if len(landmark_indices) == 0:
                continue
                
            landmark_points = np.array([hand_3d.landmarks_3d[i]['point_3d_world'] 
                                       for i in landmark_indices])
            
            # Update or create point cloud actor
            if hand_id in self.ghost_point_actors:
                # Update existing actor's points
                actor = self.ghost_point_actors[hand_id]
                mesh = actor.GetMapper().GetInput()
                mesh.points = landmark_points
                mesh.Modified()
            else:
                # Create new point cloud with transparency
                point_cloud = pv.PolyData(landmark_points)
                actor = plotter.add_mesh(point_cloud, color=color, 
                                       opacity=ghost_opacity,
                                       point_size=10, 
                                       render_points_as_spheres=True,
                                       label=f'Ghost {hand_id} ({hand_3d.handedness})')
                self.ghost_point_actors[hand_id] = actor
            
            # Create lines for connections
            lines = []
            for conn in HAND_CONNECTIONS:
                if conn[0] in hand_3d.landmarks_3d and conn[1] in hand_3d.landmarks_3d:
                    idx0 = landmark_indices.index(conn[0])
                    idx1 = landmark_indices.index(conn[1])
                    lines.append(2)  # Number of points in this line
                    lines.append(idx0)
                    lines.append(idx1)
            
            # Update or create line actor
            if len(lines) > 0:
                line_cells = np.array(lines)
                
                if hand_id in self.ghost_line_actors:
                    # Update existing actor's points and lines
                    actor = self.ghost_line_actors[hand_id]
                    mesh = actor.GetMapper().GetInput()
                    mesh.points = landmark_points
                    mesh.lines = line_cells
                    mesh.Modified()
                else:
                    # Create new line mesh with transparency
                    line_mesh = pv.PolyData(landmark_points)
                    line_mesh.lines = line_cells
                    actor = plotter.add_mesh(line_mesh, color=color, 
                                           opacity=ghost_opacity,
                                           line_width=3)
                    self.ghost_line_actors[hand_id] = actor
    
    def run_realtime(self, show_3d: bool = True):
        """Run real-time hand reconstruction."""
        logger.info("Starting real-time reconstruction...")
        logger.info("Press 'q' to quit, 's' to save frame, 'v' to toggle 3D view")
        logger.info("")
        
        # Open cameras
        self.open_cameras()
        
        # Create visualization window if requested
        plotter = None
        if show_3d:
            plotter = pv.Plotter(window_size=[1280, 960], title="3D Hand Reconstruction")
            plotter.show(interactive_update=True, auto_close=False)
            # Initialize static geometries once
            self.initialize_static_geometries(plotter)
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frames
                frames = self.capture_frames()
                
                # Reconstruct hands
                reconstructed_hands = self.reconstruct_hands(frames)
                
                # Update statistics
                self.frame_count += 1
                self.total_reconstructions += len(reconstructed_hands)
                
                # Visualize 2D
                self.visualize_2d(frames, reconstructed_hands)
                
                # Update 3D visualization (efficient update without clearing)
                if show_3d and plotter is not None:
                    # Update only hand geometries (static geometries remain)
                    self.update_hand_geometries(plotter, reconstructed_hands)
                    
                    # Update ghost hand geometries
                    self.update_ghost_hand_geometries(plotter)
                    
                    # Update camera view to rotate around hands
                    if len(reconstructed_hands) > 0:
                        hands_center = self.calculate_hands_center(reconstructed_hands)
                        self.update_camera_view(plotter, hands_center)
                    
                    # Render the updated scene
                    plotter.render()
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / max(elapsed, 0.001)
                
                # Count raw detections for info
                raw_detection_count = sum(len(self.hand_detector.detect(frame)) for frame in frames)
                
                # Show info
                logger.info(f"Frame {self.frame_count}: Raw detections={raw_detection_count}, "
                          f"Reconstructed={len(reconstructed_hands)} hands, FPS: {fps:.1f}")
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    self.save_frame(frames, reconstructed_hands)
                elif key == ord('v'):
                    show_3d = not show_3d
                    if show_3d and plotter is None:
                        plotter = pv.Plotter(window_size=[1280, 960], title="3D Hand Reconstruction")
                        plotter.show(interactive_update=True, auto_close=False)
                        # Initialize static geometries for new window
                        self.initialize_static_geometries(plotter)
                    elif not show_3d and plotter is not None:
                        plotter.close()
                        plotter = None
                        # Reset state for next window
                        self.static_geometries_initialized = False
                        self.hand_point_actors.clear()
                        self.hand_line_actors.clear()
        
        finally:
            # Cleanup
            logger.info("")
            logger.info("Cleaning up...")
            
            for cam in self.cameras:
                cam.stop_live()
                cam.release()
            
            self.hand_detector.release()
            cv2.destroyAllWindows()
            
            if plotter is not None:
                plotter.close()
            
            # Print statistics
            logger.info("")
            logger.info("="*80)
            logger.info("Statistics")
            logger.info("="*80)
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total reconstructions: {self.total_reconstructions}")
            logger.info(f"Avg hands per frame: {self.total_reconstructions / max(1, self.frame_count):.2f}")
            logger.info("="*80)
    
    def save_frame(self, frames: List[np.ndarray], reconstructed_hands: List[Hand3DReconstruction]):
        """Save current frame and reconstructions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_dir = self.output_dir / f"frame_{timestamp}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        camera_names = list(self.camera_calibrations.keys())
        for idx, (frame, cam_name) in enumerate(zip(frames, camera_names)):
            img_path = frame_dir / f"{cam_name}.png"
            cv2.imwrite(str(img_path), frame)
        
        # Save reconstruction data
        reconstruction_data = {
            'timestamp': timestamp,
            'frame_count': self.frame_count,
            'num_hands': len(reconstructed_hands),
            'hands': []
        }
        
        for hand_3d in reconstructed_hands:
            hand_data = {
                'hand_id': hand_3d.hand_id,
                'handedness': hand_3d.handedness,
                'landmarks': {
                    idx: {
                        'point_3d_world': data['point_3d_world'].tolist() if isinstance(data['point_3d_world'], np.ndarray) else data['point_3d_world'],
                        'ray_distance': data['ray_distance'],
                        'reprojection_error': data['reprojection_error'],
                        'reprojection_error_median': data.get('reprojection_error_median'),
                        'residual_mean': data.get('residual_mean'),
                        'residual_median': data.get('residual_median'),
                        'confidence': data['confidence'],
                        'views': data['views'],
                        'metadata': {
                            key: (value.tolist() if isinstance(value, np.ndarray) else value)
                            for key, value in data.get('metadata', {}).items()
                        }
                    }
                    for idx, data in hand_3d.landmarks_3d.items()
                }
            }
            reconstruction_data['hands'].append(hand_data)
        
        json_path = frame_dir / "reconstruction.json"
        with open(json_path, 'w') as f:
            json.dump(reconstruction_data, f, indent=2)
        
        # Save point clouds for each hand
        for hand_3d in reconstructed_hands:
            landmark_indices = sorted(hand_3d.landmarks_3d.keys())
            landmark_points = np.array([hand_3d.landmarks_3d[i]['point_3d_world'] 
                                       for i in landmark_indices])
            
            if len(landmark_points) > 0:
                # Create PyVista point cloud
                point_cloud = pv.PolyData(landmark_points)
                
                # Save as PLY file
                ply_path = frame_dir / f"hand_{hand_3d.hand_id}_reconstruction.ply"
                point_cloud.save(str(ply_path))
        
        logger.info(f"✓ Saved frame to {frame_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-camera hand 3D reconstruction"
    )
    parser.add_argument(
        "--config",
        default="calibration_config.yaml",
        help="Path to camera config (default: calibration_config.yaml)"
    )
    parser.add_argument(
        "--calibration-config",
        default="calibration_config.yaml",
        help="Path to calibration config (default: calibration_config.yaml)"
    )
    parser.add_argument(
        "--hand3d-config",
        default="hand_3d_reconstruction.yaml",
        help="Path to hand 3D reconstruction parameters (default: hand_3d_reconstruction.yaml)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show 3D visualization window"
    )
    parser.add_argument(
        "--move-camera",
        action="store_true",
        help="Enable automatic camera rotation around detected hands in 3D view"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output (verbose logging)"
    )
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        system = MultiCameraHand3DSystem(
            config_path=args.config,
            calibration_config_path=args.calibration_config,
            hand3d_config_path=args.hand3d_config,
            move_camera=args.move_camera,
            debug=args.debug
        )
        system.load_calibrations()
        system.run_realtime(show_3d=args.visualize)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
