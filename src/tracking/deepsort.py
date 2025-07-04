

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger

from .tracker import BaseTracker, Track
from ..detection.detector import Detection


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in image space.
    
    The state vector is [x, y, s, r, dx, dy, ds, dr] where:
    - (x, y) is the center position
    - s is the scale (area)
    - r is the aspect ratio
    - d* are the velocities
    """
    
    count = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """Initialize Kalman filter with initial bounding box."""
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement function (observe position and scale)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.kf.P *= 10.0
        
        # Process noise covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update the Kalman filter with observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Predict the next state."""
        # Prevent negative scale
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        
        # Validate the predicted state
        if np.any(np.isnan(self.kf.x)) or np.any(np.isinf(self.kf.x)):
            # Reset to a reasonable state if prediction becomes invalid
            logger.warning(f"Invalid Kalman state detected for track {self.id}, resetting")
            # Keep position but reset scale and velocities
            self.kf.x[2] = max(abs(self.kf.x[2]), 100.0)  # Minimum scale
            self.kf.x[3] = max(abs(self.kf.x[3]), 0.5)    # Reasonable aspect ratio
            self.kf.x[4:] = 0.0  # Reset velocities
        
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current bounding box estimate."""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bounding box to measurement vector [x, y, s, r]."""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bounding box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        w = max(x2 - x1, 1)  # Minimum width of 1
        h = max(y2 - y1, 1)  # Minimum height of 1
        
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h  # scale (area)
        r = w / float(h)  # aspect ratio (h is guaranteed > 0)
        
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert state vector to bounding box [x1, y1, x2, y2]."""
        # Handle potential NaN or negative values
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # Return a default bounding box if state is invalid
            return (0, 0, 1, 1)
        
        # Ensure scale and aspect ratio are positive
        scale = max(abs(float(x[2])), 1.0)  # Minimum scale of 1
        aspect_ratio = max(abs(float(x[3])), 0.1)  # Minimum aspect ratio
        
        w = np.sqrt(scale * aspect_ratio)
        h = scale / w if w > 0 else 1.0
        
        # Ensure width and height are positive
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        # Calculate bounding box coordinates
        center_x = float(x[0])
        center_y = float(x[1])
        
        x1 = int(max(0, center_x - w / 2.0))
        y1 = int(max(0, center_y - h / 2.0))
        x2 = int(center_x + w / 2.0)
        y2 = int(center_y + h / 2.0)
        
        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
            
        return (x1, y1, x2, y2)


class DeepSORTTrack(Track):
    """Extended track for DeepSORT with appearance features."""
    
    def __init__(self, track_id: int, detection: Detection, feature: Optional[np.ndarray] = None):
        super().__init__(track_id)
        self.kalman = KalmanBoxTracker(detection.bbox)
        self.features = [feature] if feature is not None else []
        self.update(detection)
    
    def update(self, detection: Detection, feature: Optional[np.ndarray] = None):
        """Update track with new detection and feature."""
        super().update(detection)
        self.kalman.update(detection.bbox)
        
        if feature is not None:
            self.features.append(feature)
            # Keep only recent features (sliding window)
            if len(self.features) > 100:
                self.features = self.features[-100:]
    
    def predict(self):
        """Predict next state using Kalman filter."""
        super().predict()
        predicted_bbox = self.kalman.predict()
        return predicted_bbox
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current bounding box from Kalman filter."""
        return self.kalman.get_state()
    
    def get_feature(self) -> Optional[np.ndarray]:
        """Get averaged appearance feature."""
        if not self.features:
            return None
        
        # Return mean of recent features
        features = np.array(self.features[-10:])  # Use last 10 features
        return np.mean(features, axis=0)


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT tracker implementation.
    
    Combines Kalman filtering for motion prediction with appearance features
    for robust data association across occlusions.
    """
    
    def __init__(
        self,
        max_disappeared: int = 70,
        max_distance: float = 0.7,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        nn_budget: int = 100,
        appearance_weight: float = 0.6,
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_disappeared: Maximum frames before track deletion
            max_distance: Maximum cosine distance for appearance matching
            max_iou_distance: Maximum IoU distance for motion matching
            max_age: Maximum age of track
            n_init: Number of consecutive detections before track confirmation
            nn_budget: Maximum number of features per track
            appearance_weight: Weight for appearance-based cost in blended cost matrix
        """
        super().__init__(max_disappeared, max_distance)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.appearance_weight = appearance_weight
        
        logger.info("DeepSORT tracker initialized")
    
    def update(self, detections: List[Detection], features: Optional[List[np.ndarray]] = None) -> List[Track]:
        """
        Update tracker with new detections and features.
        
        Args:
            detections: List of detections from current frame
            features: Optional list of appearance features for each detection
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks.values():
            if isinstance(track, DeepSORTTrack):
                track.predict()
        
        # Associate detections with tracks
        if detections:
            matches, unmatched_dets, unmatched_trks = self._associate(
                detections, list(self.tracks.values()), features
            )
            
            # Update matched tracks
            for det_idx, trk_idx in matches:
                track_id = list(self.tracks.keys())[trk_idx]
                feature = features[det_idx] if features else None
                self.tracks[track_id].update(detections[det_idx], feature)
            
            # Mark unmatched tracks as missed
            for trk_idx in unmatched_trks:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id].mark_missed()
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                feature = features[det_idx] if features else None
                self._create_new_track(detections[det_idx], feature)
        
        else:
            # No detections, mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
        
        # Cleanup old tracks
        self.cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _create_new_track(self, detection: Detection, feature: Optional[np.ndarray] = None):
        """Create a new DeepSORT track."""
        track = DeepSORTTrack(self.next_id, detection, feature)
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _build_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[DeepSORTTrack],
        features: Optional[List[np.ndarray]]
    ) -> np.ndarray:
        """Compute blended distance matrix (appearance + IoU)."""
        n_det = len(detections)
        n_trk = len(tracks)
        
        if n_det == 0 or n_trk == 0:
            return np.array([]).reshape(n_det, n_trk)
        
        C = np.ones((n_det, n_trk), dtype=np.float32)  # Initialize with max cost
        
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(tracks):
                # IoU part
                if trk.current_bbox:
                    iou = self._compute_iou(det.bbox, trk.current_bbox)
                    iou = max(0.0, min(1.0, iou))  # Clamp to [0, 1]
                    iou_dist = 1.0 - iou
                else:
                    iou_dist = 1.0  # Maximum distance when no bbox
                
                # Appearance part
                cos_dist = 1.0  # Default to maximum distance
                if features is not None and d_idx < len(features) and trk.get_feature() is not None:
                    try:
                        cos_dist = self._cosine_distance(features[d_idx], trk.get_feature())
                        cos_dist = max(0.0, min(2.0, cos_dist))  # Clamp to reasonable range
                    except Exception as e:
                        logger.debug(f"Failed to compute cosine distance: {e}")
                        cos_dist = 1.0
                
                # Blended cost with validation
                try:
                    blended_cost = (
                        self.appearance_weight * cos_dist + 
                        (1.0 - self.appearance_weight) * iou_dist
                    )
                    # Ensure finite and positive cost
                    if np.isfinite(blended_cost) and blended_cost >= 0:
                        C[d_idx, t_idx] = blended_cost
                    else:
                        C[d_idx, t_idx] = 1.0  # Fallback to maximum cost
                except Exception as e:
                    logger.debug(f"Error computing blended cost: {e}")
                    C[d_idx, t_idx] = 1.0
        
        return C

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[DeepSORTTrack],
        features: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate with blended cost matrix (no separate cascades)."""
        if not tracks:
            return [], list(range(len(detections))), []
        
        if not detections:
            return [], [], list(range(len(tracks)))
        
        cost_matrix = self._build_cost_matrix(detections, tracks, features)
        
        # Validate cost matrix
        if cost_matrix.size == 0:
            logger.warning("Empty cost matrix, returning no matches")
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Check for invalid values
        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
            logger.warning("Cost matrix contains invalid values, using fallback IoU matching")
            return self._fallback_iou_matching(detections, tracks)
        
        # Check if cost matrix is reasonable
        if np.all(cost_matrix > self.max_distance):
            logger.warning("All costs exceed threshold, using fallback IoU matching")
            return self._fallback_iou_matching(detections, tracks)
        
        try:
            # Hungarian algorithm
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.warning(f"Hungarian algorithm failed: {e}, using fallback IoU matching")
            return self._fallback_iou_matching(detections, tracks)
        
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        for r, c in zip(row_idx, col_idx):
            # Gate cost
            if cost_matrix[r, c] > self.max_distance:
                continue
            matches.append((r, c))
            if r in unmatched_dets:
                unmatched_dets.remove(r)
            if c in unmatched_trks:
                unmatched_trks.remove(c)
        
        return matches, unmatched_dets, unmatched_trks
    
    def _fallback_iou_matching(
        self,
        detections: List[Detection],
        tracks: List[DeepSORTTrack]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Fallback IoU-based matching when sophisticated matching fails."""
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(tracks):
                if trk.current_bbox:
                    iou_matrix[d_idx, t_idx] = self._compute_iou(det.bbox, trk.current_bbox)
        
        # Simple greedy matching
        while True:
            # Find best IoU match
            max_iou = np.max(iou_matrix)
            if max_iou < 0.1:  # Minimum IoU threshold
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, trk_idx = max_idx
            
            matches.append((det_idx, trk_idx))
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if trk_idx in unmatched_trks:
                unmatched_trks.remove(trk_idx)
            
            # Remove matched detection and track from consideration
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, trk_idx] = 0
        
        return matches, unmatched_dets, unmatched_trks
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two feature vectors."""
        try:
            # Ensure inputs are valid numpy arrays
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                return 1.0
            
            a = a.flatten()
            b = b.flatten()
            
            # Check for empty or mismatched arrays
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return 1.0
            
            # Check for invalid values
            if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isinf(a)) or np.any(np.isinf(b)):
                return 1.0
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            # Avoid division by zero
            if norm_a == 0 or norm_b == 0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
                return 1.0
            
            cosine_sim = dot_product / (norm_a * norm_b)
            
            # Clamp cosine similarity to [-1, 1] to handle numerical errors
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            
            # Convert to distance (0 = identical, 2 = opposite)
            distance = 1.0 - cosine_sim
            
            # Ensure result is finite and positive
            if not np.isfinite(distance) or distance < 0:
                return 1.0
            
            return distance
            
        except Exception:
            # Return maximum distance on any error
            return 1.0
    
    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union 