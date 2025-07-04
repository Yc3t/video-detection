

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from ..detection.detector import Detection


@dataclass
class Track:
    """Tracked person across frames."""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    last_seen: int = 0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        self.detections.append(detection)
        self.last_seen = detection.frame_id
        self.hits += 1
        self.time_since_update = 0
        
        # Confirm track after minimum hits
        if self.hits >= 3 and self.state == "tentative":
            self.state = "confirmed"
    
    def predict(self):
        """Predict next state (to be implemented by specific trackers)."""
        self.age += 1
        self.time_since_update += 1
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        
        # Delete track if not seen for too long
        if self.time_since_update > 30:
            self.state = "deleted"
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the most recent bounding box."""
        if self.detections:
            return self.detections[-1].bbox
        return None
    
    @property
    def current_confidence(self) -> Optional[float]:
        """Get the most recent confidence score."""
        if self.detections:
            return self.detections[-1].confidence
        return None


class BaseTracker(ABC):
    """Abstract base class for person trackers."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize base tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        pass
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (confirmed) tracks."""
        return [
            track for track in self.tracks.values() 
            if track.state == "confirmed"
        ]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (including tentative)."""
        return list(self.tracks.values())
    
    def cleanup_tracks(self):
        """Remove deleted tracks."""
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.state == "deleted"
        ]
        for track_id in to_remove:
            del self.tracks[track_id]


class PersonTracker(BaseTracker):
    """
    Simple IoU-based person tracker.
    
    Uses intersection over union (IoU) for data association.
    Good baseline tracker for comparison with more sophisticated methods.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0, iou_threshold: float = 0.3):
        """
        Initialize IoU tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing
            max_distance: Maximum distance for association (not used in IoU tracker)
            iou_threshold: Minimum IoU for association
        """
        super().__init__(max_disappeared, max_distance)
        self.iou_threshold = iou_threshold
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with IoU-based association."""
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks.values():
            track.predict()
        
        # Associate detections with tracks
        if detections and self.tracks:
            matches, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
                detections, list(self.tracks.values())
            )
            
            # Update matched tracks
            for det_idx, trk_idx in matches:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id].update(detections[det_idx])
            
            # Mark unmatched tracks as missed
            for trk_idx in unmatched_trks:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id].mark_missed()
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_new_track(detections[det_idx])
        
        elif detections:
            # No existing tracks, create new ones
            for detection in detections:
                self._create_new_track(detection)
        
        else:
            # No detections, mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
        
        # Cleanup deleted tracks
        self.cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _create_new_track(self, detection: Detection):
        """Create a new track from detection."""
        track = Track(track_id=self.next_id)
        track.update(detection)
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Detection], 
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using IoU.
        
        Returns:
            matches: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if not tracks:
            return [], list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                if track.current_bbox is not None:
                    iou_matrix[d, t] = self._compute_iou(detection.bbox, track.current_bbox)
        
        # Find matches using Hungarian algorithm (simplified greedy approach)
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Greedy matching
        while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            # Find best match
            best_iou = 0
            best_det = -1
            best_trk = -1
            
            for d in unmatched_detections:
                for t in unmatched_tracks:
                    if iou_matrix[d, t] > best_iou:
                        best_iou = iou_matrix[d, t]
                        best_det = d
                        best_trk = t
            
            # Check if best match is above threshold
            if best_iou >= self.iou_threshold:
                matches.append((best_det, best_trk))
                unmatched_detections.remove(best_det)
                unmatched_tracks.remove(best_trk)
            else:
                break
        
        return matches, unmatched_detections, unmatched_tracks
    
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