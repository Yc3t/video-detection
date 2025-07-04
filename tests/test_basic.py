"""
Basic unit tests for the person detection and tracking system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.detector import PersonDetector, Detection
from src.tracking.tracker import PersonTracker, Track
from src.pipeline import PersonTrackingPipeline


class TestDetection:
    """Test person detection functionality."""
    
    def test_detection_creation(self):
        """Test Detection dataclass creation."""
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=0,
            frame_id=1,
            timestamp=1.5
        )
        
        assert detection.bbox == (10, 20, 100, 200)
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.frame_id == 1
        assert detection.timestamp == 1.5
    
    def test_detector_initialization(self):
        """Test PersonDetector initialization."""
        detector = PersonDetector(
            model_name="yolov8n.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
        
        assert detector.model_name == "yolov8n.pt"
        assert detector.confidence_threshold == 0.5
        assert detector.device == "cpu"
        
        # Test model info
        info = detector.get_model_info()
        assert "model_name" in info
        assert "confidence_threshold" in info
    
    def test_detector_empty_frame(self):
        """Test detector with empty frame."""
        detector = PersonDetector(model_name="yolov8n.pt", device="cpu")
        
        # Test with None frame
        detections = detector.detect(None)
        assert detections == []
        
        # Test with empty frame
        empty_frame = np.array([])
        detections = detector.detect(empty_frame)
        assert detections == []
    
    def test_detector_dummy_frame(self):
        """Test detector with dummy frame."""
        detector = PersonDetector(model_name="yolov8n.pt", device="cpu")
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Should not crash (may or may not detect anything)
        detections = detector.detect(frame, frame_id=1, timestamp=1.0)
        assert isinstance(detections, list)
        
        # All detections should be valid
        for det in detections:
            assert isinstance(det, Detection)
            assert det.frame_id == 1
            assert det.timestamp == 1.0
            assert 0 <= det.confidence <= 1.0


class TestTracking:
    """Test person tracking functionality."""
    
    def test_track_creation(self):
        """Test Track creation and updates."""
        track = Track(track_id=1)
        
        assert track.track_id == 1
        assert track.state == "tentative"
        assert track.hits == 0
        assert track.age == 0
        
        # Test update
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=0,
            frame_id=1,
            timestamp=1.0
        )
        
        track.update(detection)
        assert track.hits == 1
        assert track.last_seen == 1
        assert len(track.detections) == 1
        assert track.current_bbox == (10, 20, 100, 200)
        assert track.current_confidence == 0.85
    
    def test_track_state_transitions(self):
        """Test track state transitions."""
        track = Track(track_id=1)
        
        # Create dummy detections
        for i in range(5):
            detection = Detection(
                bbox=(10, 20, 100, 200),
                confidence=0.85,
                class_id=0,
                frame_id=i,
                timestamp=i * 0.1
            )
            track.update(detection)
        
        # Should be confirmed after 3 hits
        assert track.state == "confirmed"
        assert track.hits == 5
    
    def test_tracker_initialization(self):
        """Test PersonTracker initialization."""
        tracker = PersonTracker(max_disappeared=30, iou_threshold=0.3)
        
        assert tracker.max_disappeared == 30
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1
    
    def test_tracker_single_detection(self):
        """Test tracker with single detection."""
        tracker = PersonTracker()
        
        detection = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=0,
            frame_id=1,
            timestamp=1.0
        )
        
        tracks = tracker.update([detection])
        
        # Should create one track
        assert len(tracker.tracks) == 1
        assert len(tracks) == 0  # Not confirmed yet
        
        # Update again to confirm
        tracks = tracker.update([detection])
        tracks = tracker.update([detection])
        
        assert len(tracks) == 1  # Now confirmed
        assert tracks[0].track_id == 1
    
    def test_iou_computation(self):
        """Test IoU computation."""
        bbox1 = (0, 0, 10, 10)  # Area = 100
        bbox2 = (5, 5, 15, 15)  # Area = 100, overlap = 25
        
        iou = PersonTracker._compute_iou(bbox1, bbox2)
        expected_iou = 25 / (100 + 100 - 25)  # intersection / union
        
        assert abs(iou - expected_iou) < 1e-6
        
        # Test no overlap
        bbox3 = (20, 20, 30, 30)
        iou = PersonTracker._compute_iou(bbox1, bbox3)
        assert iou == 0.0
        
        # Test perfect overlap
        iou = PersonTracker._compute_iou(bbox1, bbox1)
        assert iou == 1.0


class TestPipeline:
    """Test complete pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = PersonTrackingPipeline()
        
        assert pipeline.detector is not None
        assert pipeline.tracker is not None
        assert "total_frames_processed" in pipeline.stats
    
    def test_pipeline_single_frame(self):
        """Test pipeline with single frame."""
        pipeline = PersonTrackingPipeline()
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = pipeline.process_frame(frame, frame_id=1, timestamp=1.0)
        
        assert result.frame_id == 1
        assert result.timestamp == 1.0
        assert isinstance(result.detections, list)
        assert isinstance(result.tracks, list)
        assert result.processing_time > 0
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics tracking."""
        pipeline = PersonTrackingPipeline()
        
        # Initial stats
        stats = pipeline.get_statistics()
        assert stats["total_frames_processed"] == 0
        
        # Process a frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame)
        
        # Check updated stats
        stats = pipeline.get_statistics()
        assert stats["total_frames_processed"] == 1
        assert stats["average_processing_time"] > 0
        
        # Reset stats
        pipeline.reset_statistics()
        stats = pipeline.get_statistics()
        assert stats["total_frames_processed"] == 0
    
    def test_pipeline_visualization(self):
        """Test pipeline visualization."""
        pipeline = PersonTrackingPipeline()
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        
        # Should not crash
        vis_frame = pipeline.visualize_frame(frame, result)
        assert vis_frame.shape == frame.shape
        assert vis_frame.dtype == frame.dtype


def test_integration():
    """Integration test with multiple frames."""
    pipeline = PersonTrackingPipeline()
    
    # Process multiple frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_id=i, timestamp=i * 0.1)
        
        assert result.frame_id == i
        assert isinstance(result.detections, list)
        assert isinstance(result.tracks, list)
    
    # Check final statistics
    stats = pipeline.get_statistics()
    assert stats["total_frames_processed"] == 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 