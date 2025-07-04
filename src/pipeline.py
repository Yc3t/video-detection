import cv2
import numpy as np
import yaml
from typing import List, Dict, Optional, Any, Generator, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
import time
import subprocess
import tempfile
import os

from .detection import PersonDetector, Detection
from .tracking import PersonTracker, DeepSORTTracker, Track
from .features.feature_extractor import FeatureExtractor


@dataclass
class ProcessingResult:
    """Result of processing a single frame."""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": [asdict(det) for det in self.detections],
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox": track.current_bbox,
                    "confidence": track.current_confidence,
                    "state": track.state,
                    "hits": track.hits,
                    "age": track.age
                }
                for track in self.tracks
            ],
            "processing_time": self.processing_time
        }


@dataclass
class VideoResult:
    """Result of processing an entire video."""
    video_path: str
    total_frames: int
    fps: float
    duration: float
    frame_results: List[ProcessingResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "frame_results": [result.to_dict() for result in self.frame_results],
            "summary": self.summary
        }


class PersonTrackingPipeline:
    """
    Complete person tracking pipeline.
    
    Integrates detection, tracking, and re-identification for comprehensive
    person analysis in video streams.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the tracking pipeline.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for processing (default: "auto")
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config = self._load_config(config_path)
        # Device override ('auto', 'cpu', 'gpu' -> 'cuda:0')
        device = device.lower() if device else "auto"
        if device == "gpu":
            device = "cuda:0"
        self.override_device = device
        
        # Initialize components
        self.detector = self._init_detector()
        self.tracker = self._init_tracker()
        # Use ResNet-50 based feature extractor for richer embeddings
        self.extractor = FeatureExtractor(device=self.override_device, method="resnet50")
        
        # Statistics
        self.stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "total_tracks_created": 0,  # Cumulative count of tracks created
            "average_processing_time": 0.0
        }
        
        logger.info("Person tracking pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "detection": {
                "model_name": "yolov8m.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "auto",
                "classes": [0],
                "max_det": 300,
                "multi_scale": True
            },
            "tracking": {
                "tracker_type": "deepsort",
                "max_disappeared": 30,
                "max_distance": 50
            },
            "processing": {
                "batch_size": 1,
                "frame_skip": 1,
                "resize_height": 640
            }
        }
    
    def _init_detector(self) -> PersonDetector:
        """Initialize person detector."""
        det_config = self.config["detection"].copy()

        # Override device if requested
        if self.override_device != "auto":
            det_config["device"] = self.override_device

        return PersonDetector(
            model_name=det_config["model_name"],
            confidence_threshold=det_config["confidence_threshold"],
            iou_threshold=det_config["iou_threshold"],
            device=det_config["device"],
            classes=det_config["classes"],
            max_det=det_config["max_det"],
            multi_scale=det_config.get("multi_scale", False),
            scales=det_config.get("scales"),
            merge_iou_threshold=det_config.get("merge_iou_threshold", 0.5),
        )
    
    def _init_tracker(self) -> PersonTracker:
        """Initialize person tracker."""
        track_config = self.config["tracking"]
        tracker_type = track_config.get("tracker_type", "deepsort")
        
        if tracker_type == "deepsort":
            return DeepSORTTracker(
                max_disappeared=track_config.get("max_disappeared", 70),
                max_distance=track_config.get("max_distance", 0.7)
            )
        else:
            return PersonTracker(
                max_disappeared=track_config.get("max_disappeared", 30),
                max_distance=track_config.get("max_distance", 50.0)
            )
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_id: int = 0, 
        timestamp: float = 0.0
    ) -> ProcessingResult:
        """
        Process a single frame.
        
        Args:
            frame: Input frame as numpy array
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            ProcessingResult with detections and tracks
        """
        start_time = time.time()
        
        # Person detection
        detections = self.detector.detect(frame)
        
        # Extract features for detected persons (batch extraction)
        bboxes = [d.bbox for d in detections]
        features: List[np.ndarray] = []
        if bboxes:
            extracted = self.extractor.extract_features(frame, bboxes)
            if extracted is not None and len(extracted) == len(bboxes):
                features = extracted
            else:
                # Fallback to zero vectors if extraction failed or size mismatch
                features = [np.zeros(self.extractor.feature_dim, dtype=np.float32) for _ in bboxes]
        
        # Update tracker. DeepSORTTracker requires appearance features, IoU tracker does not.
        if isinstance(self.tracker, DeepSORTTracker):
            tracks = self.tracker.update(detections, features)  # type: ignore[arg-type]
        else:
            tracks = self.tracker.update(detections)  # type: ignore[arg-type]
        
        # Update statistics
        self.stats["total_frames_processed"] += 1
        self.stats["total_detections"] += len(detections)
        
        # Count new tracks (track_id not seen before)
        current_track_ids = {track.track_id for track in tracks}
        if not hasattr(self, '_seen_track_ids'):
            self._seen_track_ids = set()
        
        new_tracks = current_track_ids - self._seen_track_ids
        self.stats["total_tracks_created"] += len(new_tracks)
        self._seen_track_ids.update(new_tracks)
        
        processing_time = time.time() - start_time
        
        # Update average processing time
        total_frames = self.stats["total_frames_processed"]
        current_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (current_avg * (total_frames - 1) + processing_time) / total_frames
        
        return ProcessingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            tracks=tracks,
            processing_time=processing_time
        )
    
    def process_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        visualize: bool = False,
        save_results: bool = True,
        preserve_audio: bool = True
    ) -> VideoResult:
        """
        Process an entire video file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            visualize: Whether to create visualization video
            save_results: Whether to save processing results
            preserve_audio: Whether to preserve audio in output video
            
        Returns:
            VideoResult with processing information
        """
        logger.info(f"Processing video: {video_path}")
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        
        # Setup output video writer if needed
        out_writer = None
        temp_output_path = None
        
        if visualize and output_path:
            # Use temporary file for video without audio
            temp_output_path = tempfile.mktemp(suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        frame_results = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_id / fps if fps > 0 else frame_id
                
                # Process frame
                result = self.process_frame(frame, frame_id, timestamp)
                frame_results.append(result)
                
                # Create visualization if requested
                if visualize and out_writer:
                    vis_frame = self.visualize_frame(frame, result)
                    out_writer.write(vis_frame)
                
                frame_id += 1
                
                # Progress logging
                if frame_id % 100 == 0:
                    progress = (frame_id / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_id}/{total_frames})")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
        
        # Generate summary
        summary = self._generate_summary(frame_results, fps)
        
        # Create video result
        video_result = VideoResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            frame_results=frame_results,
            summary=summary
        )
        
        # Save results if requested
        if save_results:
            self._save_results(video_result, video_path)
        
        # Handle audio preservation and final output
        if visualize and output_path and temp_output_path:
            if preserve_audio:
                self._combine_video_with_audio(video_path, temp_output_path, output_path)
                # Clean up temporary file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            else:
                # Just move the temporary file to final location
                if os.path.exists(temp_output_path):
                    import shutil
                    shutil.move(temp_output_path, output_path)
        
        logger.info(f"Video processing completed: {len(frame_results)} frames processed")
        return video_result
    
    def process_stream(
        self, 
        source: str, 
        max_frames: Optional[int] = None
    ) -> Generator[ProcessingResult, None, None]:
        """
        Process a video stream (camera or network stream).
        
        Args:
            source: Stream source (camera index or URL)
            max_frames: Maximum number of frames to process (None for unlimited)
            
        Yields:
            ProcessingResult for each frame
        """
        logger.info(f"Starting stream processing: {source}")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if unknown
        frame_id = 0
        
        try:
            while True:
                if max_frames and frame_id >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    break
                
                timestamp = frame_id / fps
                result = self.process_frame(frame, frame_id, timestamp)
                
                yield result
                frame_id += 1
        
        finally:
            cap.release()
            logger.info(f"Stream processing ended: {frame_id} frames processed")
    
    def visualize_frame(self, frame: np.ndarray, result: ProcessingResult) -> np.ndarray:
        """
        Create visualization for a single frame.
        
        Args:
            frame: Original frame
            result: Processing result
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw tracks
        for track in result.tracks:
            if track.current_bbox is not None:
                x1, y1, x2, y2 = track.current_bbox
                
                # Different color for tracked objects
                color = (255, 0, 0)  # Blue for tracks
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID
                track_text = f"ID: {track.track_id}"
                cv2.putText(vis_frame, track_text, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add frame info
        info_text = f"Frame: {result.frame_id} | Detections: {len(result.detections)} | Tracks: {len(result.tracks)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        processing_text = f"Processing: {result.processing_time*1000:.1f}ms"
        cv2.putText(vis_frame, processing_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def _generate_summary(self, frame_results: List[ProcessingResult], fps: float) -> Dict[str, Any]:
        """Generate processing summary."""
        if not frame_results:
            return {}
        
        total_detections = sum(len(r.detections) for r in frame_results)
        total_tracks = len(set(track.track_id for r in frame_results for track in r.tracks))
        avg_processing_time = sum(r.processing_time for r in frame_results) / len(frame_results)
        
        return {
            "total_frames": len(frame_results),
            "total_detections": total_detections,
            "total_unique_tracks": total_tracks,
            "average_detections_per_frame": total_detections / len(frame_results),
            "average_processing_time_ms": avg_processing_time * 1000,
            "processing_fps": 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }
    
    def _save_results(self, video_result: VideoResult, video_path: str):
        """Save processing results to JSON file."""
        output_dir = Path(video_path).parent
        output_file = output_dir / f"{Path(video_path).stem}_results.json"
        
        try:
            with open(output_file, 'w') as f:
                import json
                json.dump(video_result.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset pipeline statistics."""
        self.stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "total_tracks_created": 0,
            "average_processing_time": 0.0
        }
        if hasattr(self, '_seen_track_ids'):
            self._seen_track_ids.clear()
        logger.info("Pipeline statistics reset")
    
    def _combine_video_with_audio(self, original_video: str, processed_video: str, output_path: str):
        """Combine processed video with original audio using ffmpeg."""
        try:
            cmd = [
                'ffmpeg',
                '-i', processed_video,  # Video input
                '-i', original_video,   # Audio input
                '-c:v', 'copy',         # Copy video codec
                '-c:a', 'aac',          # Audio codec
                '-map', '0:v:0',        # Map video from first input
                '-map', '1:a:0',        # Map audio from second input
                '-y',                   # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully combined video with audio: {output_path}")
            else:
                logger.warning(f"ffmpeg failed: {result.stderr}")
                # Fallback: just copy the processed video
                import shutil
                shutil.copy2(processed_video, output_path)
                logger.info(f"Copied processed video without audio: {output_path}")
                
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Saving video without audio.")
            import shutil
            shutil.copy2(processed_video, output_path)
        except Exception as e:
            logger.error(f"Error combining video with audio: {e}")
            import shutil
            shutil.copy2(processed_video, output_path) 
