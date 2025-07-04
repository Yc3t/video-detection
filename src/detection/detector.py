

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """Represents a single person detection."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    frame_id: int
    timestamp: float


class PersonDetector:
    """
    YOLOv8-based person detector.
    
    Detects people in video frames and returns bounding boxes with confidence scores.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        multi_scale: bool = False,
        scales: Optional[List[float]] = None,
        merge_iou_threshold: float = 0.5,
    ):
        """
        Initialize the person detector.
        
        Args:
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('auto', 'cpu', 'cuda:0', etc.)
            classes: List of class IDs to detect (None for all, [0] for person only)
            max_det: Maximum number of detections per image
            multi_scale: Enable multi-scale inference for improved recall
            scales: List of scale factors (e.g., [0.5, 1.0, 1.5]) when multi_scale is True
            merge_iou_threshold: IoU threshold when merging detections from different scales
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or [0]  # Default to person class only
        self.max_det = max_det
        self.multi_scale = multi_scale
        self.scales = sorted(list(set(scales or [1.0] + ([0.5, 1.5] if multi_scale else []))))
        self.merge_iou_threshold = merge_iou_threshold
        
        # Initialize model
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        # Move to specified device
        if device != "auto":
            self.model.to(device)
            
        logger.info(f"Person detector initialized with {model_name} on {device}")
    
    @staticmethod
    def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _apply_nms(self, detections: List["Detection"], iou_thr: float) -> List["Detection"]:
        """Apply simple IoU-based NMS on a list of Detection objects."""
        if not detections:
            return []

        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        while detections_sorted:
            best = detections_sorted.pop(0)
            kept.append(best)
            detections_sorted = [d for d in detections_sorted if self._iou(best.bbox, d.bbox) < iou_thr]
        return kept

    def detect(
        self, 
        frame: np.ndarray, 
        frame_id: int = 0, 
        timestamp: float = 0.0
    ) -> List[Detection]:
        """
        Detect people in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_id: Frame identifier
            timestamp: Timestamp of the frame
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        try:
            if not self.multi_scale:
                frames_to_run = [(1.0, frame)]
            else:
                frames_to_run = [(s, cv2.resize(frame, (int(frame.shape[1]*s), int(frame.shape[0]*s)))) for s in self.scales]

            detections_all: List[Detection] = []

            for scale, img in frames_to_run:
                results = self.model(
                    img,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=self.classes,
                    max_det=self.max_det,
                    verbose=False,
                )

                for result in results:
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box / scale  # scale back to original image size
                        detection = Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                        detections_all.append(detection)

            # Merge duplicates via NMS if multi-scale gathered several boxes
            final_detections = self._apply_nms(detections_all, self.merge_iou_threshold)

            logger.debug(f"Detected {len(final_detections)} people in frame {frame_id} (raw {len(detections_all)})")
            return final_detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def detect_batch(
        self, 
        frames: List[np.ndarray], 
        frame_ids: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[List[Detection]]:
        """
        Detect people in a batch of frames.
        
        Args:
            frames: List of input frames
            frame_ids: List of frame identifiers
            timestamps: List of timestamps
            
        Returns:
            List of detection lists (one per frame)
        """
        if not frames:
            return []
        
        if frame_ids is None:
            frame_ids = list(range(len(frames)))
        if timestamps is None:
            timestamps = [0.0] * len(frames)
        
        batch_detections = []
        
        try:
            # Run batch inference
            results = self.model(
                frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_det,
                verbose=False
            )
            
            # Process each frame's results
            for i, result in enumerate(results):
                frame_detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        detection = Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            frame_id=frame_ids[i],
                            timestamp=timestamps[i]
                        )
                        frame_detections.append(detection)
                
                batch_detections.append(frame_detections)
                
        except Exception as e:
            logger.error(f"Error during batch detection: {e}")
            batch_detections = [[] for _ in frames]
        
        return batch_detections
    
    def visualize_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Detection],
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize detections on a frame.
        
        Args:
            frame: Input frame
            detections: List of detections to visualize
            show_confidence: Whether to show confidence scores
            color: BGR color for bounding boxes
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            if show_confidence:
                label = f"Person: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    vis_frame, 
                    (x1, y1 - label_size[1] - 10), 
                    (x1 + label_size[0], y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    vis_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
        
        return vis_frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.classes,
            "max_det": self.max_det,
            "multi_scale": self.multi_scale,
            "scales": self.scales,
            "merge_iou_threshold": self.merge_iou_threshold,
        } 