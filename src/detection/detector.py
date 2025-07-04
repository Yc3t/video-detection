import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """representa una single detección persona."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    frame_id: int
    timestamp: float


class PersonDetector:
    """
    detector personas basado yolov8.
    
    detecta personas en frames video y retorna bounding boxes con scores confianza.
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
        inicializa detector personas.
        
        args:
            model_name: nombre modelo yolov8 (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: confianza mínima para detecciones
            iou_threshold: umbral iou para nms
            device: dispositivo ejecutar inferencia ('auto', 'cpu', 'cuda:0', etc.)
            classes: lista ids clases detectar (none para todos, [0] solo personas)
            max_det: número máximo detecciones por imagen
            multi_scale: habilita inferencia multi-escala para mejorar recall
            scales: lista factores escala (ej. [0.5, 1.0, 1.5]) cuando multi_scale es true
            merge_iou_threshold: umbral iou cuando merge detecciones de diferentes escalas
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or [0]  # por defecto solo clase persona
        self.max_det = max_det
        self.multi_scale = multi_scale
        self.scales = sorted(list(set(scales or [1.0] + ([0.5, 1.5] if multi_scale else []))))
        self.merge_iou_threshold = merge_iou_threshold
        
        # inicializa modelo
        logger.info(f"cargando modelo yolo: {model_name}")
        self.model = YOLO(model_name)
        
        # mueve a dispositivo especificado
        if device != "auto":
            self.model.to(device)
            
        logger.info(f"detector personas inicializado con {model_name} en {device}")
    
    @staticmethod
    def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """computa iou entre dos bboxes."""
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
        """aplica nms simple basado iou en lista objetos detection."""
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
        detecta personas en single frame.
        
        args:
            frame: frame entrada como numpy array (formato bgr)
            frame_id: identificador frame
            timestamp: timestamp del frame
            
        returns:
            lista objetos detection
        """
        if frame is None or frame.size == 0:
            logger.warning("frame vacío recibido")
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
                        x1, y1, x2, y2 = box / scale  # escala de vuelta a tamaño imagen original
                        detection = Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            frame_id=frame_id,
                            timestamp=timestamp,
                        )
                        detections_all.append(detection)

            # merge duplicados via nms si multi-escala reunió varias cajas
            final_detections = self._apply_nms(detections_all, self.merge_iou_threshold)

            logger.debug(f"detectadas {len(final_detections)} personas en frame {frame_id} (raw {len(detections_all)})")
            return final_detections

        except Exception as e:
            logger.error(f"error durante detección: {e}")
            return []
    
    def detect_batch(
        self, 
        frames: List[np.ndarray], 
        frame_ids: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[List[Detection]]:
        """
        detecta personas en batch frames.
        
        args:
            frames: lista frames entrada
            frame_ids: lista identificadores frame
            timestamps: lista timestamps
            
        returns:
            lista listas detecciones (una por frame)
        """
        if not frames:
            return []
        
        if frame_ids is None:
            frame_ids = list(range(len(frames)))
        if timestamps is None:
            timestamps = [0.0] * len(frames)
        
        batch_detections = []
        
        try:
            # ejecuta inferencia batch
            results = self.model(
                frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_det,
                verbose=False
            )
            
            # procesa resultados cada frame
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
            logger.error(f"error durante detección batch: {e}")
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
        visualiza detecciones en frame.
        
        args:
            frame: frame entrada
            detections: lista detecciones visualizar
            show_confidence: si mostrar scores confianza
            color: color bgr para bounding boxes
            
        returns:
            frame con detecciones visualizadas
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # dibuja bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # dibuja score confianza
            if show_confidence:
                label = f"persona: {detection.confidence:.2f}"
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
        """obtiene información sobre modelo cargado."""
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