import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger

from .tracker import BaseTracker, Track
from ..detection.detector import Detection


class KalmanBoxTracker:
    """
    filtro kalman para tracking de bounding boxes en espacio imagen.
    
    vector estado: [x, y, s, r, dx, dy, ds, dr] donde:
    - (x, y) posición centro
    - s escala (área)
    - r ratio aspecto
    - d* velocidades
    """
    
    count = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """inicializa filtro kalman con bbox inicial."""
        # modelo velocidad constante
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # matriz transición estado (velocidad constante)
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
        
        # función medición (observa posición y escala)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # ruido medición
        self.kf.R[2:, 2:] *= 10.0
        
        # ruido proceso
        self.kf.P[4:, 4:] *= 1000.0  # alta incertidumbre velocidades
        self.kf.P *= 10.0
        
        # covarianza ruido proceso
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # inicializa estado
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """actualiza filtro kalman con bbox observado."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """predice siguiente estado."""
        # previene escala negativa
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        
        # valida estado predicho
        if np.any(np.isnan(self.kf.x)) or np.any(np.isinf(self.kf.x)):
            # resetea a estado razonable si predicción inválida
            logger.warning(f"estado kalman inválido detectado para track {self.id}, reseteando")
            # mantiene posición pero resetea escala y velocidades
            self.kf.x[2] = max(abs(self.kf.x[2]), 100.0)  # escala mínima
            self.kf.x[3] = max(abs(self.kf.x[3]), 0.5)    # ratio aspecto razonable
            self.kf.x[4:] = 0.0  # resetea velocidades
        
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """obtiene estimación bbox actual."""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """convierte bbox a vector medición [x, y, s, r]."""
        x1, y1, x2, y2 = bbox
        
        # asegura bbox válido
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        w = max(x2 - x1, 1)  # ancho mínimo 1
        h = max(y2 - y1, 1)  # alto mínimo 1
        
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h  # escala (área)
        r = w / float(h)  # ratio aspecto
        
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
        """convierte vector estado a bbox [x1, y1, x2, y2]."""
        # maneja valores nan o negativos
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # retorna bbox por defecto si estado inválido
            return (0, 0, 1, 1)
        
        # asegura escala y ratio aspecto positivos
        scale = max(abs(float(x[2])), 1.0)  # escala mínima 1
        aspect_ratio = max(abs(float(x[3])), 0.1)  # ratio aspecto mínimo
        
        w = np.sqrt(scale * aspect_ratio)
        h = scale / w if w > 0 else 1.0
        
        # asegura ancho y alto positivos
        w = max(w, 1.0)
        h = max(h, 1.0)
        
        # calcula coordenadas bbox
        center_x = float(x[0])
        center_y = float(x[1])
        
        x1 = int(max(0, center_x - w / 2.0))
        y1 = int(max(0, center_y - h / 2.0))
        x2 = int(center_x + w / 2.0)
        y2 = int(center_y + h / 2.0)
        
        # asegura x2 > x1 y y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
            
        return (x1, y1, x2, y2)


class DeepSORTTrack(Track):
    """track extendido para deepsort con features apariencia."""
    
    def __init__(self, track_id: int, detection: Detection, feature: Optional[np.ndarray] = None):
        super().__init__(track_id)
        self.kalman = KalmanBoxTracker(detection.bbox)
        self.features = [feature] if feature is not None else []
        self.update(detection)
    
    def update(self, detection: Detection, feature: Optional[np.ndarray] = None):
        """actualiza track con nueva detección y feature."""
        super().update(detection)
        self.kalman.update(detection.bbox)
        
        if feature is not None:
            self.features.append(feature)
            # mantiene solo features recientes (ventana deslizante)
            if len(self.features) > 100:
                self.features = self.features[-100:]
    
    def predict(self):
        """predice siguiente estado usando filtro kalman."""
        super().predict()
        predicted_bbox = self.kalman.predict()
        return predicted_bbox
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """obtiene bbox actual del filtro kalman."""
        return self.kalman.get_state()
    
    def get_feature(self) -> Optional[np.ndarray]:
        """obtiene feature apariencia promediado."""
        if not self.features:
            return None
        
        # retorna promedio features recientes
        features = np.array(self.features[-10:])  # usa últimos 10 features
        return np.mean(features, axis=0)


class DeepSORTTracker(BaseTracker):
    """
    implementación tracker deepsort.
    
    combina filtrado kalman para predicción movimiento con features apariencia
    para asociación datos robusta a través oclusiones.
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
        inicializa tracker deepsort.
        
        args:
            max_disappeared: frames máximos antes eliminación track
            max_distance: distancia coseno máxima para matching apariencia
            max_iou_distance: distancia iou máxima para matching movimiento
            max_age: edad máxima track
            n_init: detecciones consecutivas antes confirmación track
            nn_budget: número máximo features por track
            appearance_weight: peso para costo basado apariencia en matriz costo mezclada
        """
        super().__init__(max_disappeared, max_distance)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.appearance_weight = appearance_weight
        
        logger.info("tracker deepsort inicializado")
    
    def update(self, detections: List[Detection], features: Optional[List[np.ndarray]] = None) -> List[Track]:
        """
        actualiza tracker con nuevas detecciones y features.
        
        args:
            detections: lista detecciones frame actual
            features: lista opcional features apariencia para cada detección
            
        returns:
            lista tracks activos
        """
        self.frame_count += 1
        
        # predice tracks existentes
        for track in self.tracks.values():
            if isinstance(track, DeepSORTTrack):
                track.predict()
        
        # asocia detecciones con tracks
        if detections:
            matches, unmatched_dets, unmatched_trks = self._associate(
                detections, list(self.tracks.values()), features
            )
            
            # actualiza tracks matched
            for det_idx, trk_idx in matches:
                track_id = list(self.tracks.keys())[trk_idx]
                feature = features[det_idx] if features else None
                self.tracks[track_id].update(detections[det_idx], feature)
            
            # marca tracks unmatched como missed
            for trk_idx in unmatched_trks:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id].mark_missed()
            
            # crea nuevos tracks para detecciones unmatched
            for det_idx in unmatched_dets:
                feature = features[det_idx] if features else None
                self._create_new_track(detections[det_idx], feature)
        
        else:
            # sin detecciones, marca todos tracks como missed
            for track in self.tracks.values():
                track.mark_missed()
        
        # limpia tracks viejos
        self.cleanup_tracks()
        
        return self.get_active_tracks()
    
    def _create_new_track(self, detection: Detection, feature: Optional[np.ndarray] = None):
        """crea nuevo track deepsort."""
        track = DeepSORTTrack(self.next_id, detection, feature)
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _build_cost_matrix(
        self,
        detections: List[Detection],
        tracks: List[DeepSORTTrack],
        features: Optional[List[np.ndarray]]
    ) -> np.ndarray:
        """computa matriz distancia mezclada (apariencia + iou)."""
        n_det = len(detections)
        n_trk = len(tracks)
        
        if n_det == 0 or n_trk == 0:
            return np.array([]).reshape(n_det, n_trk)
        
        C = np.ones((n_det, n_trk), dtype=np.float32)  # inicializa con costo máximo
        
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(tracks):
                # parte iou
                if trk.current_bbox:
                    iou = self._compute_iou(det.bbox, trk.current_bbox)
                    iou = max(0.0, min(1.0, iou))  # clamp a [0, 1]
                    iou_dist = 1.0 - iou
                else:
                    iou_dist = 1.0  # distancia máxima cuando no bbox
                
                # parte apariencia
                cos_dist = 1.0  # por defecto distancia máxima
                if features is not None and d_idx < len(features) and trk.get_feature() is not None:
                    try:
                        cos_dist = self._cosine_distance(features[d_idx], trk.get_feature())
                        cos_dist = max(0.0, min(2.0, cos_dist))  # clamp a rango razonable
                    except Exception as e:
                        logger.debug(f"fallo computar distancia coseno: {e}")
                        cos_dist = 1.0
                
                # costo mezclado con validación - german, esto es critical para matching
                try:
                    blended_cost = (
                        self.appearance_weight * cos_dist + 
                        (1.0 - self.appearance_weight) * iou_dist
                    )
                    # asegura costo finito y positivo
                    if np.isfinite(blended_cost) and blended_cost >= 0:
                        C[d_idx, t_idx] = blended_cost
                    else:
                        C[d_idx, t_idx] = 1.0  # fallback a costo máximo
                except Exception as e:
                    logger.debug(f"error computando costo mezclado: {e}")
                    C[d_idx, t_idx] = 1.0
        
        return C

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[DeepSORTTrack],
        features: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """asocia con matriz costo mezclada (sin cascadas separadas)."""
        if not tracks:
            return [], list(range(len(detections))), []
        
        if not detections:
            return [], [], list(range(len(tracks)))
        
        cost_matrix = self._build_cost_matrix(detections, tracks, features)
        
        # valida matriz costo
        if cost_matrix.size == 0:
            logger.warning("matriz costo vacía, retornando sin matches")
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # verifica valores inválidos
        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
            logger.warning("matriz costo contiene valores inválidos, usando fallback iou matching")
            return self._fallback_iou_matching(detections, tracks)
        
        # verifica si matriz costo es razonable
        if np.all(cost_matrix > self.max_distance):
            logger.warning("todos costos exceden umbral, usando fallback iou matching")
            return self._fallback_iou_matching(detections, tracks)
        
        try:
            # algoritmo húngaro
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.warning(f"algoritmo húngaro falló: {e}, usando fallback iou matching")
            return self._fallback_iou_matching(detections, tracks)
        
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        for r, c in zip(row_idx, col_idx):
            # filtra costo
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
        """fallback matching basado iou cuando matching sofisticado falla."""
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))
        
        # computa matriz iou
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(tracks):
                if trk.current_bbox:
                    iou_matrix[d_idx, t_idx] = self._compute_iou(det.bbox, trk.current_bbox)
        
        # matching greedy simple
        while True:
            # encuentra mejor match iou
            max_iou = np.max(iou_matrix)
            if max_iou < 0.1:  # umbral iou mínimo
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            det_idx, trk_idx = max_idx
            
            matches.append((det_idx, trk_idx))
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if trk_idx in unmatched_trks:
                unmatched_trks.remove(trk_idx)
            
            # remueve detección y track matched de consideración
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, trk_idx] = 0
        
        return matches, unmatched_dets, unmatched_trks
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """computa distancia coseno entre dos vectores feature."""
        try:
            # asegura inputs son arrays numpy válidos
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                return 1.0
            
            a = a.flatten()
            b = b.flatten()
            
            # verifica arrays vacíos o no coincidentes
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return 1.0
            
            # verifica valores inválidos
            if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isinf(a)) or np.any(np.isinf(b)):
                return 1.0
            
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            # evita división por cero
            if norm_a == 0 or norm_b == 0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
                return 1.0
            
            cosine_sim = dot_product / (norm_a * norm_b)
            
            # clamp similitud coseno a [-1, 1] para manejar errores numéricos
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            
            # convierte a distancia (0 = idéntico, 2 = opuesto)
            distance = 1.0 - cosine_sim
            
            # asegura resultado finito y positivo
            if not np.isfinite(distance) or distance < 0:
                return 1.0
            
            return distance
            
        except Exception:
            # retorna distancia máxima en cualquier error
            return 1.0
    
    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """computa intersection over union (iou) de dos bounding boxes - german, esto es para matching geométrico."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # computa intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # computa unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union 