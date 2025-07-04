"""
extractor features simple para re-identificación personas.

este módulo provee extracción básica features apariencia para aplicaciones tracking.
para mejor rendimiento, considera usar modelos reid dedicados como osnet o resnet.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
except ImportError:  # torch puede no estar disponible en algunos entornos
    torch = None  # type: ignore
    T = None  # type: ignore
    models = None  # type: ignore


class FeatureExtractor:
    """
    extractor features simple para apariencia personas.
    
    usa features imagen básicas como histogramas color y descriptores hog
    para tracking basado apariencia. esto es baseline ligero que
    puede reemplazarse con modelos reid más sofisticados.
    """
    
    def __init__(self, device: str = "auto", feature_dim: int = 512, method: str = "hog"):
        """
        inicializa extractor features.
        
        args:
            device: dispositivo usar (auto, cpu, cuda)
            feature_dim: dimensión features salida (ignorado para modelos deep)
            method: método extracción features ("hog" para hog+color clásico, "resnet50" para cnn deep)
        """
        self.device = device
        self.method = method.lower()

        # hog clásico + histograma color
        if self.method.startswith("hog"):
            self.feature_dim = feature_dim
            self._setup_hog()
            logger.info(f"featureextractor (hog) inicializado con features {self.feature_dim}-d")

        # features cnn deep (ej. backbone resnet-50)
        elif self.method in {"resnet50", "cnn"}:
            if torch is None or models is None:
                raise ImportError("pytorch y torchvision requeridos para extracción features deep")
            self._setup_deep_model()
            logger.info(f"featureextractor (resnet50) inicializado con features {self.feature_dim}-d en {self.device}")

        else:
            raise ValueError(f"método extracción features desconocido: {self.method}")
    
    def _setup_hog(self):
        """configura descriptor hog."""
        # parámetros hog
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
    
    def _setup_deep_model(self):
        """carga cnn preentrenada (resnet-50) para embeddings apariencia."""
        # resuelve dispositivo
        resolved_device = self.device
        if self.device == "auto":
            resolved_device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # carga resnet-50 preentrenada (maneja diferencias versión torchvision)
        try:
            # torchvision >= 0.13
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            # versiones torchvision más viejas
            self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = torch.nn.Identity()  # remueve capa clasificación final
        self.cnn.eval().to(resolved_device)

        # dimensión salida capa penúltima resnet-50
        self.feature_dim = 2048

        # pipeline pre-procesamiento (stats imagenet)
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._deep_device = resolved_device
    
    def extract_features(
        self, 
        frame: np.ndarray, 
        bboxes: List[Tuple[int, int, int, int]]
    ) -> Optional[List[np.ndarray]]:
        """
        extrae features apariencia de personas detectadas.
        
        args:
            frame: frame entrada
            bboxes: lista bounding boxes [(x1, y1, x2, y2), ...]
            
        returns:
            lista vectores features para cada bbox, o none si extracción falla
        """
        if not bboxes or frame is None or frame.size == 0:
            return None
        
        try:
            features = []
            
            for bbox in bboxes:
                if self.method.startswith("hog"):
                    feature = self._extract_single(frame, bbox)
                else:
                    feature = self._extract_single_deep(frame, bbox)
                if feature is not None:
                    features.append(feature)
                else:
                    # usa vector cero como fallback
                    features.append(np.zeros(self.feature_dim, dtype=np.float32))
            
            return features
            
        except Exception as e:
            logger.warning(f"extracción features falló: {e}")
            return None
    
    def _extract_single(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """extrae features de un single bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            
            # valida bounding box
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # extrae roi
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # redimensiona a tamaño estándar
            roi = cv2.resize(roi, (64, 128))
            
            # extrae features
            color_features = self._extract_color_features(roi)
            hog_features = self._extract_hog_features(roi)
            
            # combina features
            combined = np.concatenate([color_features, hog_features])
            
            # normaliza y redimensiona a dimensión objetivo
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            
            # rellena o trunca a dimensión objetivo
            if len(combined) < self.feature_dim:
                feature = np.zeros(self.feature_dim, dtype=np.float32)
                feature[:len(combined)] = combined
            else:
                feature = combined[:self.feature_dim].astype(np.float32)
            
            return feature
            
        except Exception as e:
            logger.debug(f"extracción single feature falló: {e}")
            return None
    
    def _extract_single_deep(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """extrae features cnn deep para un single bounding box."""
        if torch is None:
            return None

        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # preprocesa y forward pass
            with torch.no_grad():
                tensor = self.preprocess(roi).unsqueeze(0).to(self._deep_device)
                embedding = self.cnn(tensor)
                vector = embedding.squeeze().cpu().numpy()

            # normaliza
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            return vector.astype(np.float32)

        except Exception as e:
            logger.debug(f"extracción features deep falló: {e}")
            return None
    
    def _extract_color_features(self, roi: np.ndarray) -> np.ndarray:
        """extrae features histograma color."""
        try:
            # convierte a hsv para mejor representación color
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # calcula histogramas para cada canal
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # concatena y normaliza
            color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            color_features = color_features / (np.sum(color_features) + 1e-8)
            
            return color_features
            
        except Exception:
            # retorna features cero en error
            return np.zeros(96, dtype=np.float32)
    
    def _extract_hog_features(self, roi: np.ndarray) -> np.ndarray:
        """extrae features hog (histogram of oriented gradients)."""
        try:
            # convierte a escala grises
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # computa features hog
            hog_features = self.hog.compute(gray)
            
            if hog_features is not None:
                return hog_features.flatten()
            else:
                return np.zeros(3780, dtype=np.float32)  # tamaño hog por defecto
                
        except Exception:
            # retorna features cero en error
            return np.zeros(3780, dtype=np.float32) 