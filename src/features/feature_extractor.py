"""
Simple feature extractor for person re-identification.

This module provides basic appearance feature extraction for tracking applications.
For better performance, consider using dedicated ReID models like OSNet or ResNet.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
except ImportError:  # Torch may not be available in some environments
    torch = None  # type: ignore
    T = None  # type: ignore
    models = None  # type: ignore


class FeatureExtractor:
    """
    Simple feature extractor for person appearance.
    
    Uses basic image features like color histograms and HOG descriptors
    for appearance-based tracking. This is a lightweight baseline that
    can be replaced with more sophisticated ReID models.
    """
    
    def __init__(self, device: str = "auto", feature_dim: int = 512, method: str = "hog"):
        """
        Initialize feature extractor.
        
        Args:
            device: Device to use (auto, cpu, cuda)
            feature_dim: Dimension of output features (ignored for deep models)
            method: Feature extraction method ("hog" for classic HOG+color, "resnet50" for deep CNN)
        """
        self.device = device
        self.method = method.lower()

        # Classic HOG + color histogram
        if self.method.startswith("hog"):
            self.feature_dim = feature_dim
            self._setup_hog()
            logger.info(f"FeatureExtractor (HOG) initialized with {self.feature_dim}-D features")

        # Deep CNN features (e.g. ResNet-50 backbone)
        elif self.method in {"resnet50", "cnn"}:
            if torch is None or models is None:
                raise ImportError("PyTorch and torchvision are required for deep feature extraction")
            self._setup_deep_model()
            logger.info(f"FeatureExtractor (ResNet50) initialized with {self.feature_dim}-D features on {self.device}")

        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")
    
    def _setup_hog(self):
        """Setup HOG descriptor."""
        # HOG parameters
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
    
    def _setup_deep_model(self):
        """Load a pretrained CNN (ResNet-50) for appearance embeddings."""
        # Resolve device
        resolved_device = self.device
        if self.device == "auto":
            resolved_device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Load pretrained ResNet-50 (handle torchvision version differences)
        try:
            # torchvision >= 0.13
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            # Older torchvision versions
            self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = torch.nn.Identity()  # Remove final classification layer
        self.cnn.eval().to(resolved_device)

        # Output dimension of ResNet-50 penultimate layer
        self.feature_dim = 2048

        # Pre-processing pipeline (ImageNet stats)
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
        Extract appearance features from detected persons.
        
        Args:
            frame: Input frame
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            List of feature vectors for each bbox, or None if extraction fails
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
                    # Use zero vector as fallback
                    features.append(np.zeros(self.feature_dim, dtype=np.float32))
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _extract_single(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract features from a single bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Validate bounding box
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Resize to standard size
            roi = cv2.resize(roi, (64, 128))
            
            # Extract features
            color_features = self._extract_color_features(roi)
            hog_features = self._extract_hog_features(roi)
            
            # Combine features
            combined = np.concatenate([color_features, hog_features])
            
            # Normalize and resize to target dimension
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            
            # Pad or truncate to target dimension
            if len(combined) < self.feature_dim:
                feature = np.zeros(self.feature_dim, dtype=np.float32)
                feature[:len(combined)] = combined
            else:
                feature = combined[:self.feature_dim].astype(np.float32)
            
            return feature
            
        except Exception as e:
            logger.debug(f"Single feature extraction failed: {e}")
            return None
    
    def _extract_single_deep(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract deep CNN features for a single bounding box."""
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

            # Preprocess and forward pass
            with torch.no_grad():
                tensor = self.preprocess(roi).unsqueeze(0).to(self._deep_device)
                embedding = self.cnn(tensor)
                vector = embedding.squeeze().cpu().numpy()

            # Normalize
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            return vector.astype(np.float32)

        except Exception as e:
            logger.debug(f"Deep feature extraction failed: {e}")
            return None
    
    def _extract_color_features(self, roi: np.ndarray) -> np.ndarray:
        """Extract color histogram features."""
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Concatenate and normalize
            color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            color_features = color_features / (np.sum(color_features) + 1e-8)
            
            return color_features
            
        except Exception:
            # Return zero features on error
            return np.zeros(96, dtype=np.float32)
    
    def _extract_hog_features(self, roi: np.ndarray) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Compute HOG features
            hog_features = self.hog.compute(gray)
            
            if hog_features is not None:
                return hog_features.flatten()
            else:
                return np.zeros(3780, dtype=np.float32)  # Default HOG size
                
        except Exception:
            # Return zero features on error
            return np.zeros(3780, dtype=np.float32) 