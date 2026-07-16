"""Recognizer V1 package for 16x16 pressure-cushion posture recognition."""

from .feature_extractor import FEATURE_DIM, extract_features
from .recognizer import PosturePrediction, PrototypeRecognizer, RecognizerConfig
from .recognizer_api import Recognizer
from .seat_detector import SeatDetector, SeatPhase

__all__ = [
    "FEATURE_DIM",
    "PosturePrediction",
    "Recognizer",
    "PrototypeRecognizer",
    "RecognizerConfig",
    "SeatDetector",
    "SeatPhase",
    "extract_features",
]
