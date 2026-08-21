"""
Models layer - Data structures and entities.
"""
from .detection_models import Detection, DetectionResult, ModelInfo, SystemStats

__all__ = [
    'Detection',
    'DetectionResult',
    'SystemStats',
    'ModelInfo'
]
