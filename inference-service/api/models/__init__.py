# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

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
