"""Detectors for tests and for running without a model."""

from __future__ import annotations

import numpy as np

from stereovision.detection.base import Detection


class NullDetector:
    """Finds nothing."""

    classes: tuple[str, ...] = ()

    def detect(self, image: np.ndarray) -> list[Detection]:
        return []


class FixedDetector:
    """Always returns the same detections (boxes as fractions of the image size)."""

    def __init__(self, detections: list[tuple[str, float, tuple[float, float, float, float]]]):
        self._dets = detections
        self.classes = tuple(dict.fromkeys(label for label, _, _ in detections))

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(label, conf, (fx0 * w, fy0 * h, fx1 * w, fy1 * h))
                for label, conf, (fx0, fy0, fx1, fy1) in self._dets]
