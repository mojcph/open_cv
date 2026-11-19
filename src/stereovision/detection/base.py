"""Detector interface, independent of any ML library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the input image


class Detector(Protocol):
    @property
    def classes(self) -> tuple[str, ...]: ...

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR image."""
        ...
