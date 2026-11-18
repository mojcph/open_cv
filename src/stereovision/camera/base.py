"""Hardware-independent camera interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray      # HxWx3, BGR order (OpenCV convention)
    timestamp_ns: int      # sensor timestamp; used to pair left/right frames


class FrameSource(Protocol):
    """Anything that delivers frames: a Pi camera, a video file, a test fake."""

    def start(self) -> None: ...

    def read(self) -> Frame:
        """Block until the next frame is available and return it."""
        ...

    def close(self) -> None: ...
