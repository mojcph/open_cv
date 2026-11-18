"""In-memory FrameSource for tests and offline development."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable

import numpy as np

from stereovision.camera.base import Frame


class FakeSource:
    """Cycles through the given images. Optionally raises on start to simulate a failure."""

    def __init__(
        self,
        images: Iterable[np.ndarray],
        fps: float = 30.0,
        start_error: Exception | None = None,
    ) -> None:
        self._images = list(images)
        if not self._images:
            raise ValueError("FakeSource needs at least one image")
        self._period_ns = int(1e9 / fps)
        self._start_error = start_error
        self._counter = itertools.count()
        self.started = False
        self.closed = False

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    def read(self) -> Frame:
        if not self.started or self.closed:
            raise RuntimeError("FakeSource not running")
        i = next(self._counter)
        time.sleep(self._period_ns / 1e9)
        image = self._images[i % len(self._images)]
        return Frame(image=image.copy(), timestamp_ns=i * self._period_ns)

    def close(self) -> None:
        self.closed = True
