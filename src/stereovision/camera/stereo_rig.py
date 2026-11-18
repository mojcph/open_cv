"""Runs two FrameSources in background threads and exposes the latest rotated pair."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from stereovision.camera.base import Frame, FrameSource

_ROTATE_CODES = {
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def rotate(image: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "none":
        return image
    try:
        return cv2.rotate(image, _ROTATE_CODES[rotation])
    except KeyError:
        raise ValueError(f"unknown rotation {rotation!r}") from None


@dataclass(frozen=True)
class StereoFrame:
    left: Frame
    right: Frame

    @property
    def dt_ns(self) -> int:
        """Right minus left sensor timestamp; large values mean the pair is out of sync."""
        return self.right.timestamp_ns - self.left.timestamp_ns


class StereoRig:
    SIDES = ("left", "right")

    def __init__(self, left: FrameSource, right: FrameSource, rotation: str = "none") -> None:
        rotate(np.zeros((2, 2, 3), np.uint8), rotation)  # validate early
        self._sources = {"left": left, "right": right}
        self.rotation = rotation
        self._latest: dict[str, Frame | None] = {"left": None, "right": None}
        self._status = {"left": "Starting", "right": "Starting"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for side in self.SIDES:
            t = threading.Thread(target=self._run, args=(side,), daemon=True, name=f"cam-{side}")
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    def _set_status(self, side: str, value: str) -> None:
        with self._lock:
            self._status[side] = value

    def _run(self, side: str) -> None:
        source = self._sources[side]
        try:
            source.start()
            self._set_status(side, "Running")
            while not self._stop.is_set():
                frame = source.read()
                frame = Frame(rotate(frame.image, self.rotation), frame.timestamp_ns)
                with self._lock:
                    self._latest[side] = frame
        except Exception as exc:  # report any camera failure on /status instead of dying silently
            self._set_status(side, f"ERROR: {exc}")
        finally:
            try:
                source.close()
            except Exception as exc:
                self._set_status(side, f"ERROR closing: {exc}")

    def status(self) -> dict[str, str]:
        with self._lock:
            return dict(self._status)

    def latest(self) -> StereoFrame | None:
        """Most recent left/right frames, or None until both cameras have delivered one."""
        with self._lock:
            left, right = self._latest["left"], self._latest["right"]
        if left is None or right is None:
            return None
        return StereoFrame(left, right)
