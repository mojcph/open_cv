"""Runs two FrameSources in background threads and exposes timestamp-paired, rotated frames."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
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


def pair_nearest(
    left: Sequence[Frame], right: Sequence[Frame], max_dt_ns: int | None = None
) -> StereoFrame | None:
    """Pair the newest frame of the side that lags behind with its nearest-in-time partner.

    Using the lagging side's newest frame means its partner has already arrived on the
    other side. Returns None if either side is empty or the best gap exceeds max_dt_ns.
    """
    if not left or not right:
        return None
    if left[-1].timestamp_ns <= right[-1].timestamp_ns:
        anchor = left[-1]
        partner = min(right, key=lambda f: abs(f.timestamp_ns - anchor.timestamp_ns))
        pair = StereoFrame(anchor, partner)
    else:
        anchor = right[-1]
        partner = min(left, key=lambda f: abs(f.timestamp_ns - anchor.timestamp_ns))
        pair = StereoFrame(partner, anchor)
    if max_dt_ns is not None and abs(pair.dt_ns) > max_dt_ns:
        return None
    return pair


class StereoRig:
    SIDES = ("left", "right")

    def __init__(
        self,
        left: FrameSource,
        right: FrameSource,
        rotation: str = "none",
        max_pair_dt_ns: int | None = None,
        history: int = 4,
    ) -> None:
        rotate(np.zeros((2, 2, 3), np.uint8), rotation)  # validate early
        self._sources = {"left": left, "right": right}
        self.rotation = rotation
        self.max_pair_dt_ns = max_pair_dt_ns
        self._frames: dict[str, deque[Frame]] = {s: deque(maxlen=history) for s in self.SIDES}
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
            self._set_status(side, "Waiting for first frame")
            while not self._stop.is_set():
                frame = source.read()
                frame = Frame(rotate(frame.image, self.rotation), frame.timestamp_ns)
                with self._lock:
                    self._frames[side].append(frame)
                    self._status[side] = "Running"
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

    def _snapshot(self) -> tuple[list[Frame], list[Frame]]:
        with self._lock:
            return list(self._frames["left"]), list(self._frames["right"])

    def latest(self) -> StereoFrame | None:
        """Newest time-matched left/right pair, or None if none is within max_pair_dt_ns."""
        return pair_nearest(*self._snapshot(), self.max_pair_dt_ns)

    def sync_info(self) -> dict:
        """Timing of the best available pair, ignoring the limit — for /status."""
        pair = pair_nearest(*self._snapshot())
        dt_ns = pair.dt_ns if pair is not None else None
        limit = self.max_pair_dt_ns
        return {
            "pair_dt_ms": round(dt_ns / 1e6, 3) if dt_ns is not None else None,
            "max_pair_dt_ms": round(limit / 1e6, 3) if limit is not None else None,
            "in_sync": dt_ns is not None and (limit is None or abs(dt_ns) <= limit),
        }
