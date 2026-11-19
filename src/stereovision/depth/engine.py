"""Depth pipeline: rectify -> disparity -> depth, and a background worker sharing results."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from stereovision.camera.stereo_rig import StereoFrame, StereoRig
from stereovision.config import DepthConfig
from stereovision.depth.disparity import compute_disparity, make_matcher
from stereovision.depth.measure import (
    disparity_to_depth,
    range_from_midpoint,
    region_distance,
)
from stereovision.depth.rectify import Rectifier


@dataclass(frozen=True)
class Optics:
    """Rectified-left camera geometry, full-resolution pixels."""

    focal_px: float
    cx: float
    cy: float
    baseline_mm: float


@dataclass(frozen=True)
class DepthFrame:
    left: np.ndarray      # rectified left image, full resolution (BGR)
    depth: np.ndarray     # depth Z in mm (along the camera axis) at the working scale, NaN = none
    scale: float          # depth-array pixels per full-resolution pixel
    timestamp_ns: int
    optics: Optics | None = None  # None: distance_at falls back to plain depth Z

    def depth_at(self, x: float, y: float, radius_px: int) -> float | None:
        """Depth Z (mm) at full-resolution pixel (x, y): median over radius_px."""
        s = self.scale
        return region_distance(self.depth, x * s, y * s, max(1, round(radius_px * s)))

    def range_at(self, x: float, y: float, z: float) -> float:
        """Straight-line distance from the midpoint between the cameras, for depth z at
        full-resolution pixel (x, y)."""
        if self.optics is None:
            return z
        o = self.optics
        return range_from_midpoint(x, y, z, o.focal_px, o.cx, o.cy, o.baseline_mm)

    def distance_at(self, x: float, y: float, radius_px: int) -> float | None:
        """Distance (mm) from the midpoint between the cameras to what is seen at (x, y)."""
        z = self.depth_at(x, y, radius_px)
        return None if z is None else self.range_at(x, y, z)


class DepthEngine:
    """Stateless per-frame computation. rectifier=None means the frames are already
    rectified (used in tests)."""

    def __init__(self, rectifier: Rectifier | None, focal_px: float, baseline_mm: float,
                 cfg: DepthConfig, cx: float | None = None, cy: float | None = None) -> None:
        self.rectifier = rectifier
        self.focal_px = focal_px
        self.baseline_mm = baseline_mm
        self.cfg = cfg
        self.cx, self.cy = cx, cy  # principal point; None = image centre
        self._matcher = make_matcher(cfg)

    @classmethod
    def from_rectifier(cls, rectifier: Rectifier, cfg: DepthConfig) -> DepthEngine:
        cal = rectifier.calibration
        # Rectified baseline from P2 = [f 0 cx f*Tx]; Tx is negative for a right camera.
        baseline = -float(cal.P2[0, 3]) / float(cal.P2[0, 0])
        return cls(rectifier, cal.focal_px, baseline, cfg,
                   cx=float(cal.P1[0, 2]), cy=float(cal.P1[1, 2]))

    @property
    def min_distance_mm(self) -> float:
        """Closest distance the disparity search range can measure."""
        return self.focal_px * self.cfg.scale * self.baseline_mm / self.cfg.num_disparities

    def compute(self, pair: StereoFrame) -> DepthFrame:
        left, right = pair.left.image, pair.right.image
        if self.rectifier is not None:
            left, right = self.rectifier.rectify(left, right)
        disparity = compute_disparity(self._matcher, left, right, self.cfg.scale)
        depth = disparity_to_depth(disparity, self.focal_px, self.baseline_mm)
        h, w = left.shape[:2]
        optics = Optics(self.focal_px,
                        (w - 1) / 2 if self.cx is None else self.cx,
                        (h - 1) / 2 if self.cy is None else self.cy,
                        self.baseline_mm)
        return DepthFrame(left, depth, self.cfg.scale, pair.left.timestamp_ns, optics)


class DepthWorker:
    """Computes depth for the newest frame pair in one background thread.

    Consumers call latest(); the worker pauses when nobody has asked for `idle_s` seconds,
    so depth only costs CPU while a depth page is open.
    """

    def __init__(self, rig: StereoRig, engine: DepthEngine, idle_s: float = 5.0) -> None:
        self.rig = rig
        self.engine = engine
        self.idle_s = idle_s
        self._latest: DepthFrame | None = None
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="depth")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def latest(self) -> DepthFrame | None:
        self._last_request = time.monotonic()
        with self._lock:
            return self._latest

    def _run(self) -> None:
        last_ts = None
        while not self._stop.is_set():
            if time.monotonic() - self._last_request > self.idle_s:
                time.sleep(0.1)
                continue
            pair = self.rig.latest()
            if pair is None or pair.left.timestamp_ns == last_ts:
                time.sleep(0.01)
                continue
            last_ts = pair.left.timestamp_ns
            start = time.perf_counter()
            try:
                frame = self.engine.compute(pair)
            except Exception as exc:  # show on the page instead of killing the thread
                self.error = str(exc)
                time.sleep(0.5)
                continue
            self.error = None
            elapsed = time.perf_counter() - start
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / elapsed) if self.fps else 1.0 / elapsed
            with self._lock:
                self._latest = frame
