"""Depth pipeline: rectify -> disparity -> depth, and a background worker sharing results."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from stereovision.camera.stereo_rig import StereoFrame, StereoRig
from stereovision.config import DepthConfig
from stereovision.depth.disparity import compute_disparity, make_matcher
from stereovision.depth.measure import disparity_to_depth, region_distance
from stereovision.depth.rectify import Rectifier


@dataclass(frozen=True)
class DepthFrame:
    left: np.ndarray      # rectified left image, full resolution (BGR)
    depth: np.ndarray     # depth in mm at the working scale, NaN = no measurement
    scale: float          # depth-array pixels per full-resolution pixel
    timestamp_ns: int

    def distance_at(self, x: float, y: float, radius_px: int) -> float | None:
        """Distance (mm) at full-resolution pixel (x, y), median over radius_px."""
        s = self.scale
        return region_distance(self.depth, x * s, y * s, max(1, round(radius_px * s)))


class DepthEngine:
    """Stateless per-frame computation. rectifier=None means the frames are already
    rectified (used in tests)."""

    def __init__(self, rectifier: Rectifier | None, focal_px: float, baseline_mm: float,
                 cfg: DepthConfig) -> None:
        self.rectifier = rectifier
        self.focal_px = focal_px
        self.baseline_mm = baseline_mm
        self.cfg = cfg
        self._matcher = make_matcher(cfg)

    @classmethod
    def from_rectifier(cls, rectifier: Rectifier, cfg: DepthConfig) -> DepthEngine:
        cal = rectifier.calibration
        # Rectified baseline from P2 = [f 0 cx f*Tx]; Tx is negative for a right camera.
        baseline = -float(cal.P2[0, 3]) / float(cal.P2[0, 0])
        return cls(rectifier, cal.focal_px, baseline, cfg)

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
        return DepthFrame(left, depth, self.cfg.scale, pair.left.timestamp_ns)


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
