"""Runs the detector on depth frames and attaches a distance to every detection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from stereovision.camera.stereo_rig import StereoRig
from stereovision.depth.engine import DepthEngine, DepthFrame
from stereovision.depth.measure import box_distance
from stereovision.detection.base import Detection, Detector


@dataclass(frozen=True)
class MeasuredObject:
    detection: Detection
    distance_mm: float | None  # straight line from the camera midpoint; None = no depth
    depth_mm: float | None = None  # median depth Z (along the camera axis) inside the shrunk box


@dataclass(frozen=True)
class ObjectFrame:
    frame: DepthFrame                 # the frame the detector ran on (rectified left + depth)
    objects: list[MeasuredObject]
    depth_ms: float
    detect_ms: float


def measure_objects(frame: DepthFrame, detections: list[Detection]) -> list[MeasuredObject]:
    """Distance for each box: median depth inside the box, then the straight-line distance
    from the camera midpoint to the box centre. Boxes are in full-resolution pixels."""
    s = frame.scale
    out = []
    for d in detections:
        z = box_distance(frame.depth, tuple(v * s for v in d.box))
        x0, y0, x1, y1 = d.box
        distance = None if z is None else frame.range_at((x0 + x1) / 2, (y0 + y1) / 2, z)
        out.append(MeasuredObject(d, distance, z))
    return out


class DetectionWorker:
    """Background thread: newest frame pair -> depth -> detector -> distances.

    Depth is computed here, only for the frames detection runs on, one step after the
    other: running a separate depth thread at the same time would make both fight over
    the CPU cores. Idles when nobody has asked for results for idle_s seconds.
    """

    def __init__(self, rig: StereoRig, engine: DepthEngine, detector: Detector,
                 idle_s: float = 5.0):
        self.rig = rig
        self.engine = engine
        self.detector = detector
        self.idle_s = idle_s
        self._latest: ObjectFrame | None = None
        self._last_request = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps = 0.0
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def latest(self) -> ObjectFrame | None:
        self._last_request = time.monotonic()
        with self._lock:
            return self._latest

    def peek(self) -> ObjectFrame | None:
        """Newest result without counting as a request (doesn't keep the worker awake)."""
        with self._lock:
            return self._latest

    @property
    def active(self) -> bool:
        """True while someone has asked for results within the last idle_s seconds."""
        return time.monotonic() - self._last_request <= self.idle_s

    def _run(self) -> None:
        last_ts = None
        while not self._stop.is_set():
            if not self.active:
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
                mid = time.perf_counter()
                objects = measure_objects(frame, self.detector.detect(frame.left))
            except Exception as exc:  # show on the page instead of killing the thread
                self.error = str(exc)
                time.sleep(0.5)
                continue
            self.error = None
            end = time.perf_counter()
            elapsed = end - start
            self.fps = 0.8 * self.fps + 0.2 / elapsed if self.fps else 1.0 / elapsed
            with self._lock:
                self._latest = ObjectFrame(frame, objects, (mid - start) * 1000,
                                           (end - mid) * 1000)
