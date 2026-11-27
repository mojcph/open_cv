"""Share one camera's auto exposure and white balance with the other camera.

Left and right pictures must look the same for stereo matching to find the same patch in
both. Each camera's own auto exposure (AE) and auto white balance (AWB) drift apart a
little, so the leader (left) camera keeps AE/AWB on and publishes what it chose; the
follower (right) camera turns AE/AWB off and copies those values.

No picamera2 import here, so the logic can be tested without cameras.
"""

from __future__ import annotations

import threading

# Metadata keys copied from leader to follower. ExposureTime in µs, gains as factors.
EXPOSURE_KEYS = ("ExposureTime", "AnalogueGain", "ColourGains")


def _differs(a, b, tolerance: float) -> bool:
    """True if any value differs by more than tolerance (relative)."""
    if isinstance(a, (tuple, list)):
        return any(_differs(x, y, tolerance) for x, y in zip(a, b, strict=True))
    return abs(a - b) > tolerance * max(abs(a), abs(b), 1e-9)


class ExposureLink:
    """Thread-safe hand-over of exposure settings from the leader to the follower camera.

    tolerance: relative change (0.02 = 2 %) the leader's values must make before the
    follower is updated, so the follower isn't sent new controls on every frame.
    """

    def __init__(self, tolerance: float = 0.02) -> None:
        self.tolerance = tolerance
        self._lock = threading.Lock()
        self._latest: dict | None = None

    def publish(self, metadata: dict) -> None:
        """Leader: record the exposure values of its latest frame."""
        if not all(k in metadata for k in EXPOSURE_KEYS):
            return
        values = {"ExposureTime": int(metadata["ExposureTime"]),
                  "AnalogueGain": float(metadata["AnalogueGain"]),
                  "ColourGains": tuple(float(g) for g in metadata["ColourGains"])}
        with self._lock:
            self._latest = values

    def latest(self) -> dict | None:
        """The leader's latest values (for status display), or None before the first frame."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def follower_controls(self, applied: dict | None) -> dict | None:
        """Follower: controls to set now, or None if `applied` is still close enough.

        `applied` is what the follower last set (None = still on its own auto exposure).
        """
        latest = self.latest()
        if latest is None:
            return None
        if applied is not None and not any(
            _differs(latest[k], applied[k], self.tolerance) for k in EXPOSURE_KEYS
        ):
            return None
        return {"AeEnable": False, "AwbEnable": False, **latest}
