"""Picamera2-backed FrameSource. The only module that imports picamera2/libcamera."""

from __future__ import annotations

import time

from libcamera import controls
from picamera2 import Picamera2

from stereovision.camera.base import Frame

SYNC_MODES = {
    "off": None,
    "server": controls.rpi.SyncModeEnum.Server,
    "client": controls.rpi.SyncModeEnum.Client,
}


class PicamSource:
    """One Pi camera with fixed sensor mode, frame rate and manual focus.

    With sync="server"/"client", libcamera software sync aligns the frame timing of two
    cameras. read() skips frames until the camera reports SyncReady.
    """

    def __init__(
        self,
        camera_num: int,
        width: int,
        height: int,
        sensor_size: tuple[int, int] = (2304, 1296),
        frame_rate: float = 30.0,
        lens_position: float | None = 1.0,
        sync: str = "off",
        sync_timeout_s: float = 10.0,
        buffer_count: int = 4,
    ) -> None:
        if sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {list(SYNC_MODES)}, got {sync!r}")
        self.camera_num = camera_num
        self.size = (width, height)
        self.sensor_size = tuple(sensor_size)
        self.frame_rate = frame_rate
        self.lens_position = lens_position
        self.sync = sync
        self.sync_timeout_s = sync_timeout_s
        self.buffer_count = buffer_count
        self._camera: Picamera2 | None = None
        self._synced = False

    def _controls(self) -> dict:
        frame_us = round(1e6 / self.frame_rate)
        ctrl = {"FrameDurationLimits": (frame_us, frame_us)}
        if self.lens_position is not None:  # None = keep continuous autofocus
            ctrl["AfMode"] = controls.AfModeEnum.Manual
            ctrl["LensPosition"] = float(self.lens_position)
        if SYNC_MODES[self.sync] is not None:
            ctrl["SyncMode"] = SYNC_MODES[self.sync]
        return ctrl

    def start(self) -> None:
        camera = Picamera2(self.camera_num)
        try:
            # "RGB888" in libcamera terms is BGR byte order in memory — what OpenCV expects.
            config = camera.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                sensor={"output_size": self.sensor_size, "bit_depth": 10},
                controls=self._controls(),
                buffer_count=self.buffer_count,
            )
            camera.configure(config)
            camera.start()
        except Exception:
            camera.close()
            raise
        self._camera = camera
        self._synced = self.sync == "off"

    def read(self) -> Frame:
        if self._camera is None:
            raise RuntimeError(f"camera {self.camera_num} not started")
        deadline = time.monotonic() + self.sync_timeout_s
        while True:
            request = self._camera.capture_request()
            try:
                metadata = request.get_metadata()
                if not self._synced:
                    if not metadata.get("SyncReady", False):
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                f"camera {self.camera_num}: software sync not ready after "
                                f"{self.sync_timeout_s:.0f} s (is the other camera running?)"
                            )
                        continue
                    self._synced = True
                image = request.make_array("main")
            finally:
                request.release()
            return Frame(image=image, timestamp_ns=int(metadata["SensorTimestamp"]))

    def close(self) -> None:
        if self._camera is None:
            return
        camera, self._camera = self._camera, None
        try:
            camera.stop()
        finally:
            camera.close()
