"""Picamera2-backed FrameSource. The only module that imports picamera2."""

from __future__ import annotations

import time

from picamera2 import Picamera2

from stereovision.camera.base import Frame


class PicamSource:
    def __init__(self, camera_num: int, width: int, height: int, buffer_count: int = 4) -> None:
        self.camera_num = camera_num
        self.size = (width, height)
        self.buffer_count = buffer_count
        self._camera: Picamera2 | None = None

    def start(self) -> None:
        camera = Picamera2(self.camera_num)
        try:
            # "RGB888" in libcamera terms is BGR byte order in memory — what OpenCV expects.
            config = camera.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                buffer_count=self.buffer_count,
            )
            camera.configure(config)
            camera.start()
        except Exception:
            camera.close()
            raise
        self._camera = camera

    def read(self) -> Frame:
        if self._camera is None:
            raise RuntimeError(f"camera {self.camera_num} not started")
        request = self._camera.capture_request()
        try:
            image = request.make_array("main")
            timestamp = request.get_metadata().get("SensorTimestamp", time.monotonic_ns())
        finally:
            request.release()
        return Frame(image=image, timestamp_ns=int(timestamp))

    def close(self) -> None:
        if self._camera is None:
            return
        camera, self._camera = self._camera, None
        try:
            camera.stop()
        finally:
            camera.close()
