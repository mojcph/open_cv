"""MJPEG streaming: one stream per picture, sharing a single render per frame.

A page shows up to two pictures (e.g. left/right camera, or image/heatmap). Each picture is
its own MJPEG stream so the browser can lay them out freely, but the expensive work (corner
detection, drawing, ...) must run once per frame, not once per picture. SharedRenderer
renders each new source item once and hands every stream its own picture of that result.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

import cv2
import numpy as np
from flask import Response, abort, current_app

T = TypeVar("T")


def encode_jpeg(image: np.ndarray, jpeg_quality: int = 80) -> bytes | None:
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return jpeg.tobytes() if ok else None


def downscale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class SharedRenderer(Generic[T]):
    """Renders each new item from `source` once; every caller gets the same pictures.

    source() returns the newest item (or None), key(item) identifies it (a timestamp),
    render(item) returns a tuple of pictures (one per stream).
    """

    def __init__(self, source: Callable[[], T | None], key: Callable[[T], int],
                 render: Callable[[T], tuple[np.ndarray, ...]]) -> None:
        self._source = source
        self._key = key
        self._render = render
        self._lock = threading.Lock()
        self._last_key: int | None = None
        self._pictures: tuple[np.ndarray, ...] = ()
        self.renders = 0  # how many times render() actually ran (for tests/diagnostics)

    def get(self) -> tuple[int, tuple[np.ndarray, ...]] | None:
        item = self._source()
        if item is None:
            return None
        k = self._key(item)
        with self._lock:
            if k != self._last_key:
                self._pictures = self._render(item)
                self._last_key = k
                self.renders += 1
            return k, self._pictures


def picture_stream(renderer: SharedRenderer, index: int, jpeg_quality: int = 80,
                   scale: float = 1.0, max_fps: float = 30.0) -> Iterator[bytes]:
    """MJPEG parts for picture `index`, one per new frame. scale shrinks only the stream."""
    period = 1.0 / max_fps
    last = None
    while True:
        got = renderer.get()
        if got is None or got[0] == last:
            time.sleep(0.02)
            continue
        last, pictures = got
        jpeg = encode_jpeg(downscale(pictures[index], scale), jpeg_quality)
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(period)


def picture_response(renderer: SharedRenderer, index: int, count: int, jpeg_quality: int = 80,
                     scale: float = 1.0, max_fps: float = 30.0) -> Response:
    if not 0 <= index < count:
        abort(404)
    cap = current_app.config.get("STREAM_MAX_FPS")  # app-wide limit from web.max_fps
    if cap:
        max_fps = min(max_fps, cap)
    return Response(picture_stream(renderer, index, jpeg_quality, scale, max_fps),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
