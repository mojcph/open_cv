"""MJPEG streaming helpers shared by the web pages."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np
from flask import Response

from stereovision.camera.stereo_rig import StereoFrame, StereoRig

Renderer = Callable[[StereoFrame], np.ndarray]


def side_by_side(pair: StereoFrame) -> np.ndarray:
    return cv2.hconcat([pair.left.image, pair.right.image])


def encode_jpeg(image: np.ndarray, jpeg_quality: int = 80) -> bytes | None:
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return jpeg.tobytes() if ok else None


def mjpeg_stream(
    rig: StereoRig, render: Renderer, jpeg_quality: int = 80, max_fps: float = 30.0
) -> Iterator[bytes]:
    period = 1.0 / max_fps
    while True:
        pair = rig.latest()
        jpeg = encode_jpeg(render(pair), jpeg_quality) if pair is not None else None
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(period)


def mjpeg_response(
    rig: StereoRig, render: Renderer, jpeg_quality: int = 80, max_fps: float = 30.0
) -> Response:
    return Response(
        mjpeg_stream(rig, render, jpeg_quality, max_fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
