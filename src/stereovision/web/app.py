"""Flask app factory: live side-by-side MJPEG stream and camera status."""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
from flask import Flask, Response, jsonify, render_template

from stereovision.camera.stereo_rig import StereoFrame, StereoRig


def encode_side_by_side(pair: StereoFrame, jpeg_quality: int = 80) -> bytes | None:
    """Concatenate left|right and JPEG-encode. Returns None if encoding fails."""
    combined = cv2.hconcat([pair.left.image, pair.right.image])
    ok, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return jpeg.tobytes() if ok else None


def mjpeg_stream(rig: StereoRig, jpeg_quality: int, max_fps: float = 30.0) -> Iterator[bytes]:
    period = 1.0 / max_fps
    while True:
        pair = rig.latest()
        jpeg = encode_side_by_side(pair, jpeg_quality) if pair is not None else None
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(period)


def create_app(rig: StereoRig, jpeg_quality: int = 80) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video")
    def video():
        return Response(
            mjpeg_stream(rig, jpeg_quality),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status")
    def camera_status():
        return jsonify(rig.status() | rig.sync_info())

    return app
