"""Flask app factory: live side-by-side MJPEG stream, camera status, calibration capture."""

from __future__ import annotations

import cv2
from flask import Flask, jsonify, render_template

from stereovision.calibration.capture import CaptureStore
from stereovision.camera.stereo_rig import StereoRig
from stereovision.web.stream import mjpeg_response, side_by_side


def create_app(
    rig: StereoRig,
    jpeg_quality: int = 80,
    board: cv2.aruco.CharucoBoard | None = None,
    store: CaptureStore | None = None,
) -> Flask:
    """board + store enable the /calibrate pages."""
    app = Flask(__name__)
    calibration_enabled = board is not None and store is not None

    @app.route("/")
    def index():
        return render_template("index.html", calibration_enabled=calibration_enabled)

    @app.route("/video")
    def video():
        return mjpeg_response(rig, side_by_side, jpeg_quality)

    @app.route("/status")
    def camera_status():
        return jsonify(rig.status() | rig.sync_info())

    if calibration_enabled:
        from stereovision.web.calibrate import create_blueprint

        app.register_blueprint(create_blueprint(rig, board, store, jpeg_quality=jpeg_quality))

    return app
