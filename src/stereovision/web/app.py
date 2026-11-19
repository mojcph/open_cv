"""Flask app factory: live side-by-side MJPEG stream, camera status, calibration capture."""

from __future__ import annotations

import cv2
from flask import Flask, jsonify, render_template

from stereovision.calibration.capture import CaptureStore
from stereovision.camera.stereo_rig import StereoFrame, StereoRig
from stereovision.config import DepthConfig
from stereovision.depth.engine import DepthWorker
from stereovision.depth.rectify import Rectifier, draw_epipolar_lines
from stereovision.detection.worker import DetectionWorker
from stereovision.web.stream import mjpeg_response, side_by_side


def create_app(
    rig: StereoRig,
    jpeg_quality: int = 80,
    board: cv2.aruco.CharucoBoard | None = None,
    store: CaptureStore | None = None,
    preview_scale: float = 1.0,
    rectifier: Rectifier | None = None,
    depth_worker: DepthWorker | None = None,
    depth_cfg: DepthConfig | None = None,
    detection_worker: DetectionWorker | None = None,
    model_name: str = "",
) -> Flask:
    """board + store enable /calibrate; rectifier enables /rectified; depth_worker /depth."""
    app = Flask(__name__)
    calibration_enabled = board is not None and store is not None

    @app.route("/")
    def index():
        return render_template("index.html", calibration_enabled=calibration_enabled,
                               rectified_enabled=rectifier is not None,
                               depth_enabled=depth_worker is not None,
                               objects_enabled=detection_worker is not None)

    if rectifier is not None:
        def render_rectified(pair: StereoFrame):
            left, right = rectifier.rectify(pair.left.image, pair.right.image)
            spacing = max(20, left.shape[0] // 24)
            return cv2.hconcat([draw_epipolar_lines(left, spacing),
                                draw_epipolar_lines(right, spacing)])

        @app.route("/rectified")
        def rectified():
            return render_template("rectified.html")

        @app.route("/rectified/video")
        def rectified_video():
            return mjpeg_response(rig, render_rectified, jpeg_quality, max_fps=15,
                                  scale=preview_scale)

    @app.route("/video")
    def video():
        return mjpeg_response(rig, side_by_side, jpeg_quality, scale=preview_scale)

    @app.route("/status")
    def camera_status():
        return jsonify(rig.status() | rig.sync_info())

    if calibration_enabled:
        from stereovision.web.calibrate import create_blueprint

        app.register_blueprint(create_blueprint(
            rig, board, store, jpeg_quality=jpeg_quality, preview_scale=preview_scale
        ))

    if depth_worker is not None:
        from stereovision.web.depth import create_blueprint as depth_blueprint

        app.register_blueprint(depth_blueprint(
            depth_worker, depth_cfg or DepthConfig(), jpeg_quality, preview_scale
        ))

    if detection_worker is not None:
        from stereovision.web.objects import create_blueprint as objects_blueprint

        app.register_blueprint(objects_blueprint(
            detection_worker, model_name, jpeg_quality, preview_scale
        ))

    return app
