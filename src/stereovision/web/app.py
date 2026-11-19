"""Flask app factory: pages share one layout (base.html), each picture is its own stream."""

from __future__ import annotations

import cv2
from flask import Flask, jsonify, render_template, request, url_for

from stereovision.calibration.capture import CaptureStore
from stereovision.camera.stereo_rig import StereoFrame, StereoRig
from stereovision.config import DepthConfig
from stereovision.depth.engine import DepthWorker
from stereovision.depth.rectify import Rectifier, draw_epipolar_lines
from stereovision.detection.worker import DetectionWorker
from stereovision.web.stream import SharedRenderer, picture_response


def pair_renderer(rig: StereoRig, render) -> SharedRenderer[StereoFrame]:
    """SharedRenderer over the rig's newest synchronised pair."""
    return SharedRenderer(rig.latest, lambda pair: pair.left.timestamp_ns, render)


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
    """board + store enable /calibrate; rectifier /rectified; depth_worker /depth;
    detection_worker /objects. Pages that aren't enabled don't exist (404)."""
    app = Flask(__name__)
    pages: list[tuple[str, str]] = [("Live view", "index")]  # (label, endpoint) for the nav

    @app.context_processor
    def navigation():
        return {"nav": [(label, url_for(ep), request.endpoint == ep) for label, ep in pages]}

    live = pair_renderer(rig, lambda pair: (pair.left.image, pair.right.image))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video/<int:index>")
    def video(index: int):
        return picture_response(live, index, 2, jpeg_quality, preview_scale)

    @app.route("/status")
    def camera_status():
        return jsonify(rig.status() | rig.sync_info())

    if board is not None and store is not None:
        from stereovision.web.calibrate import create_blueprint

        app.register_blueprint(create_blueprint(
            rig, board, store, jpeg_quality=jpeg_quality, preview_scale=preview_scale
        ))
        pages.append(("Calibration", "calibrate.page"))

    if rectifier is not None:
        def render_rectified(pair: StereoFrame):
            left, right = rectifier.rectify(pair.left.image, pair.right.image)
            spacing = max(20, left.shape[0] // 24)
            return draw_epipolar_lines(left, spacing), draw_epipolar_lines(right, spacing)

        rectified_view = pair_renderer(rig, render_rectified)

        @app.route("/rectified")
        def rectified():
            return render_template("rectified.html")

        @app.route("/rectified/video/<int:index>")
        def rectified_video(index: int):
            return picture_response(rectified_view, index, 2, jpeg_quality, preview_scale,
                                    max_fps=15)

        pages.append(("Rectified", "rectified"))

    if depth_worker is not None:
        from stereovision.web.depth import create_blueprint as depth_blueprint

        app.register_blueprint(depth_blueprint(
            depth_worker, depth_cfg or DepthConfig(), jpeg_quality, preview_scale
        ))
        pages.append(("Distance", "depth.page"))

    if detection_worker is not None:
        from stereovision.web.objects import create_blueprint as objects_blueprint

        app.register_blueprint(objects_blueprint(
            detection_worker, model_name, jpeg_quality, preview_scale
        ))
        pages.append(("Objects", "objects.page"))

    return app
