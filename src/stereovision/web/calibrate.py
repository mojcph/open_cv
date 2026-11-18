"""Calibration capture page: live view with detected board corners and a Capture button."""

from __future__ import annotations

import cv2
from flask import Blueprint, jsonify, render_template

from stereovision.calibration.capture import CaptureStore, check_pair, detect_pair, draw_view
from stereovision.camera.stereo_rig import StereoRig
from stereovision.web.stream import mjpeg_response


def create_blueprint(
    rig: StereoRig,
    board: cv2.aruco.CharucoBoard,
    store: CaptureStore,
    min_common: int = 8,
    jpeg_quality: int = 80,
) -> Blueprint:
    bp = Blueprint("calibrate", __name__, url_prefix="/calibrate")

    def render_overlay(pair):
        det = detect_pair(board, pair.left.image, pair.right.image)
        good = check_pair(det, min_common) is None
        return cv2.hconcat([draw_view(pair.left.image, det.left, good),
                            draw_view(pair.right.image, det.right, good)])

    @bp.route("/")
    def page():
        return render_template("calibrate.html", count=store.count(), folder=str(store.folder))

    @bp.route("/video")
    def video():
        # Corner detection costs ~10-30 ms per image, so the preview runs at a lower rate.
        return mjpeg_response(rig, render_overlay, jpeg_quality, max_fps=10)

    @bp.route("/state")
    def state():
        return jsonify(count=store.count(), mismatch=store.meta_mismatch())

    @bp.route("/capture", methods=["POST"])
    def capture():
        pair = rig.latest()
        if pair is None:
            return jsonify(saved=False, reason="no synchronised frame pair yet",
                           count=store.count())
        det = detect_pair(board, pair.left.image, pair.right.image)
        reason = check_pair(det, min_common)
        result = dict(left=det.left.count, right=det.right.count, common=det.common)
        if reason is None:
            try:
                result["index"] = store.save(pair.left.image, pair.right.image)
            except (ValueError, OSError) as exc:
                reason = str(exc)
        return jsonify(saved=reason is None, reason=reason, count=store.count(), **result)

    @bp.route("/undo", methods=["POST"])
    def undo():
        removed = store.remove_last()
        return jsonify(removed=removed, count=store.count())

    return bp
