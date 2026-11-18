"""Calibration capture page: live view with detected board corners, manual and auto capture."""

from __future__ import annotations

import threading

import cv2
from flask import Blueprint, jsonify, render_template

from stereovision.calibration.capture import (
    AutoCapture,
    CaptureStore,
    check_pair,
    detect_pair,
    detect_view,
    draw_view,
)
from stereovision.camera.stereo_rig import StereoRig
from stereovision.web.stream import mjpeg_response


def create_blueprint(
    rig: StereoRig,
    board: cv2.aruco.CharucoBoard,
    store: CaptureStore,
    min_common: int = 8,
    jpeg_quality: int = 80,
    preview_scale: float = 1.0,
) -> Blueprint:
    bp = Blueprint("calibrate", __name__, url_prefix="/calibrate")
    lock = threading.Lock()
    # Seed auto-capture with the poses already on disk, so it won't re-save them.
    auto = AutoCapture(saved_views=[detect_view(board, left) for left, _ in store.load_pairs()])
    state = {"auto": False, "last": None}

    def save(pair, det) -> dict:
        """Save a pair; caller holds the lock. Returns the JSON result."""
        reason = check_pair(det, min_common)
        result = dict(left=det.left.count, right=det.right.count, common=det.common)
        if reason is None:
            try:
                result["index"] = store.save(pair.left.image, pair.right.image)
                auto.saved(det.left)
            except (ValueError, OSError) as exc:
                reason = str(exc)
        result.update(saved=reason is None, reason=reason, count=store.count())
        if reason is None:
            state["last"] = result
        return result

    def render_overlay(pair):
        det = detect_pair(board, pair.left.image, pair.right.image)
        good = check_pair(det, min_common) is None
        with lock:
            if state["auto"] and auto.update(det, pair.left.image.shape[1]):
                save(pair, det)
        return cv2.hconcat([draw_view(pair.left.image, det.left, good),
                            draw_view(pair.right.image, det.right, good)])

    @bp.route("/")
    def page():
        return render_template("calibrate.html", count=store.count(), folder=str(store.folder))

    @bp.route("/video")
    def video():
        # Corner detection costs ~30-60 ms per pair, so the preview runs at a lower rate.
        # Auto-capture runs inside this stream: it works while the page is open.
        return mjpeg_response(rig, render_overlay, jpeg_quality, max_fps=10, scale=preview_scale)

    @bp.route("/state")
    def get_state():
        with lock:
            return jsonify(count=store.count(), mismatch=store.meta_mismatch(), auto=state["auto"],
                           message=auto.message if state["auto"] else None, last=state["last"])

    @bp.route("/auto/<mode>", methods=["POST"])
    def set_auto(mode: str):
        if mode not in ("on", "off"):
            return jsonify(error="mode must be on or off"), 400
        with lock:
            state["auto"] = mode == "on"
        return jsonify(auto=state["auto"])

    @bp.route("/capture", methods=["POST"])
    def capture():
        pair = rig.latest()
        if pair is None:
            return jsonify(saved=False, reason="no synchronised frame pair yet",
                           count=store.count())
        det = detect_pair(board, pair.left.image, pair.right.image)
        with lock:
            return jsonify(save(pair, det))

    @bp.route("/undo", methods=["POST"])
    def undo():
        with lock:
            removed = store.remove_last()
            if removed is not None:
                auto.forget_last()
        return jsonify(removed=removed, count=store.count())

    return bp
