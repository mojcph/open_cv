"""Depth page: rectified left image with a measurement crosshair, plus a depth heatmap."""

from __future__ import annotations

import threading

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

from stereovision.config import DepthConfig
from stereovision.depth.engine import DepthFrame, DepthWorker
from stereovision.depth.measure import colorize_depth
from stereovision.overlay import format_distance
from stereovision.web.stream import SharedRenderer, picture_response

WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def draw_measurement(image: np.ndarray, x: int, y: int, mm: float | None, radius: int) -> None:
    """Crosshair + label, drawn in place (outlined so it reads on any background)."""
    size = max(12, image.shape[1] // 40)
    thick = max(2, image.shape[1] // 400)
    for colour, t in ((BLACK, thick + 2), (WHITE, thick)):
        cv2.line(image, (x - size, y), (x + size, y), colour, t)
        cv2.line(image, (x, y - size), (x, y + size), colour, t)
        cv2.circle(image, (x, y), radius, colour, max(1, t - 1))
    scale = image.shape[1] / 600
    org = (min(x + size, image.shape[1] - int(160 * scale)), max(y - size, int(30 * scale)))
    text = format_distance(mm)
    for colour, t in ((BLACK, int(6 * scale)), (WHITE, max(2, int(2 * scale)))):
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, colour, t,
                    cv2.LINE_AA)


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def create_blueprint(
    worker: DepthWorker,
    cfg: DepthConfig,
    jpeg_quality: int = 80,
    preview_scale: float = 1.0,
) -> Blueprint:
    bp = Blueprint("depth", __name__, url_prefix="/depth")
    lock = threading.Lock()
    point = {"fx": 0.5, "fy": 0.5}  # measurement point as a fraction of the image

    def measure(frame: DepthFrame) -> tuple[int, int, float | None]:
        h, w = frame.left.shape[:2]
        with lock:
            x, y = int(point["fx"] * (w - 1)), int(point["fy"] * (h - 1))
        return x, y, frame.distance_at(x, y, cfg.measure_radius_px)

    def render(frame: DepthFrame) -> tuple[np.ndarray, np.ndarray]:
        x, y, mm = measure(frame)
        left = frame.left.copy()
        draw_measurement(left, x, y, mm, cfg.measure_radius_px)
        heat = colorize_depth(frame.depth, cfg.display_min_mm, cfg.display_max_mm)
        heat = cv2.resize(heat, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        draw_measurement(heat, x, y, mm, cfg.measure_radius_px)
        return left, heat

    view = SharedRenderer(worker.latest, lambda frame: frame.timestamp_ns, render)

    @bp.route("/")
    def page():
        return render_template(
            "depth.html", min_mm=cfg.display_min_mm, max_mm=cfg.display_max_mm,
            closest_mm=worker.engine.min_distance_mm,
        )

    @bp.route("/video/<int:index>")
    def video(index: int):
        return picture_response(view, index, 2, jpeg_quality, preview_scale)

    @bp.route("/point", methods=["POST"])
    def set_point():
        data = request.get_json(silent=True) or {}
        try:
            fx, fy = float(data["fx"]), float(data["fy"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="send JSON {fx, fy} as fractions 0..1"), 400
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return jsonify(error="fx and fy must be between 0 and 1"), 400
        with lock:
            point.update(fx=fx, fy=fy)
        return jsonify(point)

    @bp.route("/measure")
    def get_measure():
        frame = worker.latest()
        if frame is None:
            return jsonify(distance_mm=None, error=worker.error or "no depth frame yet",
                           fps=round(worker.fps, 1))
        x, y, mm = measure(frame)
        return jsonify(
            distance_mm=None if mm is None else round(mm, 1), x=x, y=y,
            depth_mm=_round(frame.depth_at(x, y, cfg.measure_radius_px)),
            fps=round(worker.fps, 1), error=worker.error,
            valid_fraction=round(float(np.isfinite(frame.depth).mean()), 3),
        )

    return bp
