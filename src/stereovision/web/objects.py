"""Objects page: detections on the rectified left image, each labelled with its distance."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from stereovision.detection.worker import DetectionWorker
from stereovision.overlay import draw_objects, label_hex
from stereovision.web.stream import SharedRenderer, picture_response


def create_blueprint(worker: DetectionWorker, model_name: str, jpeg_quality: int = 80,
                     preview_scale: float = 1.0) -> Blueprint:
    bp = Blueprint("objects", __name__, url_prefix="/objects")

    view = SharedRenderer(worker.latest, lambda result: result.frame.timestamp_ns,
                          lambda result: (draw_objects(result.frame.left, result.objects),))

    @bp.route("/")
    def page():
        return render_template("objects.html", model=model_name,
                               classes=", ".join(worker.detector.classes))

    @bp.route("/video/<int:index>")
    def video(index: int):
        return picture_response(view, index, 1, jpeg_quality, preview_scale)

    @bp.route("/list")
    def list_objects():
        result = worker.latest()
        if result is None:
            return jsonify(objects=[], fps=round(worker.fps, 1),
                           error=worker.error or "no detections yet")
        return jsonify(
            fps=round(worker.fps, 1), depth_ms=round(result.depth_ms),
            detect_ms=round(result.detect_ms), error=worker.error,
            objects=[{
                "number": n,
                "label": o.detection.label,
                "colour": label_hex(o.detection.label),
                "confidence": round(o.detection.confidence, 3),
                "box": [round(v, 1) for v in o.detection.box],
                "distance_mm": None if o.distance_mm is None else round(o.distance_mm, 1),
                "depth_mm": None if o.depth_mm is None else round(o.depth_mm, 1),
            } for n, o in enumerate(result.objects, 1)],
        )

    return bp
