"""Objects page: detections on the rectified left image, each labelled with its distance."""

from __future__ import annotations

import time
from collections.abc import Iterator

from flask import Blueprint, Response, jsonify, render_template

from stereovision.detection.worker import DetectionWorker
from stereovision.overlay import draw_objects
from stereovision.web.stream import downscale, encode_jpeg


def create_blueprint(worker: DetectionWorker, model_name: str, jpeg_quality: int = 80,
                     preview_scale: float = 1.0) -> Blueprint:
    bp = Blueprint("objects", __name__, url_prefix="/objects")

    def stream() -> Iterator[bytes]:
        last_ts = None
        while True:
            result = worker.latest()
            if result is None or result.frame.timestamp_ns == last_ts:
                time.sleep(0.02)
                continue
            last_ts = result.frame.timestamp_ns
            image = draw_objects(result.frame.left, result.objects)
            jpeg = encode_jpeg(downscale(image, preview_scale), jpeg_quality)
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    @bp.route("/")
    def page():
        return render_template("objects.html", model=model_name,
                               classes=", ".join(worker.detector.classes))

    @bp.route("/video")
    def video():
        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
                "label": o.detection.label,
                "confidence": round(o.detection.confidence, 3),
                "box": [round(v, 1) for v in o.detection.box],
                "distance_mm": None if o.distance_mm is None else round(o.distance_mm, 1),
            } for o in sorted(result.objects, key=lambda o: o.distance_mm or float("inf"))],
        )

    return bp
