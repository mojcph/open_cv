"""YOLO-E detector (Ultralytics) with text-prompt classes baked into an NCNN export.

The only module that imports ultralytics. Export once with scripts/export_detector.py;
the export folder holds detector.json recording the model, classes and input size.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from stereovision.config import DetectionConfig
from stereovision.detection.base import Detection

META = "detector.json"


def _import_ultralytics():
    """Import ultralytics, then undo its cv2.setNumThreads(0).

    Ultralytics turns off OpenCV multi-threading for the whole process when imported, which
    would make stereo matching and rectification several times slower.
    """
    import ultralytics

    cv2.setNumThreads(-1)  # back to OpenCV's default (all cores)
    return ultralytics


def export_name(cfg: DetectionConfig) -> str:
    """Folder name for an export: changes whenever model, classes or input size change."""
    key = json.dumps([cfg.model, list(cfg.classes), cfg.imgsz])
    return f"{cfg.model}-{hashlib.sha1(key.encode()).hexdigest()[:8]}_ncnn_model"


def export_path(cfg: DetectionConfig, root: Path) -> Path:
    return root / cfg.models_dir / export_name(cfg)


def check_export(path: Path, cfg: DetectionConfig) -> str | None:
    """Why the export at path can't be used for cfg, or None if it can."""
    meta_file = path / META
    if not meta_file.exists():
        return f"no exported detector at {path} - run: python scripts/export_detector.py"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    want = {"model": cfg.model, "classes": list(cfg.classes), "imgsz": cfg.imgsz}
    if {k: meta.get(k) for k in want} != want:
        return f"{path} was exported for different settings - re-run scripts/export_detector.py"
    return None


def export_detector(cfg: DetectionConfig, root: Path) -> Path:
    """Download the YOLOE model, set the text-prompt classes and export to NCNN."""
    YOLOE = _import_ultralytics().YOLOE

    models = (root / cfg.models_dir).resolve()
    models.mkdir(parents=True, exist_ok=True)
    # Ultralytics downloads the text encoder into the current folder: keep it in models/.
    with contextlib.chdir(models):
        model = YOLOE(f"{cfg.model}.pt")
        names = list(cfg.classes)
        model.set_classes(names, model.get_text_pe(names))
        exported = (models / model.export(format="ncnn", imgsz=cfg.imgsz, verbose=False))
    target = export_path(cfg, root)
    if target.exists():
        shutil.rmtree(target)
    exported.rename(target)
    (target / META).write_text(json.dumps(
        {"model": cfg.model, "classes": names, "imgsz": cfg.imgsz}, indent=2), encoding="utf-8")
    return target


class YoloeDetector:
    def __init__(self, path: Path, cfg: DetectionConfig) -> None:
        problem = check_export(path, cfg)
        if problem:
            raise FileNotFoundError(problem)
        YOLO = _import_ultralytics().YOLO

        self._model = YOLO(str(path), task="segment")
        self.classes = cfg.classes
        self.imgsz = cfg.imgsz
        self.confidence = cfg.confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(image, imgsz=self.imgsz, conf=self.confidence,
                                     verbose=False)[0]
        boxes = result.boxes
        return [
            Detection(result.names[int(c)], float(p), tuple(float(v) for v in xyxy))
            for xyxy, c, p in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(),
                                  strict=True)
        ]
