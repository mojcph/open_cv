"""Drawing detections with distance labels onto images."""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from stereovision.detection.worker import MeasuredObject


def format_distance(mm: float | None) -> str:
    if mm is None:
        return "no depth"
    return f"{mm / 1000:.2f} m" if mm >= 1000 else f"{mm:.0f} mm"


def label_colour(label: str) -> tuple[int, int, int]:
    """Stable, bright BGR colour per class name."""
    hue = zlib.crc32(label.encode()) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_objects(image: np.ndarray, objects: list[MeasuredObject]) -> np.ndarray:
    """Copy of image with a box and 'label 1.23 m' tag per object."""
    out = image.copy()
    scale = image.shape[1] / 640
    thick = max(2, int(2 * scale))
    font = 0.6 * scale
    for obj in objects:
        d = obj.detection
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        colour = label_colour(d.label)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, thick)
        text = f"{d.label} {format_distance(obj.distance_mm)}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = y0 - 4 if y0 - th - base - 4 > 0 else y0 + th + base + 4
        cv2.rectangle(out, (x0, ty - th - base), (x0 + tw + 4, ty + base // 2), colour, -1)
        cv2.putText(out, text, (x0 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
    return out
