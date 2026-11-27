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


# Fixed colours (RGB) for the street classes; any other class gets a hashed colour.
CLASS_COLOURS_RGB: dict[str, tuple[int, int, int]] = {
    "person": (0, 200, 0),         # green
    "cyclist": (255, 140, 0),      # orange
    "car": (30, 144, 255),         # blue
    "motorcycle": (255, 0, 255),   # magenta
    "bus": (255, 215, 0),          # yellow
    "truck": (230, 30, 30),        # red
}


def label_colour(label: str) -> tuple[int, int, int]:
    """Stable, bright BGR colour per class name."""
    if label in CLASS_COLOURS_RGB:
        r, g, b = CLASS_COLOURS_RGB[label]
        return b, g, r
    hue = zlib.crc32(label.encode()) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def label_hex(label: str) -> str:
    """The class colour as a CSS hex string (#rrggbb)."""
    b, g, r = label_colour(label)
    return f"#{r:02x}{g:02x}{b:02x}"


def text_colour(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever reads better on the given background."""
    b, g, r = bgr
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255)


def draw_objects(image: np.ndarray, objects: list[MeasuredObject]) -> np.ndarray:
    """Copy of image with a box in the class colour and a 'number: 1.23 m' tag per object.
    The number is the object's position in the list + 1, as on the objects table."""
    out = image.copy()
    scale = image.shape[1] / 640
    thick = max(2, int(2 * scale))
    font = 0.6 * scale
    for number, obj in enumerate(objects, 1):
        d = obj.detection
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        colour = label_colour(d.label)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, thick)
        text = f"{number}: {format_distance(obj.distance_mm)}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        ty = y0 - 4 if y0 - th - base - 4 > 0 else y0 + th + base + 4
        cv2.rectangle(out, (x0, ty - th - base), (x0 + tw + 4, ty + base // 2), colour, -1)
        cv2.putText(out, text, (x0 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, font,
                    text_colour(colour), max(1, thick - 1), cv2.LINE_AA)
    return out
