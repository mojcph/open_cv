"""Disparity -> depth, robust distance at a point or in a box, depth colouring."""

from __future__ import annotations

import cv2
import numpy as np


def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_mm: float) -> np.ndarray:
    """Depth along the optical axis (mm): Z = f * B / d. NaN stays NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (focal_px * baseline_mm) / disparity
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth.astype(np.float32)


def _median(values: np.ndarray, min_valid: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.size < min_valid * values.size:
        return None
    return float(np.median(finite))


def region_distance(depth: np.ndarray, x: float, y: float, radius: int,
                    min_valid: float = 0.2) -> float | None:
    """Median depth of valid pixels within radius of (x, y) (depth-array coordinates).

    None if fewer than min_valid of the pixels are valid (no reliable measurement).
    """
    h, w = depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    r = max(1, int(radius))
    window = depth[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1]
    return _median(window, min_valid)


def box_distance(depth: np.ndarray, box: tuple[float, float, float, float],
                 shrink: float = 0.2, min_valid: float = 0.1) -> float | None:
    """Median depth inside a box (x0, y0, x1, y1), shrunk by `shrink` on each side so the
    background around an object's outline doesn't count."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    h, w = depth.shape
    xa, xb = int(max(0, x0 + dx)), int(min(w, x1 - dx))
    ya, yb = int(max(0, y0 + dy)), int(min(h, y1 - dy))
    if xb <= xa or yb <= ya:
        return None
    return _median(depth[ya:yb, xa:xb], min_valid)


def colorize_depth(depth: np.ndarray, min_mm: float, max_mm: float) -> np.ndarray:
    """BGR heatmap: near = red, far = blue (TURBO), invalid = black."""
    valid = np.isfinite(depth)
    norm = np.zeros(depth.shape, np.uint8)
    clipped = np.clip(depth[valid], min_mm, max_mm)
    norm[valid] = (255 * (max_mm - clipped) / (max_mm - min_mm)).astype(np.uint8)
    colour = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour
