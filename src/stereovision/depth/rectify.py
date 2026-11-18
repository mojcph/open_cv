"""Undistort + rectify left/right images so matching points lie on the same image row."""

from __future__ import annotations

import cv2
import numpy as np

from stereovision.calibration.calibrate import StereoCalibration


class Rectifier:
    """Precomputes the remap tables once; rectify() is then a cheap per-frame lookup."""

    def __init__(self, cal: StereoCalibration) -> None:
        self.calibration = cal
        self.image_size = tuple(cal.image_size)
        self._m1x, self._m1y, self._m2x, self._m2y = cal.rectify_maps()

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = left.shape[:2]
        if (w, h) != self.image_size:
            raise ValueError(f"image is {w}x{h}, calibration is for "
                             f"{self.image_size[0]}x{self.image_size[1]}")
        return (cv2.remap(left, self._m1x, self._m1y, cv2.INTER_LINEAR),
                cv2.remap(right, self._m2x, self._m2y, cv2.INTER_LINEAR))


def draw_epipolar_lines(image: np.ndarray, spacing: int = 40,
                        color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Horizontal guide lines: after rectification, features should sit on the same line in
    both images."""
    out = image.copy()
    thickness = max(1, image.shape[1] // 800)
    for y in range(spacing // 2, out.shape[0], spacing):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), color, thickness)
    return out
