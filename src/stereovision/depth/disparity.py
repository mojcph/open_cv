"""StereoSGBM disparity on rectified images, computed at a reduced working scale."""

from __future__ import annotations

import cv2
import numpy as np

from stereovision.config import DepthConfig

_MODES = {
    "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
    "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "hh": cv2.STEREO_SGBM_MODE_HH,
}


def make_matcher(cfg: DepthConfig) -> cv2.StereoSGBM:
    channels = 1  # we match grayscale images
    area = channels * cfg.block_size**2
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=8 * area,       # standard smoothness penalties from the OpenCV docs
        P2=32 * area,
        disp12MaxDiff=1,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
        mode=_MODES[cfg.mode],
    )


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def compute_disparity(
    matcher: cv2.StereoSGBM, left: np.ndarray, right: np.ndarray, scale: float
) -> np.ndarray:
    """Disparity map at the working scale, in FULL-resolution pixel units, NaN where invalid.

    The array is scale x the input size; values are divided by scale so that
    Z = focal_full * baseline / disparity holds directly.
    """
    left, right = _gray(left), _gray(right)
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    raw = matcher.compute(left, right)  # int16, 4 fractional bits; invalid = minDisparity - 1
    disparity = raw.astype(np.float32) / 16.0
    disparity[raw <= 0] = np.nan
    return disparity / scale
