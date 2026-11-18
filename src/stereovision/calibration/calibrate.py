"""Stereo calibration from ChArUco detections: intrinsics, extrinsics and rectification.

Units: board lengths are in mm, so T (and the baseline) come out in mm.
Image size is (width, height) of the rotated images the rig delivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from stereovision.calibration.capture import PairDetection, ViewDetection

MIN_PAIRS = 3
MIN_CORNERS = 6  # per view, for a stable pose estimate
FEW_VIEWS = 10   # below this, fix k3 to avoid over-fitting the distortion model


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StereoCalibration:
    image_size: tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray            # rotation from left to right camera
    T: np.ndarray            # translation from left to right camera (mm), shape (3, 1)
    R1: np.ndarray           # rectification rotations
    R2: np.ndarray
    P1: np.ndarray           # rectified projection matrices
    P2: np.ndarray
    Q: np.ndarray            # disparity-to-depth matrix
    rms_left: float          # reprojection errors (px)
    rms_right: float
    rms_stereo: float
    per_view_errors: np.ndarray  # (n_pairs, 2): stereo error per pair, left/right
    meta: dict = field(default_factory=dict)

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def focal_px(self) -> float:
        """Focal length of the rectified images (P1[0, 0])."""
        return float(self.P1[0, 0])

    def rectify_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = tuple(self.image_size)
        m1x, m1y = cv2.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1, size,
                                               cv2.CV_16SC2)
        m2x, m2y = cv2.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2, size,
                                               cv2.CV_16SC2)
        return m1x, m1y, m2x, m2y


def object_points(board: cv2.aruco.CharucoBoard, ids: np.ndarray) -> np.ndarray:
    """3D board coordinates (mm, z=0) for the given ChArUco corner ids, shape (N, 1, 3)."""
    return board.getChessboardCorners()[ids.ravel()].reshape(-1, 1, 3).astype(np.float32)


def _common_points(board, det: PairDetection):
    """Object points and left/right image points for corners seen by both cameras."""
    left = dict(zip(det.left.ids.ravel().tolist(), det.left.corners.reshape(-1, 2), strict=True))
    right = dict(zip(det.right.ids.ravel().tolist(), det.right.corners.reshape(-1, 2),
                     strict=True))
    ids = np.array(sorted(left.keys() & right.keys()), np.int32)
    img_l = np.array([left[i] for i in ids], np.float32).reshape(-1, 1, 2)
    img_r = np.array([right[i] for i in ids], np.float32).reshape(-1, 1, 2)
    return object_points(board, ids), img_l, img_r


def calibrate_intrinsics(board, views: list[ViewDetection], image_size: tuple[int, int]):
    """Camera matrix, distortion and RMS error from all views of one camera."""
    usable = [v for v in views if v.count >= MIN_CORNERS]
    if len(usable) < MIN_PAIRS:
        raise CalibrationError(
            f"need at least {MIN_PAIRS} views with {MIN_CORNERS}+ corners, got {len(usable)}"
        )
    obj = [object_points(board, v.ids) for v in usable]
    img = [v.corners.astype(np.float32) for v in usable]
    flags = cv2.CALIB_FIX_K3 if len(usable) < FEW_VIEWS else 0
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None, flags=flags)
    return K, D, float(rms)


def calibrate_stereo(
    board: cv2.aruco.CharucoBoard,
    pairs: list[PairDetection],
    image_size: tuple[int, int],
    meta: dict | None = None,
    alpha: float = 0.0,
) -> StereoCalibration:
    """Full stereo calibration.

    Intrinsics use every corner each camera saw; the stereo step uses only common corners.
    alpha=0 crops the rectified images to valid pixels only (no black borders).
    """
    usable = [p for p in pairs if p.common >= MIN_CORNERS]
    if len(usable) < MIN_PAIRS:
        raise CalibrationError(
            f"need at least {MIN_PAIRS} pairs with {MIN_CORNERS}+ common corners, "
            f"got {len(usable)}"
        )
    size = tuple(image_size)
    K1, D1, rms_l = calibrate_intrinsics(board, [p.left for p in pairs], size)
    K2, D2, rms_r = calibrate_intrinsics(board, [p.right for p in pairs], size)

    obj, img_l, img_r = zip(*(_common_points(board, p) for p in usable), strict=True)
    result = cv2.stereoCalibrateExtended(
        list(obj), list(img_l), list(img_r), K1, D1, K2, D2, size,
        np.eye(3), np.zeros((3, 1)), flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms_s, K1, D1, K2, D2, R, T = result[:7]
    per_view = result[-1]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    return StereoCalibration(
        image_size=size, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T.reshape(3, 1),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rms_left=rms_l, rms_right=rms_r, rms_stereo=float(rms_s),
        per_view_errors=np.asarray(per_view, float).reshape(-1, 2), meta=dict(meta or {}),
    )


def report(
    cal: StereoCalibration,
    expected_baseline_mm: float | None = None,
    expected_focal_px: float | None = None,
) -> str:
    """Human-readable summary with pass/warn hints."""
    def flag(ok: bool) -> str:
        return "ok" if ok else "CHECK"

    fx1, fx2 = float(cal.K1[0, 0]), float(cal.K2[0, 0])
    lines = [
        f"image size         {cal.image_size[0]} x {cal.image_size[1]}",
        f"pairs used         {len(cal.per_view_errors)}",
        f"RMS left / right   {cal.rms_left:.3f} / {cal.rms_right:.3f} px  "
        f"[{flag(max(cal.rms_left, cal.rms_right) < 0.5)}: target < 0.5]",
        f"RMS stereo         {cal.rms_stereo:.3f} px  [{flag(cal.rms_stereo < 0.5)}]",
        f"focal left / right {fx1:.1f} / {fx2:.1f} px",
    ]
    if expected_focal_px:
        worst = max(abs(fx1 - expected_focal_px), abs(fx2 - expected_focal_px))
        lines.append(f"  vs expected      ~{expected_focal_px:.0f} px  "
                     f"[{flag(worst < 0.1 * expected_focal_px)}: within 10%]")
    lines += [
        f"focal (rectified)  {cal.focal_px:.1f} px",
        f"baseline           {cal.baseline_mm:.2f} mm  (T = {np.round(cal.T.ravel(), 2)})",
    ]
    if expected_baseline_mm:
        err = cal.baseline_mm - expected_baseline_mm
        lines.append(f"  vs expected      {expected_baseline_mm:.1f} mm: {err:+.2f} mm  "
                     f"[{flag(abs(err) < 0.05 * expected_baseline_mm)}]")
    if len(cal.per_view_errors) < FEW_VIEWS:
        lines.append(f"only {len(cal.per_view_errors)} pairs - use 15-25 varied poses for "
                     "a reliable result")
    errs = cal.per_view_errors.max(axis=1)
    median = float(np.median(errs))
    bad = [i for i, e in enumerate(errs) if e > max(1.0, 2.5 * median)]
    lines.append("per-pair error     " + "  ".join(f"{i}:{e:.2f}" for i, e in enumerate(errs)))
    if bad:
        lines.append(f"outlier pairs      {bad} - consider deleting them and re-running")
    return "\n".join(lines)
