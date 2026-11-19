"""Calibration tests on synthetic data: board corners projected through a known stereo rig."""

import cv2
import numpy as np
import pytest

from stereovision.calibration.board import make_board
from stereovision.calibration.calibrate import (
    CalibrationError,
    calibrate_stereo,
    report,
)
from stereovision.calibration.capture import PairDetection, ViewDetection, capture_meta
from stereovision.calibration.io import camera_mismatch, load_calibration, save_calibration
from stereovision.config import BoardConfig, CameraConfig
from stereovision.depth.rectify import Rectifier, draw_epipolar_lines

BOARD = make_board(BoardConfig())
SIZE = (960, 1280)  # width, height (rotated portrait images)
K = np.array([[1250.0, 0, 480], [0, 1250.0, 640], [0, 0, 1]])
DIST = np.array([0.05, -0.1, 0.0, 0.0, 0.0])
R_LR = cv2.Rodrigues(np.array([0.0, 0.01, 0.0]))[0]  # small toe-in between the cameras
T_LR = np.array([[-87.0], [0.5], [0.3]])             # right camera 87 mm to the right
BASELINE = float(np.linalg.norm(T_LR))


def project(rvec, tvec, noise, rng):
    obj = BOARD.getChessboardCorners().astype(np.float64)
    pts, _ = cv2.projectPoints(obj, rvec, tvec, K, DIST)
    pts = pts.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2))
    inside = ((pts[:, 0] > 5) & (pts[:, 0] < SIZE[0] - 5)
              & (pts[:, 1] > 5) & (pts[:, 1] < SIZE[1] - 5))
    ids = np.flatnonzero(inside).astype(np.int32)
    return ViewDetection(pts[ids].reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1))


def synthetic_pairs(n=15, noise=0.1, seed=1):
    """Board poses spread over the image, 400-800 mm away, tilted up to ~35 degrees."""
    rng = np.random.default_rng(seed)
    centre = BOARD.getChessboardCorners().mean(axis=0)
    pairs = []
    for _ in range(n):
        angles = rng.uniform(-0.6, 0.6, 3) * np.array([1, 1, 0.3])
        rot = cv2.Rodrigues(angles)[0]
        z = rng.uniform(400, 800)
        target = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.35, 0.35) * z, z])
        tvec = target - rot @ centre  # board centre lands at `target` in the left camera
        rvec = cv2.Rodrigues(rot)[0]
        left = project(rvec, tvec, noise, rng)
        # right camera: X_r = R_LR @ X_l + T_LR
        r_rot = R_LR @ rot
        r_t = R_LR @ tvec.reshape(3, 1) + T_LR
        right = project(cv2.Rodrigues(r_rot)[0], r_t, noise, rng)
        pairs.append(PairDetection(left, right))
    return pairs


@pytest.fixture(scope="module")
def calibration():
    meta = capture_meta(CameraConfig(width=1280, height=960), BoardConfig())
    return calibrate_stereo(BOARD, synthetic_pairs(), SIZE, meta=meta)


def test_recovers_intrinsics(calibration):
    assert calibration.K1[0, 0] == pytest.approx(1250, rel=0.01)
    assert calibration.K2[1, 1] == pytest.approx(1250, rel=0.01)
    assert calibration.K1[0, 2] == pytest.approx(480, abs=5)
    assert calibration.rms_left < 0.3 and calibration.rms_right < 0.3


def test_recovers_baseline_and_rotation(calibration):
    assert calibration.baseline_mm == pytest.approx(BASELINE, abs=0.5)
    assert calibration.T[0, 0] == pytest.approx(-87.0, abs=0.5)
    assert np.allclose(calibration.R, R_LR, atol=2e-3)
    assert calibration.rms_stereo < 0.3


def test_rectified_points_share_rows(calibration):
    """After rectification, a corner's y coordinate must match in both images."""
    pair = synthetic_pairs(n=1, noise=0.0, seed=7)[0]
    common = sorted(set(pair.left.ids.ravel()) & set(pair.right.ids.ravel()))
    def rect(view, Kc, Dc, Rc, Pc):
        pos = dict(zip(view.ids.ravel(), view.corners.reshape(-1, 2), strict=True))
        pts = np.array([pos[i] for i in common], np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, Kc, Dc, R=Rc, P=Pc).reshape(-1, 2)
    c = calibration
    left = rect(pair.left, c.K1, c.D1, c.R1, c.P1)
    right = rect(pair.right, c.K2, c.D2, c.R2, c.P2)
    assert np.abs(left[:, 1] - right[:, 1]).max() < 0.5
    assert (left[:, 0] - right[:, 0]).min() > 0  # positive disparity: left x > right x


def test_report_mentions_key_numbers(calibration):
    text = report(calibration, expected_baseline_mm=87.0, expected_focal_px=1250)
    assert "baseline" in text and "CHECK" not in text


def test_too_few_pairs():
    with pytest.raises(CalibrationError, match="at least 3"):
        calibrate_stereo(BOARD, synthetic_pairs(n=2), SIZE)


def test_save_load_roundtrip(calibration, tmp_path):
    path = save_calibration(calibration, tmp_path / "stereo.npz")
    loaded = load_calibration(path)
    assert loaded.image_size == SIZE
    assert np.allclose(loaded.Q, calibration.Q)
    assert loaded.rms_stereo == pytest.approx(calibration.rms_stereo)
    assert loaded.meta == calibration.meta


def test_camera_mismatch(calibration):
    same = capture_meta(CameraConfig(width=1280, height=960), BoardConfig())["camera"]
    assert camera_mismatch(calibration, same) is None
    other = capture_meta(CameraConfig(width=640, height=480), BoardConfig())["camera"]
    assert "width" in camera_mismatch(calibration, other)


def test_rectifier_checks_size_and_runs(calibration):
    rect = Rectifier(calibration)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    left, right = rect.rectify(img, img)
    assert left.shape == img.shape
    with pytest.raises(ValueError, match="calibration is for"):
        rect.rectify(np.zeros((480, 640, 3), np.uint8), img)


def test_draw_epipolar_lines():
    img = np.zeros((100, 50, 3), np.uint8)
    out = draw_epipolar_lines(img, spacing=20)
    assert out[10, 25, 1] == 255 and img.max() == 0


def test_rectified_page_and_stream(calibration, running_rig):
    from conftest import wait_for
    from stereovision.camera import FakeSource
    from stereovision.web.app import create_app

    img = np.full((SIZE[1], SIZE[0], 3), 128, np.uint8)
    rig = running_rig(left=FakeSource([img], fps=100), right=FakeSource([img], fps=100))
    assert wait_for(lambda: rig.latest() is not None)
    client = create_app(rig, rectifier=Rectifier(calibration), preview_scale=0.5).test_client()
    assert b">Rectified<" in client.get("/").data
    assert b"same green line" in client.get("/rectified").data
    resp = client.get("/rectified/video/1")
    assert next(resp.response).startswith(b"--frame\r\nContent-Type: image/jpeg")
    resp.close()


def test_rectified_disabled_without_calibration(running_rig):
    from stereovision.web.app import create_app

    client = create_app(running_rig()).test_client()
    assert client.get("/rectified").status_code == 404
    assert b">Rectified<" not in client.get("/").data
