import cv2
import numpy as np
import pytest

from conftest import wait_for
from stereovision.camera import FakeSource, Frame, StereoFrame
from stereovision.config import DepthConfig
from stereovision.depth.disparity import compute_disparity, make_matcher
from stereovision.depth.engine import DepthEngine, DepthWorker
from stereovision.depth.measure import (
    box_distance,
    colorize_depth,
    disparity_to_depth,
    region_distance,
)
from stereovision.web.app import create_app

FOCAL, BASELINE = 1000.0, 87.0
SHIFT = 40  # px disparity at full resolution
EXPECTED_MM = FOCAL * BASELINE / SHIFT  # 2175 mm


def textured_pair(shift=SHIFT, h=480, w=640, seed=0):
    """Random texture; right image = left shifted left by `shift` px (constant disparity)."""
    rng = np.random.default_rng(seed)
    left = cv2.GaussianBlur(rng.integers(0, 256, (h, w), np.uint8), (3, 3), 0)
    right = np.roll(left, -shift, axis=1)
    return cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)


# --- measure ---------------------------------------------------------------------------------

def test_disparity_to_depth():
    d = np.array([[40.0, 0.0, np.nan, -1.0]], np.float32)
    z = disparity_to_depth(d, FOCAL, BASELINE)
    assert z[0, 0] == pytest.approx(EXPECTED_MM)
    assert np.isnan(z[0, 1:]).all()


def test_region_distance_median_ignores_invalid():
    depth = np.full((20, 20), 1000.0, np.float32)
    depth[10, 10] = 99999.0             # outlier
    depth[5:8, 5:8] = np.nan
    assert region_distance(depth, 10, 10, radius=3) == pytest.approx(1000.0)
    assert region_distance(depth, 50, 50, radius=3) is None   # outside image


def test_region_distance_needs_enough_valid_pixels():
    depth = np.full((20, 20), np.nan, np.float32)
    depth[10, 10] = 500.0
    assert region_distance(depth, 10, 10, radius=3) is None


def test_box_distance_shrinks_box():
    depth = np.full((100, 100), 3000.0, np.float32)   # background
    depth[20:80, 20:80] = 1000.0                       # object filling the box interior
    assert box_distance(depth, (10, 10, 90, 90), shrink=0.2) == pytest.approx(1000.0)
    assert box_distance(depth, (50, 50, 50, 60)) is None  # empty box


def test_colorize_depth():
    depth = np.array([[300.0, 3000.0, np.nan]], np.float32)
    img = colorize_depth(depth, 300, 3000)
    assert img.shape == (1, 3, 3)
    assert tuple(img[0, 2]) == (0, 0, 0)
    assert img[0, 0, 2] > img[0, 0, 0]   # near: more red than blue
    assert img[0, 1, 0] > img[0, 1, 2]   # far: more blue than red


# --- disparity + engine ----------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_disparity_recovers_known_shift(scale):
    cfg = DepthConfig(scale=scale, num_disparities=64 if scale == 0.5 else 96)
    left, right = textured_pair()
    disp = compute_disparity(make_matcher(cfg), left, right, scale)
    assert disp.shape == (int(480 * scale), int(640 * scale))
    centre = disp[disp.shape[0] // 4: -disp.shape[0] // 4, disp.shape[1] // 4: -disp.shape[1] // 4]
    assert np.nanmedian(centre) == pytest.approx(SHIFT, abs=0.5)
    assert np.isfinite(centre).mean() > 0.9


def make_engine(scale=0.5):
    return DepthEngine(None, FOCAL, BASELINE, DepthConfig(scale=scale, num_disparities=64))


def test_engine_distance_at_point():
    left, right = textured_pair()
    frame = make_engine().compute(StereoFrame(Frame(left, 1), Frame(right, 1)))
    assert frame.depth.shape == (240, 320)
    assert frame.distance_at(320, 240, radius_px=8) == pytest.approx(EXPECTED_MM, rel=0.02)


def test_engine_min_distance():
    assert make_engine().min_distance_mm == pytest.approx(FOCAL * 0.5 * BASELINE / 64)


def test_engine_from_rectifier_uses_rectified_baseline():
    from stereovision.calibration.calibrate import calibrate_stereo
    from stereovision.depth.rectify import Rectifier
    from test_calibrate import BOARD, SIZE, synthetic_pairs  # reuse the synthetic rig

    cal = calibrate_stereo(BOARD, synthetic_pairs(), SIZE)
    engine = DepthEngine.from_rectifier(Rectifier(cal), DepthConfig())
    assert engine.baseline_mm == pytest.approx(cal.baseline_mm, rel=1e-3)
    assert engine.focal_px == pytest.approx(cal.focal_px)


# --- worker + web ----------------------------------------------------------------------------

@pytest.fixture
def depth_rig(running_rig):
    left, right = textured_pair()
    return running_rig(left=FakeSource([left], fps=60), right=FakeSource([right], fps=60))


@pytest.fixture
def worker(depth_rig):
    w = DepthWorker(depth_rig, make_engine(), idle_s=5)
    w.start()
    yield w
    w.stop()


def test_worker_computes_on_demand(worker):
    worker.latest()  # the first request wakes the worker
    assert wait_for(lambda: worker.latest() is not None, timeout=5)
    assert worker.latest().distance_at(320, 240, 8) == pytest.approx(EXPECTED_MM, rel=0.02)
    assert worker.fps > 0


def test_worker_idles_without_requests(depth_rig):
    w = DepthWorker(depth_rig, make_engine(), idle_s=0.1)
    w.start()
    try:
        import time
        time.sleep(0.4)          # nobody asked: nothing computed
        assert w._latest is None
    finally:
        w.stop()


def test_depth_endpoints(depth_rig, worker):
    client = create_app(depth_rig, depth_worker=worker,
                        depth_cfg=DepthConfig(num_disparities=64)).test_client()
    assert b">Distance<" in client.get("/").data
    assert b"Closest measurable" in client.get("/depth/").data
    assert wait_for(lambda: client.get("/depth/measure").get_json()["distance_mm"] is not None,
                    timeout=5)
    m = client.get("/depth/measure").get_json()
    assert m["distance_mm"] == pytest.approx(EXPECTED_MM, rel=0.02)
    assert (m["x"], m["y"]) == (319, 239)  # centre by default

    assert client.post("/depth/point", json={"fx": 0.25, "fy": 0.75}).status_code == 200
    m = client.get("/depth/measure").get_json()
    assert (m["x"], m["y"]) == (159, 359)
    assert client.post("/depth/point", json={"fx": 2}).status_code == 400
    assert client.post("/depth/point", data="nope").status_code == 400

    resp = client.get("/depth/video/0")
    assert next(resp.response).startswith(b"--frame\r\nContent-Type: image/jpeg")
    resp.close()


def test_depth_disabled_without_worker(running_rig):
    client = create_app(running_rig()).test_client()
    assert client.get("/depth/").status_code == 404


@pytest.mark.parametrize("kw", [{"num_disparities": 100}, {"block_size": 4}, {"scale": 0},
                                {"mode": "fast"}, {"display_min_mm": 5000}])
def test_invalid_depth_config(kw):
    with pytest.raises(ValueError):
        DepthConfig(**kw)


def test_too_close_is_not_reported_as_a_distance():
    """A shift beyond the search range must not come back as the max-range distance."""
    cfg = DepthConfig(scale=1.0, num_disparities=32)
    left, right = textured_pair(shift=48)  # closer than 32 px of search can reach
    disp = compute_disparity(make_matcher(cfg), left, right, 1.0)
    assert not (disp >= 31).any()
