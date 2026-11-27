import json
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import wait_for
from stereovision.camera import FakeSource
from stereovision.config import DEFAULT_CONFIG_PATH, DetectionConfig, load_config
from stereovision.depth.engine import DepthFrame, DepthWorker
from stereovision.detection import Detection, NullDetector
from stereovision.detection.null import FixedDetector
from stereovision.detection.worker import DetectionWorker, MeasuredObject, measure_objects
from stereovision.detection.yoloe import META, check_export, export_name, export_path
from stereovision.overlay import draw_objects, format_distance, label_colour
from stereovision.web.app import create_app
from test_depth import EXPECTED_MM, make_engine, textured_pair

ROOT = Path(__file__).resolve().parents[1]


# --- config / export bookkeeping -------------------------------------------------------------

def test_default_detection_config():
    cfg = load_config(DEFAULT_CONFIG_PATH).detection
    assert cfg.model == "yoloe-26n-seg" and "car" in cfg.classes


@pytest.mark.parametrize("kw", [{"classes": []}, {"classes": ["cup", "cup"]}, {"imgsz": 100},
                                {"confidence": 1.5}, {"enabled": "yes"}, {"classes": [" "]}])
def test_invalid_detection_config(kw):
    with pytest.raises(ValueError):
        DetectionConfig(**kw)


def test_export_name_changes_with_settings():
    base = DetectionConfig()
    assert export_name(base) == export_name(DetectionConfig())
    assert export_name(base) != export_name(DetectionConfig(classes=("cup",)))
    assert export_name(base) != export_name(DetectionConfig(imgsz=480))
    assert export_name(base).startswith("yoloe-26n-seg-")


def test_check_export(tmp_path):
    cfg = DetectionConfig(models_dir="m")
    path = export_path(cfg, tmp_path)
    assert "run: python scripts/export_detector.py" in check_export(path, cfg)
    path.mkdir(parents=True)
    meta = {"model": cfg.model, "classes": list(cfg.classes), "imgsz": cfg.imgsz}
    (path / META).write_text(json.dumps(meta))
    assert check_export(path, cfg) is None
    other = DetectionConfig(models_dir="m", classes=("dog",))
    assert "different settings" in check_export(path, other)


# --- distances for boxes ---------------------------------------------------------------------

def depth_frame(scale=0.5):
    depth = np.full((240, 320), 3000.0, np.float32)
    depth[60:180, 80:240] = 1000.0   # object in the middle (depth-array pixels)
    return DepthFrame(np.zeros((480, 640, 3), np.uint8), depth, scale, 1)


def test_measure_objects_scales_boxes():
    dets = [Detection("cup", 0.9, (160, 120, 480, 360)),   # full-res box around the object
            Detection("wall", 0.5, (0, 0, 100, 100))]      # background only
    objs = measure_objects(depth_frame(), dets)
    assert objs[0].distance_mm == pytest.approx(1000.0)
    assert objs[1].distance_mm == pytest.approx(3000.0)


def test_measure_objects_no_depth():
    frame = depth_frame()
    frame.depth[:] = np.nan
    assert measure_objects(frame, [Detection("cup", 0.9, (0, 0, 100, 100))])[0].distance_mm is None


# --- overlay ---------------------------------------------------------------------------------

def test_format_distance():
    assert format_distance(None) == "no depth"
    assert format_distance(850.4) == "850 mm"
    assert format_distance(1234.0) == "1.23 m"


def test_label_colour_is_stable_and_distinct():
    assert label_colour("cup") == label_colour("cup")
    assert label_colour("cup") != label_colour("person")


def test_draw_objects_keeps_original():
    img = np.zeros((480, 640, 3), np.uint8)
    out = draw_objects(img, [MeasuredObject(Detection("cup", 0.9, (100, 5, 200, 150)), 812.0)])
    assert out.any() and not img.any()


# --- worker + web ----------------------------------------------------------------------------

@pytest.fixture
def detection_setup(running_rig):
    left, right = textured_pair()
    rig = running_rig(left=FakeSource([left], fps=60), right=FakeSource([right], fps=60))
    depth = DepthWorker(rig, make_engine())
    det = DetectionWorker(rig, make_engine(), FixedDetector([("cup", 0.87, (0.3, 0.3, 0.7, 0.7))]))
    depth.start()
    det.start()
    yield rig, depth, det
    det.stop()
    depth.stop()


def test_detection_worker_measures_objects(detection_setup):
    _, _, det = detection_setup
    det.latest()  # the first request wakes the worker
    assert wait_for(lambda: det.latest() is not None, timeout=5)
    obj = det.latest().objects[0]
    assert obj.detection.label == "cup"
    assert obj.distance_mm == pytest.approx(EXPECTED_MM, rel=0.02)


def test_objects_endpoints(detection_setup):
    rig, depth, det = detection_setup
    client = create_app(rig, depth_worker=depth, detection_worker=det,
                        model_name="test-model").test_client()
    assert b">Objects<" in client.get("/").data
    page = client.get("/objects/").data
    assert b"test-model" in page and b"Looking for: cup" in page
    assert wait_for(lambda: client.get("/objects/list").get_json()["objects"], timeout=5)
    data = client.get("/objects/list").get_json()
    assert data["objects"][0]["label"] == "cup"
    assert data["objects"][0]["distance_mm"] == pytest.approx(EXPECTED_MM, rel=0.02)
    resp = client.get("/objects/video/0")
    assert next(resp.response).startswith(b"--frame\r\nContent-Type: image/jpeg")
    resp.close()


def test_objects_disabled_without_worker(running_rig):
    assert create_app(running_rig()).test_client().get("/objects/").status_code == 404


def test_null_detector():
    assert NullDetector().detect(np.zeros((10, 10, 3), np.uint8)) == []


# --- real model (run with: pytest -m model) --------------------------------------------------

@pytest.mark.model
def test_yoloe_export_runs():
    from stereovision.detection.yoloe import YoloeDetector

    cfg = load_config(DEFAULT_CONFIG_PATH).detection
    path = export_path(cfg, ROOT)
    if check_export(path, cfg):
        pytest.skip("detector not exported - run scripts/export_detector.py")
    detector = YoloeDetector(path, cfg)
    image = np.full((1280, 960, 3), 127, np.uint8)
    detections = detector.detect(image)
    assert isinstance(detections, list)
    assert all(d.label in cfg.classes for d in detections)


@pytest.mark.model
def test_yoloe_keeps_opencv_multithreading():
    import cv2

    from stereovision.detection.yoloe import _import_ultralytics

    _import_ultralytics()
    assert cv2.getNumThreads() > 1


def test_measure_objects_distance_from_midpoint():
    from stereovision.depth.engine import Optics

    frame = depth_frame()
    frame = DepthFrame(frame.left, frame.depth, frame.scale, 1,
                       Optics(focal_px=500.0, cx=320.0, cy=240.0, baseline_mm=88.0))
    centred = Detection("cup", 0.9, (160, 120, 480, 360))   # object, box centred on the axis
    obj = measure_objects(frame, [centred])[0]
    assert obj.depth_mm == pytest.approx(1000.0)
    # box centre (320, 240) is on the left camera's axis: 44 mm left of the midpoint
    assert obj.distance_mm == pytest.approx((1000.0**2 + 44.0**2) ** 0.5)


def test_peek_does_not_wake_detection_worker(running_rig):
    det = DetectionWorker(running_rig(), make_engine(), NullDetector(), idle_s=0.2)
    assert not det.active
    det.peek()
    assert not det.active
    det.latest()
    assert det.active


def test_depth_worker_defers_to_active_detection(running_rig):
    left, right = textured_pair()
    rig = running_rig(left=FakeSource([left], fps=60), right=FakeSource([right], fps=60))
    det = DetectionWorker(rig, make_engine(), FixedDetector([("cup", 0.9, (0.3, 0.3, 0.7, 0.7))]),
                          idle_s=0.5)
    depth = DepthWorker(rig, make_engine(), idle_s=0.5, defer_to=det)
    det.start()
    depth.start()
    try:
        # /objects open: depth serves the detection worker's frames and computes nothing itself
        assert wait_for(lambda: det.latest() is not None, timeout=5)
        for _ in range(5):
            det.latest()
            frame = depth.latest()
            assert frame is det.peek().frame
            time.sleep(0.05)
        assert depth.deferring and depth._latest is None
        assert depth.fps == det.fps
        # /objects closed: the depth worker takes over again
        assert wait_for(lambda: (depth.latest(), not det.active)[1], timeout=3)
        assert wait_for(lambda: depth._latest is not None, timeout=5)
        assert depth.latest() is depth._latest
        assert depth.latest().distance_at(320, 240, 8) == pytest.approx(EXPECTED_MM, rel=0.02)
    finally:
        depth.stop()
        det.stop()
