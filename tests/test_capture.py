import cv2
import numpy as np
import pytest

from conftest import wait_for
from stereovision.calibration.board import make_board
from stereovision.calibration.capture import (
    CaptureStore,
    capture_meta,
    check_pair,
    detect_pair,
    draw_view,
)
from stereovision.camera import FakeSource
from stereovision.config import BoardConfig, CameraConfig
from stereovision.web.app import create_app

BOARD_CFG = BoardConfig()
BOARD = make_board(BOARD_CFG)


def board_image(width=640, height=480, shift=0):
    """BGR image with the board on a white background, optionally shifted sideways."""
    gray = BOARD.generateImage((540, 360), marginSize=0, borderBits=1)
    img = np.full((height, width), 255, np.uint8)
    y, x = (height - 360) // 2, (width - 540) // 2 + shift
    img[y : y + 360, x : x + 540] = gray
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def blank_image(width=640, height=480):
    return np.full((height, width, 3), 255, np.uint8)


META = capture_meta(CameraConfig(), BOARD_CFG)


def test_pair_with_board_in_both_views_is_good():
    det = detect_pair(BOARD, board_image(), board_image(shift=-40))
    assert det.left.count == det.right.count == det.common == 40
    assert check_pair(det, min_common=8) is None


def test_board_missing_in_one_view():
    det = detect_pair(BOARD, board_image(), blank_image())
    assert check_pair(det, 8) == "board not found in right image"


def test_too_few_common_corners():
    det = detect_pair(BOARD, board_image(), board_image())
    assert "only 40 corners" in check_pair(det, min_common=41)


def test_draw_view_keeps_original():
    img = board_image()
    det = detect_pair(BOARD, img, img).left
    out = draw_view(img, det, good=True)
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(img, board_image())


def test_store_saves_numbered_pairs(tmp_path):
    store = CaptureStore(tmp_path / "caps", META)
    assert store.count() == 0
    assert store.save(board_image(), board_image()) == 0
    assert store.save(board_image(), board_image()) == 1
    assert store.count() == 2
    assert (tmp_path / "caps" / "001_right.png").exists()
    left, right = store.load_pairs()[0]
    assert left.shape == (480, 640, 3)


def test_store_remove_last(tmp_path):
    store = CaptureStore(tmp_path, META)
    assert store.remove_last() is None
    store.save(blank_image(), blank_image())
    store.save(blank_image(), blank_image())
    assert store.remove_last() == 1
    assert store.count() == 1
    assert store.save(blank_image(), blank_image()) == 1  # numbering continues after undo


def test_store_refuses_mixed_settings(tmp_path):
    CaptureStore(tmp_path, META).save(blank_image(), blank_image())
    other = capture_meta(CameraConfig(lens_position=2.0), BoardConfig(square_mm=25, marker_mm=18))
    store = CaptureStore(tmp_path, other)
    mismatch = store.meta_mismatch()
    assert "board.square_mm" in mismatch and "camera.lens_position" in mismatch
    with pytest.raises(ValueError, match="different camera settings"):
        store.save(blank_image(), blank_image())


def test_store_same_settings_ok_after_reload(tmp_path):
    CaptureStore(tmp_path, META).save(blank_image(), blank_image())
    assert CaptureStore(tmp_path, capture_meta(CameraConfig(), BOARD_CFG)).meta_mismatch() is None


# --- web endpoints -------------------------------------------------------------------------

def make_client(running_rig, tmp_path, left_img, right_img):
    rig = running_rig(left=FakeSource([left_img], fps=100), right=FakeSource([right_img], fps=100))
    assert wait_for(lambda: rig.latest() is not None)
    store = CaptureStore(tmp_path, META)
    return create_app(rig, board=BOARD, store=store).test_client(), store


def test_capture_endpoint_saves_good_pair(running_rig, tmp_path):
    client, store = make_client(running_rig, tmp_path, board_image(), board_image(shift=-40))
    r = client.post("/calibrate/capture").get_json()
    assert r["saved"] is True and r["index"] == 0 and r["common"] == 40
    assert store.count() == 1
    assert client.post("/calibrate/undo").get_json() == {"removed": 0, "count": 0}


def test_capture_endpoint_rejects_missing_board(running_rig, tmp_path):
    client, store = make_client(running_rig, tmp_path, board_image(), blank_image())
    r = client.post("/calibrate/capture").get_json()
    assert r["saved"] is False and "right" in r["reason"]
    assert store.count() == 0


def test_calibrate_page_and_overlay_stream(running_rig, tmp_path):
    client, _ = make_client(running_rig, tmp_path, board_image(), board_image())
    assert b"Capture (space)" in client.get("/calibrate/").data
    assert b">Calibration<" in client.get("/").data
    resp = client.get("/calibrate/video/0")
    assert next(resp.response).startswith(b"--frame\r\nContent-Type: image/jpeg")
    resp.close()


def test_calibration_disabled_without_board(running_rig):
    client = create_app(running_rig()).test_client()
    assert client.get("/calibrate/").status_code == 404
    assert b">Calibration<" not in client.get("/").data
