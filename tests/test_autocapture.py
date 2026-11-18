import cv2
import numpy as np
import pytest

from conftest import wait_for
from stereovision.calibration.capture import (
    AutoCapture,
    AutoCaptureSettings,
    CaptureStore,
    detect_pair,
    mean_shift,
)
from stereovision.camera import FakeSource
from stereovision.config import WebConfig
from stereovision.web.app import create_app
from test_capture import BOARD, META, blank_image, board_image

W = 640


def det(shift=0, right=None):
    left = board_image(shift=shift)
    return detect_pair(BOARD, left, right if right is not None else board_image(shift=shift - 40))


def feed(auto, detection, n):
    return [auto.update(detection, W) for _ in range(n)]


def test_mean_shift():
    a, b = det(0).left, det(10).left
    assert mean_shift(a, a) == pytest.approx(0.0, abs=0.01)
    assert mean_shift(a, b) == pytest.approx(10.0, abs=0.5)
    assert mean_shift(a, detect_pair(BOARD, blank_image(), blank_image()).left) is None


def test_saves_after_holding_still():
    auto = AutoCapture(AutoCaptureSettings(still_frames=4))
    results = feed(auto, det(0), 5)
    # frame 1 has nothing to compare with; frames 2-5 are still -> saves on the 5th
    assert results == [False, False, False, False, True]


def test_moving_board_never_saves():
    auto = AutoCapture()
    results = [auto.update(det(shift), W) for shift in range(0, 50, 10)]  # 10 px per frame
    assert not any(results)
    assert "moving" in auto.message


def test_same_pose_not_saved_twice():
    auto = AutoCapture(AutoCaptureSettings(still_frames=2))
    d = det(0)
    assert feed(auto, d, 3)[-1] is True
    auto.saved(d.left)
    assert not any(feed(auto, d, 6))
    assert "already captured" in auto.message


def test_new_pose_is_saved():
    auto = AutoCapture(AutoCaptureSettings(still_frames=2, novelty_frac=0.06))
    first = det(0)
    feed(auto, first, 3)
    auto.saved(first.left)
    moved = det(50)  # 50 px > 6% of 640 = 38 px
    assert feed(auto, moved, 3)[-1] is True


def test_missing_board_reports_reason():
    auto = AutoCapture()
    d = detect_pair(BOARD, board_image(), blank_image())
    assert not any(feed(auto, d, 5))
    assert auto.message == "board not found in right image"


def test_forget_last_allows_recapture():
    auto = AutoCapture(AutoCaptureSettings(still_frames=2))
    d = det(0)
    auto.saved(d.left)
    auto.forget_last()
    assert feed(auto, d, 3)[-1] is True


def test_seeded_with_existing_poses():
    d = det(0)
    auto = AutoCapture(AutoCaptureSettings(still_frames=2), saved_views=[d.left])
    assert not any(feed(auto, d, 4))


# --- web -------------------------------------------------------------------------------------

def client_for(running_rig, tmp_path, scale=1.0):
    rig = running_rig(left=FakeSource([board_image()], fps=100),
                      right=FakeSource([board_image(shift=-40)], fps=100))
    assert wait_for(lambda: rig.latest() is not None)
    store = CaptureStore(tmp_path, META)
    return create_app(rig, board=BOARD, store=store, preview_scale=scale).test_client(), store


def read_frames(resp, n):
    chunks = [next(resp.response) for _ in range(n)]
    resp.close()
    return chunks


def test_auto_capture_endpoint_saves_once(running_rig, tmp_path):
    client, store = client_for(running_rig, tmp_path)
    assert client.post("/calibrate/auto/on").get_json() == {"auto": True}
    read_frames(client.get("/calibrate/video"), 10)
    assert store.count() == 1  # static scene: one save, then "already captured"
    state = client.get("/calibrate/state").get_json()
    assert state["auto"] is True and "already captured" in state["message"]
    assert state["last"]["index"] == 0
    assert client.post("/calibrate/auto/off").get_json() == {"auto": False}
    assert client.post("/calibrate/auto/maybe").status_code == 400


def test_auto_off_does_not_save(running_rig, tmp_path):
    client, store = client_for(running_rig, tmp_path)
    read_frames(client.get("/calibrate/video"), 8)
    assert store.count() == 0


def test_preview_scale_halves_stream(running_rig, tmp_path):
    client, _ = client_for(running_rig, tmp_path, scale=0.5)
    chunk = read_frames(client.get("/video"), 1)[0]
    jpeg = chunk[chunk.index(b"\xff\xd8"):]
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 640, 3)  # two 640x480 images side by side, halved


@pytest.mark.parametrize("kw", [{"preview_scale": 0.0}, {"preview_scale": 1.5},
                                {"jpeg_quality": 0}])
def test_invalid_web_config(kw):
    with pytest.raises(ValueError):
        WebConfig(**kw)
