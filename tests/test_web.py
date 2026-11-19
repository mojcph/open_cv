import cv2
import numpy as np

from conftest import make_image, wait_for
from stereovision.web.app import create_app
from stereovision.web.stream import SharedRenderer, encode_jpeg, picture_stream


def decode(chunk: bytes) -> np.ndarray:
    jpeg = chunk[chunk.index(b"\xff\xd8"):]
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_encode_jpeg_roundtrip():
    jpeg = encode_jpeg(make_image(64, 48))
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)


# --- SharedRenderer --------------------------------------------------------------------------

def test_shared_renderer_renders_once_per_item():
    item = {"key": 1}
    calls = []

    def render(it):
        calls.append(it["key"])
        return make_image(value=10), make_image(value=20)

    r = SharedRenderer(lambda: item, lambda it: it["key"], render)
    k1, pics1 = r.get()
    k2, pics2 = r.get()            # same item: cached
    assert (k1, k2) == (1, 1) and calls == [1] and pics1 is pics2
    item["key"] = 2
    assert r.get()[0] == 2 and calls == [1, 2] and r.renders == 2


def test_shared_renderer_without_source_item():
    assert SharedRenderer(lambda: None, lambda it: 0, lambda it: ()).get() is None


def test_picture_stream_picks_its_picture():
    item = {"key": 1}
    r = SharedRenderer(lambda: item, lambda it: it["key"],
                       lambda it: (make_image(value=10), make_image(value=200)))
    first = next(picture_stream(r, 0))
    second = next(picture_stream(r, 1))
    assert decode(first)[20, 30, 0] < 50 and decode(second)[20, 30, 0] > 150


# --- pages -----------------------------------------------------------------------------------

def test_index_page_has_two_pictures_and_nav(running_rig):
    page = create_app(running_rig()).test_client().get("/").data
    assert b"/video/0" in page and b"/video/1" in page
    assert b'aria-current="page">Live view' in page
    assert b"rig-status" in page


def test_nav_lists_only_enabled_pages(running_rig):
    page = create_app(running_rig()).test_client().get("/").data
    for label in (b">Calibration<", b">Rectified<", b">Distance<", b">Objects<"):
        assert label not in page


def test_status_reports_cameras_and_sync(running_rig):
    rig = running_rig()
    assert wait_for(lambda: rig.latest() is not None)
    data = create_app(rig).test_client().get("/status").get_json()
    assert data["left"] == "Running" and data["right"] == "Running"
    assert isinstance(data["pair_dt_ms"], float | int)


def test_each_camera_has_its_own_stream(running_rig):
    rig = running_rig()   # left image value 10, right value 20 (see conftest)
    assert wait_for(lambda: rig.latest() is not None)
    client = create_app(rig).test_client()
    for index, value in ((0, 10), (1, 20)):
        resp = client.get(f"/video/{index}")
        img = decode(next(resp.response))
        resp.close()
        assert img.shape == (48, 64, 3)                 # one camera, not side by side
        assert abs(int(img[20, 30, 0]) - value) < 8
    assert client.get("/video/2").status_code == 404
