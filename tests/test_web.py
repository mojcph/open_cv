import cv2
import numpy as np
import pytest

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


def test_max_fps_limits_stream_rate(running_rig):
    import time

    rig = running_rig()   # fake cameras run at 200 fps
    assert wait_for(lambda: rig.latest() is not None)
    resp = create_app(rig, max_fps=5).test_client().get("/video/0")
    frames = resp.response
    next(frames)
    start = time.monotonic()
    for _ in range(3):
        next(frames)
    resp.close()
    assert time.monotonic() - start >= 0.55   # 3 frames at <= 5 fps take >= 0.6 s


def test_invalid_max_fps():
    import pytest

    from stereovision.config import WebConfig

    for bad in (0, 60):
        with pytest.raises(ValueError, match="max_fps"):
            WebConfig(max_fps=bad)


def test_shared_renderer_fps(monkeypatch):
    import stereovision.web.stream as stream

    clock = [100.0]
    monkeypatch.setattr(stream.time, "monotonic", lambda: clock[0])
    item = {"key": 0}
    r = SharedRenderer(lambda: item, lambda it: it["key"], lambda it: ())
    assert r.fps() == 0.0
    for k in range(5):  # a new picture every 0.1 s -> 10 fps
        item["key"] = k
        r.get()
        clock[0] += 0.1
    assert r.fps() == pytest.approx(10.0)
    clock[0] += 10  # nobody watching any more
    assert r.fps() == 0.0


def test_status_reports_camera_and_picture_rates(running_rig):
    rig = running_rig()
    client = create_app(rig).test_client()
    assert wait_for(lambda: client.get("/status").get_json()["camera_fps"] is not None)
    data = client.get("/status").get_json()
    assert data["camera_fps"] > 0 and data["picture_fps"] == {}
    stream = client.get("/video/0", buffered=False)
    chunks = stream.response  # read a few pictures
    for _ in range(3):
        next(chunks)
    stream.close()
    assert "video" in client.get("/status").get_json()["picture_fps"]
    html = client.get("/").get_data(as_text=True)
    assert "Picture rate" in html and '"video"' in html
