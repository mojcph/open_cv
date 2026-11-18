import cv2
import numpy as np

from conftest import make_image, wait_for
from stereovision.camera import Frame, StereoFrame
from stereovision.web.app import create_app
from stereovision.web.stream import encode_jpeg, side_by_side


def test_encode_side_by_side_doubles_width():
    pair = StereoFrame(Frame(make_image(64, 48), 0), Frame(make_image(64, 48), 5))
    jpeg = encode_jpeg(side_by_side(pair))
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 128, 3)


def test_index_page(running_rig):
    client = create_app(running_rig()).test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"/video" in resp.data


def test_status_reports_cameras_and_sync(running_rig):
    rig = running_rig()
    assert wait_for(lambda: rig.latest() is not None)
    data = create_app(rig).test_client().get("/status").get_json()
    assert data["left"] == "Running" and data["right"] == "Running"
    assert isinstance(data["pair_dt_ms"], float | int)


def test_video_stream_yields_jpeg(running_rig):
    rig = running_rig()
    assert wait_for(lambda: rig.latest() is not None)
    resp = create_app(rig).test_client().get("/video")
    chunk = next(resp.response)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    resp.close()
