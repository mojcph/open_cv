"""Runs only on the Pi with both cameras attached: pytest -m hardware"""

import pytest

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("camera_num", [0, 1])
def test_picam_delivers_frame(camera_num):
    from stereovision.camera.picam import PicamSource

    source = PicamSource(camera_num, 640, 480)
    source.start()
    try:
        frame = source.read()
    finally:
        source.close()
    assert frame.image.shape == (480, 640, 3)
    assert frame.timestamp_ns > 0
