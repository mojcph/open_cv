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


def test_software_sync_aligns_cameras():
    import threading

    from stereovision.camera.picam import PicamSource

    sources = [PicamSource(0, 640, 480, sync="server"), PicamSource(1, 640, 480, sync="client")]
    stamps: dict[int, list[int]] = {0: [], 1: []}

    def grab(i):
        sources[i].start()
        try:
            stamps[i].extend(sources[i].read().timestamp_ns for _ in range(20))
        finally:
            sources[i].close()

    threads = [threading.Thread(target=grab, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    gaps = [min(abs(a - b) for b in stamps[0]) for a in stamps[1][5:]]
    assert max(gaps) < 1_000_000  # under 1 ms; unsynced cameras are typically ~13 ms apart
