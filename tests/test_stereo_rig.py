import numpy as np
import pytest

from conftest import make_image, wait_for
from stereovision.camera import FakeSource, rotate


@pytest.mark.parametrize(
    ("rotation", "marker_at"),
    [("none", (0, 0)), ("ccw90", (63, 0)), ("cw90", (0, 47)), ("180", (47, 63))],
)
def test_rotate_moves_top_left_marker(rotation, marker_at):
    out = rotate(make_image(64, 48), rotation)
    assert tuple(out[marker_at]) == (255, 0, 0)


def test_rotate_rejects_unknown():
    with pytest.raises(ValueError):
        rotate(make_image(), "sideways")


def test_rig_delivers_rotated_pair(running_rig):
    rig = running_rig(rotation="ccw90")
    assert wait_for(lambda: rig.latest() is not None)
    pair = rig.latest()
    assert pair.left.image.shape == (64, 48, 3)  # 64x48 portrait after rotation
    assert pair.left.image[1, 1, 0] == 10 and pair.right.image[1, 1, 0] == 20
    assert rig.status() == {"left": "Running", "right": "Running"}


def test_rig_reports_camera_error(running_rig):
    broken = FakeSource([make_image()], start_error=RuntimeError("no camera"))
    rig = running_rig(right=broken)
    assert wait_for(lambda: rig.status()["right"].startswith("ERROR"))
    assert "no camera" in rig.status()["right"]
    assert rig.latest() is None
    assert broken.closed


def test_rig_stop_closes_sources(running_rig):
    left, right = FakeSource([make_image()]), FakeSource([make_image()])
    rig = running_rig(left=left, right=right)
    assert wait_for(lambda: rig.latest() is not None)
    rig.stop()
    assert left.closed and right.closed


def test_fake_source_requires_images():
    with pytest.raises(ValueError):
        FakeSource([])


def test_pair_dt_ns(running_rig):
    rig = running_rig()
    assert wait_for(lambda: rig.latest() is not None)
    assert isinstance(rig.latest().dt_ns, int)
    assert np.isfinite(rig.latest().dt_ns)
