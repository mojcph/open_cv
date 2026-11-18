import time

import numpy as np
import pytest

from stereovision.camera import FakeSource, StereoRig


def make_image(width=64, height=48, value=0):
    """Image with a marker in the top-left corner so rotations are detectable."""
    img = np.full((height, width, 3), value, np.uint8)
    img[0, 0] = (255, 0, 0)
    return img


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running_rig():
    rigs = []

    def _make(left=None, right=None, rotation="none", max_pair_dt_ns=None):
        rig = StereoRig(
            left or FakeSource([make_image(value=10)], fps=200),
            right or FakeSource([make_image(value=20)], fps=200),
            rotation=rotation,
            max_pair_dt_ns=max_pair_dt_ns,
        )
        rig.start()
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.stop()
