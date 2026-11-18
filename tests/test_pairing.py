import pytest

from conftest import make_image, wait_for
from stereovision.camera import FakeSource, Frame
from stereovision.camera.stereo_rig import pair_nearest

IMG = make_image(4, 4)
MS = 1_000_000


def frames(*times_ms):
    return [Frame(IMG, int(t * MS)) for t in times_ms]


def test_empty_side_gives_none():
    assert pair_nearest([], frames(0)) is None
    assert pair_nearest(frames(0), []) is None


def test_identical_timestamps_pair_newest():
    pair = pair_nearest(frames(0, 33, 66), frames(0, 33, 66))
    assert pair.left.timestamp_ns == pair.right.timestamp_ns == 66 * MS
    assert pair.dt_ns == 0


def test_lagging_side_is_anchor():
    # Right has a newer frame (99) whose left partner hasn't arrived yet:
    # pair left's newest (66) with right's nearest (66), not right's newest.
    pair = pair_nearest(frames(0, 33, 66), frames(33, 66, 99))
    assert (pair.left.timestamp_ns, pair.right.timestamp_ns) == (66 * MS, 66 * MS)


def test_nearest_partner_chosen():
    pair = pair_nearest(frames(0, 33, 66), frames(30, 64, 100))
    assert pair.left.timestamp_ns == 66 * MS
    assert pair.right.timestamp_ns == 64 * MS
    assert pair.dt_ns == -2 * MS


@pytest.mark.parametrize(("max_dt_ms", "expect_pair"), [(5, False), (15, True)])
def test_max_dt_limit(max_dt_ms, expect_pair):
    pair = pair_nearest(frames(0, 33), frames(10, 43), max_dt_ns=max_dt_ms * MS)
    assert (pair is not None) == expect_pair


def test_rig_drops_out_of_sync_pairs(running_rig):
    # 200 fps = 5 ms period; a 2 ms offset is always the nearest gap
    rig = running_rig(
        left=FakeSource([IMG], fps=200),
        right=FakeSource([IMG], fps=200, offset_ns=2 * MS),
        max_pair_dt_ns=1 * MS,
    )
    assert wait_for(lambda: rig.status()["right"] == "Running")
    assert rig.latest() is None
    info = rig.sync_info()
    assert info["in_sync"] is False
    assert abs(info["pair_dt_ms"]) > 1.0  # 2 ms or -3 ms, depending on thread timing
    assert info["max_pair_dt_ms"] == 1.0


def test_rig_in_sync(running_rig):
    rig = running_rig(max_pair_dt_ns=1 * MS)
    assert wait_for(lambda: rig.latest() is not None)
    assert rig.sync_info()["in_sync"] is True
