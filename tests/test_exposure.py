import pytest

from stereovision.camera.exposure import ExposureLink
from stereovision.web.app import exposure_status

META = {"ExposureTime": 20000, "AnalogueGain": 2.0, "ColourGains": (1.8, 1.5), "Lux": 300}


def test_follower_waits_for_leader():
    assert ExposureLink().follower_controls(None) is None


def test_follower_copies_leader_with_auto_off():
    link = ExposureLink()
    link.publish(META)
    ctrl = link.follower_controls(None)
    assert ctrl == {"AeEnable": False, "AwbEnable": False, "ExposureTime": 20000,
                    "AnalogueGain": 2.0, "ColourGains": (1.8, 1.5)}


def test_follower_updates_only_on_real_change():
    link = ExposureLink(tolerance=0.02)
    link.publish(META)
    applied = link.follower_controls(None)
    link.publish(META | {"ExposureTime": 20200})          # +1 %: ignored
    assert link.follower_controls(applied) is None
    link.publish(META | {"ColourGains": (1.8, 1.6)})      # +6.7 % blue gain: copied
    assert link.follower_controls(applied)["ColourGains"] == (1.8, 1.6)


def test_incomplete_metadata_is_ignored():
    link = ExposureLink()
    link.publish({"ExposureTime": 1000})
    assert link.latest() is None


def test_exposure_status():
    assert exposure_status(None) is None
    link = ExposureLink()
    assert exposure_status(link) is None
    link.publish(META)
    assert exposure_status(link) == {"exposure_ms": 20.0, "gain": 2.0}


def test_match_exposure_must_be_bool():
    from stereovision.config import CameraConfig

    with pytest.raises(ValueError):
        CameraConfig(match_exposure="yes")
