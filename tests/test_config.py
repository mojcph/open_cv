import pytest

from stereovision.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.camera.rotation == "ccw90"
    assert (cfg.camera.left_index, cfg.camera.right_index) == (0, 1)


@pytest.mark.parametrize(("swap", "expected"), [("false", (0, 1)), ("true", (1, 0))])
def test_swap_left_right(tmp_path, swap, expected):
    path = tmp_path / "c.yaml"
    path.write_text(f"camera:\n  swap_left_right: {swap}\n")
    assert load_config(path).camera.camera_indices == expected


def test_swap_must_be_bool(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  swap_left_right: yes please\n")
    with pytest.raises(ValueError, match="swap_left_right"):
        load_config(path)


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  widht: 640\n")
    with pytest.raises(ValueError, match="widht"):
        load_config(path)


def test_bad_rotation_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  rotation: sideways\n")
    with pytest.raises(ValueError, match="rotation"):
        load_config(path)


def test_phase2_defaults():
    cam = load_config(DEFAULT_CONFIG_PATH).camera
    assert cam.sensor_size == (2304, 1296)
    assert cam.sync is True
    assert cam.lens_position == 1.0


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("lens_position: 40", "lens_position"),
        ("frame_rate: 0", "frame_rate"),
        ("sensor_size: [2304]", "sensor_size"),
        ("max_pair_dt_ms: -1", "max_pair_dt_ms"),
        ("sync: maybe", "sync"),
    ],
)
def test_invalid_camera_values(tmp_path, line, match):
    path = tmp_path / "c.yaml"
    path.write_text(f"camera:\n  {line}\n")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_autofocus_allowed_with_null(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera:\n  lens_position: null\n")
    assert load_config(path).camera.lens_position is None
