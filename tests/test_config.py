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
