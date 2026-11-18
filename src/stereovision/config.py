"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
ROTATIONS = ("none", "cw90", "ccw90", "180")


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    rotation: str = "ccw90"
    left_index: int = 0
    right_index: int = 1
    swap_left_right: bool = False
    sensor_size: tuple[int, int] = (2304, 1296)
    frame_rate: float = 30.0
    lens_position: float | None = 1.0
    sync: bool = True
    max_pair_dt_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}, got {self.rotation!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        for name in ("swap_left_right", "sync"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if len(self.sensor_size) != 2 or min(self.sensor_size) <= 0:
            raise ValueError("sensor_size must be [width, height]")
        object.__setattr__(self, "sensor_size", tuple(self.sensor_size))
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if self.lens_position is not None and not 0.0 <= self.lens_position <= 32.0:
            raise ValueError("lens_position must be between 0 and 32 dioptres, or null")
        if self.max_pair_dt_ms <= 0:
            raise ValueError("max_pair_dt_ms must be positive")

    @property
    def camera_indices(self) -> tuple[int, int]:
        """(left, right) Picamera2 camera numbers, with swap_left_right applied."""
        if self.swap_left_right:
            return self.right_index, self.left_index
        return self.left_index, self.right_index


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    jpeg_quality: int = 80
    preview_scale: float = 0.5  # shrink the browser stream only (saves SSH/Wi-Fi bandwidth)

    def __post_init__(self) -> None:
        if not 0.1 <= self.preview_scale <= 1.0:
            raise ValueError("preview_scale must be between 0.1 and 1.0")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class BoardConfig:
    """ChArUco calibration board. square_mm/marker_mm must be the MEASURED printed sizes."""

    squares_x: int = 9
    squares_y: int = 6
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_5X5_100"

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("squares_x and squares_y must be at least 3")
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError("marker_mm must be positive and smaller than square_mm")
        if not self.dictionary.startswith("DICT_"):
            raise ValueError(f"dictionary must be an ArUco DICT_* name, got {self.dictionary!r}")


SGBM_MODES = ("sgbm", "3way", "hh")


@dataclass(frozen=True)
class DepthConfig:
    """Stereo matching (OpenCV StereoSGBM) and distance display settings."""

    scale: float = 0.5            # match on images shrunk by this factor (speed vs detail)
    num_disparities: int = 128    # search range at the working scale; multiple of 16
    block_size: int = 5           # odd, 3-11; larger = smoother, fewer details
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    mode: str = "3way"
    display_min_mm: float = 300.0   # heatmap colour range
    display_max_mm: float = 3000.0
    measure_radius_px: int = 8      # full-resolution pixels around the measured point

    def __post_init__(self) -> None:
        if not 0.1 <= self.scale <= 1.0:
            raise ValueError("depth.scale must be between 0.1 and 1.0")
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("depth.num_disparities must be a positive multiple of 16")
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise ValueError("depth.block_size must be odd")
        if self.mode not in SGBM_MODES:
            raise ValueError(f"depth.mode must be one of {SGBM_MODES}")
        if not 0 < self.display_min_mm < self.display_max_mm:
            raise ValueError("depth.display_min_mm must be positive and below display_max_mm")
        if self.measure_radius_px < 1:
            raise ValueError("depth.measure_radius_px must be at least 1")


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    web: WebConfig = field(default_factory=WebConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)


def _build(cls: type, data: dict[str, Any] | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - {"camera", "web", "board", "depth"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return Config(
        camera=_build(CameraConfig, raw.get("camera")),
        web=_build(WebConfig, raw.get("web")),
        board=_build(BoardConfig, raw.get("board")),
        depth=_build(DepthConfig, raw.get("depth")),
    )
