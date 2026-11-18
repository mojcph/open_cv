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


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    web: WebConfig = field(default_factory=WebConfig)


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
    unknown = set(raw) - {"camera", "web"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return Config(
        camera=_build(CameraConfig, raw.get("camera")),
        web=_build(WebConfig, raw.get("web")),
    )
