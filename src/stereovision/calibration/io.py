"""Save/load StereoCalibration as .npz, with the capture settings it is valid for."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from stereovision.calibration.calibrate import StereoCalibration
from stereovision.calibration.capture import diff_keys

_SCALARS = ("rms_left", "rms_right", "rms_stereo")


def save_calibration(cal: StereoCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f.name: getattr(cal, f.name) for f in fields(cal)
              if f.name not in ("meta", "image_size", *_SCALARS)}
    np.savez(
        path,
        image_size=np.array(cal.image_size, np.int32),
        scalars=np.array([getattr(cal, s) for s in _SCALARS]),
        meta=np.array(json.dumps(cal.meta, sort_keys=True)),
        **arrays,
    )
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load_calibration(path: str | Path) -> StereoCalibration:
    with np.load(path, allow_pickle=False) as data:
        kwargs = {f.name: data[f.name] for f in fields(StereoCalibration)
                  if f.name not in ("meta", "image_size", *_SCALARS)}
        kwargs["image_size"] = tuple(int(v) for v in data["image_size"])
        kwargs.update(zip(_SCALARS, (float(v) for v in data["scalars"]), strict=True))
        kwargs["meta"] = json.loads(str(data["meta"]))
    return StereoCalibration(**kwargs)


def camera_mismatch(cal: StereoCalibration, camera_meta: dict) -> str | None:
    """Why the calibration doesn't fit the current camera settings, or None if it does."""
    stored = cal.meta.get("camera")
    if stored is None:
        return "calibration file has no camera settings recorded"
    diff = diff_keys(stored, camera_meta)
    if diff:
        return f"calibration was made with different camera settings: {', '.join(diff)}"
    return None
