"""Start the dual-camera web stream. Run on the Pi: python scripts/run_web.py [config.yaml]

Pages: /  live view,  /calibrate  calibration capture,  /rectified  calibration check,
       /depth  distance measurement,  /status  camera state (JSON).
"""

import argparse
from pathlib import Path

from stereovision.calibration.board import make_board
from stereovision.calibration.capture import CaptureStore, capture_meta
from stereovision.calibration.io import camera_mismatch, load_calibration
from stereovision.camera.picam import PicamSource
from stereovision.camera.stereo_rig import StereoRig
from stereovision.config import DEFAULT_CONFIG_PATH, CameraConfig, load_config
from stereovision.depth.engine import DepthEngine, DepthWorker
from stereovision.depth.rectify import Rectifier
from stereovision.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_DIR = ROOT / "calibration" / "captures"
DEFAULT_CALIBRATION = ROOT / "calibration" / "stereo.npz"


def make_source(cam: CameraConfig, camera_num: int, sync_role: str) -> PicamSource:
    return PicamSource(
        camera_num,
        cam.width,
        cam.height,
        sensor_size=cam.sensor_size,
        frame_rate=cam.frame_rate,
        lens_position=cam.lens_position,
        sync=sync_role if cam.sync else "off",
    )


def load_rectifier(path: Path, camera_meta: dict) -> Rectifier | None:
    """Rectifier from a calibration file, or None (with a message) if missing or stale."""
    if not path.exists():
        print(f"No calibration at {path} - /rectified disabled (run scripts/calibrate.py)")
        return None
    cal = load_calibration(path)
    mismatch = camera_mismatch(cal, camera_meta)
    if mismatch:
        print(f"Ignoring {path}: {mismatch}")
        return None
    print(f"Loaded calibration {path} (baseline {cal.baseline_mm:.1f} mm)")
    return Rectifier(cal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="folder for calibration image pairs")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help="stereo calibration file (.npz) for /rectified")
    args = parser.parse_args()
    cfg = load_config(args.config)

    cam = cfg.camera
    left_num, right_num = cam.camera_indices
    rig = StereoRig(
        make_source(cam, left_num, "server"),
        make_source(cam, right_num, "client"),
        rotation=cam.rotation,
        max_pair_dt_ns=int(cam.max_pair_dt_ms * 1e6),
    )
    meta = capture_meta(cam, cfg.board)
    store = CaptureStore(args.captures, meta)
    rectifier = load_rectifier(args.calibration, meta["camera"])
    depth_worker = None
    if rectifier is not None:
        depth_worker = DepthWorker(rig, DepthEngine.from_rectifier(rectifier, cfg.depth))
    rig.start()
    if depth_worker is not None:
        depth_worker.start()
    try:
        app = create_app(rig, cfg.web.jpeg_quality, board=make_board(cfg.board), store=store,
                         preview_scale=cfg.web.preview_scale, rectifier=rectifier,
                         depth_worker=depth_worker, depth_cfg=cfg.depth)
        app.run(host=cfg.web.host, port=cfg.web.port, threaded=True, use_reloader=False)
    finally:
        if depth_worker is not None:
            depth_worker.stop()
        rig.stop()


if __name__ == "__main__":
    main()
