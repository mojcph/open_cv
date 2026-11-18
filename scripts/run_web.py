"""Start the dual-camera web stream. Run on the Pi: python scripts/run_web.py [config.yaml]

Pages: /  live view,  /calibrate  calibration capture,  /status  camera state (JSON).
"""

import argparse
from pathlib import Path

from stereovision.calibration.board import make_board
from stereovision.calibration.capture import CaptureStore, capture_meta
from stereovision.camera.picam import PicamSource
from stereovision.camera.stereo_rig import StereoRig
from stereovision.config import DEFAULT_CONFIG_PATH, CameraConfig, load_config
from stereovision.web.app import create_app

DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parents[1] / "calibration" / "captures"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="folder for calibration image pairs")
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
    store = CaptureStore(args.captures, capture_meta(cam, cfg.board))
    rig.start()
    try:
        app = create_app(rig, cfg.web.jpeg_quality, board=make_board(cfg.board), store=store)
        app.run(host=cfg.web.host, port=cfg.web.port, threaded=True, use_reloader=False)
    finally:
        rig.stop()


if __name__ == "__main__":
    main()
