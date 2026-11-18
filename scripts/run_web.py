"""Start the dual-camera web stream. Run on the Pi: python scripts/run_web.py [config.yaml]"""

import argparse

from stereovision.camera.picam import PicamSource
from stereovision.camera.stereo_rig import StereoRig
from stereovision.config import DEFAULT_CONFIG_PATH, load_config
from stereovision.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    cfg = load_config(parser.parse_args().config)

    cam = cfg.camera
    left_num, right_num = cam.camera_indices
    rig = StereoRig(
        PicamSource(left_num, cam.width, cam.height),
        PicamSource(right_num, cam.width, cam.height),
        rotation=cam.rotation,
    )
    rig.start()
    try:
        app = create_app(rig, cfg.web.jpeg_quality)
        app.run(host=cfg.web.host, port=cfg.web.port, threaded=True, use_reloader=False)
    finally:
        rig.stop()


if __name__ == "__main__":
    main()
