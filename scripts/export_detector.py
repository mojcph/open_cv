"""Download the YOLO-E model and export it to NCNN with the config's text-prompt classes.

Run on the Pi after changing detection.model, detection.classes or detection.imgsz:

    python scripts/export_detector.py [config.yaml]

Takes about a minute; needs internet the first time (model + text encoder download).
"""

import argparse
import sys
from pathlib import Path

from stereovision.config import DEFAULT_CONFIG_PATH, load_config
from stereovision.detection.yoloe import check_export, export_detector, export_path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="re-export even if up to date")
    args = parser.parse_args()
    cfg = load_config(args.config).detection

    target = export_path(cfg, ROOT)
    if not args.force and check_export(target, cfg) is None:
        print(f"Up to date: {target}")
        return 0
    print(f"Exporting {cfg.model} for {len(cfg.classes)} classes: {', '.join(cfg.classes)}")
    path = export_detector(cfg, ROOT)
    print(f"Exported {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
