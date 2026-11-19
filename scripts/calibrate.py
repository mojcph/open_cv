"""Run stereo calibration on captured board pairs. Run on the Pi:

    python scripts/calibrate.py [captures_folder] [-o calibration/stereo.npz]

The board settings come from the folder's meta.json (what the pairs were captured with),
not from the current config.
"""

import argparse
import json
import sys
from pathlib import Path

from stereovision.calibration.board import make_board
from stereovision.calibration.calibrate import CalibrationError, calibrate_stereo, report
from stereovision.calibration.capture import CaptureStore, detect_pair
from stereovision.calibration.io import save_calibration
from stereovision.config import BoardConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURES = ROOT / "calibration" / "captures"
DEFAULT_OUT = ROOT / "calibration" / "stereo.npz"
EXPECTED_BASELINE_MM = 87.0
# imx708: 4.74 mm lens, 1.4 um pixels; 2304x1296 mode (2x2 binned) cropped to 1728x1296 (4:3)
# and scaled to the output width -> pixel pitch 2.8 um * 1728 / output_width.
LENS_MM, BINNED_PITCH_UM, CROP_WIDTH_PX = 4.74, 2.8, 1728


def expected_focal_px(meta: dict) -> float | None:
    cam = meta.get("camera", {})
    if cam.get("sensor_size") != [2304, 1296] or not cam.get("width"):
        return None
    pitch_mm = BINNED_PITCH_UM * CROP_WIDTH_PX / cam["width"] / 1000
    return LENS_MM / pitch_mm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", nargs="?", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline", type=float, default=EXPECTED_BASELINE_MM,
                        help="expected camera spacing in mm, for the sanity check")
    parser.add_argument("--square-mm", type=float,
                        help="corrected measured square size: overrides the captures' meta.json "
                             "(marker size scales along); no need to recapture")
    args = parser.parse_args()

    meta_path = args.captures / CaptureStore.META
    if not meta_path.exists():
        print(f"No {CaptureStore.META} in {args.captures} - is this a captures folder?")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pairs = CaptureStore(args.captures, meta).load_pairs()
    if args.square_mm:
        # Only the scale changes: corners are detected the same way, so the captures stay
        # valid. The result records the corrected size.
        old = meta["board"]["square_mm"]
        meta["board"]["marker_mm"] = round(meta["board"]["marker_mm"] * args.square_mm / old, 3)
        meta["board"]["square_mm"] = args.square_mm
        print(f"Using square {args.square_mm} mm (captured with {old} mm in meta.json)")
    board = make_board(BoardConfig(**meta["board"]))

    print(f"{len(pairs)} pairs in {args.captures}")
    detections = [detect_pair(board, left, right) for left, right in pairs]
    for i, d in enumerate(detections):
        print(f"  pair {i:3d}: left {d.left.count:2d}  right {d.right.count:2d}  "
              f"common {d.common:2d}")

    height, width = pairs[0][0].shape[:2] if pairs else (0, 0)
    try:
        cal = calibrate_stereo(board, detections, (width, height), meta=meta)
    except CalibrationError as exc:
        print(f"Calibration failed: {exc}")
        return 1

    print()
    print(report(cal, expected_baseline_mm=args.baseline,
                 expected_focal_px=expected_focal_px(meta)))
    out = save_calibration(cal, args.out)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
