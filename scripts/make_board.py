"""Generate a printable ChArUco calibration board PDF (A4 landscape) from the config.

Run on the Pi:  python scripts/make_board.py [config.yaml] [-o out.pdf] [--dpi 300]
"""

import argparse
from pathlib import Path

from PIL import Image

from stereovision.calibration.board import render_page
from stereovision.config import DEFAULT_CONFIG_PATH, load_config

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "calibration" / "charuco_board.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    board = load_config(args.config).board
    page = render_page(board, dpi=args.dpi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # resolution= sets the PDF page size, so the board prints at its real size at 100% scale.
    Image.fromarray(page).save(args.out, "PDF", resolution=args.dpi)
    print(f"Wrote {args.out}")
    print(
        f"{board.squares_x}x{board.squares_y} squares of {board.square_mm:g} mm. Print at 100% "
        "(actual size, no 'fit to page'), then measure a square and update board.square_mm "
        "and board.marker_mm in the config."
    )


if __name__ == "__main__":
    main()
