"""ChArUco board: build it from config and render a printable page.

All board lengths are in millimetres, so calibration results (baseline, depth) come out in mm.
"""

from __future__ import annotations

import cv2
import numpy as np

from stereovision.config import BoardConfig

MM_PER_INCH = 25.4
A4_LANDSCAPE_MM = (297.0, 210.0)


def make_board(cfg: BoardConfig) -> cv2.aruco.CharucoBoard:
    try:
        dict_id = getattr(cv2.aruco, cfg.dictionary)
    except AttributeError:
        raise ValueError(f"unknown ArUco dictionary {cfg.dictionary!r}") from None
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y), cfg.square_mm, cfg.marker_mm, dictionary
    )
    n_markers = len(board.getIds())
    if n_markers > dictionary.bytesList.shape[0]:
        raise ValueError(f"{cfg.dictionary} has too few markers for a {n_markers}-marker board")
    return board


def board_size_mm(cfg: BoardConfig) -> tuple[float, float]:
    return cfg.squares_x * cfg.square_mm, cfg.squares_y * cfg.square_mm


def _mm_to_px(mm: float, dpi: int) -> int:
    return round(mm / MM_PER_INCH * dpi)


def render_page(
    cfg: BoardConfig,
    dpi: int = 300,
    paper_mm: tuple[float, float] = A4_LANDSCAPE_MM,
    min_margin_mm: float = 8.0,
) -> np.ndarray:
    """White page (grayscale uint8) with the board centred, a label and a 100 mm scale bar."""
    board_w, board_h = board_size_mm(cfg)
    paper_w, paper_h = paper_mm
    if board_w + 2 * min_margin_mm > paper_w or board_h + 2 * min_margin_mm > paper_h:
        raise ValueError(
            f"board {board_w:.0f}x{board_h:.0f} mm does not fit on "
            f"{paper_w:.0f}x{paper_h:.0f} mm paper with {min_margin_mm:.0f} mm margins"
        )

    page = np.full((_mm_to_px(paper_h, dpi), _mm_to_px(paper_w, dpi)), 255, np.uint8)
    bw, bh = _mm_to_px(board_w, dpi), _mm_to_px(board_h, dpi)
    board_img = make_board(cfg).generateImage((bw, bh), marginSize=0, borderBits=1)
    x0 = (page.shape[1] - bw) // 2
    y0 = (page.shape[0] - bh) // 2
    page[y0 : y0 + bh, x0 : x0 + bw] = board_img

    # Label and 100 mm scale bar in the bottom margin, for checking the print scale.
    margin_px = page.shape[0] - (y0 + bh)
    text_y = y0 + bh + margin_px // 2 + _mm_to_px(1.0, dpi)
    scale = dpi / 300 * 0.8
    thick = max(1, dpi // 150)
    label = (
        f"ChArUco {cfg.squares_x}x{cfg.squares_y}  square {cfg.square_mm:g} mm  "
        f"marker {cfg.marker_mm:g} mm  {cfg.dictionary}  - print at 100% / actual size"
    )
    cv2.putText(page, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)
    bar_x1 = x0 + bw - _mm_to_px(100.0, dpi)
    bar_y = text_y - _mm_to_px(1.0, dpi)
    cv2.line(page, (bar_x1, bar_y), (x0 + bw, bar_y), 0, thick * 2)
    for x in (bar_x1, x0 + bw):
        cv2.line(page, (x, bar_y - _mm_to_px(1.5, dpi)), (x, bar_y + _mm_to_px(1.5, dpi)), 0, thick)
    cv2.putText(
        page, "100 mm", (bar_x1, bar_y - _mm_to_px(2.0, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA,
    )
    return page


def detect_corners(board: cv2.aruco.CharucoBoard, gray: np.ndarray):
    """Detect ChArUco corners. Returns (corners Nx1x2 float32, ids Nx1 int32), or (None, None)."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) == 0:
        return None, None
    return corners, ids
