import cv2
import pytest

from stereovision.calibration.board import (
    board_size_mm,
    detect_corners,
    make_board,
    render_page,
)
from stereovision.config import BoardConfig

DEFAULT = BoardConfig()


def test_default_board_fits_a4():
    assert board_size_mm(DEFAULT) == (270.0, 180.0)


def test_page_is_a4_landscape_at_dpi():
    page = render_page(DEFAULT, dpi=100)
    assert page.shape == (827, 1169)  # 210 x 297 mm at 100 dpi


def test_rendered_page_detects_all_corners():
    page = render_page(DEFAULT, dpi=100)
    corners, ids = detect_corners(make_board(DEFAULT), page)
    inner = (DEFAULT.squares_x - 1) * (DEFAULT.squares_y - 1)
    assert len(ids) == inner == 40


def test_detects_partially_visible_board():
    page = render_page(DEFAULT, dpi=100)
    half = page[:, : page.shape[1] // 2]
    _, ids = detect_corners(make_board(DEFAULT), half)
    assert ids is not None and 0 < len(ids) < 40


def test_blank_image_detects_nothing():
    page = render_page(DEFAULT, dpi=100)
    page[:] = 255
    assert detect_corners(make_board(DEFAULT), page) == (None, None)


def test_board_too_big_for_paper():
    with pytest.raises(ValueError, match="does not fit"):
        render_page(BoardConfig(square_mm=40.0, marker_mm=30.0), dpi=50)


def test_unknown_dictionary():
    with pytest.raises(ValueError, match="dictionary"):
        make_board(BoardConfig(dictionary="DICT_NOPE"))


def test_dictionary_too_small():
    with pytest.raises(ValueError, match="too few markers"):
        make_board(BoardConfig(dictionary="DICT_4X4_50", squares_x=12, squares_y=9,
                               square_mm=20, marker_mm=15))


@pytest.mark.parametrize(("kw", "match"), [
    ({"marker_mm": 30.0}, "marker_mm"),
    ({"squares_x": 2}, "squares"),
    ({"dictionary": "5X5"}, "dictionary"),
])
def test_invalid_board_config(kw, match):
    with pytest.raises(ValueError, match=match):
        BoardConfig(**kw)


def test_board_uses_mm_units():
    board = make_board(DEFAULT)
    assert board.getSquareLength() == pytest.approx(30.0)
    assert board.getMarkerLength() == pytest.approx(22.0)
    assert isinstance(board, cv2.aruco.CharucoBoard)
