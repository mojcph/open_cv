"""Detect the board in left/right frames, draw the detections and store good pairs to disk."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from stereovision.calibration.board import detect_corners

GREEN = (0, 200, 0)
RED = (0, 0, 230)

# Camera settings that change the images' geometry; calibration is only valid for these.
CAMERA_META_KEYS = ("width", "height", "rotation", "sensor_size", "lens_position",
                    "left_index", "right_index", "swap_left_right")


def capture_meta(camera_cfg, board_cfg) -> dict:
    """JSON-safe record of the settings a set of captures was taken with."""
    meta = {"camera": {k: getattr(camera_cfg, k) for k in CAMERA_META_KEYS},
            "board": asdict(board_cfg)}
    return json.loads(json.dumps(meta))  # tuples -> lists, so it compares equal to meta.json


@dataclass(frozen=True)
class ViewDetection:
    corners: np.ndarray | None  # Nx1x2 float32 pixel positions
    ids: np.ndarray | None      # Nx1 int32 ChArUco corner ids

    @property
    def count(self) -> int:
        return 0 if self.ids is None else len(self.ids)

    @property
    def id_set(self) -> set[int]:
        return set() if self.ids is None else {int(i) for i in self.ids.ravel()}


@dataclass(frozen=True)
class PairDetection:
    left: ViewDetection
    right: ViewDetection

    @property
    def common(self) -> int:
        """Corners seen by both cameras — only these are useful for stereo calibration."""
        return len(self.left.id_set & self.right.id_set)


def detect_view(board: cv2.aruco.CharucoBoard, image: np.ndarray) -> ViewDetection:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids = detect_corners(board, gray)
    return ViewDetection(corners, ids)


def detect_pair(board, left: np.ndarray, right: np.ndarray) -> PairDetection:
    return PairDetection(detect_view(board, left), detect_view(board, right))


def check_pair(det: PairDetection, min_common: int) -> str | None:
    """Reason the pair is unusable, or None if it is good enough to save."""
    if det.left.count == 0 or det.right.count == 0:
        missing = [side for side, v in (("left", det.left), ("right", det.right)) if v.count == 0]
        return f"board not found in {' and '.join(missing)} image"
    if det.common < min_common:
        return f"only {det.common} corners seen by both cameras (need {min_common})"
    return None


def draw_view(image: np.ndarray, det: ViewDetection, good: bool) -> np.ndarray:
    """Copy of image with detected corners and a count label."""
    out = image.copy()
    if det.corners is not None:
        # ids=None draws the corner markers only; id labels clutter the preview.
        cv2.aruco.drawDetectedCornersCharuco(out, det.corners, None, GREEN)
    color = GREEN if good else RED
    scale = image.shape[1] / 600  # keep the label readable at any resolution
    cv2.putText(out, f"{det.count} corners", (int(10 * scale), int(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color, max(2, int(2 * scale)), cv2.LINE_AA)
    return out


def mean_shift(a: ViewDetection, b: ViewDetection, min_shared: int = 4) -> float | None:
    """Mean pixel distance between corners with the same id in two detections.

    None if they share fewer than min_shared corners (too different to compare).
    """
    if a.ids is None or b.ids is None:
        return None
    pos_a = {int(i): p for i, p in zip(a.ids.ravel(), a.corners.reshape(-1, 2), strict=True)}
    pos_b = {int(i): p for i, p in zip(b.ids.ravel(), b.corners.reshape(-1, 2), strict=True)}
    shared = pos_a.keys() & pos_b.keys()
    if len(shared) < min_shared:
        return None
    return float(np.mean([np.linalg.norm(pos_a[i] - pos_b[i]) for i in shared]))


@dataclass(frozen=True)
class AutoCaptureSettings:
    """Thresholds as fractions of the image width, so they work at any resolution."""

    still_frac: float = 0.003    # corners moved less than this per frame = "still"
    still_frames: int = 4        # consecutive still frames before saving
    novelty_frac: float = 0.06   # must differ this much from every saved pose
    min_common: int = 8


class AutoCapture:
    """Decides, frame by frame, when to save a pair: board visible, held still, new pose.

    Pure logic (no I/O): feed it detections with update(), tell it about saves with
    saved()/forget_last(). Uses the left view for movement and novelty.
    """

    def __init__(self, settings: AutoCaptureSettings | None = None,
                 saved_views: list[ViewDetection] | None = None) -> None:
        self.settings = settings or AutoCaptureSettings()
        self._saved = list(saved_views or [])
        self._prev: ViewDetection | None = None
        self._still = 0
        self.message = "waiting for board"

    def update(self, det: PairDetection, image_width: int) -> bool:
        """True if this pair should be saved now. Sets self.message for the UI."""
        s = self.settings
        reason = check_pair(det, s.min_common)
        prev, self._prev = self._prev, det.left
        if reason is not None:
            self._still = 0
            self.message = reason
            return False

        shift = mean_shift(prev, det.left) if prev is not None else None
        if shift is None or shift > s.still_frac * image_width:
            self._still = 0
            self.message = "moving - hold still"
            return False
        self._still += 1
        if self._still < s.still_frames:
            self.message = "hold still..."
            return False

        distances = [mean_shift(v, det.left) for v in self._saved]
        nearest = min((d for d in distances if d is not None), default=None)
        if nearest is not None and nearest < s.novelty_frac * image_width:
            self.message = "pose already captured - move to a new position or angle"
            return False
        return True

    def saved(self, view: ViewDetection) -> None:
        self._saved.append(view)
        self._still = 0
        self.message = "saved - move to a new pose"

    def forget_last(self) -> None:
        if self._saved:
            self._saved.pop()


def diff_keys(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    for key in sorted(a.keys() | b.keys()):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff_keys(va, vb, f"{prefix}{key}.")
        elif va != vb:
            out.append(f"{prefix}{key}")
    return out


class CaptureStore:
    """Numbered image pairs in a folder: 000_left.png, 000_right.png, ... plus meta.json.

    meta.json records the camera settings; captures with different settings can't be mixed
    in one calibration, so saving is refused if they don't match.
    """

    META = "meta.json"

    def __init__(self, folder: str | Path, camera_meta: dict) -> None:
        self.folder = Path(folder)
        self.camera_meta = camera_meta
        self._lock = threading.Lock()

    def _indices(self) -> list[int]:
        if not self.folder.is_dir():
            return []
        found = []
        for p in self.folder.glob("*_left.png"):
            stem = p.name.removesuffix("_left.png")
            if stem.isdigit() and (self.folder / f"{stem}_right.png").exists():
                found.append(int(stem))
        return sorted(found)

    def count(self) -> int:
        return len(self._indices())

    def meta_mismatch(self) -> str | None:
        path = self.folder / self.META
        if not path.exists() or self.count() == 0:
            return None
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != self.camera_meta:
            diff = diff_keys(stored, self.camera_meta)
            return (f"existing captures used different camera settings ({', '.join(diff)}); "
                    f"clear {self.folder} first")
        return None

    def save(self, left: np.ndarray, right: np.ndarray) -> int:
        with self._lock:
            mismatch = self.meta_mismatch()
            if mismatch:
                raise ValueError(mismatch)
            self.folder.mkdir(parents=True, exist_ok=True)
            indices = self._indices()
            index = indices[-1] + 1 if indices else 0
            for side, image in (("left", left), ("right", right)):
                path = self.folder / f"{index:03d}_{side}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"could not write {path}")
            (self.folder / self.META).write_text(
                json.dumps(self.camera_meta, indent=2, sort_keys=True), encoding="utf-8"
            )
            return index

    def remove_last(self) -> int | None:
        with self._lock:
            indices = self._indices()
            if not indices:
                return None
            index = indices[-1]
            for side in ("left", "right"):
                (self.folder / f"{index:03d}_{side}.png").unlink(missing_ok=True)
            return index

    def load_pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs = []
        for i in self._indices():
            left = cv2.imread(str(self.folder / f"{i:03d}_left.png"))
            right = cv2.imread(str(self.folder / f"{i:03d}_right.png"))
            pairs.append((left, right))
        return pairs
