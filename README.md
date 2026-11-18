# open_cv — Raspberry Pi 5 stereo camera

Two Raspberry Pi Camera Module 3 (imx708) cameras, mounted side by side about 87 mm
apart, used as a stereo pair. The goal is to measure the distance to objects, and later
to detect the objects with YOLO-E.

**Status: Phase 2 done.** Both cameras are rotated 90° anticlockwise and streamed side by side
to a web page. They run with fixed focus and a fixed sensor mode, and libcamera software sync
keeps their frames about 0.02 ms apart. The code is organised as a testable Python package.

| Phase | What | Status |
|-------|------|--------|
| 1 | Restructure into a package, config file, pytest | ✅ done |
| 2 | Lock focus and sensor mode, sync cameras, pair frames by timestamp | ✅ done |
| 3 | Stereo calibration (checkerboard / ChArUco) | planned |
| 4 | Depth map and distance measurement | planned |
| 5 | YOLO-E object detection with distance labels | planned |

## Setup (on the Pi)

```bash
sudo apt install python3-picamera2          # camera library, from apt (not pip)
python3 -m venv --system-site-packages .venv  # so the venv can see picamera2
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .                    # makes the stereovision package importable
```

Check that both cameras are detected:

```bash
rpicam-hello --list-cameras
```

## Run the stream

```bash
.venv/bin/python scripts/run_web.py           # uses config/default.yaml
.venv/bin/python scripts/run_web.py my.yaml   # or another config file
```

- Open http://127.0.0.1:5000 in a browser on the Pi.
- From your PC, use VS Code Remote-SSH: forward port **5000** in the **Ports** tab, then
  open http://localhost:5000. This happens automatically if you start the server in the
  VS Code terminal.
- http://127.0.0.1:5000/status shows the state of each camera, and `pair_dt_ms`, the time
  between the left and right frames. `in_sync` shows whether that gap is within
  `max_pair_dt_ms`.
- With sync on, the stream starts about 3–4 s after launch, while the cameras align.

The server holds both cameras. Stop it (Ctrl+C) before running anything else that uses them.

## Calibration board

```bash
.venv/bin/python scripts/make_board.py   # writes calibration/charuco_board.pdf
```

The board is a ChArUco board: 9×6 squares of 30 mm with 22 mm markers (DICT_5X5_100), on
A4 landscape.

1. Print at **100% / "Actual size"**, not "Fit to page".
2. Check the 100 mm scale bar with a ruler.
3. Measure a square (ideally across several squares, then divide). If it isn't exactly
   30 mm, set `board.square_mm` and `board.marker_mm` (marker = square × 22/30) in the config.
4. Glue the print onto something flat and stiff (foam board, glass, a clipboard). A curved
   board ruins the calibration.

## Calibration capture

1. Set the measured `board.square_mm` / `board.marker_mm` in the config **before** capturing.
   The board settings are stored with the captures, and mixing captures taken with different
   settings is refused.
2. Start `scripts/run_web.py` and open **/calibrate** (link on the main page).
3. Detected corners are drawn on both images. The count turns green when the pair is usable:
   the board is seen by both cameras, with at least 8 common corners.
4. Press **Capture pair** or the space bar. **Delete last** removes a bad pair.
5. Collect 20–30 pairs: cover the centre, the corners and the edges of the image, vary the
   distance (0.5–1.5 m) and the tilt (up to ~45°), and hold still.

Pairs are saved as PNG to `calibration/captures/` (`000_left.png`, `000_right.png`, …),
with `meta.json` recording the camera and board settings. To start over, delete that folder.
`--captures <folder>` selects another folder, e.g. one per board.

## Configuration — `config/default.yaml`

| Key | Meaning |
|-----|---------|
| `camera.width`, `camera.height` | Capture size, before rotation |
| `camera.rotation` | `none`, `cw90`, `ccw90` or `180`, applied to both images |
| `camera.left_index`, `camera.right_index` | Picamera2 camera numbers |
| `camera.swap_left_right` | `true` swaps the two cameras. Nearby objects should appear further **left** in the right image; if they don't, toggle this. |
| `camera.sensor_size` | Fixed sensor mode. `[2304, 1296]` is the full field of view (1536×864 is a centre crop). |
| `camera.frame_rate` | Fixed frame rate (both cameras) |
| `camera.lens_position` | Manual focus in dioptres = 1 / distance in m (`1.0` = 1 m, `0` = infinity). `null` = autofocus, which breaks calibration. |
| `camera.sync` | `true` = libcamera software sync. The left camera is the server, the right the client. |
| `camera.max_pair_dt_ms` | Drop left/right pairs further apart than this. Use `5` with sync, about `20` without. |
| `web.host`, `web.port`, `web.jpeg_quality` | Web server settings |
| `board.*` | ChArUco board: squares, measured square/marker size in mm, ArUco dictionary |

The sensor mode, focus and output size must stay the same between calibration (Phase 3)
and measuring.

Unknown keys and invalid values are rejected at startup.

## Tests

```bash
.venv/bin/pytest              # all tests that run without cameras
.venv/bin/pytest -m hardware  # needs both cameras connected and not in use
.venv/bin/ruff check src tests scripts
```

## Project layout

```
config/default.yaml           runtime configuration
scripts/run_web.py            start the web stream
scripts/make_board.py         generate the printable ChArUco board PDF
calibration/                  generated board PDF, later calibration results
src/stereovision/
  config.py                   load and validate the YAML config
  camera/base.py              Frame + FrameSource interface
  camera/picam.py             Picamera2 source (the only module that imports picamera2)
  camera/fake.py              fake source for tests / offline development
  camera/stereo_rig.py        one capture thread per camera, rotation, timestamp pairing
  calibration/board.py        ChArUco board from config, printable page, corner detection
  calibration/capture.py      pair detection, overlay drawing, CaptureStore (saved pairs)
  web/stream.py               MJPEG streaming helpers
  web/calibrate.py            /calibrate page: overlay stream, capture, delete last
  web/app.py                  Flask app: /, /video (MJPEG), /status, registers /calibrate
  web/templates, web/static   HTML and CSS
tests/                        pytest suite (hardware tests marked `hardware`)
```

Keep all camera hardware access behind `FrameSource`, so everything else can be tested with
`FakeSource`.
