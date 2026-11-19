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
| 3 | Stereo calibration (ChArUco): board, capture, calibrate, rectified check | ✅ code done — needs a real capture set |
| 4 | Depth map and distance measurement | ✅ code done — tape-measure check after real calibration |
| 5 | YOLO-E object detection with distance labels | ✅ done — accuracy follows the calibration |

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

All pages share one layout: picture 1 | picture 2 | side panel (page text and controls,
then navigation). The header shows the camera sync status. As the window gets narrower,
the panel moves below the pictures, and then everything stacks into one column. Pictures
scale down to fit the window; only the display shrinks, not the stream. Each picture is
its own stream (`/video/0`, `/video/1`, `/depth/video/0`, …).

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
   Or turn on **Auto capture**: it saves a pair when the board is visible in both images,
   the image has been still for a moment (no motion blur), and the pose differs clearly from
   every pair already saved. Move to a pose, hold still for about 1 s, then move on. Auto
   capture runs while the /calibrate page is open.
5. Collect 20–30 pairs: cover the centre, the corners and the edges of the image, vary the
   distance (0.5–1.5 m) and the tilt (up to ~45°), and hold still.

Pairs are saved as PNG to `calibration/captures/` (`000_left.png`, `000_right.png`, …),
with `meta.json` recording the camera and board settings. To start over, delete that folder.
`--captures <folder>` selects another folder, e.g. one per board.

## Calibrate

```bash
.venv/bin/python scripts/calibrate.py calibration/captures   # -> calibration/stereo.npz
```

The script reads the board settings from the captures folder's `meta.json`, then calibrates
each camera, then the pair, and prints a report:

| Line | Good result |
|------|-------------|
| `RMS left / right`, `RMS stereo` | below 0.5 px |
| `focal left / right` | about 1254 px at 1280×960 (imx708, 4.74 mm lens) |
| `baseline` | about 87 mm, mostly in T's x component; y and z near 0 |
| `per-pair error` | no pair far above the others. Delete outliers and re-run. |

A good result needs 15–25 **varied** poses, including tilted ones. Straight-on views alone
can't pin down the focal length.

Restart `run_web.py` to load the new calibration, then open **/rectified**. With a good
calibration, each feature sits on the same green line in both images. The calibration is
only loaded if it was made with the current camera settings.

## Distance measurement

Needs a valid `calibration/stereo.npz`. Open **/depth**:

- Left: the rectified left image with a crosshair and the distance at that point (median over
  a small circle). Right: the depth heatmap (red = near, blue = far, black = no depth).
- Click either image to move the crosshair. **Centre** resets it.
- Depth is computed at `depth.scale` (default half resolution, about 6 frames/s on a Pi 5) in
  one background thread. It pauses when no depth page is open.
- No depth on plain, textureless surfaces, or in the band down the left edge that only the
  left camera sees.
- Closest measurable distance ≈ focal × scale × baseline / `num_disparities`, about 0.43 m
  with the defaults. Raise `num_disparities` to measure closer, at the cost of speed.

Expected accuracy with the 87 mm baseline: about 1 cm at 1 m, about 5 cm at 3 m. The error
grows with distance². Check it with a tape measure at 0.5, 1, 2 and 3 m.

## Object detection

YOLO-E (Ultralytics) with **text prompts**: the classes to look for are listed in the config.
The model runs on the Pi CPU through an NCNN export.

```bash
.venv/bin/python scripts/export_detector.py   # after changing detection.model / classes / imgsz
```

The first run downloads the model and its text encoder into `models/` (git-ignored) and needs
internet. The export bakes in the class list, and `run_web.py` refuses an export made for
different settings. Open **/objects**: each detection shows a box and "label distance",
plus a table sorted by distance.

- Distance = median depth inside the middle 60% of the box, so the background around the
  object's edges doesn't count. "no depth" means the object has no texture, or is closer
  than the closest measurable distance.
- Speed on a Pi 5 (`yoloe-26n-seg`, 640 px): depth ~140 ms + detection ~240 ms ≈ 2.5
  measured frames/s. `yoloe-26s-seg` is more reliable but about 2.5× slower.
- Use one page at a time: /depth and /objects together share the 4 CPU cores and both slow down.
- Wrong labels usually mean the real object isn't in the class list. Add it (e.g. `ruler`,
  `phone`) and re-export, or raise `detection.confidence`.

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
| `web.preview_scale` | Size of the browser stream (0.5 = half). Capture and processing always use full resolution. |
| `web.max_fps` | Frame-rate limit per picture in the browser (default 10). At 30 fps, one live picture is about 8 Mbit/s through the SSH tunnel, enough to make VS Code's connection drop. |
| `board.*` | ChArUco board: squares, measured square/marker size in mm, ArUco dictionary |
| `detection.*` | `enabled`, `model`, `classes` (text prompts), `imgsz`, `confidence`, `models_dir` |
| `depth.*` | Stereo matching: working `scale`, `num_disparities` (search range), `block_size`, SGBM `mode`, heatmap range, measurement radius |

The sensor mode, focus and output size must stay the same between calibration (Phase 3)
and measuring.

Unknown keys and invalid values are rejected at startup.

## Tests

```bash
.venv/bin/pytest              # all tests that run without cameras
.venv/bin/pytest -m hardware  # needs both cameras connected and not in use
.venv/bin/pytest -m model     # needs the exported YOLO-E detector
.venv/bin/ruff check src tests scripts
```

## Project layout

```
config/default.yaml           runtime configuration
scripts/run_web.py            start the web stream
scripts/make_board.py         generate the printable ChArUco board PDF
scripts/calibrate.py          compute the stereo calibration from captured pairs
scripts/export_detector.py    export YOLO-E with the configured classes to NCNN
calibration/                  generated board PDF, later calibration results
src/stereovision/
  config.py                   load and validate the YAML config
  camera/base.py              Frame + FrameSource interface
  camera/picam.py             Picamera2 source (the only module that imports picamera2)
  camera/fake.py              fake source for tests / offline development
  camera/stereo_rig.py        one capture thread per camera, rotation, timestamp pairing
  calibration/board.py        ChArUco board from config, printable page, corner detection
  calibration/capture.py      pair detection, overlay, CaptureStore, AutoCapture
  calibration/calibrate.py    intrinsics, stereo extrinsics, rectification, report
  calibration/io.py           save/load calibration .npz, camera settings check
  depth/rectify.py            Rectifier (remap tables), epipolar guide lines
  depth/disparity.py          StereoSGBM matcher, disparity at working scale
  depth/measure.py            disparity -> depth, point/box distance, heatmap
  depth/engine.py             DepthEngine (per frame), DepthWorker (background thread)
  web/depth.py                /depth page: crosshair distance, heatmap, click to measure
  detection/base.py           Detection + Detector interface (no ML library needed)
  detection/yoloe.py          YOLO-E export/load (the only module importing ultralytics)
  detection/worker.py         DetectionWorker: frame -> depth -> detections -> distances
  overlay.py                  boxes with 'label distance' tags
  web/objects.py              /objects page and /objects/list JSON
  web/stream.py               SharedRenderer (one render per frame) + per-picture MJPEG streams
  web/calibrate.py            /calibrate page: overlay stream, capture, delete last
  web/app.py                  Flask app, navigation, /, /video/<n>, /status, /rectified
  web/templates/base.html     shared page layout (pictures, side panel, navigation)
  web/templates, web/static   HTML and CSS
tests/                        pytest suite (hardware tests marked `hardware`)
```

Keep all camera hardware access behind `FrameSource`, so everything else can be tested with
`FakeSource`.
