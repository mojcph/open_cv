# stereovision: trade-offs and settings

This rig measures distance with two Raspberry Pi Camera Module 3 cameras about 89 mm
apart. Almost every setting trades one thing against another: accuracy against speed,
sharpness against noise, or near range against far range. This page explains those
trade-offs for this rig and what to change for which goal. All settings are in
[config/default.yaml](../../config/default.yaml).

## How a distance is made

```
 left + right camera ──► pair by timestamp ──► rectify ──► match patches (SGBM)
   (synced, same           (≤ 5 ms apart)     (calibration)    │
    exposure)                                                  ▼
                       box from YOLO-E ──► median depth in box ──► distance
```

The matcher looks for the same small patch of detail in the left and right picture. How
far the patch has shifted sideways (the **disparity**, in pixels) gives the depth:

    depth Z = focal length (px) × baseline (mm) / disparity (px)

Everything below either changes how reliably patches can be matched, or how many pixels
of disparity one metre of distance is worth.

## Key numbers for this rig

From the current calibration (`calibration/stereo.npz`):

| Quantity | Value |
|---|---|
| Baseline (distance between cameras) | 88.8 mm |
| Focal length (rectified, full resolution) | 1285 px |
| Picture size after rotation | 960 × 1280 (portrait) |
| Closest measurable distance | ~0.45 m (`scale` 0.5, `num_disparities` 128) |

Disparity shrinks with distance, and the depth error grows with the **square** of the
distance. The error column below assumes the matcher is good to about ¼ pixel at
its working resolution. That is typical for a textured scene in good light, and worse on
plain surfaces or in poor light.

| Distance | Disparity (full res) | Error at `scale` 0.5 | Error at `scale` 1.0 |
|---|---|---|---|
| 1 m | 114 px | ± 4 mm | ± 2 mm |
| 5 m | 23 px | ± 0.11 m | ± 0.05 m |
| 10 m | 11 px | ± 0.44 m | ± 0.22 m |
| 20 m | 5.7 px | ± 1.8 m | ± 0.9 m |
| 30 m | 3.8 px | ± 3.9 m | ± 2.0 m |

So the rig is precise at room scale and rough at street scale. Beyond about 20 m the
distances are only indicative.

## Hardware: baseline

The cameras' distance apart is fixed by the mount, but it is the strongest lever there is.

- **Wider baseline:** proportionally more disparity per metre, so the error at a given
  distance drops by the same factor. Doubling to ~180 mm halves every error in the table.
- **Cost:** the closest measurable distance grows by the same factor. The two cameras also
  see less of the same scene up close, and the rig must be re-calibrated.
- For a street 10–30 m away, a baseline of 200–300 mm would be a real improvement.

## Camera settings

### Resolution and sensor mode (`width`, `height`, `sensor_size`)

1280 × 960 from the full-field 2304 × 1296 binned sensor mode. The binned mode keeps the
whole field of view and has less noise than a cropped mode. Higher resolution gives more
pixels of disparity, but costs CPU in every later step. **Changing any of these invalidates
the calibration.**

### Focus (`lens_position`)

Focus is fixed, because autofocus moves the lens and changes the focal length, which breaks
the calibration. The value is in dioptres (1 / distance in metres).

| Value | Sharp around | Good for |
|---|---|---|
| `1.0` (current) | 1 m | desk / room, calibration at close range |
| `0.2`–`0.25` | 4–5 m, and roughly 2 m → far | street, general outdoor |
| `0.0` | infinity | very distant scenes only |

Out-of-focus areas lose the fine detail the matcher needs, so the street at 1 m focus gives
fewer and noisier depths. **Changing focus requires a new calibration.** Calibrate with the
board at roughly the distance you focus at, or as close as the board still looks sharp.

### Frame rate, exposure and light (`frame_rate`, `match_exposure`)

At 30 fps, each frame can be exposed for at most ~33 ms. Auto exposure first lengthens
the exposure time, then raises the gain:

- **Long exposure:** moving objects smear. A car at 50 km/h, 15 m away, moves about 40
  pixels during 33 ms, but only about 12 pixels during 10 ms. Smeared edges match badly.
- **High gain:** more noise. Noise differs between the two cameras, so it creates false
  matches.
- **Lower `frame_rate`:** allows longer exposures and less gain, at the cost of more motion
  blur and slower updates. Higher `frame_rate` does the opposite.

The header of every page shows the current exposure as `exposure ms × gain`. A gain above
~4 means light is getting poor and depth will get noisier.

**`match_exposure: true` (default):** the left camera runs auto exposure and auto white
balance, and the right camera copies its exposure time, gain and colour gains. Without
this, each camera adjusts on its own, so the same object can be a little brighter in one
picture than the other, which the matcher sees as a different patch. When the light
changes, the right camera follows a few frames later. Those few frames can differ slightly,
which matters only when the light changes suddenly.

Set `match_exposure: false` if one camera is shaded differently, for example by a lens hood
or a smudge, and you'd rather each picture be well exposed on its own.

### Frame synchronisation (`sync`, `max_pair_dt_ms`)

libcamera software sync starts both sensors' exposures within ~0.02 ms of each other.
Pairs further apart than `max_pair_dt_ms` (5 ms) are dropped. Without sync, the cameras can
be up to half a frame (~17 ms) apart, and a moving car shifts between the left and right
pictures. That shift looks like disparity and gives a wrong distance. Keep `sync: true`
for anything that moves.

## Stereo matching (`depth:`)

| Setting | Current | Raise it | Lower it |
|---|---|---|---|
| `scale` | 0.5 | more detail, half the error at 1.0, ~4× slower | faster, coarser, larger errors |
| `num_disparities` | 128 | measures closer objects, slower | faster, closest distance moves out |
| `block_size` | 5 | smoother, fewer holes, blurs object edges | more detail, more noise |
| `uniqueness_ratio` | 10 | rejects more doubtful matches (more "no depth") | more depths, more wrong ones |
| `speckle_window` / `speckle_range` | 100 / 2 | removes more small isolated blobs | keeps more fine structure |
| `mode` | `3way` | `hh`: best quality, slowest | `sgbm`: middle |

For a street scene, nothing is close, so **`num_disparities` can drop to 64** (closest
distance becomes ~0.9 m). The time saved can pay for **`scale: 0.75`**, which gives more
precision at range. `display_max_mm` only sets the colour range of the depth heatmap.
Raise it to 20000–30000 for the street, or everything beyond 3 m shows as one colour.

Matching uses grayscale, so it needs **texture**: contrast and detail. Plain car doors,
smooth asphalt, sky and deep shadow give "no depth" or unreliable values in any setting.

## Distance of an object

The distance shown for an object is the **median depth inside its box**, shrunk 20% on each
side so the background around the outline doesn't count. The value is then converted to
the straight-line distance from the midpoint between the cameras to the box centre. The
median ignores scattered bad matches, but if most of the box is a plain surface, few
pixels are left to vote. This is why large, plain vehicles can show "no depth" while a
textured cyclist doesn't.

## Object detection (`detection:`)

| Setting | Trade-off |
|---|---|
| `model` | `yoloe-26n-seg`: ~9 detections/s, misses more. `yoloe-26s-seg`: ~4/s, more reliable, especially for small or distant objects |
| `imgsz` | network input size. Larger finds smaller, more distant objects but is slower; 640 is a good balance |
| `confidence` | lower (e.g. 0.15) finds more distant or partly hidden objects but gives more false boxes |
| `classes` | text prompts. Fewer classes are slightly faster and give fewer mix-ups. Wording matters: "cyclist" vs "bicycle" give different results |

After changing `model`, `classes` or `imgsz`, run `python scripts/export_detector.py`.

The Pi has four cores to share. While the Objects page is open, the depth page uses the
detection worker's depth maps instead of computing its own, so the two don't compete
for CPU. Both workers pause about 5 s after their page is closed.

## Browser stream (`web:`)

These change only what is sent to the browser, never what is measured.

| Setting | Trade-off |
|---|---|
| `preview_scale` | 0.5 = 640 × 480 stream. 1.0 is sharper on large screens; it needs ~4× the bandwidth and more CPU for JPEG encoding |
| `max_fps` | 10 fps ≈ 2.7 Mbit/s per picture through the SSH tunnel. Higher is smoother but heavier |
| `jpeg_quality` | 80. Higher is sharper with bigger frames; lower is blockier with smaller frames |

## What invalidates the calibration

The calibration stores these camera settings, and is ignored (with a message at startup)
if any of them change: `width`, `height`, `rotation`, `sensor_size`, `lens_position`,
`left_index`, `right_index`, `swap_left_right`. Physically moving one camera relative to
the other, even slightly, also invalidates the calibration, although the software can't
detect that.

Everything else, including frame rate, exposure, all `depth:`, `detection:` and `web:`
settings, can change freely.

## Suggested starting points

| | Desk / room (0.5–3 m) | Street (5–30 m) |
|---|---|---|
| `camera.lens_position` | 1.0 | 0.2 (then re-calibrate) |
| `camera.match_exposure` | true | true |
| `depth.scale` | 0.5 | 0.75 |
| `depth.num_disparities` | 128 | 64 |
| `depth.display_max_mm` | 3000 | 25000 |
| `detection.model` | yoloe-26n-seg | yoloe-26s-seg if 4/s is enough |
| `detection.confidence` | 0.25 | 0.15–0.25 |
| Hardware | 89 mm baseline is fine | a wider baseline (200–300 mm) helps the most |
