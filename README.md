# SAE cone detection, keypoints and pose

Perception work for a Formula Student / SAE driverless car. It covers YOLO cone detection, an 8-keypoint
cone pose model, PnP cone localisation, a synthetic Blender test set, and a web app with an iPhone capture app
that turns photos and videos into a 3D cone map with the camera's trajectory.

Python is managed with [uv](https://docs.astral.sh/uv/) (`uv sync --locked`, then `uv run --locked ...`).
The project uses Python 3.12+, Ultralytics YOLO and a CUDA GPU (developed on an RTX 5060).

Large data is not in git: datasets, renders, training runs, base weights, logs and app runtime data
(see `.gitignore`). The only checkpoint in the repo is `runs/by-n-E-far/weights/best.pt`, the model the
cone app serves.

## Contents

| Path | What it is |
|---|---|
| `cone_app/` | Cone pose web app: FastAPI backend, keypoint → PnP → cone-only SLAM pipeline, Three.js 3D viewer |
| `cone_app/ios/ConeCapture/` | iPhone app that records photo/video with per-frame ARKit intrinsics and uploads to `cone_app` |
| `web_view.py` | Older Gradio app for box-only cone detection (blue/yellow) on images and videos |
| `cone_pose_predict.py` | Command line: run the pose model on images/videos, export overlays and JSONL keypoints |
| `train_*.py`, `run_*.py`, `augment_*.py`, `clean_brt_pose.py`, `adjust_brt_labels.py` | BRT cone pose dataset preparation and training |
| `evaluate_brt_pose.py`, `compare_brt_pose_sizes.py`, `audit_recall.py`, `benchmark_brt_speed.py` | Evaluation and benchmarking |
| `synth/` | Blender synthetic cone test set, PnP evaluation, keypoint bias correction, reports ([synth/README.md](synth/README.md)) |
| `label_audit/` | Audit of BRT keypoint label quality and model bias ([label_audit/REPORT.md](label_audit/REPORT.md)) |
| `demos/` | Standalone HTML cone PnP demo |
| `datasets/brt-cone-pose/*.md` | Dataset assessment, full training flow, model size comparison |

## Cone pose web app (`cone_app/`)

An iPhone captures a photo or video together with ARKit **camera intrinsics only**. ARKit's camera pose is
never used. The capture is uploaded to the server, which:

1. detects cones and their 8 keypoints with `runs/by-n-E-far/weights/best.pt` at 1920 px;
2. applies a box-size-dependent keypoint bias correction (`cone_app/assets/kp_correction_E_1920.json`,
   fitted on synthetic renders);
3. solves planar PnP per cone for its base position in the camera frame;
4. for a **photo**: fits a ground plane through the cones and places camera and cones in a z-up world frame;
5. for a **video**: runs a small cone-only visual SLAM. Cones are the only landmarks, with no image features
   or IMU. It uses an optional flat-ground prior, Hungarian data association, robust tracking, relocalisation,
   and sliding-window plus global bundle adjustment on keypoint reprojection. The result is the camera
   trajectory and a cone map.

The web UI shows the media with box and keypoint overlays next to a 3D viewport. The viewport has cone
meshes, a camera frustum moving along the trajectory, and orbit, follow, top-down and through-the-camera
views. A timeline is synced to the video. Results can be exported as JSON or a landmarks CSV, and jobs can
be re-run with different options.

The API, capture format and result schema are specified in [cone_app/SPEC.md](cone_app/SPEC.md).

### Run the server

```sh
uv run --locked python -m cone_app.server --host 127.0.0.1 --port 7870
```

On the dev machine it runs as a systemd user service and is published privately on the tailnet:

```sh
systemctl --user status cone-app          # unit: ~/.config/systemd/user/cone-app.service
journalctl --user -u cone-app -f          # logs
tailscale serve --bg --https 8444 http://127.0.0.1:7870   # https://<machine>.<tailnet>.ts.net:8444/
```

Environment variables:

| Variable | Purpose |
|---|---|
| `CONE_APP_TOKEN` | Require a bearer token. The web UI asks for it once; set the same value in the iOS app. |
| `CONE_APP_DATA_DIR` | Job storage. Default: `cone_app_data/` |
| `CONE_APP_WEIGHTS` | Model checkpoint |
| `CONE_APP_IMGSZ` | Default inference size |
| `CONE_APP_BATCH_SIZE` | Detector batch size |

Jobs, media, results and intrinsics profiles are stored in `cone_app_data/`. The web page can also take a
manual photo/video upload; it reuses intrinsics saved from an earlier iOS capture from the same device at
the same resolution.

### Tests

```sh
uv run --locked python -m cone_app.tests.test_geometry
uv run --locked python -m cone_app.tests.test_slam       # simulated multi-minute drive, ~1 min
uv run --locked python -m cone_app.tests.test_api        # needs GPU + synth_data/
uv run --locked python -m cone_app.tests.test_pipeline   # needs GPU + synth_data/
```

To develop the frontend without the backend, run `node cone_app/static/dev/gen_mock.js` once, then open
`cone_app/static/index.html?mock=1`.

### iPhone app

See [cone_app/ios/ConeCapture/README.md](cone_app/ios/ConeCapture/README.md). In short:
`brew install xcodegen && cd cone_app/ios/ConeCapture && xcodegen && open ConeCapture.xcodeproj`. Set the
signing team, run it on the phone, and enter the server URL in Settings.

### Accuracy and limitations

- On synthetic renders, per-cone PnP has a median position error of about 1.4–1.7% of range. The keypoint
  correction table was fitted on synthetic images and has not been checked against measured real distances.
- SLAM has only been validated in simulation. On a 225 m simulated drive the trajectory error was about
  0.15 m and the cone error about 0.12 m. There is **no loop closure**, so revisited cones can appear twice.
- ARKit images are treated as pinhole (no distortion model).

## Older detector web view (`web_view.py`)

```sh
uv run --locked web_view.py --port 7860   # loopback; published with tailscale serve on :8443
uv run test_web_view.py                   # image/video check against the running app
```

It does box detection only, using `runs/yolo26m-640-small-cones/weights/best.pt` (not in git; pass
`--model PATH`). Images show annotated boxes and class counts. Videos are annotated per frame and
re-encoded to MP4, which needs system FFmpeg with `libopenh264`.

## Cone pose model

Training uses the BRT cone pose dataset under `datasets/brt-cone-pose/`; see
[FULL_TRAINING.md](datasets/brt-cone-pose/FULL_TRAINING.md). Classes: 0 BLUE, 1 ORANGE_BIG, 2 ORANGE,
3 UNDEFINED, 4 YELLOW. Keypoints are ring pairs (left, right) from top to bottom. Small cones use 6 points;
points 6–7 are used only for big orange cones.

- `runs/brt-pose-m-960-30k`: medium model, test box/pose mAP50–95 0.869 / 0.920, median keypoint error
  1.14 px.
- `runs/by-n-E-far`: nano model fine-tuned on cleaned blue/yellow labels plus far-range synthetic tiles.
  It has the best far-cone PnP and is the cone app's default.

```sh
uv run --locked cone_pose_predict.py path/to/image-or-video --output pose-predictions
```

This writes overlays and `predictions.jsonl` with original-pixel boxes and eight keypoints with confidences.

## Synthetic test set and PnP evaluation (`synth/`)

Blender Cycles renders of FS tracks with exact cone poses and keypoints, used to measure keypoint and PnP
accuracy by range, to fit the keypoint bias correction, and to build far-cone training tiles. Rendering,
evaluation and report commands are in [synth/README.md](synth/README.md).
