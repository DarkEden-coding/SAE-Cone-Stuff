# Synthetic cone keypoint / PnP test set

Blender Cycles renders of FS driverless tracks for evaluating keypoint models and PnP.
Blender 4.5 LTS is installed at `/mnt/Primary/tools/blender-4.5.14-linux-x64/`.

```sh
B=/mnt/Primary/tools/blender-4.5.14-linux-x64/blender
$B -b -P synth/render_blender.py -- --out synth/output/test --start 0 --count 1000 --skip-existing
uv run synth/postprocess.py synth/output/test --viz
```

- `cone_model.py`: cone geometry (FSG/WEMAS dimensions), keypoint definitions, silhouette math, camera model.
- `render_blender.py`: samples a track scene per frame (seeded by `--seed + index`), renders a linear
  multilayer EXR with an object-index pass, undistorted with overscan, plus `raw/<id>.json`.
- `postprocess.py`: lens distortion, chromatic aberration, softness, vignetting, auto exposure,
  RGGB sensor noise, demosaic, tone curve, sharpening, grain, JPEG; writes `images/`,
  `labels/` (YOLO pose, BRT classes), `annotations/` and `dataset.json`.

Scenes: blue left / yellow right boundaries, 3.0–4.5 m wide, spaced 2–5 m by curvature; 30% of frames
have a start line with big orange cones; 20% have a small-orange exit lane; 45% show a second track
section. Camera: 1920x1200, 80° HFOV, 0.95–1.15 m high, pitched 3–8° down. Depth of field and
motion blur (forward speed plus vibration) are rendered physically; ground truth is the mid-exposure pose.
Defect severity is drawn per frame (0–1) and every parameter is recorded in the annotation.

See `dataset.json` → `conventions` for frames, units and visibility codes. Per cone, annotations hold
`T_cam_cone` (translation = bottom face centre in the camera frame), the view-dependent silhouette
keypoints in the cone and camera frames, 2D keypoints in the distorted image, and boxes, visibility and range.
The fixed approximation is in `cone_models.*.planar_model_xyz_m`.

Assets: Poly Haven HDRIs and asphalt textures (CC0), in `assets/`.

## Keypoint model + PnP evaluation

```sh
uv run --locked synth/eval_pnp.py synth_data      # predictions cached in synth_data/eval/predictions/
uv run --locked synth/report_pnp.py synth_data    # summary.json + report.html
```

Runs nano/small/medium BRT pose checkpoints at 640/800/960/1280 px (batch 1, latency with CUDA sync),
matches detections to ground truth (IoU >= 0.5), solves planar IPPE PnP on keypoints with conf >= 0.5
(falls back to SQPnP/iterative when OpenCV 5.0 IPPE returns NaN), and compares with the true
bottom-face pose. Baselines run the same PnP on ground-truth keypoints. Per-cone rows: `eval/cones.csv`.

Static PDF (charts at 960 px, plus detections, keypoint close-ups and top-down PnP for one frame per model):
`uv run --locked synth/report_pdf.py synth_data [--frame 000780] [--out ~/Downloads/cone-pnp-report.pdf]`
