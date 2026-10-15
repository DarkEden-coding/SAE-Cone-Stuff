# Cone Pose Web App — shared spec (contract between backend, SLAM, frontend, iOS)

Goal: an iPhone 14 Pro captures a photo or a video together with **ARKit per-frame camera intrinsics**
(intrinsics ONLY — the app must never read or send ARKit camera pose / transform / anchors), uploads it
directly to a FastAPI server on this machine (exposed privately over Tailscale). The server runs the cone
keypoint model (YOLO pose, 8 keypoints), solves per-cone PnP, and for videos runs a **cone-only visual
SLAM** (cones are the only landmarks/features — no ORB/optical flow/IMU) to recover the camera trajectory
and a cone map. A Three.js single-page app shows the media with overlays and a **3D viewport** (cones,
camera frustum(s), trajectory) for both photo and video jobs.

There is no in-app checkerboard/ChArUco calibration: intrinsics come only from the iOS app's ARKit data.

## Layout (file ownership in parentheses — do not edit files you don't own)

```
cone_app/
  SPEC.md                      (lead)
  __init__.py                  (backend)
  config.py                    (backend)   paths, defaults, env vars
  server.py                    (backend)   FastAPI app + static mount; `uv run --locked python -m cone_app.server`
  jobs.py                      (backend)   persistent job store + single GPU worker thread
  intrinsics.py                (backend)   capture.json parsing/validation, K scaling, intrinsics profiles
  detector.py                  (backend)   YOLO pose wrapper
  pipeline.py                  (backend)   photo/video processing -> cone-result/1 JSON
  geometry.py                  (slam)      cone model, keypoint correction, PnP, ground plane, single-frame scene
  slam.py                      (slam)      cone-only SLAM
  assets/kp_correction_*.json  (backend)   fitted correction table(s)
  tests/                       (backend: test_api.py, test_pipeline.py; slam: test_geometry.py, test_slam.py)
  static/                      (frontend)  index.html, app.js, *.js modules, style.css, vendor/three/...
  ios/ConeCapture/             (ios)       SwiftUI + ARKit capture/upload app, XcodeGen project.yml
cone_app_data/                 runtime data (jobs, media, results) — created by backend; /mnt/Primary disk
```

Python: use the repo's uv env (`uv run --locked ...`). Available: fastapi, uvicorn, python-multipart,
numpy, scipy, opencv 5.0 (has FFMPEG, decodes H.264), torch + CUDA (RTX 5060, 8 GB), ultralytics.
Do NOT add dependencies unless unavoidable (if so, `uv add` and say why). The home disk is nearly full:
keep all data under the repo (/mnt/Primary).

Existing code to reuse (read it, import from `synth/` or copy minimal pieces): `synth/cone_model.py`
(cone dimensions, `canonical_rings`, CLASS_NAMES), `synth/eval_pnp.py` (`planar_model`, `solve_pnp`),
`synth/kp_correction.py` (box-height keypoint bias correction + range scale; tables in
`synth_data/eval/kp_correction_hires.json` for model D — refit for model E), `web_view.py` (existing
Gradio app: loopback-only serving pattern).

Model: `runs/by-n-E-far/weights/best.pt`, default imgsz 1920. Classes: 0 BLUE, 1 ORANGE_BIG, 2 ORANGE,
3 UNDEFINED, 4 YELLOW. Keypoints per cone: rings top→bottom, each ring a (left, right) pair in image
(even index = left, odd = right). Small cones (0,2,3,4) use 3 rings → keypoints 0–5 (6,7 unused);
big orange (1) uses 4 rings → 0–7. Keypoint 3D model: ring at height z with radius r (from
`canonical_rings`), the two points are the silhouette tangent points ±r perpendicular to the viewing ray.

## Coordinate conventions

- Camera frame: OpenCV (x right, y down, z forward), metres.
- World frame: right-handed, **z up**, metres. Origin = ground point below the first initialized camera
  (or the camera centre if no ground plane); world x = first camera's forward direction projected onto
  the ground; y = left. If no ground plane can be estimated, world = camera frame rotated so cam z→world x,
  cam x→world −y, cam y→world −z, camera at origin.
- Matrices in JSON: nested lists, row-major. `T_world_cam` maps camera-frame points to world:
  `p_world = T_world_cam @ [p_cam, 1]`.
- Cone position = centre of the cone's base (ground contact), on its axis.

## capture.json (iOS → server), schema `cone-capture/1`

```json
{
  "schema": "cone-capture/1",
  "kind": "photo" | "video",
  "device": {"model": "iPhone15,2", "system": "iOS 26.0", "app_version": "1.0"},
  "created_at": "2026-10-15T12:00:00Z",
  "image_size": [3840, 2160],
  "frames": [
    {"index": 0, "t": 0.0, "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
  ],
  "distortion": null,
  "name": "optional user label",
  "notes": "optional"
}
```

- Media pixels are stored **as captured by ARKit (sensor-native landscape, no rotation)**; `K` is in the
  pixel coordinates of the stored media (`image_size` = media width/height). If the app ever scales
  images it must scale K identically.
- ARKit `camera.intrinsics` is a column-major `simd_float3x3`: fx = m[0][0], fy = m[1][1],
  cx = m[2][0], cy = m[2][1]. Convert to the row-major K above.
- `frames`: photo → exactly one entry; video → one per encoded video frame in order, `index` = encoded
  frame index, `t` = presentation time in seconds relative to the first frame (matches the MP4 PTS).
- `distortion: null` = pinhole (ARKit images are treated as undistorted).
- Video codec: **H.264** in MP4 (the server's system ffmpeg has no HEVC decoder; OpenCV decodes H.264;
  browsers play H.264). Photo: JPEG.

## HTTP API (server on 127.0.0.1, default port 7870; Tailscale serve proxies to it)

Auth: if env `CONE_APP_TOKEN` is set, every `/api/*` request except `GET /api/health` must send
`Authorization: Bearer <token>` (the web UI asks for the token once and stores it in localStorage; media URLs
also accept `?token=` so `<video src>` works). If unset, no auth (tailnet-only).

| Method | Path | Body / notes | Response |
|---|---|---|---|
| GET | `/api/health` | | `{"ok": true, "model": str, "imgsz": int, "device": str, "version": str, "auth": bool}` |
| POST | `/api/captures` | multipart: `media` (file), `capture` (capture.json string), optional `options` (JSON string) — used by iOS | 201 `Job` |
| POST | `/api/jobs` | multipart: `media` (file), and either `capture` (JSON string) or `profile_id` (str); optional `options`, `name` — used by web UI for manual photo/video upload with a stored intrinsics profile | 201 `Job` |
| GET | `/api/jobs` | | `[Job]` newest first |
| GET | `/api/jobs/{id}` | | `Job` |
| GET | `/api/jobs/{id}/media` | HTTP Range supported | original media file |
| GET | `/api/jobs/{id}/result` | 404 until done | `cone-result/1` JSON |
| POST | `/api/jobs/{id}/rerun` | JSON `options` (partial, merged) | `Job` (re-queued) |
| DELETE | `/api/jobs/{id}` | | 204 |
| GET | `/api/intrinsics-profiles` | profiles auto-saved from every iOS capture: per device model + image_size, median K | `[{"id","device","image_size","K","n_captures","updated_at"}]` |
| GET | `/` and `/static/*` | the SPA | |

`Job`:
```json
{"id": "20261015-120000-ab12", "name": "...", "kind": "photo"|"video", "status": "queued"|"running"|"done"|"failed",
 "stage": "decoding"|"detecting"|"slam"|"writing"|..., "progress": 0.0-1.0, "error": null|str,
 "created_at": iso, "updated_at": iso, "options": Options, "image_size": [w,h], "n_frames": int|null,
 "duration_s": float|null, "media_url": "/api/jobs/{id}/media", "result_url": "/api/jobs/{id}/result"|null,
 "source": "ios"|"web"}
```

`Options` (all optional, server fills defaults and echoes the effective values):
```json
{"stride": null, "target_fps": 10, "imgsz": 1920, "conf": 0.25, "kp_conf": 0.5,
 "kp_correction": true, "flat_ground": true, "max_range_m": 40.0}
```
(`stride` = process every Nth frame; if null, derived from `target_fps`.)

## Result JSON, schema `cone-result/1`

```json
{
  "schema": "cone-result/1",
  "job_id": "...", "kind": "photo"|"video", "image_size": [w, h],
  "model": {"weights": "runs/by-n-E-far/weights/best.pt", "imgsz": 1920, "kp_correction": "assets/...json"|null},
  "options": {...effective Options...},
  "class_names": {"0": "BLUE", "1": "ORANGE_BIG", "2": "ORANGE", "3": "UNDEFINED", "4": "YELLOW"},
  "cone_dims": {"0": {"height": 0.325, "base": 0.228}, "1": {"height": 0.505, "base": 0.285}, ...},
  "frames": [
    {
      "index": 0, "t": 0.0, "K": [[...]],
      "T_world_cam": [[...4x4...]] | null,
      "status": "single" | "init" | "tracked" | "lost",
      "detections": [
        {"det_id": 0, "class_id": 4, "class_name": "YELLOW", "conf": 0.91,
         "box_xyxy": [x1, y1, x2, y2],
         "keypoints": [[x, y, c], ... 8 entries, after correction if enabled],
         "keypoints_raw": [[x, y, c], ...],
         "p_cam": [x, y, z] | null,
         "range_m": float | null,
         "reproj_rms_px": float | null,
         "landmark_id": int | null}
      ]
    }
  ],
  "landmarks": [
    {"id": 0, "class_id": 4, "class_name": "YELLOW", "p_world": [x, y, z],
     "sigma_m": [sx, sy, sz], "n_obs": 12, "first_frame": 0, "last_frame": 40}
  ],
  "ground_plane": {"camera_height_m": 1.1, "normal_world": [0, 0, 1]} | null,
  "stats": {"n_frames_total": 900, "n_frames_processed": 300, "n_tracked": 290, "n_lost": 10,
            "n_landmarks": 85, "runtime_s": 42.0, "detector_ms_per_frame": 25.0}
}
```
For a photo: one frame with status `"single"`; `landmarks` = one per solved detection (`n_obs` 1),
`landmark_id` set accordingly. `frames[i].index` is the **video frame index**, `t` its time in seconds —
the frontend syncs to `<video>.currentTime` by nearest `t`.

## Python interfaces between backend and SLAM modules

`cone_app/geometry.py` (slam owner):
```python
CLASS_NAMES: dict[int, str]
def cone_dims(class_id: int) -> dict            # {"height": m, "base": m}
def ring_model(class_id: int) -> list[tuple[float, float]]   # (z, r) per ring, top->bottom
def n_keypoints(class_id: int) -> int           # 6 or 8
class KeypointCorrection:                       # box-height keypoint bias + range scale
    @classmethod
    def load(cls, path) -> "KeypointCorrection"
    def apply(self, kpts: np.ndarray, box_xyxy) -> np.ndarray   # (8,3) -> (8,3)
    def range_scale(self, box_h_px: float) -> float                  # multiply PnP translation
@dataclass
class ConeMeasurement:
    class_id: int; p_cam: np.ndarray; cov_cam: np.ndarray; reproj_rms_px: float; n_kp: int
def solve_cone(class_id, kpts, K, kp_conf=0.5, range_scale=1.0) -> ConeMeasurement | None
def project_cone_keypoints(class_id, p_world, up_world, T_world_cam, K) -> np.ndarray  # (n_kp, 2)
def fit_ground_plane(points_cam, weights=None) -> tuple[np.ndarray, float] | None  # (up normal in cam frame, camera height)
def single_frame_scene(detections: list[dict], K, flat_ground: bool) -> dict
    # detections: [{"det_id","class_id","conf","box_xyxy","keypoints"(8,3 corrected),"range_scale"}]
    # returns {"T_world_cam": 4x4 ndarray, "ground_plane": dict|None,
    #          "measurements": {det_id: ConeMeasurement}, "landmarks": [landmark dicts as in result JSON]}
```
`cone_app/slam.py` (slam owner):
```python
def run_slam(frames: list[dict], options: dict, progress=None) -> dict
    # frames: [{"index","t","K"(3x3 ndarray),"detections":[same as single_frame_scene]}] in time order
    # options: effective Options; progress(frac: float, stage: str) optional
    # returns {"frames": [{"index","T_world_cam": ndarray|None,"status": "init"|"tracked"|"lost",
    #                      "assoc": {det_id: landmark_id}, "measurements": {det_id: ConeMeasurement}}],
    #          "landmarks": [...], "ground_plane": dict|None, "stats": {...}}
```
The pipeline (backend) turns these into the result JSON (ndarray → lists).
