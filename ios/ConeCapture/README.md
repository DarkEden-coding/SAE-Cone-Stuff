# ConeCapture (iOS)

SwiftUI + ARKit app for iPhone 14 Pro (iOS 17+) that records photos/videos together with **per-frame ARKit
camera intrinsics** and uploads them to the cone-pose server (`cone_app/SPEC.md`, schema `cone-capture/1`,
`POST /api/captures`). No third-party dependencies.

**Hard rule:** ARKit supplies only the camera image and `camera.intrinsics` (plus, in Live mode, the 2D
`displayTransform` used to draw overlays on the preview). The app never reads or sends `camera.transform`,
euler angles, view/projection matrices, anchors or any tracking pose. CoreMotion gravity drives the level
indicator and capture orientation, and in Live mode it is the SLAM's tilt prior; raw gravity readings are
never uploaded.

Built and signed with Xcode 27 on macOS. The manifest and intrinsics unit tests pass on an iPhone 14 Pro.
Remaining device-specific assumptions are marked `// VERIFY:` in the source.

## Build

```sh
brew install xcodegen
cd ios/ConeCapture
xcodegen                      # generates ConeCapture.xcodeproj and Support/Info.plist
open ConeCapture.xcodeproj
```

1. Select the `ConeCapture` target > Signing & Capabilities; choose your Team (or set `DEVELOPMENT_TEAM` in
   `project.yml` and re-run `xcodegen`). Change the bundle id from `com.example.ConeCapture` to something unique
   (also change `Uploader.sessionIdentifier` and the Keychain service in `Keychain.swift` to match, optional).
2. Plug in the iPhone, enable Developer Mode (Settings > Privacy & Security), pick it as run destination, Run.
3. Tests (`ConeCaptureTests`: intrinsics, manifest, horizon, gravity rotation, JPEG and H.264 pixel checks): Cmd-U.
4. ARKit needs a real device for the camera; the simulator only runs the unit tests / UI shell.

If `xcodebuild` reports that the active developer directory is CommandLineTools, prefix build commands with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. For example, after generating the project:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project ConeCapture.xcodeproj -scheme ConeCapture \
  -destination 'platform=iOS,id=YOUR_IPHONE_UDID' \
  -allowProvisioningUpdates DEVELOPMENT_TEAM=YOUR_TEAM_ID build
```

On first launch, trust the development certificate under Settings > General > VPN & Device Management.
The optional `ConeCaptureSmoke` scheme runs a camera UI test requiring a physical iPhone and camera access.
Its separate test runner also needs an available app slot when using free developer signing. It saves a short
video and a photo locally, with automatic upload disabled during capture, and restores the upload setting.
The standard `ConeCapture` scheme runs only unit tests and creates no test recordings.

Swift language mode is pinned to 5 in `project.yml` (works with Swift 5.9 through 6 toolchains); concurrency is
written to be Swift-6-friendly but not strict-checked.

## Usage

1. Settings: server URL (e.g. `https://pc.tailnet.ts.net:8444`, as exposed by `tailscale serve`), optional
   bearer token (Keychain; must match server `CONE_APP_TOKEN`), **Test connection** (`GET /api/health`).
2. Video format: defaults to ARKit's recommended 4K format (3840x2160). Autofocus is off by default.
3. Capture: Photo (high-res still, JPEG q0.95) or Video (H.264 MP4, ~50 Mbps at 4K). Media is stored in
   the nearest gravity-upright orientation using 0/90/180/270-degree turns. The rotation is chosen at shutter
   press / recording start and stays fixed for the entire video, independent of the UI and rotation lock.
   Pixels are physically rotated; `K` and `image_size` match the resulting JPEG / decoded MP4 frames.
   Pointing nearly straight up/down keeps the last reliable orientation (initial fallback: portrait).
4. Recordings list: pending / uploading (progress) / uploaded (job id, "Open in Safari" ->
   `{base}/#/job/<id>`) / failed (Retry). Local files live in `Documents/Recordings/<timestamp>/`
   (`video.mp4`|`photo.jpg`, `capture.json`, `meta.json`). Uploads use a background URLSession; the multipart
   body is first written to `Caches/Uploads` (needs free space equal to the video size).

## Live mode

The **Live** mode runs everything on the phone at about 10–15 fps:

1. **Detection:** the YOLO26n pose model as CoreML (`ConeCapture/Resources/ConePose1280x736.mlpackage` and
   `ConePose736x1280.mlpackage`). The camera frame is rotated to the gravity-upright quarter turn chosen when
   Live mode starts, scaled to 1280×720 (or 720×1280), letterboxed to the model input, and decoded with
   class-aware NMS in Swift (`ConeDetector.swift`).
2. **Keypoint correction:** the box-height bias table from the server (`Resources/kp_correction.json`). It was
   fitted for the 1920 px server model; it is looked up by box height in inference pixels, which is only an
   approximation at 1280 px. Refit it on the synthetic set at 1280 for accurate ranges (setting: *Keypoint
   correction*).
3. **Per-cone position and live SLAM:** the `ConeSLAM` Swift package (`ConeSLAM/`, pure Swift, no
   dependencies). It ports the online half of `cone_app/slam.py`: pixel-space Hungarian association, robust
   pose refinement, relocalisation, landmark fusion, keyframes, windowed and periodic global bundle adjustment
   with the camera blocks eliminated, and duplicate merging. With gravity known, each cone's position is a
   3-parameter fit of the silhouette model instead of planar PnP. Gravity (CoreMotion, 100 Hz, interpolated
   to the frame timestamp) is a soft tilt prior (σ 0.015 rad); heading and position come only from cones. The
   server's offline passes (re-localising every frame against the final map, gap filling) are not run live.
4. **Streaming:** frame, map, recording and session messages (`cone-live/1`, `cone_app/SPEC.md`) go over a
   WebSocket to `{server}/api/live/ingest`. Only metadata is sent, never images. Open **Live** in the web UI
   to watch or replay a session.

On screen: boxes, keypoints and ranges over the preview, a heading-up minimap of the cone map (double-tap
it to reset the map), and a HUD line with tracking status, fps and timings. The shutter records video as in
Video mode; the recording also gets `live.jsonl`, the phone's own log of the live messages, which is
uploaded with the video so the web replay can sync the live results to the video.

Tests: `cd ConeSLAM && swift test -c release` runs the SLAM on the same simulated drives as
`cone_app/tests/test_slam.py` (fixtures from `ConeSLAM/Tests/make_fixtures.py`), checks projection parity with
the Python geometry and the CoreML output decoding against ultralytics. `ConeCaptureTests` checks the
orientation/gravity conversions and runs the CoreML detector on a real cone image in the simulator.
To re-export the models after retraining: `sh ios/ConeCapture/scripts/export_model.sh [weights.pt]`.

## Field-recording checklist

- [ ] Phone charged (>50%), plenty of free storage (4K50Mbps = ~375 MB/min, plus a temporary copy for upload).
- [ ] Settings: server URL set, **Test connection** shows OK (on the same network path you will upload from).
- [ ] 4K format selected; autofocus OFF; remove any lens filter/case glare; clean the lens.
- [ ] Let the camera run ~5 s at the working distance, set focus once, then do not change format/autofocus
      mid-session (a locked focus keeps fx/fy stable; changing it changes intrinsics).
- [ ] Hold portrait or landscape; use the level indicator (roll within 1 degree) at the start. The UI and
      horizon adapt to the interface orientation; saved media uses gravity at capture start. Keep a consistent
      hold during video: its output orientation stays fixed until recording stops.
- [ ] Keep all cones in view with the cone tops/bases visible; avoid pure rotation without translation (SLAM
      needs parallax); move smoothly, no digital zoom, avoid motion blur (good light).
- [ ] Watch the HUD: fps near the format fps; `drop` should stay near 0 (dropped frames are skipped in
      `capture.json` too, so the data stays consistent, but gaps hurt tracking).
- [ ] Give the capture a name (Settings) for easy identification; stop recording before leaving the app
      (recording auto-stops when the app is backgrounded).
- [ ] After recording, open Recordings and confirm "Uploaded" with a job id; tap "Open in Safari" to check.

## Assumptions

- Server returns the `Job` JSON with an `id` for `201` on `POST /api/captures` (per SPEC).
- Multipart fields: `capture` (JSON string, `application/json`) then `media` (`video/mp4` or `image/jpeg`).
- iPhone 14 Pro, iOS 17+. `UIRequiredDeviceCapabilities: arkit`, portrait and both landscape orientations.
- ATS is opened (`NSAllowsArbitraryLoads`) only so a plain `http://` tailnet address works for testing;
  Tailscale `serve` URLs are HTTPS anyway.
