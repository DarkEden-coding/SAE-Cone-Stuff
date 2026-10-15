# ConeCapture (iOS)

SwiftUI + ARKit app for iPhone 14 Pro (iOS 17+) that records photos/videos together with **per-frame ARKit
camera intrinsics** and uploads them to the cone-pose server (`cone_app/SPEC.md`, schema `cone-capture/1`,
`POST /api/captures`). No third-party dependencies.

**Hard rule:** ARKit supplies only the camera image and `camera.intrinsics`. The app never reads or sends
`camera.transform`, euler angles, view/projection matrices, anchors or any tracking pose. (The level
indicator uses CoreMotion gravity and is never uploaded.)

This code was written on Linux and has **never been compiled**. Expect to fix a few compile errors; places
of doubt are marked `// VERIFY:` (`grep -rn "VERIFY:" ConeCapture`).

## Build

```sh
brew install xcodegen
cd cone_app/ios/ConeCapture
xcodegen                      # generates ConeCapture.xcodeproj and Support/Info.plist
open ConeCapture.xcodeproj
```

1. Select the `ConeCapture` target > Signing & Capabilities; choose your Team (or set `DEVELOPMENT_TEAM` in
   `project.yml` and re-run `xcodegen`). Change the bundle id from `com.example.ConeCapture` to something unique
   (also change `Uploader.sessionIdentifier` and the Keychain service in `Keychain.swift` to match, optional).
2. Plug in the iPhone, enable Developer Mode (Settings > Privacy & Security), pick it as run destination, Run.
3. Tests (`ConeCaptureTests`: IntrinsicsConverter + CaptureManifest key layout): Cmd-U on any iOS simulator.
4. ARKit needs a real device for the camera; the simulator only runs the unit tests / UI shell.

Swift language mode is pinned to 5 in `project.yml` (works with Swift 5.9 through 6 toolchains); concurrency is
written to be Swift-6-friendly but not strict-checked.

## Usage

1. Settings: server URL (e.g. `https://pc.tailnet.ts.net:8444`, as exposed by `tailscale serve`), optional
   bearer token (Keychain; must match server `CONE_APP_TOKEN`), **Test connection** (`GET /api/health`).
2. Video format: defaults to ARKit's recommended 4K format (3840x2160). Autofocus is off by default.
3. Capture: Photo (high-res still, JPEG q0.95) or Video (H.264 MP4, ~50 Mbps at 4K). Media is stored in
   sensor-native landscape with no rotation; `K` is in those pixel coordinates.
4. Recordings list: pending / uploading (progress) / uploaded (job id, "Open in Safari" ->
   `{base}/#/job/<id>`) / failed (Retry). Local files live in `Documents/Recordings/<timestamp>/`
   (`video.mp4`|`photo.jpg`, `capture.json`, `meta.json`). Uploads use a background URLSession; the multipart
   body is first written to `Caches/Uploads` (needs free space equal to the video size).

## Field-recording checklist

- [ ] Phone charged (>50%), plenty of free storage (4K50Mbps = ~375 MB/min, plus a temporary copy for upload).
- [ ] Settings: server URL set, **Test connection** shows OK (on the same network path you will upload from).
- [ ] 4K format selected; autofocus OFF; remove any lens filter/case glare; clean the lens.
- [ ] Let the camera run ~5 s at the working distance, set focus once, then do not change format/autofocus
      mid-session (a locked focus keeps fx/fy stable; changing it changes intrinsics).
- [ ] Hold landscape; use the level indicator (roll within 1 degree) at the start.
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
- iPhone 14 Pro, iOS 17+. `UIRequiredDeviceCapabilities: arkit`, landscape-only.
- ATS is opened (`NSAllowsArbitraryLoads`) only so a plain `http://` tailnet address works for testing;
  Tailscale `serve` URLs are HTTPS anyway.
