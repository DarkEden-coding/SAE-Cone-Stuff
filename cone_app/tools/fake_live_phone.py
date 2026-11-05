"""Stream a synthetic ``cone-live/1`` session to a running server, paced in real time.

uv run --locked python -m cone_app.tools.fake_live_phone --url ws://127.0.0.1:7870 [--token T] [--rate 12]
"""

import argparse
from datetime import datetime, timezone
import json
import time
from urllib.parse import quote
from uuid import uuid4

import numpy as np
from websockets.sync.client import connect

from cone_app.geometry import project_cone_keypoints
from cone_app.tests.test_slam import K as SIM_K, simulate

IMAGE_SIZE = [1280, 720]
SCALE = IMAGE_SIZE[0] / 1920  # simulate() renders a 1920x1080 camera
WAITING_FRAMES = 5
MAP_EVERY_S, KEYFRAME_EVERY_S = 0.5, 0.5


def link_detections(frame: dict, pose: np.ndarray, landmarks: np.ndarray, classes: np.ndarray) -> list[dict]:
    """simulate() does not say which landmark a detection came from; recover it by reprojection."""
    projected = {i: project_cone_keypoints(int(classes[i]), landmarks[i], [0, 0, 1], pose, SIM_K).mean(axis=0)
                 for i in range(len(landmarks))}
    inverse = np.linalg.inv(pose)
    out = []
    for det in frame["detections"]:
        centre = np.asarray(det["keypoints"])[:, :2].mean(axis=0)
        near = [(np.linalg.norm(projected[i] - centre), i) for i in projected if classes[i] == det["class_id"]]
        distance, landmark = min((d, i) for d, i in near if np.isfinite(d)) if any(np.isfinite(d) for d, _ in near) \
            else (np.inf, None)
        known = landmark is not None and distance < 15
        keypoints = np.c_[np.asarray(det["keypoints"])[:, :2] * SCALE, np.asarray(det["keypoints"])[:, 2]]
        out.append({"class_id": det["class_id"], "conf": det["conf"],
                    "box_xyxy": [round(v * SCALE, 1) for v in det.get("box_xyxy", [*keypoints[:, :2].min(0), *keypoints[:, :2].max(0)])],
                    "keypoints": np.round(keypoints, 2).tolist(),
                    "p_cam": np.round((inverse @ [*landmarks[landmark], 1])[:3], 3).tolist() if known else None,
                    "landmark_id": int(landmark) if known else None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:7870")
    parser.add_argument("--token")
    parser.add_argument("--rate", type=float, default=12, help="frames per second")
    parser.add_argument("--name", default="Fake phone")
    parser.add_argument("--frames", type=int, default=600, help="one full lap is 600 frames")
    args = parser.parse_args()

    frames, truth, landmarks, classes, _ = simulate(args.frames)
    K = (SIM_K * SCALE).round(3)
    K[2, 2] = 1.0
    K_json = K.tolist()
    session_id = str(uuid4())
    recording_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rec_start, rec_stop = args.frames * 35 // 100, args.frames * 65 // 100
    url = args.url.rstrip("/") + "/api/live/ingest" + (f"?token={quote(args.token)}" if args.token else "")

    observations: dict[int, int] = {}
    keyframes: list[dict] = []
    next_map = next_keyframe = 0.0
    start = time.monotonic()
    with connect(url, max_size=None) as ws:
        def send(message: dict) -> None:
            ws.send(json.dumps(message, separators=(",", ":")))
            try:
                while True:
                    ws.recv(timeout=0)  # drain acks
            except TimeoutError:
                pass

        send({"type": "hello", "schema": "cone-live/1", "session_id": session_id, "t": 0.0,
              "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "name": args.name,
              "device": {"model": "FakePhone1,1", "system": "synthetic", "app_version": "dev"},
              "image_size": IMAGE_SIZE, "K": K_json, "model": {"name": "simulated", "imgsz": [736, 1280]},
              "options": {"conf": 0.25, "kp_conf": 0.5, "kp_correction": True, "flat_ground": True,
                          "gravity_prior": True}})
        print(f"session {session_id}", flush=True)
        for i, frame in enumerate(frames):
            t = i / args.rate
            time.sleep(max(0.0, start + t - time.monotonic()))
            if i == rec_start:
                send({"type": "recording", "t": round(t, 3), "event": "start", "recording_id": recording_id,
                      "name": "Fake lap"})
            waiting = i < WAITING_FRAMES
            detections = link_detections(frame, truth[i], landmarks, classes)
            for det in detections:
                if waiting:  # no map yet: detections only
                    det["p_cam"] = det["landmark_id"] = None
                elif det["landmark_id"] is not None:
                    observations[det["landmark_id"]] = observations.get(det["landmark_id"], 0) + 1
            send({"type": "frame", "i": i, "t": round(t, 3),
                  "status": "waiting" if waiting else "init" if i == WAITING_FRAMES else "tracked",
                  "T_world_cam": None if waiting else np.round(truth[i], 5).tolist(), "K": K_json,
                  "image_size": IMAGE_SIZE, "detections": detections,
                  "n_inliers": sum(d["landmark_id"] is not None for d in detections),
                  "rec": {"recording_id": recording_id, "t": round(t - rec_start / args.rate, 3)}
                  if rec_start <= i < rec_stop else None,
                  "stats": {"det_ms": 18.0, "slam_ms": 6.0, "fps": args.rate}})
            if not waiting and t >= next_keyframe:
                keyframes.append({"i": i, "t": round(t, 3), "T_world_cam": np.round(truth[i], 5).tolist()})
                next_keyframe = t + KEYFRAME_EVERY_S
            if not waiting and t >= next_map:
                send({"type": "map", "t": round(t, 3), "keyframes": keyframes,
                      "landmarks": [{"id": lid, "class_id": int(classes[lid]), "p_world": np.round(landmarks[lid], 3).tolist(),
                                     "sigma_m": [0.05, 0.05, 0.03], "n_obs": n, "confirmed": n >= 3}
                                    for lid, n in sorted(observations.items())]})
                next_map = t + MAP_EVERY_S
            if i + 1 == rec_stop:
                send({"type": "recording", "t": round(t, 3), "event": "stop", "recording_id": recording_id,
                      "name": "Fake lap"})
        send({"type": "bye", "t": round(len(frames) / args.rate, 3)})
    print("done", flush=True)


if __name__ == "__main__":
    main()
