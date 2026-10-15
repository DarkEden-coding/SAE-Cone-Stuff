"""Media validation, capture intrinsics, and persistent device profiles."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inspect_media(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is not None:
        h, w = image.shape[:2]
        return {"kind": "photo", "image_size": [w, h], "n_frames": 1, "duration_s": None, "fps": None}
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError("Media must be a readable image or video")
        w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, count = video.get(cv2.CAP_PROP_FPS), int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = video.read()
        if not ok or frame is None or min(w, h) <= 0 or not np.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError("Video has no readable frames, frame count, or valid frame rate")
        return {"kind": "video", "image_size": [w, h], "n_frames": count, "duration_s": count / fps, "fps": fps}
    finally:
        video.release()


def scale_K(K: Any, source_size: list[int], target_size: list[int]) -> np.ndarray:
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0 or abs((sw / sh) / (tw / th) - 1) > 0.005:
        raise ValueError("Capture and media aspect ratios differ; intrinsics cannot be scaled")
    out = np.asarray(K, dtype=float).copy()
    out[0] *= tw / sw
    out[1] *= th / sh
    return out


def validate_K(value: Any, image_size: list[int]) -> np.ndarray:
    K = np.asarray(value, dtype=float)
    w, h = image_size
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be a finite 3x3 matrix")
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6:
        raise ValueError("K must be a row-major pinhole intrinsics matrix")
    if not (0.05 * w < K[0, 0] < 20 * w and 0.05 * h < K[1, 1] < 20 * h):
        raise ValueError("K focal lengths are implausible for the image size")
    if not (0 <= K[0, 2] <= w and 0 <= K[1, 2] <= h):
        raise ValueError("K principal point lies outside the image")
    return K


def validate_capture(value: Any, media: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != "cone-capture/1":
        raise ValueError("Expected schema cone-capture/1")
    if value.get("kind") != media["kind"]:
        raise ValueError("Capture kind does not match media")
    try:
        timestamp = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("created_at must include a timezone")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("created_at must be an ISO 8601 timestamp with a timezone") from error
    size = value.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("image_size must contain positive integer width and height")
    device = value.get("device")
    if not isinstance(device, dict) or not isinstance(device.get("model"), str) or not device["model"].strip():
        raise ValueError("device.model is required")
    if value.get("distortion") is not None:
        raise ValueError("Only undistorted pinhole captures (distortion: null) are supported")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames or (media["kind"] == "photo" and len(frames) != 1):
        raise ValueError("Capture requires frames; photos require exactly one frame")
    clean = []
    previous_index, previous_time = -1, -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each capture frame must be an object")
        index, t = frame.get("index"), frame.get("t")
        if type(index) is not int or index <= previous_index or index >= media["n_frames"]:
            raise ValueError("Frame indices must increase and lie within the media frame count")
        if type(t) not in (float, int) or not np.isfinite(t) or t < 0 or t <= previous_time:
            raise ValueError("Frame times must be finite, nonnegative, and strictly increasing")
        if media["duration_s"] is not None and t > media["duration_s"] + 1 / media["fps"]:
            raise ValueError("Capture frame time exceeds media duration")
        K = scale_K(validate_K(frame.get("K"), size), size, media["image_size"])
        clean.append({"index": index, "t": float(t), "K": K.tolist()})
        previous_index, previous_time = index, float(t)
    if clean[0]["index"] != 0 or abs(clean[0]["t"]) > 1e-3:
        raise ValueError("The first capture frame must have index 0 and time 0")
    return {**value, "image_size": media["image_size"], "frames": clean}


class CaptureIntrinsics:
    def __init__(self, capture: dict):
        self.frames = capture["frames"]
        self.by_index = {f["index"]: f for f in self.frames}
        self.times = np.asarray([f["t"] for f in self.frames])

    def frame(self, index: int, t: float) -> dict:
        found = self.by_index.get(index)
        if found is None:
            position = int(np.searchsorted(self.times, t))
            candidates = [max(0, position - 1), min(position, len(self.frames) - 1)]
            found = self.frames[min(candidates, key=lambda j: abs(self.times[j] - t))]
        return {"K": np.asarray(found["K"], dtype=float), "t": found["t"] if index in self.by_index else t}


class ProfileStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.records = json.loads(path.read_text()) if path.exists() else {}

    def list(self) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in item.items() if k != "samples"} for item in self.records.values()]

    def update(self, capture: dict) -> dict:
        device, size = capture["device"], capture["image_size"]
        key = hashlib.sha256(json.dumps([device["model"], size]).encode()).hexdigest()[:16]
        median = np.median([f["K"] for f in capture["frames"]], axis=0).tolist()
        with self.lock:
            record = self.records.get(key, {"id": key, "device": device, "image_size": size, "samples": []})
            record["samples"].append(median)
            record.update(device=device, K=np.median(record["samples"], axis=0).tolist(),
                          n_captures=len(record["samples"]), updated_at=now())
            self.records[key] = record
            write_json(self.path, self.records)
            return {k: v for k, v in record.items() if k != "samples"}

    def capture(self, profile_id: str, media: dict) -> dict:
        with self.lock:
            if profile_id not in self.records:
                raise ValueError("Unknown intrinsics profile")
            record = self.records[profile_id]
            K = scale_K(record["K"], record["image_size"], media["image_size"]).tolist()
            return {"schema": "cone-capture/1", "kind": media["kind"], "device": record["device"],
                    "created_at": now(), "image_size": media["image_size"], "distortion": None,
                    "frames": [{"index": i, "t": i / media["fps"] if media["fps"] else 0, "K": K}
                               for i in range(media["n_frames"])]}
