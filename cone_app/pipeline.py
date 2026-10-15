"""Bounded-memory media inference and assembly of cone-result/1."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .geometry import CLASS_NAMES, KeypointCorrection, cone_dims, single_frame_scene
from .intrinsics import CaptureIntrinsics

Progress = Callable[[float, str], None]


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def process_job(job: dict, media_path: Path, capture: dict, detector: Any,
                settings: Settings, progress: Progress) -> dict:
    started = perf_counter()
    options = job["options"]
    path = settings.correction_path(options["imgsz"]) if options["kp_correction"] else None
    correction = KeypointCorrection.load(path) if path else None
    intrinsics = CaptureIntrinsics(capture)
    frames: list[dict] = []
    detector_ms = 0.0

    def detect_batch(images: list[np.ndarray], metadata: list[tuple[int, float]]) -> None:
        nonlocal detector_ms
        predictions = detector.detect(images, imgsz=options["imgsz"], conf=options["conf"])
        if len(predictions) != len(metadata):
            raise ValueError("Detector returned the wrong number of frames")
        detector_ms += detector.last_ms_per_frame * len(images)
        for (index, t), detections in zip(metadata, predictions):
            intr = intrinsics.frame(index, t)
            for detection in detections:
                raw = np.asarray(detection["keypoints"], float).copy()
                detection["keypoints_raw"] = raw
                detection["keypoints"] = correction.apply(raw, detection["box_xyxy"]) if correction else raw.copy()
                height = detection["box_xyxy"][3] - detection["box_xyxy"][1]
                detection["range_scale"] = correction.range_scale(height) if correction else 1.0
            frames.append({"index": index, "t": intr["t"], "K": intr["K"], "detections": detections})

    progress(0.01, "decoding")
    if job["kind"] == "photo":
        image = cv2.imread(str(media_path))
        if image is None:
            raise ValueError("Photo could not be decoded")
        progress(0.05, "detecting")
        detect_batch([image], [(0, 0.0)])
        del image
    else:
        video = cv2.VideoCapture(str(media_path))
        try:
            if not video.isOpened():
                raise ValueError("Video could not be opened")
            fps = float(video.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Video has no valid FPS")
            stride = options["stride"] or max(1, round(fps / options["target_fps"]))
            images, metadata = [], []
            index, previous_t = 0, -1.0
            while video.grab():
                if index % stride == 0:
                    ok, image = video.retrieve()
                    if not ok:
                        raise ValueError(f"Cannot decode video frame {index}")
                    t = float(video.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    if not np.isfinite(t) or t <= previous_t:
                        t = index / fps
                    previous_t = t
                    images.append(image)
                    metadata.append((index, t))
                    del image
                    if len(images) >= settings.batch_size:
                        detect_batch(images, metadata)
                        images.clear()
                        metadata.clear()
                        progress(0.05 + 0.65 * min(1, (index + 1) / job["n_frames"]), "detecting")
                index += 1
            if images:
                detect_batch(images, metadata)
                images.clear()
            if index < job["n_frames"]:
                raise ValueError(f"Video ended after {index} of {job['n_frames']} declared frames")
        finally:
            video.release()
    if not frames:
        raise ValueError("Media produced no frames")

    # The scene interface has a fixed default confidence threshold. Binary masks
    # honor the requested threshold without changing confidences in the result.
    scene_frames = []
    for frame in frames:
        detections = []
        for detection in frame["detections"]:
            kpts = detection["keypoints"].copy()
            kpts[:, 2] = (kpts[:, 2] >= options["kp_conf"]).astype(float)
            detections.append({**detection, "keypoints": kpts})
        scene_frames.append({**frame, "detections": detections})

    progress(0.72, "slam")
    if job["kind"] == "photo":
        scene = single_frame_scene(scene_frames[0]["detections"], frames[0]["K"], options["flat_ground"])
        scene["frames"] = [{"index": 0, "T_world_cam": scene["T_world_cam"], "status": "single",
                            "measurements": scene["measurements"],
                            "assoc": {key: key for key in scene["measurements"]}}]
    else:
        from .slam import run_slam

        # Confidence masks above are already thresholded.
        scene = run_slam(scene_frames, {**options, "kp_conf": 0.5},
                         progress=lambda frac, stage: progress(0.72 + 0.25 * frac, stage))
    solved_frames = {frame["index"]: frame for frame in scene["frames"]}
    allowed_landmarks = set()
    for frame in frames:
        solved = solved_frames.get(frame["index"], {})
        frame.update(T_world_cam=solved.get("T_world_cam"), status=solved.get("status", "lost"))
        measurements, assoc = solved.get("measurements", {}), solved.get("assoc", {})
        for detection in frame["detections"]:
            det_id = detection["det_id"]
            measurement = measurements.get(det_id)
            if measurement is not None and np.linalg.norm(measurement.p_cam) > options["max_range_m"]:
                measurement = None
            landmark_id = assoc.get(det_id) if measurement is not None else None
            if landmark_id is not None:
                allowed_landmarks.add(landmark_id)
            detection.update(class_name=CLASS_NAMES[detection["class_id"]],
                             p_cam=measurement.p_cam if measurement else None,
                             range_m=float(np.linalg.norm(measurement.p_cam)) if measurement else None,
                             reproj_rms_px=measurement.reproj_rms_px if measurement else None,
                             landmark_id=landmark_id)
            detection.pop("range_scale", None)
    landmarks = [landmark for landmark in scene["landmarks"] if landmark["id"] in allowed_landmarks]
    progress(0.99, "writing")
    weights = settings.weights.resolve()
    from .config import ROOT

    result = {"schema": "cone-result/1", "job_id": job["id"], "kind": job["kind"],
              "image_size": job["image_size"], "options": options,
              "model": {"weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
                        "imgsz": options["imgsz"],
                        "kp_correction": str(path.relative_to(ROOT / "cone_app")) if path else None},
              "class_names": CLASS_NAMES, "cone_dims": {key: cone_dims(key) for key in CLASS_NAMES},
              "frames": frames, "landmarks": landmarks, "ground_plane": scene["ground_plane"],
              "stats": {**scene.get("stats", {}), "n_frames_total": job["n_frames"],
                        "n_frames_processed": len(frames),
                        "n_tracked": sum(frame["status"] in ("single", "init", "tracked") for frame in frames),
                        "n_lost": sum(frame["status"] == "lost" for frame in frames),
                        "n_landmarks": len(landmarks), "runtime_s": perf_counter() - started,
                        "detector_ms_per_frame": detector_ms / len(frames)}}
    return json_value(result)
