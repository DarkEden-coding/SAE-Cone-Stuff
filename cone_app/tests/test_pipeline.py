"""Real-model integration tests; run with uv run --locked python -m unittest."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cone_app.config import Options, ROOT, Settings
from cone_app.detector import Detector
from cone_app.intrinsics import CaptureIntrinsics, ProfileStore, inspect_media, validate_capture
from cone_app.pipeline import process_job


def synth_capture(kind: str = "photo", count: int = 1, fps: float = 10) -> dict:
    calibration = json.loads((ROOT / "synth_data/dataset.json").read_text())["calibration"]
    return {"schema": "cone-capture/1", "kind": kind, "device": {"model": "synthetic-test"},
            "created_at": "2026-10-15T00:00:00Z", "image_size": [1920, 1200], "distortion": None,
            "frames": [{"index": i, "t": i / fps, "K": calibration["K"]} for i in range(count)]}


def make_video(path: Path, count: int = 8, fps: float = 10) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
                    "-framerate", str(fps), "-i", str(ROOT / "synth_data/images/000001.jpg"),
                    "-frames:v", str(count), "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
                    "-b:v", "8M", "-movflags", "+faststart", str(path)], check=True)


def iou_matrix(boxes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    lo = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    intersection = np.prod(np.maximum(hi - lo, 0), axis=2)
    area_a = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    area_b = np.prod(truth[:, 2:] - truth[:, :2], axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


class IntrinsicsTests(unittest.TestCase):
    def test_validation_scaling_and_nearest_time(self):
        capture = synth_capture("video", 4)
        media = {"kind": "video", "image_size": [960, 600], "n_frames": 4, "duration_s": .4, "fps": 10}
        capture["frames"] = [capture["frames"][0], capture["frames"][3]]
        scaled = validate_capture(capture, media)
        self.assertAlmostEqual(scaled["frames"][0]["K"][0][0], capture["frames"][0]["K"][0][0] / 2)
        self.assertEqual(CaptureIntrinsics(scaled).frame(2, .21)["t"], .21)
        with self.assertRaises(ValueError):
            validate_capture(capture, {**media, "image_size": [960, 540]})
        capture["frames"][1]["t"] = -1
        with self.assertRaises(ValueError):
            validate_capture(capture, media)

    def test_profiles_persist_median_and_scale(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cone_app_data") as directory:
            path = Path(directory) / "profiles.json"
            store = ProfileStore(path)
            capture = synth_capture()
            first = store.update(capture)
            capture["frames"][0]["K"][0][0] += 10
            second = store.update(capture)
            self.assertEqual(second["n_captures"], 2)
            self.assertAlmostEqual(second["K"][0][0], first["K"][0][0] + 5)
            loaded = ProfileStore(path)
            self.assertEqual(loaded.list()[0]["n_captures"], 2)
            media = {"kind": "photo", "image_size": [960, 600], "n_frames": 1, "fps": None}
            derived = loaded.capture(first["id"], media)
            self.assertAlmostEqual(derived["frames"][0]["K"][0][0], second["K"][0][0] / 2)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.detector = Detector(cls.settings.weights)

    def test_photo_metric_pose(self):
        errors = []
        with tempfile.TemporaryDirectory(dir=ROOT / "cone_app_data") as directory:
            for stem in ("000001", "000003", "000005"):
                annotation = json.loads((ROOT / f"synth_data/annotations/{stem}.json").read_text())
                K = np.asarray(annotation["camera"]["K"])
                # The synthetic corpus has lens distortion. The API contract is
                # pinhole, so rectify its real image with the supplied true K.
                image = cv2.imread(str(ROOT / f"synth_data/images/{stem}.jpg"))
                image = cv2.undistort(image, K, np.asarray(annotation["camera"]["dist"]))
                path = Path(directory) / "frame.jpg"
                cv2.imwrite(str(path), image)
                job = {"id": "test-photo", "kind": "photo", "image_size": [1920, 1200],
                       "n_frames": 1, "options": Options().model_dump()}
                result = process_job(job, path, synth_capture(), self.detector, self.settings, lambda *_: None)
                self.assertEqual(result["schema"], "cone-result/1")
                self.assertEqual(result["frames"][0]["status"], "single")
                detections = [d for d in result["frames"][0]["detections"] if d["p_cam"] is not None]
                truth = [c for c in annotation["cones"] if c["class_id"] in (0, 4) and 2 <= c["distance_m"] <= 30]
                overlaps = iou_matrix(np.array([d["box_xyxy"] for d in detections]),
                                      np.array([c["bbox_xyxy_clipped"] for c in truth]))
                for d, g in zip(*linear_sum_assignment(-overlaps)):
                    if overlaps[d, g] < .5 or detections[d]["class_id"] != truth[g]["class_id"]:
                        continue
                    position = np.asarray(truth[g]["T_cam_cone"])[:3, 3]
                    errors.append(np.linalg.norm(np.asarray(detections[d]["p_cam"]) - position) / np.linalg.norm(position))
                json.dumps(result, allow_nan=False)
        median = float(np.median(errors))
        print(f"\nPhoto GPU accuracy: n={len(errors)}, median relative position error={100 * median:.3f}%", flush=True)
        self.assertGreater(len(errors), 15)
        self.assertLess(median, .05)

    def test_h264_video_pipeline(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "cone_app_data") as directory:
            path = Path(directory) / "test.mp4"
            make_video(path)
            media = inspect_media(path)
            capture = validate_capture(synth_capture("video", 8), media)
            options = Options(stride=2).model_dump()
            job = {"id": "test-video", **media, "options": options}
            progress = []
            result = process_job(job, path, capture, self.detector, self.settings,
                                 lambda frac, stage: progress.append((frac, stage)))
            self.assertEqual([f["index"] for f in result["frames"]], [0, 2, 4, 6])
            np.testing.assert_allclose([f["t"] for f in result["frames"]], [0, .2, .4, .6])
            self.assertEqual(result["stats"]["n_frames_total"], 8)
            self.assertGreater(result["stats"]["n_tracked"], 0)
            self.assertGreater(len(result["landmarks"]), 0)
            self.assertTrue(all(len(d["keypoints_raw"]) == 8 for f in result["frames"] for d in f["detections"]))
            json.dumps(result, allow_nan=False)
            print(f"\nVideo GPU: {result['stats']}", flush=True)


if __name__ == "__main__":
    unittest.main()
