"""One YOLO model, owned by the GPU worker, with original-pixel keypoints."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, weights: Path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights))
        self.last_ms_per_frame = 0.0

    def detect(self, images: list[np.ndarray], *, imgsz: int, conf: float) -> list[list[dict]]:
        started = perf_counter()
        predictions = self.model.predict(images, imgsz=imgsz, conf=conf, device=self.device,
                                         quantize=16 if self.device.startswith("cuda") else None, verbose=False)
        output = []
        for prediction in predictions:
            detections = []
            if prediction.boxes is not None and prediction.keypoints is not None:
                # Ultralytics scales both xyxy and keypoints.data back to orig_shape.
                boxes = prediction.boxes.xyxy.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy()
                scores = prediction.boxes.conf.cpu().numpy()
                keypoints = prediction.keypoints.data.cpu().numpy()
                for i, (box, cls, score, kpts) in enumerate(zip(boxes, classes, scores, keypoints)):
                    if kpts.shape != (8, 3):
                        raise ValueError(f"Expected 8 pose keypoints with confidence, got {kpts.shape}")
                    detections.append({"det_id": i, "class_id": int(cls), "conf": float(score),
                                       "box_xyxy": box.tolist(), "keypoints": kpts.astype(float)})
            output.append(detections)
        # The predictor caches its last source/results, including orig_img.
        # Retain the model, but release full-resolution pixels after each batch.
        self.model.predictor.results = None
        self.model.predictor.batch = None
        self.model.predictor.dataset = None
        self.last_ms_per_frame = (perf_counter() - started) * 1000 / len(images)
        return output
