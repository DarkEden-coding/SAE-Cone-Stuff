"""Compare detection recall by labeled cone size on the validation split."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGES = sorted((ROOT / "dataset/valid/images").glob("*.jpg"))
MODELS = {
    "640 baseline": ROOT / "runs/yolo26m-640/weights/best.pt",
    "small-cone fine-tune": ROOT / "runs/yolo26m-640-small-cones/weights/best.pt",
}
BINS = (0, 4, 8, 16, 32, 1000)


def evaluate(name: str, weights: Path) -> None:
    """Match each labeled cone to a same-class detection at IoU >= 0.5."""
    counts = defaultdict(lambda: [0, 0])
    near_misses = defaultdict(int)
    model = YOLO(str(weights))
    for prediction in model.predict(source=str(ROOT / "dataset/valid/images"), stream=True, imgsz=640,
                                    conf=0.25, iou=0.7, device=0, batch=4, verbose=False):
        path = Path(prediction.path)
        targets = np.array([list(map(float, line.split())) for line in
                            (ROOT / "dataset/valid/labels" / f"{path.stem}.txt").read_text().splitlines()])
        if not len(targets):
            continue
        labels = targets[:, 0].astype(int)
        x, y, w, h = targets[:, 1:].T * 640
        ground_truth = np.stack((x-w/2, y-h/2, x+w/2, y+h/2), axis=1)
        boxes = prediction.boxes.xyxy.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        if len(boxes):
            lower = np.maximum(ground_truth[:, None, :2], boxes[None, :, :2])
            upper = np.minimum(ground_truth[:, None, 2:], boxes[None, :, 2:])
            intersect = np.maximum(upper - lower, 0).prod(axis=2)
            area_gt = (ground_truth[:, 2:] - ground_truth[:, :2]).prod(axis=1)
            area_pred = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
            overlaps = intersect / np.maximum(area_gt[:, None] + area_pred[None, :] - intersect, 1e-9)
            same_class = overlaps.copy()
            same_class[:, :] *= labels[:, None] == classes[None, :]
            matched = np.zeros(len(targets), dtype=bool)
            while same_class.size and same_class.max() >= 0.5:
                gi, pi = np.unravel_index(same_class.argmax(), same_class.shape)
                matched[gi] = True
                same_class[gi, :] = -1
                same_class[:, pi] = -1
        else:
            overlaps = np.zeros((len(targets), 0))
            matched = np.zeros(len(targets), dtype=bool)
        pixels = np.maximum(w, h)
        for index, size in enumerate(pixels):
            bucket = int(np.searchsorted(BINS, size, side="right") - 1)
            key = (BINS[bucket], BINS[bucket + 1])
            counts[key][0] += int(matched[index])
            counts[key][1] += 1
            if not matched[index] and overlaps.shape[1] and overlaps[index].max() >= 0.5:
                near_misses[key] += 1
    print(name)
    for key in sorted(counts):
        hit, total = counts[key]
        print(f"  longest box side {key[0]:>2}-{key[1]:<4} px: {hit:>5}/{total:<5} = {hit/total:.1%}; overlap without match: {near_misses[key]}")
    hit, total = map(sum, zip(*counts.values()))
    print(f"  total {hit}/{total} = {hit/total:.1%}")


if __name__ == "__main__":
    for name, weights in MODELS.items():
        evaluate(name, weights)
