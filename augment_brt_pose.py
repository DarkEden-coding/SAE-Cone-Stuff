"""Build deterministic, training-only cone pose augmentations with transformed labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasets/brt-cone-pose/brt-cone-pose-11k"
OUTPUT = ROOT / "datasets/brt-cone-pose/augmented-30k"


def distort(points, width, height, k):
    center = np.array([width / 2, height / 2])
    scale = max(width, height) / 2
    p = (points - center) / scale
    return center + scale * p * (1 + k * (p * p).sum(axis=-1, keepdims=True))


def geometry(image, rows, rng):
    h, w = image.shape[:2]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.92, 1.04)
    affine = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale), [0, 0, 1]])
    affine[:2, 2] += rng.uniform(-0.015, 0.015, 2) * [w, h]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    target = corners + (rng.uniform(-0.012, 0.012, (4, 2)) * [w, h]).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target) @ affine
    image = cv2.warpPerspective(image, matrix, (w, h), borderValue=(114, 114, 114))
    k = rng.uniform(-0.05, 0.05) if rng.random() < 0.35 else 0.0
    if k:
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        dest = np.stack([xx, yy], axis=-1)
        center = np.array([w / 2, h / 2], dtype=np.float32)
        normalized = (dest - center) / (max(w, h) / 2)
        src = normalized.copy()
        for _ in range(7):
            src = normalized / (1 + k * (src * src).sum(-1, keepdims=True))
        src = center + src * (max(w, h) / 2)
        image = cv2.remap(image, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    transformed = []
    for row in rows:
        xc, yc, bw, bh = row[1:5] * [w, h, w, h]
        left, top, right, bottom = xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2
        t = np.linspace(0, 1, 9)
        edge = np.concatenate([np.c_[left+t*bw, np.full(9, top)],
                               np.c_[left+t*bw, np.full(9, bottom)],
                               np.c_[np.full(9, left), top+t*bh],
                               np.c_[np.full(9, right), top+t*bh]])
        keypoints = row[5:].reshape(8, 3).copy()
        points = np.concatenate([edge, keypoints[:, :2] * [w, h]])
        homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
        points = homogeneous[:, :2] / homogeneous[:, 2:]
        points = distort(points, w, h, k)
        low, high = points[:36].min(0), points[:36].max(0)
        unclipped = np.prod(high-low)
        low, high = np.maximum(low, 0), np.minimum(high, [w, h])
        size = high-low
        if np.min(size) < 2 or np.prod(size) / max(unclipped, 1e-9) < 0.6:
            continue
        kp = points[36:]
        visible = (keypoints[:, 2] > 0) & (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
        keypoints[visible, :2] = kp[visible] / [w, h]
        keypoints[~visible] = 0
        if visible.sum() < 3:
            continue
        transformed.append(np.r_[row[0], (low+high)/2/[w,h], size/[w,h], keypoints.ravel()])
    return image, np.asarray(transformed).reshape(-1, 29), {"rotation":float(angle), "scale":float(scale), "radial_k":float(k)}


def generate(job):
    cv2.setNumThreads(1)
    image_path, variant_ids = job
    image_path = Path(image_path)
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Cannot decode {image_path}")
    # Preserve tiny-cone detail while limiting very large sensor frames.
    if max(original.shape[:2]) > 1600:
        factor = 1600 / max(original.shape[:2])
        original = cv2.resize(original, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    label_path = SOURCE / "labels/train" / f"{image_path.stem}.txt"
    rows = np.asarray([list(map(float,line.split())) for line in label_path.read_text().splitlines()
                       if line.strip()]).reshape(-1,29)
    records = []
    for index in variant_ids:
        seed = 20261010 + index
        rng = np.random.default_rng(seed)
        image, labels, params = geometry(original.copy(), rows, rng)
        if len(labels) == 0 and len(rows) > 0:
            image, labels, params = original.copy(), rows.copy(), {"geometry_fallback":True}
        blur = 0
        if rng.random() < 0.6:
            blur = int(rng.choice([3, 5, 7, 9]))
            kernel = np.zeros((blur, blur), np.float32)
            cv2.line(kernel, (0, blur//2), (blur-1, blur//2), 1, 1)
            rotation = cv2.getRotationMatrix2D((blur/2-0.5, blur/2-0.5), float(rng.uniform(0,180)), 1)
            kernel = cv2.warpAffine(kernel, rotation, (blur,blur))
            kernel /= max(kernel.sum(), 1e-9)
            image = cv2.filter2D(image, -1, kernel)
        contrast, brightness = float(rng.uniform(0.8,1.2)), float(rng.uniform(-22,22))
        image = np.clip(image.astype(np.float32)*contrast+brightness,0,255)
        if rng.random() < 0.3:
            image += rng.normal(0, rng.uniform(1,4), image.shape).astype(np.float32)
        image = np.clip(image,0,255).astype(np.uint8)
        name = f"{image_path.stem}__aug_{index:05d}"
        target = OUTPUT / "images/train" / f"{name}.jpg"
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(82,98))]):
            raise OSError(target)
        assert labels.shape[1] == 29 and np.isfinite(labels).all()
        assert ((labels[:,1:5] >= 0) & (labels[:,1:5] <= 1)).all()
        kp = labels[:,5:].reshape(-1,8,3)
        assert ((kp[:,:,:2] >= 0) & (kp[:,:,:2] <= 1)).all()
        np.savetxt(OUTPUT / "labels/train" / f"{name}.txt", labels, fmt=["%d"]+["%.8f"]*28)
        records.append({"image":str(target), "source":str(image_path), "seed":seed,
                        "objects":len(labels), "motion_blur_kernel":blur,
                        "contrast":contrast, "brightness":brightness, **params})
    return records


def main():
    global SOURCE, OUTPUT  # overridable from the CLI; read by the forked worker processes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source", type=Path, default=SOURCE, help="YOLO pose dataset to augment")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE, OUTPUT = args.source.resolve(), args.output.resolve()
    images = sorted((SOURCE / "images/train").glob("*.jpg"))
    if args.total < len(images):
        raise ValueError("Total must include all source training images")
    for sub in ("images/train","labels/train"):
        (OUTPUT/sub).mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(20261010)
    assignments = {str(p):[] for p in images}
    shuffled = rng.permutation(len(images))
    for index in range(args.total-len(images)):
        assignments[str(images[shuffled[index % len(images)]])].append(index)
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(generate, [(p, ids) for p, ids in assignments.items() if ids]):
            records.extend(result)
            if len(records)//1000 != (len(records)-len(result))//1000:
                print(f"Augmented {len(records)}/{args.total-len(images)}",flush=True)
    paths = [str(p) for p in images]+[r["image"] for r in records]
    assert len(paths)==args.total and len(set(paths))==args.total
    (OUTPUT/"train.txt").write_text("\n".join(paths)+"\n")
    config = yaml.safe_load((SOURCE/"data.yaml").read_text())
    config.update(path=str(SOURCE), train=str(OUTPUT/"train.txt"),
                  val=str(SOURCE/"images/val"), test=str(SOURCE/"images/test"))
    (OUTPUT/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    (OUTPUT/"provenance.json").write_text(json.dumps(records))
    print(f"Ready: {len(images)} original + {len(records)} augmented = {len(paths)} train images",flush=True)


if __name__ == "__main__":
    main()
