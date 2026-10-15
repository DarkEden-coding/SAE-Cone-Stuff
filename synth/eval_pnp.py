"""Evaluate YOLO pose models + PnP on the synthetic set: latency and cone pose error.

uv run --locked synth/eval_pnp.py synth_data [--imgsz 640 800 960 1280]

For each model and input size: batch-1 prediction on every frame (end-to-end latency with
CUDA sync, image already decoded), detections matched to ground truth by box IoU >= 0.5,
then planar IPPE PnP on keypoints with confidence >= 0.5 using the fixed cone model.
Baselines run the same PnP on ground-truth 2D keypoints (planar model, and the exact
view-dependent 3D points as a sanity check). Writes eval/cones.csv and eval/summary.json.
"""
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'synth'))
import cone_model as cm  # noqa: E402

MODELS = {
    'nano': ROOT / 'runs/brt-pose-n-960-30k-40/weights/best.pt',
    'small': ROOT / 'runs/brt-pose-s-960-30k-40/weights/best.pt',
    'medium': ROOT / 'runs/brt-pose-m-960-30k/weights/best.pt',
}
KP_CONF = 0.5
IOU_MATCH = 0.5
RANGE_BINS = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60)]


def planar_model(class_id):
    """Fixed model in a camera-facing plane: x lateral, y down (−height), z = 0."""
    return np.array([[s * r, -z, 0.0] for z, r in cm.canonical_rings(class_id) for s in (-1, 1)])


def solve_pnp(obj, img_pts, K, dist):
    """Planar IPPE (both mirror solutions); falls back to SQPnP, then iterative LM, when IPPE
    returns NaN (an OpenCV 5.0 edge case on near-symmetric point sets). Lowest reprojection RMS wins."""
    rvecs = tvecs = None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img_pts, K, dist, flags=flag)
        except cv2.error:
            continue
        if rvecs and all(np.isfinite(t).all() and np.isfinite(r).all() for r, t in zip(rvecs, tvecs)):
            break
        rvecs = tvecs = None
    best = None
    for rv, tv in zip(rvecs or (), tvecs or ()):
        if tv[2, 0] <= 0:
            continue
        pr = cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2)
        e = float(np.sqrt(np.mean(np.sum((pr - img_pts) ** 2, axis=1))))
        if best is None or e < best[2]:
            best = (cv2.Rodrigues(rv)[0], tv.ravel(), e)
    return best


def pose_errors(R, t, gt, T_wc, up_axis=(0, -1, 0)):
    """up_axis: the model-frame direction of the cone axis (planar model: −y)."""
    T = np.array(gt['T_cam_cone'])
    tg = T[:3, 3]
    d = t - tg
    ray = tg / np.linalg.norm(tg)
    up = R @ np.array(up_axis, float)
    cosang = np.clip(up @ T[:3, 2], -1, 1)
    gw = T_wc[:3, :3] @ d  # error vector in world frame (z up)
    return dict(t_err=float(np.linalg.norm(d)), rel_err=float(np.linalg.norm(d) / np.linalg.norm(tg)),
                range_err=float(np.linalg.norm(t) - np.linalg.norm(tg)),
                lat_err=float(np.linalg.norm(d - (d @ ray) * ray)),
                ground_err=float(np.hypot(gw[0], gw[1])),
                axis_err_deg=float(np.degrees(np.arccos(cosang))))


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def detectable(c):
    return c['visible_pixels'] >= 12 and c['visible_fraction'] >= 0.1


def base_row(frame, gi, c, ann):
    occ = any(k[2] == 1 for k in c['keypoints_2d'])
    return dict(frame=frame, gt=gi, cls=c['class_name'], range_m=round(c['ground_range_m'], 3),
                distance_m=round(c['distance_m'], 3), detectable=int(detectable(c)), occluded=int(occ),
                visible_fraction=round(c['visible_fraction'], 3), truncated=int(c['truncated']),
                context=ann['capture']['extras']['car_context'], severity=round(ann['defects']['severity'], 3),
                box_h=round(c['bbox_xyxy_clipped'][3] - c['bbox_xyxy_clipped'][1], 1))


def run_predictions(name, imgsz, frames, root, cache):
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    model = YOLO(str(MODELS[name]))
    warm = cv2.imread(str(root / frames[0][1]['image']))
    for _ in range(20):
        model.predict(warm, imgsz=imgsz, conf=0.25, device=0, verbose=False)
    out = []
    for stem, ann in frames:
        img = cv2.imread(str(root / ann['image']))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=0.25, device=0, verbose=False)[0]
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        out.append(dict(latency_ms=lat, speed=r.speed, boxes=r.boxes.xyxy.cpu().numpy(),
                        cls=r.boxes.cls.cpu().numpy().astype(int), conf=r.boxes.conf.cpu().numpy(),
                        kpts=r.keypoints.data.cpu().numpy()))
    del model
    torch.cuda.empty_cache()
    cache.write_bytes(pickle.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--imgsz', type=int, nargs='+', default=[640, 800, 960, 1280])
    args = ap.parse_args()
    root = Path(args.root)
    ev = root / 'eval'
    (ev / 'predictions').mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / 'dataset.json').read_text())
    K = np.array(meta['calibration']['K'])
    dist = np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    rows, configs = [], []

    # Baselines on ground-truth keypoints.
    for name in ('gt_kp_planar', 'gt_kp_exact'):
        for stem, ann in frames:
            T_wc = np.array(ann['camera']['T_world_cam'])
            for gi, c in enumerate(ann['cones']):
                if not detectable(c):
                    continue
                kp = np.array(c['keypoints_2d'])
                n = 8 if c['class_id'] == 1 else 6
                use = np.where(kp[:n, 2] > 0)[0]
                row = base_row(stem, gi, c, ann) | dict(model=name, imgsz=0, matched=1, n_kp=len(use))
                if len(use) >= 4:
                    if name == 'gt_kp_planar':
                        obj, up = planar_model(c['class_id'])[use], (0, -1, 0)
                    else:  # exact silhouette points, cone frame (z up); coplanar via the apex
                        obj, up = np.array(c['keypoints_3d_cone'][:n], float)[use], (0, 0, 1)
                    sol = solve_pnp(obj.astype(np.float64), kp[use, :2].astype(np.float64), K, dist)
                    if sol:
                        row |= dict(pnp_ok=1, rms_px=sol[2]) | pose_errors(sol[0], sol[1], c, T_wc, up)
                rows.append(row)

    for name in MODELS:
        for imgsz in args.imgsz:
            preds = run_predictions(name, imgsz, frames, root, ev / 'predictions' / f'{name}_{imgsz}.pkl')
            pnp_ms, fps = [], 0
            for (stem, ann), p in zip(frames, preds):
                T_wc = np.array(ann['camera']['T_world_cam'])
                gts = ann['cones']
                gb = np.array([c['bbox_xyxy_clipped'] for c in gts]) if gts else np.zeros((0, 4))
                iou = iou_matrix(p['boxes'], gb)
                pi, gi_ = linear_sum_assignment(-iou) if iou.size else ([], [])
                match = {g: d for d, g in zip(pi, gi_) if iou[d, g] >= IOU_MATCH}
                fps += sum(1 for d in range(len(p['boxes'])) if d not in match.values())
                t0 = time.perf_counter()
                sols = {}
                for d in range(len(p['boxes'])):  # PnP on every detection, as deployed
                    cls = int(p['cls'][d])
                    n = 8 if cls == 1 else 6
                    kp = p['kpts'][d]
                    use = np.where(kp[:n, 2] >= KP_CONF)[0]
                    sols[d] = (solve_pnp(planar_model(cls)[use], kp[use, :2].astype(np.float64), K, dist)
                               if len(use) >= 4 else None, len(use))
                pnp_ms.append((time.perf_counter() - t0) * 1000)
                for gi, c in enumerate(gts):
                    if not detectable(c) and gi not in match:
                        continue
                    row = base_row(stem, gi, c, ann) | dict(model=name, imgsz=imgsz, matched=int(gi in match))
                    if gi in match:
                        d = match[gi]
                        sol, nkp = sols[d]
                        kp, gk = p['kpts'][d], np.array(c['keypoints_2d'])
                        ok = (kp[:, 2] >= KP_CONF) & (gk[:, 2] > 0)
                        row |= dict(iou=round(float(iou[d, gi]), 3), cls_correct=int(p['cls'][d] == c['class_id']),
                                    conf=round(float(p['conf'][d]), 3), n_kp=nkp,
                                    kp_err_px=float(np.mean(np.linalg.norm(kp[ok, :2] - gk[ok, :2], axis=1))) if ok.any() else None)
                        if sol:
                            row |= dict(pnp_ok=1, rms_px=sol[2]) | pose_errors(sol[0], sol[1], c, T_wc)
                    rows.append(row)
            lat = np.array([p['latency_ms'] for p in preds])
            configs.append(dict(model=name, imgsz=imgsz, latency_median_ms=float(np.median(lat)),
                                latency_p95_ms=float(np.percentile(lat, 95)),
                                pnp_median_ms=float(np.median(pnp_ms)), false_positives=fps,
                                speed_median={k: float(np.median([p['speed'][k] for p in preds])) for k in preds[0]['speed']}))
            print(f'{name} {imgsz}: {np.median(lat):.1f} ms', flush=True)

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('model', 'imgsz', 'frame', 'gt'), k))
    with open(ev / 'cones.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    (ev / 'configs.json').write_text(json.dumps(configs, indent=1))
    print('rows', len(rows))


if __name__ == '__main__':
    main()
