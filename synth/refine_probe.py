"""Would a high-resolution in-box keypoint refiner help? Training-free probes on the synthetic set.

uv run --locked synth/refine_probe.py synth_data --ckpt runs/by-n-D-aug/weights/last.pt [--name D_aug30]

Uses the 960 px detections (cached by kp_correction.py / compare_models.py) of blue/yellow cones whose predicted box
is 12-80 px tall, then compares keypoint sources on exactly the same cones:
  960    the full-frame keypoints at 960 px input
  1920   full-frame keypoints at native resolution (matched to the same ground-truth cone)
  crop   the same network on a native-resolution crop around the 960 box (2.6x box height square), upscaled so the
         box is ~CROP_H px tall: a stand-in for a dedicated crop refiner (no new pixels, but more network resolution)
Every source gets the same correction as kp_correction.py (per-keypoint bias by box height, then range scale;
fitted on even frames) and PnP is scored on odd frames. Also reported: robust keypoint scatter after bias removal
in native pixels, and the PnP error that ground-truth keypoints with Gaussian pixel noise would give (the floor a
refiner with that accuracy could reach).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pnp as ev  # noqa: E402
import kp_correction as kc  # noqa: E402

CROP_H = 96
CTX = 2.6
BINS = ((12, 20), (20, 30), (30, 40), (40, 80))


def crop_predict(model, img, box):
    x0, y0, x1, y1 = box
    side = CTX * max(y1 - y0, x1 - x0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = CROP_H / (y1 - y0)  # upscale factor
    out = int(round(side * s))
    M = np.array([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]])
    crop = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    r = model.predict(crop, imgsz=max(256, int(np.ceil(out / 32) * 32)), conf=0.1, device=0, verbose=False)[0]
    if not len(r.boxes):
        return None
    b = r.boxes.xyxy.cpu().numpy()
    tb = np.array([[out / 2 - s * (x1 - x0) / 2, out / 2 - s * (y1 - y0) / 2, out / 2 + s * (x1 - x0) / 2, out / 2 + s * (y1 - y0) / 2]])
    iou = ev.iou_matrix(b, tb)[:, 0]
    d = int(np.argmax(iou))
    if iou[d] < 0.5:
        return None
    kp = r.keypoints.data.cpu().numpy()[d].copy()
    kp[:, 0] = (kp[:, 0] - M[0, 2]) / s
    kp[:, 1] = (kp[:, 1] - M[1, 2]) / s
    return kp


def scatter(recs, tab):
    """Robust sd (1.4826 * MAD) of bias-corrected keypoint errors in native px, pooled over kp 0-5 and x/y, per bin."""
    out = []
    for lo, hi in BINS:
        e = []
        for r in recs:
            if lo <= r['hp'] < hi:
                kp, gt = kc.correct(r, tab), np.array(r['c']['keypoints_2d'], float)
                ok = (kp[:6, 2] >= ev.KP_CONF) & (gt[:6, 2] > 0)
                e.extend((kp[:6][ok, :2] - gt[:6][ok, :2]).ravel())
        e = np.array(e)
        out.append(1.4826 * np.median(np.abs(e - np.median(e))) if len(e) else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--ckpt', default='runs/by-n-D-aug/weights/last.pt')
    ap.add_argument('--name', default='D_aug30')
    ap.add_argument('--out')
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    ev.MODELS[args.name] = Path(args.ckpt)
    pred = lambda s: ev.run_predictions(args.name, s, frames, root, root / 'eval' / 'predictions' / f'xp_{args.name}_{s}.pkl')
    base = [r for r in kc.collect(frames, pred(960)) if 12 <= r['hp'] < 80 and 2 <= r['c']['ground_range_m'] <= 30]
    full = {(r['fi'], id(r['c'])): r['kp'] for r in kc.collect(frames, pred(1920))}

    cache = root / 'eval' / 'predictions' / f'crop_{args.name}_{CROP_H}.npy'
    if cache.exists():
        crops = list(np.load(cache, allow_pickle=True))
    else:
        model = YOLO(args.ckpt)
        boxes = {}
        for fi, p in enumerate(pred(960)):
            boxes[fi] = p['boxes']
        crops, img, cur = [], None, -1
        for r in base:
            if r['fi'] != cur:
                cur, img = r['fi'], cv2.imread(str(root / frames[r['fi']][1]['image']))
            # the matched 960 box: the detection whose keypoints are r['kp']
            p = pred(960)[r['fi']]
            d = int(np.argmin(np.abs(p['kpts'] - r['kp']).sum(axis=(1, 2))))
            crops.append(crop_predict(model, img, p['boxes'][d]))
        np.save(cache, np.array(crops, dtype=object), allow_pickle=True)

    keep = [i for i, r in enumerate(base) if crops[i] is not None and (r['fi'], id(r['c'])) in full]
    sources = {
        '960': [base[i] for i in keep],
        '1920': [dict(base[i], kp=full[(base[i]['fi'], id(base[i]['c']))]) for i in keep],
        f'crop{CROP_H}': [dict(base[i], kp=crops[i]) for i in keep],
    }
    print(f'{len(keep)} cones (of {len(base)} matched at 960 with box 12-80 px); crop found the cone for '
          f'{sum(c is not None for c in crops)}')
    T = lambda r: np.array(frames[r['fi']][1]['camera']['T_world_cam'])
    res = {}
    hdr = '  '.join(f'{lo}-{hi}px' for lo, hi in BINS)
    print(f"\nodd frames, corrected (bias + range scale): median rel error / >10% outliers per box height\n{'source':>8}   {hdr}")
    for name, recs in sources.items():
        tab, _ = kc.fit([r for r in recs if r['fi'] % 2 == 0])
        scale, _ = kc.fit_range([r for r in recs if r['fi'] % 2 == 0], tab, K, dist, frames)
        test = [r for r in recs if r['fi'] % 2 == 1]
        row = []
        for lo, hi in BINS:
            rel = [e['rel_err'] * 100 for r in test if lo <= r['hp'] < hi
                   and (e := kc.pnp(kc.correct(r, tab), r['c'], K, dist, T(r), scale(r['hp'])))]
            row.append((float(np.median(rel)), float(np.mean(np.array(rel) > 10) * 100), len(rel)))
        res[name] = dict(pnp=row, kp_scatter_px=scatter(test, tab))
        print(f'{name:>8}   ' + '  '.join(f'{m:4.1f}%/{o:4.1f}%' for m, o, _ in row))
    print(f"\nbias-corrected keypoint scatter, robust sd in native px\n{'source':>8}   {hdr}")
    for name, v in res.items():
        print(f'{name:>8}   ' + '  '.join(f'{x:9.2f}' for x in v['kp_scatter_px']))

    rng = np.random.default_rng(0)
    test = [r for r in sources['960'] if r['fi'] % 2 == 1]
    print(f"\nfloor: ground-truth keypoints + Gaussian noise (sd px, x and y): median rel error per box height\n{'sd px':>8}   {hdr}")
    res['noise_floor'] = {}
    for sd in (0.25, 0.5, 0.75, 1.0, 1.5):
        row = []
        for lo, hi in BINS:
            rel = []
            for r in test:
                if lo <= r['hp'] < hi:
                    kp = np.array(r['c']['keypoints_2d'], float)
                    kp[:6, :2] += rng.normal(0, sd, (6, 2))
                    kp[:, 2] = 1.0
                    if (e := kc.pnp(kp, r['c'], K, dist, T(r))):
                        rel.append(e['rel_err'] * 100)
            row.append(float(np.median(rel)))
        res['noise_floor'][sd] = row
        print(f'{sd:8.2f}   ' + '  '.join(f'{x:8.1f}%' for x in row))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
