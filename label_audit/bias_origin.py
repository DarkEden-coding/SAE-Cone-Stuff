"""Where does the size-dependent keypoint bias come from? Compares cone keypoint geometry across four sources.

uv run --locked label_audit/bias_origin.py [--ckpt runs/by-n-D-aug/weights/last.pt]

Sources (blue/yellow cones, all six small-cone keypoints present):
  synth_gt     exact synthetic keypoints (synth_data, all frames), ground-truth box
  synth_model  model keypoints on synthetic frames (cached 960 px predictions), matched ground-truth box
  real_label   human labels of the cleaned BRT dataset (train+val+test), label box
  real_model   model keypoints on the cleaned BRT val+test images at 960 px, matched label box
Geometry per cone, in units of the box height h: height above the box bottom of each keypoint pair (mean of L/R),
and pair widths. Box-free shape ratios (stripe-bottom position along the top-foot span, widths / span) do not depend
on how boxes were drawn. Binned by box height in *model-input* pixels (box height x 960 / long image side), the
resolution the network actually sees.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'synth'))
import eval_pnp as ev  # noqa: E402

DATA = ROOT / 'datasets/brt-cone-pose/brt-cone-pose-by-clean'
BINS = ((4, 8), (8, 12), (12, 16), (16, 24), (24, 40), (40, 80), (80, 400))
IMGSZ = 960


def geom(kp, box):
    """kp: 6x2 pixel keypoints, box xyxy -> dict of geometry features."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    ys = [(kp[2 * p, 1] + kp[2 * p + 1, 1]) / 2 for p in range(3)]
    ws = [kp[2 * p + 1, 0] - kp[2 * p, 0] for p in range(3)]
    span = ys[2] - ys[0]
    return dict(top=(y1 - ys[0]) / h, bot=(y1 - ys[1]) / h, foot=(y1 - ys[2]) / h,
                w_top=ws[0] / h, w_foot=ws[2] / h, box_aspect=(x1 - x0) / h,
                s_bot=(ys[1] - ys[0]) / span if span > 0 else np.nan,
                s_wtop=ws[0] / span if span > 0 else np.nan, s_wfoot=ws[2] / span if span > 0 else np.nan)


def real_rows(split_files):
    for lp in split_files:
        ip = next((DATA / 'images' / lp.parent.name).glob(lp.stem + '.*'))
        rows = [list(map(float, r.split())) for r in lp.read_text().splitlines() if r.strip()]
        if not rows:
            continue
        with Image.open(ip) as im:
            W, H = im.size
        yield ip, W, H, rows


def label_kp(v, W, H):
    box = ((v[1] - v[3] / 2) * W, (v[2] - v[4] / 2) * H, (v[1] + v[3] / 2) * W, (v[2] + v[4] / 2) * H)
    kp = np.array([(v[5 + 3 * k] * W, v[6 + 3 * k] * H, v[7 + 3 * k]) for k in range(6)])
    return box, kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'runs/by-n-D-aug/weights/last.pt'))
    ap.add_argument('--name', default='D_aug30')
    args = ap.parse_args()
    src = {k: [] for k in ('synth_gt', 'synth_model', 'real_label', 'real_model')}

    # synthetic
    sroot = ROOT / 'synth_data'
    frames = [json.loads(p.read_text()) for p in sorted((sroot / 'annotations').glob('*.json'))]
    preds = pickle.loads((sroot / 'eval/predictions' / f'xp_{args.name}_{IMGSZ}.pkl').read_bytes())
    for ann, p in zip(frames, preds):
        f = IMGSZ / max(ann['width'], ann['height'])
        gts = [c for c in ann['cones'] if c['class_id'] in (0, 4) and ev.detectable(c) and not c['truncated']]
        if not gts:
            continue
        gb = np.array([c['bbox_xyxy_clipped'] for c in gts])
        for c, box in zip(gts, gb):
            kp = np.array(c['keypoints_2d'])[:6]
            if (kp[:, 2] > 0).all():
                src['synth_gt'].append(((box[3] - box[1]) * f, geom(kp[:, :2], box)))
        iou = ev.iou_matrix(p['boxes'], gb)
        if iou.size:
            for d, g in zip(*linear_sum_assignment(-iou)):
                kp = p['kpts'][d][:6]
                if iou[d, g] >= 0.5 and (kp[:, 2] >= ev.KP_CONF).all():
                    src['synth_model'].append(((gb[g, 3] - gb[g, 1]) * f, geom(kp[:, :2], gb[g])))

    # real labels (all splits)
    for split in ('train', 'val', 'test'):
        for ip, W, H, rows in real_rows(sorted((DATA / 'labels' / split).glob('*.txt'))):
            f = IMGSZ / max(W, H)
            for v in rows:
                box, kp = label_kp(v, W, H)
                if (kp[:, 2] > 0).all() and box[1] > 1 and box[3] < H - 1 and box[0] > 1 and box[2] < W - 1:
                    src['real_label'].append(((box[3] - box[1]) * f, geom(kp[:, :2], box)))

    # model on real val+test
    cache = ROOT / 'label_audit/work' / f'real_preds_{args.name}_{IMGSZ}.pkl'
    if cache.exists():
        rp = pickle.loads(cache.read_bytes())
    else:
        from ultralytics import YOLO
        model = YOLO(args.ckpt)
        rp = {}
        for split in ('val', 'test'):
            for ip in sorted((DATA / 'images' / split).iterdir()):
                r = model.predict(str(ip), imgsz=IMGSZ, conf=0.25, device=0, verbose=False)[0]
                rp[f'{split}/{ip.stem}'] = (r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy())
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(rp))
    for split in ('val', 'test'):
        for ip, W, H, rows in real_rows(sorted((DATA / 'labels' / split).glob('*.txt'))):
            f = IMGSZ / max(W, H)
            boxes, kps = rp[f'{split}/{ip.stem}']
            lb = [label_kp(v, W, H) for v in rows]
            gb = np.array([b for b, _ in lb])
            iou = ev.iou_matrix(boxes, gb)
            if not iou.size:
                continue
            for d, g in zip(*linear_sum_assignment(-iou)):
                kp = kps[d][:6]
                b = gb[g]
                if iou[d, g] >= 0.5 and (kp[:, 2] >= ev.KP_CONF).all() and b[1] > 1 and b[3] < H - 1:
                    src['real_model'].append(((b[3] - b[1]) * f, geom(kp[:, :2], b)))

    feats = ['top', 'bot', 'foot', 'w_top', 'w_foot', 'box_aspect', 's_bot', 's_wtop', 's_wfoot']
    out = {}
    for feat in feats:
        print(f"\n{feat} (median; box-height bins in model-input px)\n{'source':>12} " +
              ' '.join(f'{f"{lo}-{hi}":>13}' for lo, hi in BINS))
        out[feat] = {}
        for name, rows in src.items():
            hs = np.array([h for h, _ in rows])
            vals = np.array([g[feat] for _, g in rows])
            cells, rec = [], []
            for lo, hi in BINS:
                m = (hs >= lo) & (hs < hi) & np.isfinite(vals)
                med = float(np.median(vals[m])) if m.sum() >= 20 else None
                rec.append(dict(n=int(m.sum()), median=med))
                cells.append(f'{med:7.3f} ({m.sum():>4})' if med is not None else f'{"-":>13}')
            out[feat][name] = rec
            print(f'{name:>12} ' + ' '.join(cells))
    (ROOT / 'label_audit/bias_origin.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
