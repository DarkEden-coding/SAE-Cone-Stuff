"""Long-range PnP on the synthetic test set at full sensor resolution, before/after far-cone training.

uv run --locked synth/far_eval.py synth_data NAME=CHECKPOINT [...] [--imgsz 1920] [--real-val] [--out FILE]

Per model: batch-1 predictions on all frames (cached as eval/predictions/xp_<name>_<imgsz>.pkl), the keypoint
correction of kp_correction.py refitted for that model on even frames (2-60 m for the range scale), and results on
odd frames per range bin. Denominator: every odd-frame blue/yellow cone that is not truncated and has no occluded
keypoint, at any pixel size. Found = matched to a detection (IoU >= 0.5); solved = PnP succeeded. --real-val also
reports pose/box mAP on the real cleaned validation split at 960 px.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pnp as ev  # noqa: E402
import kp_correction as kc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINS = [(2, 10), (10, 20), (20, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 60)]


def evaluate(frames, preds, K, dist):
    recs = kc.collect(frames, preds)
    even = [r for r in recs if r['fi'] % 2 == 0]
    tab, _ = kc.fit(even)
    scale, _ = kc.fit_range([r for r in even if 2 <= r['c']['ground_range_m'] <= 60], tab, K, dist, frames)
    allc = [(fi, c) for fi, (_, a) in enumerate(frames) if fi % 2 for c in a['cones']
            if c['class_id'] in kc.BY and not c['truncated'] and not any(k[2] == 1 for k in c['keypoints_2d'])]
    got = {(r['fi'], id(r['c'])): r for r in recs if r['fi'] % 2}
    rows = []
    for lo, hi in BINS:
        sub = [(fi, c) for fi, c in allc if lo <= c['ground_range_m'] < hi]
        raw, cor, ab, found = [], [], [], 0
        for fi, c in sub:
            r = got.get((fi, id(c)))
            if r is None:
                continue
            found += 1
            T = np.array(frames[fi][1]['camera']['T_world_cam'])
            e1 = kc.pnp(r['kp'], c, K, dist, T)
            e2 = kc.pnp(kc.correct(r, tab), c, K, dist, T, scale(r['hp']))
            if e1 and e2:
                raw.append(e1['range_err'] / c['distance_m'] * 100)
                cor.append(e2['range_err'] / c['distance_m'] * 100)
                ab.append(e2['t_err'])
        raw, cor = np.array(raw), np.array(cor)
        q = lambda x, f: float(f(x)) if len(x) else float('nan')
        rows.append(dict(
            bin=f'{lo}-{hi}', n=len(sub), found=found / len(sub) * 100, solved=len(cor) / len(sub) * 100,
            raw_med=q(np.abs(raw), np.median), raw_p90=q(np.abs(raw), lambda x: np.percentile(x, 90)),
            raw_bias=q(raw, np.median),
            cor_med=q(np.abs(cor), np.median), cor_p90=q(np.abs(cor), lambda x: np.percentile(x, 90)),
            cor_gt10=q(np.abs(cor) > 10, np.mean) * 100, cor_bias=q(cor, np.median), abs_m=q(np.array(ab), np.median)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('models', nargs='+', help='NAME=CHECKPOINT')
    ap.add_argument('--imgsz', type=int, default=1920)
    ap.add_argument('--real-val', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    res = {}
    for name, ckpt in (m.split('=', 1) for m in args.models):
        ev.MODELS[name] = Path(ckpt)
        preds = ev.run_predictions(name, args.imgsz, frames, root,
                                   root / 'eval' / 'predictions' / f'xp_{name}_{args.imgsz}.pkl')
        res[name] = dict(bins=evaluate(frames, preds, K, dist),
                         latency_ms=float(np.median([p['latency_ms'] for p in preds])))
        if args.real_val:
            from ultralytics import YOLO
            m = YOLO(ckpt).val(data=str(ROOT / 'datasets/brt-cone-pose/brt-cone-pose-by-clean/data.yaml'),
                               split='val', imgsz=960, batch=8, device=0, plots=False, verbose=False)
            res[name]['real_val'] = dict(pose_map=float(m.pose.map), pose_map50=float(m.pose.map50),
                                         box_map=float(m.box.map), box_map50=float(m.box.map50))
        print(f"\n== {name} @{args.imgsz}  latency {res[name]['latency_ms']:.1f} ms"
              + (f"  real val pose mAP50-95 {res[name]['real_val']['pose_map']:.3f}" if args.real_val else ''))
        print(f"{'range m':>8} {'n':>5} {'solved':>6} | raw med/p90/bias   | corr med/p90/bias/>10% | abs m")
        for r in res[name]['bins']:
            print(f"{r['bin']:>8} {r['n']:5d} {r['solved']:5.0f}% | {r['raw_med']:4.1f}/{r['raw_p90']:5.1f}/{r['raw_bias']:+5.1f} | "
                  f"{r['cor_med']:4.1f}/{r['cor_p90']:5.1f}/{r['cor_bias']:+5.1f}/{r['cor_gt10']:4.0f}% | {r['abs_m']:5.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
