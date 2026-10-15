"""Compare pose checkpoints on the synthetic set: blue/yellow cones only, PnP error and keypoint placement.

uv run --locked synth/compare_models.py synth_data NAME=CHECKPOINT [NAME=CHECKPOINT ...] [--imgsz 960] [--out FILE]

Per model: batch-1 predictions (cached in eval/predictions/xp_<name>_<imgsz>.pkl), matching to ground-truth blue/
yellow cones (IoU >= 0.5), planar PnP as in eval_pnp.py. Pose error is compared on the common set of cones every
model solved (2-30 m), with a cone-level bootstrap 95% CI on the median, plus a breakdown by box height. Keypoint placement is the signed offset of
each predicted keypoint from the exact synthetic keypoint, in % of box height: inward (+ = inside the silhouette)
and vertical (+ = below), over matched cones >= 40 px tall with all six keypoints confident.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pnp as ev  # noqa: E402

BY = {0, 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('models', nargs='+', help='NAME=CHECKPOINT')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--out')
    ap.add_argument('--frames', choices=['all', 'even', 'odd'], default='all',
                    help='evaluate on a subset of frames (predictions always cover all frames)')
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    models = dict(m.split('=', 1) for m in args.models)

    per = {}
    for name, ckpt in models.items():
        ev.MODELS[name] = Path(ckpt)
        preds = ev.run_predictions(name, args.imgsz, frames, root, root / 'eval' / 'predictions' / f'xp_{name}_{args.imgsz}.pkl')
        cones, offs = {}, []
        for i, ((stem, ann), p) in enumerate(zip(frames, preds)):
            if args.frames != 'all' and i % 2 != (args.frames == 'odd'):
                continue
            T_wc = np.array(ann['camera']['T_world_cam'])
            gts = [(i, c) for i, c in enumerate(ann['cones']) if c['class_id'] in BY and ev.detectable(c)]
            if not gts:
                continue
            gb = np.array([c['bbox_xyxy_clipped'] for _, c in gts])
            iou = ev.iou_matrix(p['boxes'], gb)
            match = {}
            if iou.size:
                for d, g in zip(*linear_sum_assignment(-iou)):
                    if iou[d, g] >= ev.IOU_MATCH:
                        match[g] = d
            for g, (gi, c) in enumerate(gts):
                rec = dict(range=c['ground_range_m'], matched=g in match, solved=False,
                           h=c['bbox_xyxy_clipped'][3] - c['bbox_xyxy_clipped'][1])
                if g in match:
                    d = match[g]
                    kp = p['kpts'][d]
                    use = np.where(kp[:6, 2] >= ev.KP_CONF)[0]
                    if len(use) >= 4:
                        sol = ev.solve_pnp(ev.planar_model(c['class_id'])[use], kp[use, :2].astype(np.float64), K, dist)
                        if sol:
                            e = ev.pose_errors(sol[0], sol[1], c, T_wc)
                            rec.update(solved=True, rel=e['rel_err'], range_err_rel=e['range_err'] / c['distance_m'])
                    gk = np.array(c['keypoints_2d'])[:6]
                    h = c['bbox_xyxy_clipped'][3] - c['bbox_xyxy_clipped'][1]
                    if h >= 40 and (kp[:6, 2] >= ev.KP_CONF).all() and (gk[:, 2] > 0).all():
                        dx = (kp[:6, 0] - gk[:, 0]) * np.array([1, -1] * 3)  # + = inward
                        offs.append(np.c_[dx, kp[:6, 1] - gk[:, 1]] / h * 100)
                cones[(stem, gi)] = rec
        per[name] = (cones, np.array(offs))

    evalset = lambda cones: {k: v for k, v in cones.items() if 2 <= v['range'] <= 30}
    common = set.intersection(*[{k for k, v in evalset(c).items() if v['solved']} for c, _ in per.values()])
    rng = np.random.default_rng(0)
    out = dict(imgsz=args.imgsz, frames=args.frames, common_n=len(common), models={})
    names = ['stripe_top_L', 'stripe_top_R', 'stripe_bot_L', 'stripe_bot_R', 'foot_L', 'foot_R']
    for name, (cones, offs) in per.items():
        es = evalset(cones)
        rel = np.array([cones[k]['rel'] for k in common]) * 100
        rb = np.array([cones[k]['range_err_rel'] for k in common]) * 100
        boots = np.median(rng.choice(rel, (500, len(rel))), axis=1)
        bins = {}
        for lo, hi in ((2, 10), (10, 20), (20, 30)):
            sub = [v for v in es.values() if lo <= v['range'] < hi]
            sol = [v['rel'] * 100 for v in sub if v['solved']]
            bins[f'{lo}-{hi}'] = dict(n=len(sub), success=len(sol) / len(sub), rel_median=float(np.median(sol)) if sol else None)
        by_h = {}
        for lo, hi in ((12, 20), (20, 30), (30, 40), (40, 80), (80, 9999)):
            sol = [v for v in cones.values() if v['solved'] and lo <= v['h'] < hi]
            by_h[f'{lo}-{hi}px'] = dict(n=len(sol), rel_median=float(np.median([v['rel'] for v in sol]) * 100) if sol else None,
                                        range_bias_median=float(np.median([v['range_err_rel'] for v in sol]) * 100) if sol else None)
        out['models'][name] = dict(
            checkpoint=models[name], n=len(es),
            recall=float(np.mean([v['matched'] for v in es.values()])),
            success=float(np.mean([v['solved'] for v in es.values()])),
            common_rel_median=float(np.median(rel)), common_rel_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            common_rel_p90=float(np.percentile(rel, 90)), common_outliers=float(np.mean(rel > 10)),
            common_range_bias_median=float(np.median(rb)), by_range=bins, by_box_height=by_h,
            kp_offsets_pctH=dict(n_cones=len(offs), **{
                n: dict(inward=float(np.median(offs[:, i, 0])), below=float(np.median(offs[:, i, 1])),
                        abs_err=float(np.median(np.hypot(offs[:, i, 0], offs[:, i, 1])))) for i, n in enumerate(names)}),
        )
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"common set: {out['common_n']} blue/yellow cones at 2-30 m solved by every model (imgsz {args.imgsz}, frames: {args.frames})")
    print(f"{'model':>14} {'recall':>7} {'success':>8} {'median rel [95% CI]':>22} {'p90':>6} {'>10%':>6} {'range bias':>10}")
    for name, m in out['models'].items():
        print(f"{name:>14} {100*m['recall']:6.1f}% {100*m['success']:7.1f}% {m['common_rel_median']:7.2f}% "
              f"[{m['common_rel_ci95'][0]:.2f}-{m['common_rel_ci95'][1]:.2f}] {m['common_rel_p90']:5.1f}% "
              f"{100*m['common_outliers']:5.1f}% {m['common_range_bias_median']:+9.2f}%")
    print('\nPnP by box height (all solved cones, any range): median rel error / range bias')
    for name, m in out['models'].items():
        print(f"{name:>14} " + '  '.join(f"{k}: {v['rel_median']:.2f}%/{v['range_bias_median']:+.2f}%" for k, v in m['by_box_height'].items() if v['n']))
    print('\nkeypoint offset vs exact synthetic keypoints, median % of box height (inward / below):')
    print(f"{'model':>14} " + ' '.join(f'{n:>15}' for n in names))
    for name, m in out['models'].items():
        o = m['kp_offsets_pctH']
        print(f"{name:>14} " + ' '.join(f"{o[n]['inward']:+6.2f}/{o[n]['below']:+6.2f}" for n in names) + f"  (n={o['n_cones']})")


if __name__ == '__main__':
    main()
