"""Box-height-dependent keypoint bias correction, fitted on even synthetic frames and tested on odd ones.

uv run --locked synth/kp_correction.py synth_data NAME=CHECKPOINT[@IMGSZ] [...] [--out FILE]

For each blue/yellow cone matched to a detection (IoU >= 0.5), the offset of every predicted keypoint (0-5) from
the exact synthetic keypoint is expressed in units of the *predicted* box height (available at run time), split
into inward (toward the box centre) and vertical. Per keypoint, the median offset is taken in log-spaced box-height
bins over the even frames and linearly interpolated in log(h); applying it subtracts the interpolated bias. PnP
(as in eval_pnp.py) then runs on the odd frames with raw and corrected keypoints; the ground-truth keypoint PnP
on the same cones is the floor. Writes the fitted tables with --out (JSON).
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
EDGES = np.array([10, 14, 18, 23, 30, 40, 55, 80, 120, 200, 2000], float)
SIGN = np.array([1, -1] * 3)  # + = inward: left points (even) move right
H_BINS = ((12, 20), (20, 30), (30, 40), (40, 80), (80, 9999))


def collect(frames, preds):
    """Matched blue/yellow cones: frame index, gt cone, predicted keypoints (8x3), predicted box height."""
    out = []
    for fi, ((stem, ann), p) in enumerate(zip(frames, preds)):
        gts = [c for c in ann['cones'] if c['class_id'] in BY and ev.detectable(c)]
        if not gts or not len(p['boxes']):
            continue
        iou = ev.iou_matrix(p['boxes'], np.array([c['bbox_xyxy_clipped'] for c in gts]))
        for d, g in zip(*linear_sum_assignment(-iou)):
            if iou[d, g] >= ev.IOU_MATCH:
                b = p['boxes'][d]
                out.append(dict(fi=fi, c=gts[g], kp=p['kpts'][d].copy(), hp=float(b[3] - b[1])))
    return out


def fit(recs):
    """Median offset table [bin, kp, (inward, below)] in units of predicted box height; NaN where too few samples."""
    tab = np.full((len(EDGES) - 1, 6, 2), np.nan)
    count = np.zeros((len(EDGES) - 1, 6), int)
    for b in range(len(EDGES) - 1):
        sub = [r for r in recs if EDGES[b] <= r['hp'] < EDGES[b + 1]]
        for k in range(6):
            o = [((r['kp'][k, 0] - r['c']['keypoints_2d'][k][0]) * SIGN[k] / r['hp'],
                  (r['kp'][k, 1] - r['c']['keypoints_2d'][k][1]) / r['hp'])
                 for r in sub if r['kp'][k, 2] >= ev.KP_CONF and r['c']['keypoints_2d'][k][2] > 0]
            count[b, k] = len(o)
            if len(o) >= 30:
                tab[b, k] = np.median(np.array(o), axis=0)
    return tab, count


def offset(tab, h):
    centres = np.log(np.sqrt(EDGES[:-1] * EDGES[1:]))
    out = np.zeros((6, 2))
    for k in range(6):
        for j in range(2):
            ok = np.isfinite(tab[:, k, j])
            out[k, j] = np.interp(np.log(h), centres[ok], tab[ok, k, j]) if ok.any() else 0.0
    return out


def correct(r, tab):
    kp = r['kp'].copy()
    o = offset(tab, r['hp']) * r['hp']
    kp[:6, 0] -= o[:, 0] * SIGN
    kp[:6, 1] -= o[:, 1]
    return kp


def pnp(kp, c, K, dist, T_wc, scale=1.0):
    """scale multiplies the solved translation (range correction along the viewing ray)."""
    use = np.where(kp[:6, 2] >= ev.KP_CONF)[0]
    if len(use) < 4:
        return None
    sol = ev.solve_pnp(ev.planar_model(c['class_id'])[use], kp[use, :2].astype(np.float64), K, dist)
    return ev.pose_errors(sol[0], sol[1] * scale, c, T_wc) if sol else None


def fit_range(recs, tab, K, dist, frames):
    """Median relative range error of corrected-keypoint PnP per box-height bin, interpolated in log(h)."""
    centres = np.log(np.sqrt(EDGES[:-1] * EDGES[1:]))
    errs = [(r['hp'], e['range_err'] / r['c']['distance_m']) for r in recs
            if (e := pnp(correct(r, tab), r['c'], K, dist, np.array(frames[r['fi']][1]['camera']['T_world_cam'])))]
    hp, rb = np.array(errs).T
    med = np.array([np.median(rb[(hp >= a) & (hp < b)]) if ((hp >= a) & (hp < b)).sum() >= 30 else np.nan
                    for a, b in zip(EDGES[:-1], EDGES[1:])])
    ok = np.isfinite(med)
    return lambda h: 1 / (1 + np.interp(np.log(h), centres[ok], med[ok])), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('models', nargs='+', help='NAME=CHECKPOINT[@IMGSZ]')
    ap.add_argument('--out')
    args = ap.parse_args()
    root = Path(args.root)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    tables = {}
    print('PnP on odd frames, blue/yellow cones 2-30 m matched by the model; median relative error / median range bias')
    print(f"{'model':>16} {'variant':>10} {'n':>6} {'success':>8} {'median':>7} {'>10%':>6} {'bias':>7}   " +
          '  '.join(f'{lo}-{hi if hi < 9999 else "+"}px' for lo, hi in H_BINS))
    for spec in args.models:
        name, rest = spec.split('=', 1)
        ckpt, imgsz = (rest.split('@') + ['960'])[:2]
        imgsz = int(imgsz)
        ev.MODELS[name] = Path(ckpt)
        preds = ev.run_predictions(name, imgsz, frames, root, root / 'eval' / 'predictions' / f'xp_{name}_{imgsz}.pkl')
        recs = collect(frames, preds)
        tab, count = fit([r for r in recs if r['fi'] % 2 == 0])
        tables[f'{name}@{imgsz}'] = dict(edges_px=EDGES.tolist(), offsets_inward_below_frac_h=np.where(np.isfinite(tab), tab, None).tolist(),
                                         counts=count.tolist())
        even = [r for r in recs if r['fi'] % 2 == 0 and 2 <= r['c']['ground_range_m'] <= 30]
        scale, med = fit_range(even, tab, K, dist, frames)
        tables[f'{name}@{imgsz}']['range_bias_frac'] = np.where(np.isfinite(med), med, None).tolist()
        test = [r for r in recs if r['fi'] % 2 == 1 and 2 <= r['c']['ground_range_m'] <= 30]
        variants = {}
        for vname, kpf, sf in (('raw', lambda r: r['kp'], None), ('corrected', lambda r: correct(r, tab), None),
                               ('corr+range', lambda r: correct(r, tab), scale),
                               ('gt_kp', lambda r: np.array(r['c']['keypoints_2d'], float), None)):
            res = []
            for r in test:
                e = pnp(kpf(r), r['c'], K, dist, np.array(frames[r['fi']][1]['camera']['T_world_cam']),
                        sf(r['hp']) if sf else 1.0)
                res.append(None if e is None else (e['rel_err'] * 100, e['range_err'] / r['c']['distance_m'] * 100))
            variants[vname] = res
        common = [i for i in range(len(test)) if all(v[i] is not None for v in variants.values())]
        for vname, res in variants.items():
            rel = np.array([res[i][0] for i in common])
            rb = np.array([res[i][1] for i in common])
            hb = []
            for lo, hi in H_BINS:
                idx = [j for j, i in enumerate(common) if lo <= test[i]['hp'] < hi]
                hb.append(f'{np.median(rel[idx]):4.1f}/{np.median(rb[idx]):+4.1f}' if idx else '   -   ')
            succ = np.mean([x is not None for x in res])
            print(f"{name + '@' + str(imgsz):>16} {vname:>10} {len(common):6d} {100 * succ:7.1f}% {np.median(rel):6.2f}% "
                  f"{100 * np.mean(rel > 10):5.1f}% {np.median(rb):+6.2f}%   " + '  '.join(f'{s:>9}' for s in hb))
    if args.out:
        Path(args.out).write_text(json.dumps(tables, indent=1))


if __name__ == '__main__':
    main()
