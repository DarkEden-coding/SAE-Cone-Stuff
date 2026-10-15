"""Apply keypoint label corrections to the cleaned blue/yellow dataset (labels only; images are hard-linked).

uv run --locked adjust_brt_labels.py [--variant v1|v2]

Offsets are in % of box height. Points moved up slide along that side's edge line (through the side's
original stripe-top and foot points) so they stay on the silhouette; inward moves are toward the cone axis.

v1 (brt-cone-pose-by-clean-adj): the label audit's real-minus-synthetic medians (label_audit/REPORT.md):
   stripe top up 1.89 blue / 2.10 yellow, stripe bottom up 1.24 / 0.72, foot inward 1.27 / 0.81.
   Overshot: models trained on it placed stripe tops 1.4-1.9 %H too high and feet ~1 %H inside.
v2 (brt-cone-pose-by-clean-adj2): the systematic keypoint error of a model trained on the uncorrected labels
   (run by-n-C-clean), measured against exact synthetic keypoints on the even synthetic frames only
   (synth_data/eval/C_residuals_even.json); vertical corrections >= 0.3 %H are applied, horizontal ones are not:
   stripe top up 0 blue / 0.38 yellow, stripe bottom up 2.57 / 2.02, foot up 1.33 / 1.11.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'datasets/brt-cone-pose/brt-cone-pose-by-clean'
VARIANTS = {
    # up: %H per pair (stripe top 0/1, stripe bottom 2/3, foot 4/5); foot_in: %H for pair 4/5
    'v1': dict(dst='brt-cone-pose-by-clean-adj', up={0: (1.89, 1.24, 0.0), 4: (2.10, 0.72, 0.0)}, foot_in={0: 1.27, 4: 0.81}),
    'v2': dict(dst='brt-cone-pose-by-clean-adj2', up={0: (0.0, 2.57, 1.33), 4: (0.38, 2.02, 1.11)}, foot_in={0: 0.0, 4: 0.0}),
}


def adjust(row, aspect, up, foot_in):
    """aspect = image height / width, converting a %-of-box-height offset into normalised x units."""
    v = [float(x) for x in row]
    cls = int(v[0])
    h = v[4]  # normalised box height; all offsets scale with it (y normalised by image height)
    kp = [v[5 + 3 * k:8 + 3 * k] for k in range(8)]
    cx = v[1]
    for side in (0, 1):  # 0 = left (even indices), 1 = right (odd)
        top, bot, foot = kp[side], kp[2 + side], kp[4 + side]
        slope = None
        if top[2] > 0 and foot[2] > 0 and abs(foot[1] - top[1]) > 1e-9:
            slope = (foot[0] - top[0]) / (foot[1] - top[1])  # dx/dy along this side's edge, normalised units
        for p, amount in zip((top, bot, foot), up[cls]):
            if p[2] > 0 and amount:
                dy = -amount / 100 * h
                p[1] += dy
                if slope is not None:
                    p[0] += slope * dy
        if foot[2] > 0 and foot_in[cls]:
            inward = foot_in[cls] / 100 * h * aspect
            foot[0] += inward if foot[0] < cx else -inward
    for k in range(8):
        v[5 + 3 * k:8 + 3 * k] = kp[k]
    return v


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=VARIANTS, default='v2')
    args = ap.parse_args()
    cfg = VARIANTS[args.variant]
    dst = SRC.parent / cfg['dst']
    n = 0
    for split in ('train', 'val', 'test'):
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
        link = dst / 'images' / split
        if not link.exists():
            # Hard links, not a directory symlink: Ultralytics resolves dataset paths, and a resolved
            # symlink would point it at the source dataset's labels.
            link.mkdir(parents=True)
            for f in (SRC / 'images' / split).iterdir():
                os.link(f, link / f.name)
        for lp in sorted((SRC / 'labels' / split).glob('*.txt')):
            rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
            out = []
            if rows:
                ip = next((SRC / 'images' / split).glob(lp.stem + '.*'))
                with Image.open(ip) as im:
                    W, H = im.size
                for r in rows:
                    v = adjust(r, H / W, cfg['up'], cfg['foot_in'])
                    for k in range(8):  # keep inside the image
                        if v[7 + 3 * k] > 0:
                            v[5 + 3 * k] = min(max(v[5 + 3 * k], 0.0), 1.0)
                            v[6 + 3 * k] = min(max(v[6 + 3 * k], 0.0), 1.0)
                    out.append(' '.join([str(int(v[0]))] + [f'{x:.6f}' for x in v[1:5]] +
                                        [f'{x:.6f}' if i % 3 != 2 else str(int(x)) for i, x in enumerate(v[5:])]))
                    n += 1
            (dst / 'labels' / split / lp.name).write_text(''.join(o + '\n' for o in out))
    (dst / 'data.yaml').write_text((SRC / 'data.yaml').read_text().replace(str(SRC), str(dst)))
    (dst / 'adjustment.json').write_text(json.dumps(dict(source=str(SRC), variant=args.variant, up_pct_H=cfg['up'],
                                                         foot_inward_pct_H=cfg['foot_in'], doc=__doc__.strip()), indent=1))
    print('adjusted', n, 'cones ->', dst)


if __name__ == '__main__':
    main()
