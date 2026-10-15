"""Build a cleaned blue/yellow-only copy of the BRT cone pose dataset.

uv run --locked clean_brt_pose.py [--dry-run]

Keeps only BLUE (0) and YELLOW (4) label rows; class ids and the 8x3 keypoint layout are unchanged,
so existing checkpoints and tools work as-is. Cones with broken keypoint labels are found with the
geometry checks from label_audit/ (thresholds set from exact synthetic labels): swapped left/right pair,
stripe/foot order violation, left/right height asymmetry > 15% of box height, cone narrowing downward,
keypoint outside the box, or wrong keypoint count. Checks apply to cones >= 30 px tall, as in the audit.
Broken cones are painted out and their labels removed (deleting only the label would leave an unlabelled
cone that training treats as background). The padded box is replaced by a background patch copied from
beside it (left/right first, avoiding every labelled cone of any class), blended with a feathered edge;
Telea inpainting is the fallback when no clean patch exists. If a broken cone's padded box covers more
than 5% of any other cone's box (any class) the image is dropped instead, so no fragments remain.
Orange cones stay in the images unlabelled, so the model learns them as not-a-target.
"""
import argparse
import collections
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'datasets/brt-cone-pose/brt-cone-pose-11k'
DST = ROOT / 'datasets/brt-cone-pose/brt-cone-pose-by-clean'
KEEP = {0, 4}
MIN_H = 30
ASYM, ORDER, WGROW, OUTBOX = 0.15, 0.08, -0.03, 0.05
PAD, MAX_OVERLAP = 0.15, 0.05


def flags(row, W, H):
    """Geometry checks for one small (6-keypoint) cone; returns the list of failed checks."""
    v = list(map(float, row))
    bx, by, bw, bh = v[1] * W, v[2] * H, v[3] * W, v[4] * H
    x0, y0, x1, y1 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    kp = [(v[5 + 3 * k] * W, v[6 + 3 * k] * H, v[7 + 3 * k]) for k in range(8)]
    f = []
    if sum(k[2] > 0 for k in kp[:6]) != 6:
        return ['miscount']
    widths, ymid = [], []
    for p in range(3):
        (xl, yl, _), (xr, yr, _) = kp[2 * p], kp[2 * p + 1]
        if xl >= xr:
            f.append('swap')
        if abs(yl - yr) / bh > ASYM:
            f.append('asym')
        m = OUTBOX * bh
        if any(x < x0 - m or x > x1 + m or y < y0 - m or y > y1 + m for x, y in ((xl, yl), (xr, yr))):
            f.append('outbox')
        widths.append(xr - xl)
        ymid.append((yl + yr) / 2)
    for p in range(2):
        if (ymid[p + 1] - ymid[p]) / bh < ORDER:
            f.append('order')
        if (widths[p + 1] - widths[p]) / bh < WGROW:
            f.append('narrowing')
    return sorted(set(f))


def box(r, W, H, pad):
    cx, cy, w, h = (float(v) for v in r[1:5])
    w, h = w * W * (1 + 2 * pad), h * H * (1 + 2 * pad)
    return (max(0.0, cx * W - w / 2), max(0.0, cy * H - h / 2), min(W, cx * W + w / 2), min(H, cy * H + h / 2))


def overlap(a, g):
    """Fraction of good box g covered by box a."""
    iw = max(0.0, min(a[2], g[2]) - max(a[0], g[0]))
    ih = max(0.0, min(a[3], g[3]) - max(a[1], g[1]))
    return iw * ih / max((g[2] - g[0]) * (g[3] - g[1]), 1e-9)


def paint_out(img, b, broken, protect):
    """Replace box b with a feathered background patch from beside it; returns the method used."""
    Hh, Ww = img.shape[:2]
    x0, y0, x1, y1 = int(b[0]), int(b[1]), int(np.ceil(b[2])), int(np.ceil(b[3]))
    w, h = x1 - x0, y1 - y0
    f = max(3, int(0.12 * min(w, h)))  # feather width
    X0, Y0, X1, Y1 = max(0, x0 - f), max(0, y0 - f), min(Ww, x1 + f), min(Hh, y1 + f)
    inner = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    inner[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = 255
    dist = cv2.distanceTransform(255 - inner, cv2.DIST_L2, 3)
    alpha = np.clip(1 - dist / f, 0, 1)
    for p in protect:  # never touch other cones
        px0, py0, px1, py1 = (int(max(v, 0)) for v in p)
        alpha[max(py0 - Y0, 0):max(py1 - Y0, 0), max(px0 - X0, 0):max(px1 - X0, 0)] = 0
    avoid = [grow(p, 0.05) for p in protect] + list(broken)
    region = img[Y0:Y1, X0:X1].astype(np.float32)
    ring = (alpha > 0) & (alpha < 1)  # feather band: patch should match the surroundings here
    region_black = (region.max(axis=2) < 8).mean()
    best = None
    for dx, dy in ((-1.15, 0), (1.15, 0), (-1.15, -0.3), (1.15, -0.3), (-1.15, 0.3), (1.15, 0.3),
                   (-2.3, 0), (2.3, 0), (-2.3, -0.3), (2.3, -0.3), (0, 1.2), (0, -1.2)):
        ox, oy = int(round(dx * (w + 2 * f))), int(round(dy * (h + 2 * f)))
        sx0, sy0, sx1, sy1 = X0 + ox, Y0 + oy, X1 + ox, Y1 + oy
        if sx0 < 0 or sy0 < 0 or sx1 > Ww or sy1 > Hh:
            continue
        if any(overlap((sx0, sy0, sx1, sy1), a) > 0 or overlap(a, (sx0, sy0, sx1, sy1)) > 0 for a in avoid):
            continue
        patch = img[sy0:sy1, sx0:sx1].astype(np.float32)
        if (patch.max(axis=2) < 8).mean() > region_black + 0.01:  # would copy letterbox padding in
            continue
        score = float(np.abs(patch[ring] - region[ring]).mean()) if ring.any() else 0.0
        if best is None or score < best[0]:
            best = (score, patch)
    if best is not None:
        img[Y0:Y1, X0:X1] = np.clip(region * (1 - alpha[..., None]) + best[1] * alpha[..., None], 0, 255).astype(np.uint8)
        return 'patch'
    mask = np.zeros((Hh, Ww), np.uint8)
    mask[Y0:Y1, X0:X1] = (alpha > 0.5).astype(np.uint8) * 255
    img[:] = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    return 'telea'


def grow(bx, g):
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    return (bx[0] - g * w, bx[1] - g * h, bx[2] + g * w, bx[3] + g * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    stats = collections.defaultdict(collections.Counter)
    plan = []
    for split in ('train', 'val', 'test'):
        for lp in sorted((SRC / 'labels' / split).glob('*.txt')):
            ip = next(p for p in (SRC / 'images' / split).glob(lp.stem + '.*'))
            rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
            kept = [r for r in rows if int(float(r[0])) in KEEP]
            s = stats[split]
            s['images'] += 1
            s['rows_in'] += len(rows)
            s['rows_other_class'] += len(rows) - len(kept)
            bad, good = [], []
            if kept:
                with Image.open(ip) as im:
                    W, H = im.size
                for r in kept:
                    f = flags(r, W, H) if float(r[4]) * H >= MIN_H else []
                    (bad if f else good).append((r, f))
                    for x in f:
                        s['flag_' + x] += 1
            action = None
            if bad:
                bad_rows = [r for r, _ in bad]
                others = [box(r, W, H, 0.0) for r in rows if r not in bad_rows]
                bb = [box(r, W, H, PAD) for r in bad_rows]
                if max((overlap(b, o) for b in bb for o in others), default=0.0) > MAX_OVERLAP:
                    s['images_dropped_overlap'] += 1
                    s['good_cones_lost_with_image'] += len(good)
                    continue
                action = (bb, others)
                s['images_painted'] += 1
                s['broken_cones_removed'] += len(bad)
            if rows and not kept:
                s['images_now_background'] += 1
            s['images_kept'] += 1
            s['cones_kept'] += len(good)
            plan.append((split, ip, lp.name, [r for r, _ in good], action, [f for _, f in bad]))
    for split, s in stats.items():
        print(split, dict(s))
    if args.dry_run:
        return
    if DST.exists():
        raise SystemExit(f'{DST} exists; remove it first to rebuild')
    inpainted = {}
    for split, ip, lname, kept, action, bad_flags in plan:
        (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)
        out = DST / 'images' / split / ip.name
        if action is None:
            shutil.copy2(ip, out)
        else:
            img = cv2.imread(str(ip))
            methods = [paint_out(img, b, action[0], action[1]) for b in action[0]]
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            inpainted[f'{split}/{ip.name}'] = dict(boxes_xyxy=[[round(v, 1) for v in b] for b in action[0]], flags=bad_flags, method=methods)
        (DST / 'labels' / split / lname).write_text(''.join(' '.join(r) + '\n' for r in kept))
    (DST / 'data.yaml').write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape:\n- 8\n- 3\nnames:\n  0: BLUE\n  1: ORANGE_BIG\n  2: ORANGE\n  3: UNDEFINED\n  4: YELLOW\n")
    (DST / 'cleaning.json').write_text(json.dumps(dict(
        source=str(SRC), kept_classes=['BLUE', 'YELLOW'], min_box_height_px=MIN_H,
        thresholds=dict(asym=ASYM, order=ORDER, wgrow=WGROW, outbox=OUTBOX),
        policy='broken kept-class cones painted out with an adjacent background patch (Telea fallback) and unlabelled '
               '(box padded %.0f%%); image dropped if a broken box covers >%.0f%% of any other cone box; '
               'other classes removed from labels' % (100 * PAD, 100 * MAX_OVERLAP),
        inpainted=inpainted,
        stats={k: dict(v) for k, v in stats.items()}), indent=1))
    print('wrote', DST)


if __name__ == '__main__':
    main()
