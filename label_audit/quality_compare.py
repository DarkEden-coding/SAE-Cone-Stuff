"""Do far synthetic cones look like far real cones? Side-by-side crops plus simple image-quality measures.

uv run --locked label_audit/quality_compare.py

Blue/yellow cones, not touching the image border, sampled in box-height bins measured at *network input*
(box height x 960 / long image side), the scale the model sees. Crops are cut from the full-resolution image
(context 2x box height), resampled to network-input scale (as the model sees them), then enlarged with
nearest-neighbour for display. Measures, at network-input scale, per cone:
  noise     robust sd of the Laplacian in the background ring (grey levels)
  edge      10-90% rise width of the strongest horizontal edge across the cone flank (px), lower = sharper
  contrast  Lab distance between cone body and background medians
  chroma    Lab chroma of the cone body
  jpeg      8x8 blockiness at native resolution (mean step across block borders / mean step elsewhere)
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / 'datasets/brt-cone-pose/brt-cone-pose-by-clean'
SYN = ROOT / 'synth_data'
OUT = ROOT / 'label_audit/work/quality'
BINS = [(6, 9), (9, 13), (13, 18), (18, 26)]
N_SHOW, N_STAT, CELL = 8, 150, 132
IMGSZ = 960


def real_cones():
    for split in ('val', 'test', 'train'):
        for lp in sorted((REAL / 'labels' / split).glob('*.txt')):
            rows = [list(map(float, r.split())) for r in lp.read_text().splitlines() if r.strip()]
            rows = [v for v in rows if int(v[0]) in (0, 4)]
            if rows:
                yield next((REAL / 'images' / split).glob(lp.stem + '.*')), rows, lp.stem.split('_')[0]


def collect(rng):
    out = {'real': {b: [] for b in BINS}, 'synth': {b: [] for b in BINS}}
    rl = [(ip, v, src) for ip, rows, src in real_cones() for v in rows]
    for ap in sorted((SYN / 'annotations').glob('*.json')):
        a = json.loads(ap.read_text())
        f = IMGSZ / max(a['width'], a['height'])
        for c in a['cones']:
            if c['class_id'] in (0, 4) and not c['truncated'] and not any(k[2] == 1 for k in c['keypoints_2d']):
                b = c['bbox_xyxy_clipped']
                h = (b[3] - b[1]) * f
                for bn in BINS:
                    if bn[0] <= h < bn[1]:
                        out['synth'][bn].append((SYN / a['image'], b, f'{c["ground_range_m"]:.0f} m'))
    rng.shuffle(rl)
    sizes = {}
    for ip, v, src in rl:
        if ip not in sizes:
            sizes[ip] = cv2.imread(str(ip)).shape[1::-1]
        W, H = sizes[ip]
        x0, y0, x1, y1 = (v[1] - v[3] / 2) * W, (v[2] - v[4] / 2) * H, (v[1] + v[3] / 2) * W, (v[2] + v[4] / 2) * H
        if x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2:
            continue
        h = (y1 - y0) * IMGSZ / max(W, H)
        for bn in BINS:
            if bn[0] <= h < bn[1] and len(out['real'][bn]) < N_STAT:
                out['real'][bn].append((ip, (x0, y0, x1, y1), src))
        if all(len(out['real'][bn]) >= N_STAT for bn in BINS):
            break
    for bn in BINS:
        rng.shuffle(out['synth'][bn])
        out['synth'][bn] = out['synth'][bn][:N_STAT]
    return out


def crop(img, box, f):
    """Square crop, 2x box height, resampled to network-input scale f (INTER_AREA, like the letterbox resize)."""
    x0, y0, x1, y1 = box
    side = 2.0 * (y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    n = max(8, int(round(side * f)))
    M = np.array([[f, 0, n / 2 - f * cx], [0, f, n / 2 - f * cy]])
    if f < 1:
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        M = np.array([[1, 0, n / 2 - f * cx], [0, 1, n / 2 - f * cy]])
        return cv2.warpAffine(small, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def measures(c):
    n = c.shape[0]
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.mgrid[:n, :n]
    r = np.maximum(np.abs(xx - n / 2), np.abs(yy - n / 2))
    q = n / 4  # box half-height
    body = (np.abs(xx - n / 2) < q * 0.25) & (yy > n / 2 - q * 0.1) & (yy < n / 2 + q * 0.6)
    bg = r > q * 1.4
    lap = cv2.Laplacian(g, cv2.CV_32F)
    noise = 1.4826 * np.median(np.abs(lap[bg] - np.median(lap[bg])))
    mb, mg = np.median(lab[body], axis=0), np.median(lab[bg], axis=0)
    contrast = float(np.linalg.norm(mb - mg))
    chroma = float(np.hypot(mb[1], mb[2]))
    # edge width: rows through the lower body, profile from centre outward on both sides
    widths = []
    for y in range(int(n / 2), int(n / 2 + q * 0.7)):
        for prof in (g[y, int(n / 2):], g[y, int(n / 2)::-1]):
            p = prof[: int(q * 1.6)]
            if len(p) < 4:
                continue
            lo, hi = np.percentile(p, 5), np.percentile(p, 95)
            if hi - lo < 10:
                continue
            a = (p - p[0]) / (p[-1] - p[0] + 1e-6)
            try:
                i10 = np.argmax(a > 0.1)
                i90 = np.argmax(a > 0.9)
            except ValueError:
                continue
            if i90 > i10:
                widths.append(i90 - i10)
    edge = float(np.median(widths)) if widths else np.nan
    return dict(noise=float(noise), edge=edge, contrast=contrast, chroma=chroma)


def blockiness(img, box):
    x0, y0, x1, y1 = map(int, box)
    pad = int(y1 - y0)
    g = cv2.cvtColor(img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad], cv2.COLOR_BGR2GRAY).astype(np.float32)
    ox, oy = max(0, x0 - pad) % 8, max(0, y0 - pad) % 8
    d = np.abs(np.diff(g, axis=1))
    cols = (np.arange(d.shape[1]) + ox + 1) % 8 == 0
    if cols.sum() == 0 or (~cols).sum() == 0:
        return np.nan
    return float(d[:, cols].mean() / (d[:, ~cols].mean() + 1e-6))


def main():
    rng = np.random.default_rng(3)
    data = collect(rng)
    OUT.mkdir(parents=True, exist_ok=True)
    stats, cache = {}, {}
    load = lambda p: cache.setdefault(p, cv2.imread(str(p))) if len(cache) < 400 else cv2.imread(str(p))
    for bn in BINS:
        rows = []
        for kind in ('real', 'synth'):
            ms, tiles = [], []
            for i, (ip, box, tag) in enumerate(data[kind][bn]):
                img = load(ip)
                f = IMGSZ / max(img.shape[:2])
                c = crop(img, box, f)
                ms.append(dict(measures(c), jpeg=blockiness(img, box)))
                if i < N_SHOW:
                    t = cv2.resize(c, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
                    lbl = f'{tag} {(box[3] - box[1]) * f:.0f}px'
                    cv2.putText(t, lbl, (4, CELL - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(t, lbl, (4, CELL - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
                    tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
            stats.setdefault(f'{bn[0]}-{bn[1]}', {})[kind] = {k: float(np.nanmedian([m[k] for m in ms])) for k in ms[0]} | {'n': len(ms)}
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(OUT / f'cmp_{bn[0]}_{bn[1]}.png'), np.vstack(rows))
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=1))
    for b, v in stats.items():
        print(b, {k: {m: round(x, 2) for m, x in d.items()} for k, d in v.items()})


if __name__ == '__main__':
    main()
