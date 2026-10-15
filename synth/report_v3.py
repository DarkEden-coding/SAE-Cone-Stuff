"""Graphs-only PDF: original nano / small / medium pose models vs the far-cone synthetic-trained nano (run E).

uv run --locked synth/report_v3.py synth_data [--out ~/Downloads/cone-pnp-report-v3.pdf]

Blue/yellow cones on the odd synthetic frames, 2-60 m ground range, not truncated and with no occluded keypoint (the
far_eval.py denominator). Main comparison: every model at 1920 px (native sensor resolution). Each model gets its own
keypoint correction (kp_correction.py) fitted on the even frames over 2-60 m. Uses cached predictions in
eval/predictions (computes missing ones). Needs the Playwright-cached Chromium for printing, like report_v2.py.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pnp as ev  # noqa: E402
import kp_correction as kc  # noqa: E402
from report_v2 import CSS, GRAY, GRID, INK, INK2, plt, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ev.MODELS['E_far'] = ROOT / 'runs/by-n-E-far/weights/best.pt'
# categorical slots 1, 2, 3, 7 of the reference palette (validated light mode; aqua < 3:1 -> always direct-labelled)
MODELS = [('nano', 'original nano', '#2a78d6'), ('small', 'original small', '#eb6834'),
          ('medium', 'original medium', '#1baf7a'), ('E_far', 'far-cone nano (E)', '#6250d6')]
COL = {m: c for m, _, c in MODELS}
LAB = {m: l for m, l, _ in MODELS}
BINS = [(2, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
MID = [(lo + hi) / 2 for lo, hi in BINS]
SPEED = [('nano', 960), ('nano', 1280), ('nano', 1920), ('small', 960), ('small', 1280), ('small', 1920),
         ('medium', 960), ('medium', 1280), ('medium', 1920), ('E_far', 960), ('E_far', 1280), ('E_far', 1920)]


def cache(name, imgsz):
    return ROOT / 'synth_data/eval/predictions' / (f'xp_{name}_{imgsz}.pkl' if name == 'E_far' else f'{name}_{imgsz}.pkl')


def eval_cones(frames):
    return {(fi, id(c)): c for fi, (_, a) in enumerate(frames) if fi % 2 for c in a['cones']
            if c['class_id'] in kc.BY and not c['truncated'] and not any(k[2] == 1 for k in c['keypoints_2d'])
            and 2 <= c['ground_range_m'] < 60}


def evaluate(frames, K, dist, cones, name, imgsz):
    preds = ev.run_predictions(name, imgsz, frames, ROOT / 'synth_data', cache(name, imgsz))
    recs = kc.collect(frames, preds)
    even = [r for r in recs if r['fi'] % 2 == 0]
    tab, _ = kc.fit(even)
    scale, _ = kc.fit_range([r for r in even if 2 <= r['c']['ground_range_m'] <= 60], tab, K, dist, frames)
    res = {'raw': {}, 'corrected': {}}
    for r in recs:
        key = (r['fi'], id(r['c']))
        if key not in cones:
            continue
        c, T = r['c'], np.array(frames[r['fi']][1]['camera']['T_world_cam'])
        for v, kp, s in (('raw', r['kp'], 1.0), ('corrected', kc.correct(r, tab), scale(r['hp']))):
            e = kc.pnp(kp, c, K, dist, T, s)
            if e:
                res[v][key] = dict(rel=abs(e['range_err']) / c['distance_m'] * 100, bias=e['range_err'] / c['distance_m'] * 100,
                                   t=e['t_err'], rng=c['ground_range_m'])
    return dict(res=res, recs=recs, tab=tab, scale=scale, preds=preds, latency=float(np.median([p['latency_ms'] for p in preds])))


def per_bin(res, cones, field='rel', stat=np.median):
    out = []
    for lo, hi in BINS:
        v = [x[field] for x in res.values() if lo <= x['rng'] < hi]
        n = sum(1 for c in cones.values() if lo <= c['ground_range_m'] < hi)
        out.append((float(stat(v)) if len(v) >= 20 else np.nan, len(v) / n * 100))
    return out


# ---------- figures ----------

def fig_range(runs, floor, cones, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for m, lab, col in MODELS:
        b = per_bin(runs[m]['res']['corrected'], cones)
        axes[0].plot(MID, [s for _, s in b], color=col, lw=2, marker='o', ms=7, mec='white', mew=1.5, label=lab)
        axes[1].plot(MID, [e for e, _ in b], color=col, lw=2, marker='o', ms=7, mec='white', mew=1.5, label=lab)
    fb = per_bin(floor, cones)
    axes[1].plot(MID, [e for e, _ in fb], color=GRAY, lw=1.5, ls='--', label='exact keypoints (PnP floor)')
    axes[0].set_ylim(0, 104)
    axes[0].set_title('Cones detected and solved (% of all cones in range)', fontsize=10.5, loc='left', color=INK)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title('Median range error (% of distance), with correction', fontsize=10.5, loc='left', color=INK)
    for ax in axes:
        ax.set_xticks(MID, [f'{lo}–{hi}' for lo, hi in BINS])
        ax.set_xlabel('ground range (m)')
        ax.set_xlim(2, 66)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower left')
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_dumbbell(runs, path):
    """Raw vs corrected median error, near and far, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 2.9), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for ax, (lo, hi) in zip(axes, ((2, 30), (30, 60))):
        for yi, (m, lab, col) in zip(y, MODELS):
            v = [np.median([x['rel'] for x in runs[m]['res'][k].values() if lo <= x['rng'] < hi]) for k in ('raw', 'corrected')]
            ax.plot(v, [yi, yi], color=GRID, lw=4, solid_capstyle='round', zorder=1)
            ax.scatter([v[0]], [yi], s=70, facecolor='white', edgecolor=col, linewidth=2, zorder=3)
            ax.scatter([v[1]], [yi], s=70, color=col, edgecolor='white', linewidth=1.5, zorder=3)
            ax.annotate(f'{v[1]:.1f}%', (v[1], yi), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8.5, color=INK)
        ax.set_yticks(y, [l for _, l, _ in MODELS], fontsize=9.5, color=INK)
        ax.set_ylim(-0.6, len(MODELS) - 0.2)
        ax.set_xlim(left=0)
        ax.set_title(f'{lo}–{hi} m: median range error (%)', fontsize=10.5, loc='left', color=INK)
        style(ax)
    h = [plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, mew=2, label='raw keypoints'),
         plt.Line2D([], [], marker='o', ls='', color=INK2, label='with correction (labelled)')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(left=0.13, wspace=0.08, bottom=0.22)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_bias(runs, cones, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axhline(0, color='#c9c8c3', lw=1)
    for m, lab, col in MODELS:
        b = per_bin(runs[m]['res']['raw'], cones, 'bias')
        ax.plot(MID, [e for e, _ in b], color=col, lw=2, marker='o', ms=7, mec='white', mew=1.5, label=lab)
    ax.set_xticks(MID, [f'{lo}–{hi}' for lo, hi in BINS])
    ax.set_xlabel('ground range (m)')
    ax.set_title('Raw range bias (%, + = placed too far)', fontsize=10.5, loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc='best')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_cdf(runs, cones, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.3), sharey=True)
    for ax, (lo, hi) in zip(axes, ((2, 30), (30, 60))):
        n = sum(1 for c in cones.values() if lo <= c['ground_range_m'] < hi)
        for m, lab, col in MODELS:
            rel = np.sort([x['rel'] for x in runs[m]['res']['corrected'].values() if lo <= x['rng'] < hi])
            ax.plot(np.r_[0, rel], np.arange(len(rel) + 1) / n * 100, color=col, lw=2, label=lab)
        ax.axvline(5, color='#c9c8c3', lw=1, ls=':')
        ax.axvline(10, color='#c9c8c3', lw=1, ls=':')
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 100)
        ax.set_xlabel('range error (% of distance)')
        ax.set_title(f'{lo}–{hi} m: % of ALL cones solved within a given error', fontsize=10.5, loc='left', color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_speed(points, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for ax, k, title in ((axes[0], 'solved', 'Cones solved at 30–60 m (%)'),
                         (axes[1], 'within5', 'Cones at 30–60 m solved within 5% error (%)')):
        for m, lab, col in MODELS:
            pts = sorted([p for p in points if p['m'] == m], key=lambda p: p['lat'])
            ax.plot([p['lat'] for p in pts], [p[k] for p in pts], color=col, lw=1.5, zorder=2, label=lab)
            for p, mk in zip(sorted(pts, key=lambda p: p['sz']), ('s', '^', 'o')):
                ax.scatter([p['lat']], [p[k]], marker=mk, s=60, color=col, edgecolor='white', linewidth=1.5, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks([8, 10, 15, 20, 30, 40, 60], ['8', '10', '15', '20', '30', '40', '60'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('latency per frame, batch 1, RTX 5060 (ms, log)')
        ax.set_ylim(-8, 100)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
        style(ax)
    h = axes[0].get_legend_handles_labels()[0] + [plt.Line2D([], [], marker=mk, ls='', color=INK2, label=f'{sz} px')
                                                  for mk, sz in (('s', 960), ('^', 1280), ('o', 1920))]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.16, bottom=0.2)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def pick_frame(frames, cones):
    best = None
    for fi, (_, a) in enumerate(frames):
        if fi % 2 == 0 or not 0.3 <= a['defects']['severity'] <= 0.7:
            continue
        far = sum(1 for k, c in cones.items() if k[0] == fi and c['ground_range_m'] >= 30)
        if best is None or far > best[0]:
            best = (far, fi)
    return best[1]


def solutions(fi, frames, K, dist, run, cones):
    out = []
    for r in run['recs']:
        if r['fi'] != fi or (fi, id(r['c'])) not in cones:
            continue
        kp = kc.correct(r, run['tab'])
        use = np.where(kp[:6, 2] >= ev.KP_CONF)[0]
        if len(use) >= 4 and (sol := ev.solve_pnp(ev.planar_model(r['c']['class_id'])[use], kp[use, :2].astype(np.float64), K, dist)):
            out.append(dict(c=r['c'], kp=r['kp'], t=sol[1] * run['scale'](r['hp'])))
    return out


def fig_topdown(fi, frames, K, dist, runs, cones, path):
    a = frames[fi][1]
    show = ('nano', 'medium', 'E_far')
    fig, axes = plt.subplots(1, len(show), figsize=(13, 7.4), sharey=True)
    for ax, m in zip(axes, show):
        sols = solutions(fi, frames, K, dist, runs[m], cones)
        hfov = np.degrees(2 * np.arctan(a['width'] / 2 / K[0, 0]))
        for s in (-1, 1):
            ax.plot([0, 64 * np.tan(np.radians(s * hfov / 2))], [0, 64], color='#c9c8c3', lw=1, zorder=0)
        for k, c in cones.items():
            if k[0] == fi:
                x, _, z = c['base_center_cam']
                ax.scatter(x, z, s=30, color='#2a78d6' if c['class_id'] == 0 else '#eda100', edgecolor=INK2, lw=0.6, zorder=3)
        for s in sols:
            gx, _, gz = s['c']['base_center_cam']
            ax.plot([gx, s['t'][0]], [gz, s['t'][2]], color=INK2, lw=1, zorder=2)
            ax.scatter(s['t'][0], s['t'][2], s=30, marker='x', color=COL[m], lw=1.8, zorder=4)
        n = sum(1 for k in cones if k[0] == fi)
        ax.set_title(f'{LAB[m]}\n{len(sols)} of {n} cones solved', fontsize=10.5, loc='left', color=INK)
        ax.set_xlim(-26, 26)
        ax.set_ylim(0, 62)
        ax.set_aspect('equal')
        ax.set_xlabel('lateral x (m)')
        style(ax)
    axes[0].set_ylabel('forward z (m)')
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def frame_detections(fi, frames, runs, cones, img, figdir):
    """Each model's raw detections on the frame (cropped to the cone rows): boxes and keypoints (conf >= 0.5) in the model colour,
    eval cones the model missed in grey. Written at native resolution so the PDF can be zoomed."""
    a = frames[fi][1]
    boxes = [c['bbox_xyxy_clipped'] for k, c in cones.items() if k[0] == fi]
    top = max(0, int(min(b[1] for b in boxes)) - 60) // 8 * 8
    bot = min(a['height'], int(max(b[3] for b in boxes)) + 60)
    out = []
    for m, lab, col in MODELS:
        im = img.copy()
        bgr = tuple(int(col[i:i + 2], 16) for i in (5, 3, 1))
        p = runs[m]['preds'][fi]
        matched = {id(r['c']) for r in runs[m]['recs'] if r['fi'] == fi}
        for k, c in cones.items():
            if k[0] == fi and id(c) not in matched:
                x0, y_0, x1, y1 = map(int, c['bbox_xyxy_clipped'])
                cv2.rectangle(im, (x0 - 2, y_0 - 2), (x1 + 2, y1 + 2), (150, 150, 150), 1, cv2.LINE_AA)
        for b, kp in zip(p['boxes'], p['kpts']):
            x0, y_0, x1, y1 = map(int, b)
            cv2.rectangle(im, (x0, y_0), (x1, y1), bgr, 2, cv2.LINE_AA)
            for x, y, cf in kp[:6]:
                if cf >= ev.KP_CONF:
                    cv2.circle(im, (int(round(x)), int(round(y))), 2, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(im, (int(round(x)), int(round(y))), 1, bgr, -1, cv2.LINE_AA)
        cv2.imwrite(str(figdir / f'frame_{m}.jpg'), im[top:bot], [cv2.IMWRITE_JPEG_QUALITY, 93])
        n_eval = sum(1 for k in cones if k[0] == fi)
        n_found = sum(1 for k, c in cones.items() if k[0] == fi and id(c) in matched)
        out.append((m, lab, col, n_found, n_eval, len(p['boxes'])))
    return out


def zoom_strip(fi, frames, K, dist, runs, cones, img, n=6):
    """Far cones (raw network keypoints). White rings = truth, dots = prediction; no dots = not detected."""
    far = sorted([c for k, c in cones.items() if k[0] == fi and 20 <= c['ground_range_m'] < 60], key=lambda c: c['ground_range_m'])
    picks = [far[j] for j in np.linspace(0, len(far) - 1, min(n, len(far))).astype(int)]
    rows = []
    for m, lab, col in MODELS:
        recs = {id(r['c']): r for r in runs[m]['recs'] if r['fi'] == fi}
        row = []
        for c in picks:
            x0, y0, x1, y1 = c['bbox_xyxy_clipped']
            h = y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            X0, Y0 = int(cx - 0.8 * h), int(cy - 0.8 * h)
            S = 230
            s = S / (1.6 * h)
            crop = cv2.resize(img[max(0, Y0):Y0 + int(1.6 * h), max(0, X0):X0 + int(1.6 * h)], (S, S), interpolation=cv2.INTER_NEAREST)
            P = lambda x, y: (round((x - X0) * s), round((y - Y0) * s))
            for x, y, v in c['keypoints_2d'][:6]:
                cv2.circle(crop, P(x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
            r = recs.get(id(c))
            if r is not None:
                bgr = tuple(int(col[i:i + 2], 16) for i in (5, 3, 1))
                for x, y, cf in r['kp'][:6]:
                    if cf >= ev.KP_CONF:
                        cv2.circle(crop, P(x, y), 5, bgr, -1, cv2.LINE_AA)
                        cv2.circle(crop, P(x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
            txt = f"{c['ground_range_m']:.0f} m, {h:.0f} px" + ('' if r is not None else ', MISSED')
            crop = cv2.copyMakeBorder(crop, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(crop, txt, (4, S + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
            row.append(cv2.copyMakeBorder(crop, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        tag = np.full((S + 36, 170, 3), 255, np.uint8)
        cv2.circle(tag, (14, (S + 6) // 2), 7, tuple(int(col[i:i + 2], 16) for i in (5, 3, 1)), -1, cv2.LINE_AA)
        for j, wline in enumerate(lab.split(' ', 1)):
            cv2.putText(tag, wline, (28, (S + 6) // 2 - 4 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.hstack([tag] + row))
    return np.vstack(rows)


# ---------- page ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=str(Path.home() / 'Downloads' / 'cone-pnp-report-v3.pdf'))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    figdir = root / 'eval' / 'pdf_figures_v3'
    figdir.mkdir(exist_ok=True)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    cones = eval_cones(frames)

    runs = {}
    for m, _, _ in MODELS:
        print('evaluating', m, flush=True)
        runs[m] = evaluate(frames, K, dist, cones, m, 1920)
    floor = {}
    for k, c in cones.items():
        e = kc.pnp(np.array(c['keypoints_2d'], float), c, K, dist, np.array(frames[k[0]][1]['camera']['T_world_cam']))
        if e:
            floor[k] = dict(rel=abs(e['range_err']) / c['distance_m'] * 100, rng=c['ground_range_m'])

    points = []
    n_far = sum(1 for c in cones.values() if c['ground_range_m'] >= 30)
    for m, sz in SPEED:
        r = runs[m] if sz == 1920 else evaluate(frames, K, dist, cones, m, sz)
        far = [x['rel'] for x in r['res']['corrected'].values() if x['rng'] >= 30]
        points.append(dict(m=m, sz=sz, lat=r['latency'], solved=len(far) / n_far * 100,
                           within5=np.sum(np.array(far) < 5) / n_far * 100))

    fig_range(runs, floor, cones, figdir / 'range.png')
    fig_dumbbell(runs, figdir / 'dumbbell.png')
    fig_bias(runs, cones, figdir / 'bias.png')
    fig_cdf(runs, cones, figdir / 'cdf.png')
    fig_speed(points, figdir / 'speed.png')
    fi = pick_frame(frames, cones)
    fig_topdown(fi, frames, K, dist, runs, cones, figdir / 'topdown.png')
    img = cv2.imread(str(root / frames[fi][1]['image']))
    dets = frame_detections(fi, frames, runs, cones, img, figdir)
    cv2.imwrite(str(figdir / 'zoom.jpg'), zoom_strip(fi, frames, K, dist, runs, cones, img), [cv2.IMWRITE_JPEG_QUALITY, 92])

    far = lambda m, k='corrected': [x['rel'] for x in runs[m]['res'][k].values() if x['rng'] >= 30]
    best_old = max(('nano', 'small', 'medium'), key=lambda m: len(far(m)))
    tiles = [(f"{len(far('E_far')) / n_far * 100:.0f}%", f"cones at 30–60 m solved by E (best original, {LAB[best_old]}: "
              f"{len(far(best_old)) / n_far * 100:.0f}%)"),
             (f"{np.median(far('E_far')):.1f}%", f"E median range error at 30–60 m ({LAB[best_old]}: {np.median(far(best_old)):.1f}%)"),
             (f"{np.median(far('E_far', 'raw')):.1f}%", 'E at 30–60 m with no correction at all'),
             (f"{runs['E_far']['latency']:.1f} ms", f"E per 1920 frame (medium: {runs['medium']['latency']:.0f} ms)")]
    tiles_html = ''.join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles)
    a = frames[fi][1]

    page = f"""<!doctype html><html><head><meta charset="utf-8">{CSS}</head><body>
<section><h1>Long-range cone pose: original models vs far-cone synthetic training</h1>
<p class="sub">Synthetic test set (1000 Blender frames, 1920×1200, 80° HFOV); blue/yellow cones on the odd frames, 2–60 m,
fully visible ({len(cones)} cones). All models run on the full 1920 frame (native resolution). Each model has its own keypoint
correction fitted on the even frames. Generated {time.strftime('%Y-%m-%d')}.</p>
<div class="tiles">{tiles_html}</div>
<h3>The models</h3>
<ul>
<li><b>Original nano / small / medium</b>: the first BRT pose models (all cone classes, real labels only, trained at 960 px).</li>
<li><b>Far-cone nano (E)</b>: cleaned + augmented blue/yellow nano (run D), fine-tuned 20 epochs on 2,815 native-resolution
960×600 tiles of far cones from 2,000 separately rendered synthetic frames, mixed with 2,500 real images. Selected on 322
held-out synthetic tiles; never trained on the test frames.</li>
</ul>
<img class="mid" src="file://{figdir}/range.png">
</section>

<section class="pb"><h2>How much the correction matters</h2>
<p class="sub">Open circle = raw network keypoints, filled = with each model's own box-height correction. The original models need
the correction; E without any correction already beats every corrected original model.</p>
<img src="file://{figdir}/dumbbell.png">
<div class="two"><div><img src="file://{figdir}/bias.png"></div>
<div><h3>Reading this</h3><ul>
<li>The original models swing from about +6% (too far, the small-cone label template) at 20–30 m to −4% at 50–60 m. E has a
steady short bias of 1–5% that grows with range; its correction removes it.</li>
<li>Beyond 30 m the original models solve few cones, so their error curves rest on the easiest cones that were found.</li>
<li>Model size does not help: at 1920, small and medium solve no more far cones than the original nano, at 2× and 5× the latency.</li>
</ul></div></div>
</section>

<section class="pb"><h2>Error distribution</h2>
<p class="sub">Share of all eval cones in the range band that were solved within a given range error (curves end at the share solved).
Dotted lines at 5% and 10%.</p>
<img class="mid" src="file://{figdir}/cdf.png">
<h2 style="margin-top:14px">Speed vs long-range performance</h2>
<p class="sub">Each model at 960 (square), 1280 (triangle) and 1920 px (circle), latency measured on the same 1000 frames. E was
trained on native-scale tiles, i.e. for 1920; at lower input sizes far cones shrink below what it was trained on.</p>
<img class="mid" src="file://{figdir}/speed.png">
</section>

<section class="pb"><h2>Example frame {frames[fi][0]}</h2>
<p class="sub">{a['capture']['extras']['car_context'].replace('_', ' ')}, defect severity {a['defects']['severity']:.2f}. Top-down camera
frame (camera at the origin). Dots: true cones (eval set); crosses: corrected PnP estimates, joined to their cone.</p>
<img class="tall" src="file://{figdir}/topdown.png"></section>
<section class="pb"><h2>Detections in frame {frames[fi][0]} (image)</h2>
<p class="sub">The same scene in the camera image, cropped to the rows containing cones, each model at 1920. Boxes and keypoints (confidence ≥ 0.5): raw
network output in the model colour; grey boxes: eval cones the model did not match. Images are embedded at full resolution — zoom
in to see the far cones.</p>
<div class="two" style="gap:8px">{''.join(f'<div><div class="note" style="margin:2px 0"><span style="color:{col}">●</span> <b style="color:#0b0b0b">{lab}</b> — {nf} of {ne} eval cones detected, {nb} boxes</div><img style="margin:0" src="file://{figdir}/frame_{m}.jpg"></div>' for m, lab, col, nf, ne, nb in dets)}</div>
</section>

<section class="pb"><h2>Far-cone keypoints, frame {frames[fi][0]}</h2>
<p class="sub">Cones at 20–60 m, nearest to farthest, crops enlarged with nearest-neighbour from the native image. White rings:
exact keypoints; dots: raw network keypoints (confidence ≥ 0.5). MISSED = no matched detection; no dots without MISSED = detected, but keypoint confidence below 0.5.</p>
<img style="max-height:118mm" src="file://{figdir}/zoom.jpg">
<h3>Caveats</h3>
<ul>
<li>E was trained on the same renderer as this test set, so part of its gain is domain fit; the real-world gain needs real images
with measured distances. Its real-validation pose mAP dropped slightly (0.918 → 0.907 for run D → E).</li>
<li>The original models were trained at 960 px; at 1920 they see cones larger than in training, which helps recall but is not
their native setting (see the speed chart for 960 and 1280).</li>
</ul></section>
</body></html>"""
    with tempfile.TemporaryDirectory() as td:
        hp = Path(td) / 'report_v3.html'
        hp.write_text(page)
        (figdir / 'report_v3.html').write_text(page)
        exe = sorted(Path.home().glob('.cache/ms-playwright/chromium-*/chrome-linux64/chrome'))[-1]
        script = f"""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path={str(exe)!r})
    pg = b.new_page(viewport={{'width': 1100, 'height': 900}}, color_scheme='light')
    pg.goto({hp.as_uri()!r}); pg.wait_for_timeout(500)
    pg.pdf(path={args.out!r}, format='A4', landscape=True, print_background=True,
           margin={{'top': '10mm', 'bottom': '10mm', 'left': '10mm', 'right': '10mm'}})
    b.close()
"""
        subprocess.run(['uv', 'run', '--no-project', '--with', 'playwright', 'python', '-c', script], check=True)
    print('frame', frames[fi][0], '->', args.out)


if __name__ == '__main__':
    main()
