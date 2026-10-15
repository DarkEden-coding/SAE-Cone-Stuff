"""Updated PDF report: run D (cleaned + augmented nano) with the box-height keypoint correction, against the old nano.

uv run --locked synth/report_v2.py synth_data [--out ~/Downloads/cone-pnp-report-v2.pdf]

Blue/yellow cones only (run D is a blue/yellow model). The correction (kp_correction.py: per-keypoint bias by
predicted box height, then a range scale) is fitted on even frames; every number here is on odd frames.
Uses cached predictions in eval/predictions; the root-cause page reads label_audit/bias_origin.json.
Needs the Playwright-cached Chromium for printing, like report_pdf.py.
"""
import argparse
import html
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'synth'))
import eval_pnp as ev  # noqa: E402
import kp_correction as kc  # noqa: E402

ev.MODELS['D_aug30'] = ROOT / 'runs/by-n-D-aug/weights/last.pt'
CONFIGS = [('nano', 960, 'old nano'), ('D_aug30', 960, 'D'), ('D_aug30', 1280, 'D'), ('D_aug30', 1920, 'D')]
RANGE_BINS = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
H_BINS = [(10, 14), (14, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, 400)]
# reference palette (dataviz skill), light mode, fixed order
C_OLD, C_RAW, C_COR, C_HI, GRAY = '#eb6834', '#2a78d6', '#1baf7a', '#e87ba4', '#8a8984'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c3')
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def evaluate(frames, K, dist, name, imgsz):
    """Per-cone results on odd frames for raw and corrected keypoints, plus the eval-set denominator."""
    preds = ev.run_predictions(name, imgsz, frames, ROOT / 'synth_data',
                               ROOT / 'synth_data/eval/predictions' / (f'{name}_{imgsz}.pkl' if name == 'nano' else f'xp_{name}_{imgsz}.pkl'))
    recs = kc.collect(frames, preds)
    tab, _ = kc.fit([r for r in recs if r['fi'] % 2 == 0])
    scale, _ = kc.fit_range([r for r in recs if r['fi'] % 2 == 0 and 2 <= r['c']['ground_range_m'] <= 30], tab, K, dist, frames)
    test = [r for r in recs if r['fi'] % 2 == 1 and 2 <= r['c']['ground_range_m'] <= 30]
    out = {'raw': {}, 'corrected': {}}
    for r in test:
        T = np.array(frames[r['fi']][1]['camera']['T_world_cam'])
        key = (r['fi'], id(r['c']))
        for v, kp, s in (('raw', r['kp'], 1.0), ('corrected', kc.correct(r, tab), scale(r['hp']))):
            e = kc.pnp(kp, r['c'], K, dist, T, s)
            out[v][key] = None if e is None else dict(rel=e['rel_err'] * 100, bias=e['range_err'] / r['c']['distance_m'] * 100,
                                                      t=e['t_err'], rng=r['c']['ground_range_m'], hp=r['hp'])
    t_corr = timeit_correct(test, tab, scale)
    lat = float(np.median([p['latency_ms'] for p in preds]))
    return dict(res=out, tab=tab, scale=scale, recs=recs, preds=preds, latency=lat, corr_us=t_corr)


def timeit_correct(test, tab, scale, n=500):
    sub = test[:n]
    t0 = time.perf_counter()
    for r in sub:
        kc.correct(r, tab), scale(r['hp'])
    return (time.perf_counter() - t0) / max(len(sub), 1) * 1e6


def eval_set(frames):
    """Odd-frame blue/yellow detectable cones at 2-30 m: (frame index, cone id) -> cone."""
    return {(fi, id(c)): c for fi, (_, a) in enumerate(frames) if fi % 2 == 1 for c in a['cones']
            if c['class_id'] in kc.BY and ev.detectable(c) and 2 <= c['ground_range_m'] <= 30}


def gt_floor(frames, K, dist, keys, cones):
    out = {}
    for k in keys:
        c = cones[k]
        e = kc.pnp(np.array(c['keypoints_2d'], float), c, K, dist, np.array(frames[k[0]][1]['camera']['T_world_cam']))
        out[k] = None if e is None else dict(rel=e['rel_err'] * 100, t=e['t_err'], rng=c['ground_range_m'])
    return out


def summ(res, n_total, keys=None):
    vals = [v for k, v in res.items() if v is not None and (keys is None or k in keys)]
    rel = np.array([v['rel'] for v in vals])
    return dict(n=len(vals), success=len(vals) / n_total if n_total else None, rel_med=float(np.median(rel)),
                rel_p90=float(np.percentile(rel, 90)), outl=float(np.mean(rel > 10) * 100),
                bias=float(np.median([v['bias'] for v in vals])) if 'bias' in vals[0] else None,
                t_med=float(np.median([v['t'] for v in vals])))


def binned(res, key, bins, field='rel'):
    out = []
    for lo, hi in bins:
        v = [x[field] for x in res.values() if x is not None and lo <= x[key] < hi]
        out.append((float(np.median(v)), len(v)) if len(v) >= 20 else (np.nan, len(v)))
    return out


# ---------- figures ----------

def fig_headline(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    labels = [r['label'] for r in rows]
    y = np.arange(len(rows))[::-1]
    for ax, (k, title, fmt) in zip(axes, (('rel_med', 'Median relative error (%)', '{:.2f}%'),
                                          ('outl', 'Outliers, error > 10% (%)', '{:.1f}%'),
                                          ('bias', 'Median range bias (%, + = too far)', '{:+.2f}%'))):
        for yi, r in zip(y, rows):
            a, b = r['raw'][k], r['corrected'][k]
            ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle='round')
            ax.scatter([a], [yi], s=60, color=C_RAW, edgecolor='white', linewidth=1.5, zorder=3)
            ax.scatter([b], [yi], s=60, color=C_COR, edgecolor='white', linewidth=1.5, zorder=3)
            ax.annotate(fmt.format(b), (b, yi), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.5, color=INK)
        if k == 'bias':
            ax.axvline(0, color='#c9c8c3', lw=1, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [''] * len(rows), fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=10.5, loc='left', color=INK)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        style(ax)
    h = [plt.Line2D([], [], marker='o', ls='', color=C_RAW, label='raw keypoints'),
         plt.Line2D([], [], marker='o', ls='', color=C_COR, label='with correction (labelled)')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.2, wspace=0.12)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_by(series, floor, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.3))
    xr = [f'{lo}–{hi} m' for lo, hi in RANGE_BINS]
    for name, col, res in series:
        b = binned(res, 'rng', RANGE_BINS)
        axes[0].plot(range(len(xr)), [m for m, _ in b], color=col, lw=2, marker='o', ms=6, mec='white', mew=1.5, label=name)
        hb = binned(res, 'hp', H_BINS, 'bias')
        axes[1].plot(range(len(H_BINS)), [m for m, _ in hb], color=col, lw=2, marker='o', ms=6, mec='white', mew=1.5, label=name)
    fb = binned(floor, 'rng', RANGE_BINS)
    axes[0].plot(range(len(xr)), [m for m, _ in fb], color=GRAY, lw=1.5, ls='--', label='ground-truth keypoints (floor)')
    axes[0].set_xticks(range(len(xr)), xr)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title('Median relative error by ground range (%)', fontsize=10.5, loc='left', color=INK)
    axes[1].axhline(0, color='#c9c8c3', lw=1)
    axes[1].set_xticks(range(len(H_BINS)), [f'{lo}–{hi}' for lo, hi in H_BINS])
    axes[1].set_xlabel('predicted box height (native px)')
    axes[1].set_title('Median range bias by box height (%, + = too far)', fontsize=10.5, loc='left', color=INK)
    for ax in axes:
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_cdf(series, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for name, col, res in series:
        rel = np.sort([v['rel'] for v in res.values() if v is not None])
        ax.plot(rel, np.arange(1, len(rel) + 1) / len(rel) * 100, color=col, lw=2, label=name)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 100)
    ax.axvline(10, color='#c9c8c3', lw=1, ls=':')
    ax.set_xlabel('relative translation error (%)')
    ax.set_title('Share of solved cones under a given error (%)', fontsize=10.5, loc='left', color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def template_stats(recs):
    """D@960 on synthetic cones: keypoint geometry relative to the predicted box vs the truth in the same box."""
    bins = ((8, 12), (12, 16), (16, 24), (24, 40), (40, 80), (80, 400))
    rows = []
    for lo, hi in bins:
        sub = [r for r in recs if lo <= r['hp'] / 2 < hi and (r['kp'][:6, 2] >= ev.KP_CONF).all()]  # input px = native / 2
        if len(sub) < 20:
            continue
        f = dict(model_foot=[], true_foot=[], model_wtop=[], true_wtop=[], pred_aspect=[], gt_aspect=[], bottom_gap=[])
        for r in sub:
            gt = np.array(r['c']['keypoints_2d'], float)
            p = r['kp']
            gb = r['c']['bbox_xyxy_clipped']
            pb = r['box']
            h = pb[3] - pb[1]
            f['model_foot'].append((pb[3] - (p[4, 1] + p[5, 1]) / 2) / h)
            f['true_foot'].append((pb[3] - (gt[4, 1] + gt[5, 1]) / 2) / h)
            f['model_wtop'].append((p[1, 0] - p[0, 0]) / h)
            f['true_wtop'].append((gt[1, 0] - gt[0, 0]) / h)
            f['pred_aspect'].append((pb[2] - pb[0]) / h)
            f['gt_aspect'].append((gb[2] - gb[0]) / (gb[3] - gb[1]))
            f['bottom_gap'].append((gb[3] - pb[3]) / h)
        rows.append(dict(bin=f'{lo}–{hi}', n=len(sub), **{k: float(np.median(v)) for k, v in f.items()}))
    return rows


def fig_template(tstats, bias, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = range(len(tstats))
    xl = [t['bin'] for t in tstats]
    ax = axes[0]
    ax.plot(x, [t['model_foot'] for t in tstats], color=C_RAW, lw=2, marker='o', mec='white', mew=1.5, label='model foot')
    ax.plot(x, [t['true_foot'] for t in tstats], color=INK2, lw=1.5, ls='--', marker='o', mec='white', mew=1.5, label='true foot')
    ax.set_title('Foot height above predicted box bottom (× box h)', fontsize=10, loc='left', color=INK)
    ax = axes[1]
    ax.plot(x, [t['pred_aspect'] for t in tstats], color=C_RAW, lw=2, marker='o', mec='white', mew=1.5, label='predicted box')
    ax.plot(x, [t['gt_aspect'] for t in tstats], color=INK2, lw=1.5, ls='--', marker='o', mec='white', mew=1.5, label='true box')
    ax.set_title('Box width / height', fontsize=10, loc='left', color=INK)
    for ax in axes[:2]:
        ax.set_xticks(list(x), xl)
        ax.set_xlabel('box height at network input (px)')
        ax.legend(frameon=False, fontsize=8.5)
        style(ax)
    ax = axes[2]
    bins = ['4–8', '8–12', '12–16', '16–24', '24–40', '40–80', '80–400']
    for src, col, lab in (('synth_gt', INK2, 'synthetic, exact'), ('real_label', C_OLD, 'real human labels'),
                          ('real_model', C_RAW, 'model D on real val')):
        v = [c['median'] for c in bias['box_aspect'][src]]
        ax.plot(range(len(bins)), [np.nan if m is None else m for m in v], color=col, lw=2 if src != 'synth_gt' else 1.5,
                ls='--' if src == 'synth_gt' else '-', marker='o', mec='white', mew=1.5, label=lab)
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel('box height at network input (px)')
    ax.set_title('Box width / height: real labels vs geometry', fontsize=10, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_offsets(tab, path):
    centres = np.sqrt(kc.EDGES[:-1] * kc.EDGES[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), sharey=False)
    names = ['stripe top', 'stripe bottom', 'foot']
    cols = [C_RAW, C_OLD, C_COR]
    for j, (ax, title) in enumerate(zip(axes, ('Inward offset (% of box height, + = toward centre)',
                                              'Vertical offset (% of box height, + = too low)'))):
        for p in range(3):
            v = np.nanmean(tab[:, 2 * p:2 * p + 2, j], axis=1) * 100
            ok = np.isfinite(v)
            ax.plot(centres[ok], v[ok], color=cols[p], lw=2, marker='o', ms=5, mec='white', mew=1.2, label=names[p])
        ax.axhline(0, color='#c9c8c3', lw=1)
        ax.set_xscale('log')
        ax.set_xticks([12, 20, 30, 50, 80, 150, 300], ['12', '20', '30', '50', '80', '150', '300'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('predicted box height (native px, D @960)')
        ax.set_title(title, fontsize=10, loc='left', color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def pick_frame(frames):
    best = None
    for fi, (stem, a) in enumerate(frames):
        if fi % 2 == 0 or a['capture']['extras']['car_context'] == 'straight':
            continue
        cones = [c for c in a['cones'] if c['class_id'] in kc.BY and ev.detectable(c) and 2 <= c['ground_range_m'] <= 30]
        far = sum(c['ground_range_m'] > 12 for c in cones)
        if 0.3 <= a['defects']['severity'] <= 0.7 and (best is None or len(cones) + far > best[0]):
            best = (len(cones) + far, fi)
    return best[1]


def frame_solutions(fi, frames, K, dist, run, corrected):
    a = frames[fi][1]
    out = []
    for r in run['recs']:
        if r['fi'] != fi:
            continue
        kp = kc.correct(r, run['tab']) if corrected else r['kp']
        use = np.where(kp[:6, 2] >= ev.KP_CONF)[0]
        if len(use) < 4:
            continue
        sol = ev.solve_pnp(ev.planar_model(r['c']['class_id'])[use], kp[use, :2].astype(np.float64), K, dist)
        if sol:
            out.append(dict(c=r['c'], kp=kp, t=sol[1] * (run['scale'](r['hp']) if corrected else 1.0)))
    return a, out


def fig_topdown(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.2), sharey=True)
    for ax, (title, col, a, sols) in zip(axes, panels):
        hfov = np.degrees(2 * np.arctan(a['width'] / 2 / a['camera']['K'][0][0]))
        for s in (-1, 1):
            ax.plot([0, 40 * np.tan(np.radians(s * hfov / 2))], [0, 40], color='#c9c8c3', lw=1, zorder=0)
        for c in a['cones']:
            if c['class_id'] in kc.BY:
                x, _, z = c['base_center_cam']
                ax.scatter(x, z, s=36, color='#2a78d6' if c['class_id'] == 0 else '#eda100', edgecolor='white', lw=1, zorder=3)
        errs = []
        for s in sols:
            gx, _, gz = s['c']['base_center_cam']
            ex, _, ez = s['t']
            ax.plot([gx, ex], [gz, ez], color=INK2, lw=1, zorder=2)
            ax.scatter(ex, ez, s=34, marker='x', color=col, lw=1.8, zorder=4)
            if 2 <= s['c']['ground_range_m'] <= 30:
                errs.append(np.linalg.norm(s['t'] - np.array(s['c']['base_center_cam'])))
        ax.set_title(f'{title}\n{len(errs)} solved (2–30 m), median error {np.median(errs):.2f} m', fontsize=10, loc='left', color=INK)
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 38)
        ax.set_aspect('equal')
        ax.set_xlabel('lateral x (m)')
        style(ax)
    axes[0].set_ylabel('forward z (m)')
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def zoom_strip(img, a, rows, n=5):
    """Close-ups, nearest to farthest: white rings = truth, dots = raw / corrected keypoints per row."""
    cones = sorted([c for c in a['cones'] if c['class_id'] in kc.BY and ev.detectable(c) and not c['truncated']
                    and 2 <= c['ground_range_m'] <= 30], key=lambda c: c['ground_range_m'])
    picks = [cones[j] for j in np.linspace(0, len(cones) - 1, min(n, len(cones))).astype(int)]
    out = []
    for label, col, sols in rows:
        row = []
        for c in picks:
            x0, y0, x1, y1 = c['bbox_xyxy_clipped']
            pad = 0.3 * (y1 - y0)
            X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            X1, Y1 = int(min(img.shape[1], x1 + pad)), int(min(img.shape[0], y1 + pad))
            s = 280 / (Y1 - Y0)
            crop = cv2.resize(img[Y0:Y1, X0:X1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            P = lambda x, y: (round((x - X0) * s), round((y - Y0) * s))
            for x, y, v in c['keypoints_2d'][:6]:
                if v:
                    cv2.circle(crop, P(x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
            sol = next((q for q in sols if q['c'] is c), None)
            if sol is not None:
                bgr = tuple(int(col[i:i + 2], 16) for i in (5, 3, 1))
                for x, y, cf in sol['kp'][:6]:
                    if cf >= ev.KP_CONF:
                        cv2.circle(crop, P(x, y), 4, bgr, -1, cv2.LINE_AA)
                        cv2.circle(crop, P(x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
            txt = f"{c['ground_range_m']:.1f} m, {y1 - y0:.0f} px"
            cv2.putText(crop, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(crop, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
            row.append(cv2.copyMakeBorder(crop, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        h = max(r.shape[0] for r in row)
        out.append(np.hstack([cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in row]))
    w = max(r.shape[1] for r in out)
    return np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in out])


# ---------- page ----------

CSS = """<style>
@page{size:A4 landscape;margin:10mm}
html,body{background:#fff;color:#0b0b0b;font:13px/1.45 system-ui,'Inter',sans-serif;margin:0}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 4px}h3{font-size:13.5px;margin:14px 0 4px}
.sub{color:#52514e;font-size:12.5px;margin:0 0 8px}.pb{break-before:page}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.tile{border:1px solid #e4e3df;border-radius:8px;padding:10px 14px}.tile .v{font-size:22px;font-weight:600}
.tile .l{color:#52514e;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;margin:6px 0;break-inside:avoid}
th,td{padding:4px 8px;text-align:right;border-bottom:1px solid #e4e3df;white-space:nowrap}
th{color:#52514e;font-weight:500}td:first-child,th:first-child{text-align:left}
tr.hl td{font-weight:600}
img{max-width:100%;display:block;margin:6px auto 0;break-inside:avoid}img.tall{max-height:150mm}img.mid{max-height:72mm}
ul{margin:4px 0 8px 18px;padding:0}li{margin:2px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.note{color:#52514e;font-size:11.5px}
</style>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=str(Path.home() / 'Downloads' / 'cone-pnp-report-v2.pdf'))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    figdir = root / 'eval' / 'pdf_figures_v2'
    figdir.mkdir(exist_ok=True)
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    frames = [(p.stem, json.loads(p.read_text())) for p in sorted((root / 'annotations').glob('*.json'))]
    cones = eval_set(frames)
    n_total = len(cones)

    runs = {}
    for name, imgsz, label in CONFIGS:
        print('evaluating', name, imgsz, flush=True)
        runs[(name, imgsz)] = evaluate(frames, K, dist, name, imgsz)
        for r in runs[(name, imgsz)]['recs']:  # keep the matched box for the template page
            p = runs[(name, imgsz)]['preds'][r['fi']]
            r['box'] = p['boxes'][int(np.argmin(np.abs(p['kpts'] - r['kp']).sum(axis=(1, 2))))]
    floor = gt_floor(frames, K, dist, list(cones), cones)
    # common set: solved by every configuration and variant (and the floor)
    common = set(k for k, v in floor.items() if v is not None)
    for run in runs.values():
        for res in run['res'].values():
            common &= {k for k, v in res.items() if v is not None}

    rows = []
    for name, imgsz, label in CONFIGS:
        run = runs[(name, imgsz)]
        rows.append(dict(label=f'{label} @{imgsz}', latency=run['latency'], corr_us=run['corr_us'],
                         **{v: summ(run['res'][v], n_total) for v in ('raw', 'corrected')},
                         **{f'common_{v}': summ(run['res'][v], n_total, common) for v in ('raw', 'corrected')}))
    fl = summ(floor, n_total)
    fl_common = summ(floor, n_total, common)

    fig_headline(rows, figdir / 'headline.png')
    d960, d1920, old = runs[('D_aug30', 960)], runs[('D_aug30', 1920)], runs[('nano', 960)]
    series = [('old nano @960, raw', C_OLD, old['res']['raw']), ('D @960, raw', C_RAW, d960['res']['raw']),
              ('D @960, corrected', C_COR, d960['res']['corrected']), ('D @1920, corrected', C_HI, d1920['res']['corrected'])]
    fig_by(series, floor, figdir / 'by.png')
    fig_cdf(series, figdir / 'cdf.png')
    tstats = template_stats(d960['recs'])
    bias = json.loads((ROOT / 'label_audit/bias_origin.json').read_text())
    fig_template(tstats, bias, figdir / 'template.png')
    fig_offsets(d960['tab'], figdir / 'offsets.png')

    fi = pick_frame(frames)
    panels, zrows = [], []
    for title, col, run, corr in (('old nano @960, raw', C_OLD, old, False), ('D @960, raw', C_RAW, d960, False),
                                  ('D @960, corrected', C_COR, d960, True)):
        a, sols = frame_solutions(fi, frames, K, dist, run, corr)
        panels.append((title, col, a, sols))
        zrows.append((title, col, sols))
    fig_topdown(panels, figdir / 'topdown.png')
    img = cv2.imread(str(root / frames[fi][1]['image']))
    cv2.imwrite(str(figdir / 'zoom.jpg'), zoom_strip(img, frames[fi][1], zrows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ---- tables ----
    def trow(r, v, hl=False):
        s, c = r[v], r[f'common_{v}']
        return (f"<tr class='{'hl' if hl else ''}'><td>{r['label']}</td><td>{v}</td><td>{r['latency']:.1f} ms</td>"
                f"<td>{100 * s['success']:.1f}%</td><td>{c['rel_med']:.2f}%</td><td>{c['outl']:.1f}%</td>"
                f"<td>{s['rel_med']:.2f}%</td><td>{s['rel_p90']:.1f}%</td><td>{s['outl']:.1f}%</td>"
                f"<td>{s['bias']:+.2f}%</td><td>{s['t_med']:.3f} m</td></tr>")
    table = ("<table><tr><th>model</th><th>keypoints</th><th>latency</th><th>success</th><th>common median</th>"
             "<th>common outliers</th><th>median</th><th>p90</th><th>outliers</th><th>range bias</th><th>median error</th></tr>"
             + ''.join(trow(r, v, hl=(r['label'] == 'D @960' and v == 'corrected')) for r in rows for v in ('raw', 'corrected'))
             + f"<tr><td>ground-truth keypoints</td><td>–</td><td>–</td><td>{100 * fl['success']:.1f}%</td>"
               f"<td>{fl_common['rel_med']:.2f}%</td><td>{fl_common['outl']:.1f}%</td><td>{fl['rel_med']:.2f}%</td>"
               f"<td>{fl['rel_p90']:.1f}%</td><td>{fl['outl']:.1f}%</td><td>–</td><td>{fl['t_med']:.3f} m</td></tr></table>")

    def rng_table():
        head = ''.join(f'<th>{lo}–{hi} m</th>' for lo, hi in RANGE_BINS)
        body = ''
        for name, imgsz, label in CONFIGS:
            run = runs[(name, imgsz)]
            for v in ('raw', 'corrected'):
                b = binned(run['res'][v], 'rng', RANGE_BINS)
                succ = [sum(1 for k, x in run['res'][v].items() if x is not None and lo <= x['rng'] < hi) /
                        max(1, sum(1 for c in cones.values() if lo <= c['ground_range_m'] < hi)) for lo, hi in RANGE_BINS]
                body += (f'<tr><td>{label} @{imgsz} {v}</td>' +
                         ''.join(f'<td>{m:.2f}% / {100 * s:.0f}%</td>' for (m, _), s in zip(b, succ)) + '</tr>')
        fb = binned(floor, 'rng', RANGE_BINS)
        body += '<tr><td>ground-truth keypoints</td>' + ''.join(f'<td>{m:.2f}%</td>' for m, _ in fb) + '</tr>'
        return f'<table><tr><th>configuration</th>{head}</tr>{body}</table>'

    tt = ''.join(f"<tr><td>{t['bin']} px</td><td>{t['n']}</td><td>{t['pred_aspect']:.2f}</td><td>{t['gt_aspect']:.2f}</td>"
                 f"<td>{100 * t['bottom_gap']:+.1f}%</td><td>{t['model_foot']:.3f}</td><td>{t['true_foot']:.3f}</td>"
                 f"<td>{t['model_wtop']:.3f}</td><td>{t['true_wtop']:.3f}</td></tr>" for t in tstats)
    rf = bias['foot']
    fmt3 = lambda c: '–' if c['median'] is None else f"{c['median']:.3f}"
    rft = ', '.join(f'{lo}–{hi} px {fmt3(a)} / {fmt3(b)}' for (lo, hi), a, b in
                    zip(((4, 8), (8, 12), (12, 16), (16, 24), (24, 40), (40, 80), (80, 400)), bias['foot']['real_label'], bias['foot']['real_model']))

    D = {r['label']: r for r in rows}
    d9, o9, d19 = D['D @960'], D['old nano @960'], D['D @1920']
    tiles = [(f"{d9['corrected']['rel_med']:.2f}%", f"D @960 + correction: median error (raw {d9['raw']['rel_med']:.2f}%)"),
             (f"{d9['corrected']['outl']:.1f}%", f"cones off by >10% (raw {d9['raw']['outl']:.1f}%, old nano {o9['raw']['outl']:.1f}%)"),
             (f"{d9['corrected']['bias']:+.2f}%", f"median range bias (raw {d9['raw']['bias']:+.2f}%)"),
             (f"{d19['corrected']['rel_med']:.2f}%", f"D @1920 + correction, {d19['latency']:.1f} ms, {100 * d19['corrected']['success']:.0f}% solved")]
    tiles_html = ''.join(f'<div class="tile"><div class="v">{v}</div><div class="l">{html.escape(l)}</div></div>' for v, l in tiles)
    stem = frames[fi][0]
    ex = frames[fi][1]['capture']['extras']

    page = f"""<!doctype html><html><head><meta charset="utf-8">{CSS}</head><body>
<section><h1>Cone pose + PnP, updated: model D with keypoint bias correction</h1>
<p class="sub">Synthetic test set (1000 Blender frames, 1920×1200, 80° HFOV). Blue and yellow cones only, 2–30 m ground range,
odd frames ({n_total} cones). The correction is fitted on the even frames, so nothing here is scored on data it was fitted to.
Generated {time.strftime('%Y-%m-%d')}.</p>
<div class="tiles">{tiles_html}</div>
<h3>What changed since the first report</h3>
<ul>
<li><b>Model D</b> (<code>runs/by-n-D-aug</code>): YOLO26 nano pose trained on the cleaned blue/yellow BRT set
(broken cones inpainted, 20 epochs) then 10 more epochs with augmentation. Best real-validation pose mAP50-95 so far (0.915).</li>
<li><b>Keypoint correction</b> (<code>synth/kp_correction.py</code>): subtract a per-keypoint offset that depends only on the
predicted box height, then rescale the PnP range by a box-height factor. Run-time inputs are the detection itself; cost ≈
{d9['corr_us']:.0f} µs per cone in Python.</li>
<li>Raw, model D is more biased on far cones than the old nano (range bias {d9['raw']['bias']:+.2f}% vs {o9['raw']['bias']:+.2f}%).
With the correction it is the best configuration tested: {d9['corrected']['rel_med']:.2f}% median at 960 vs {o9['corrected']['rel_med']:.2f}% for the corrected old nano, and {d19['corrected']['rel_med']:.2f}% at 1920.</li>
</ul>
<h3>Headline: raw → corrected, per configuration</h3>
<img src="file://{figdir}/headline.png">
<p class="note">Median and outliers over all solved cones per configuration. Range bias is signed: positive means cones are placed too far away.</p>
</section>

<section class="pb"><h2>All configurations</h2>
<p class="sub">"Common" columns use only the {len(common)} cones that every configuration, both keypoint variants and the ground-truth
floor solved, so they compare like with like. Success = matched (IoU ≥ 0.5) and PnP solved, over all {n_total} eval cones.
Latency: batch 1, end to end, RTX 5060.</p>
{table}
<h3>By ground range: median relative error / success</h3>
{rng_table()}
<p class="note">The old nano is fitted its own correction the same way, to show what the correction alone is worth. Its
detector was trained on all cone classes; it is scored here on blue/yellow cones only.</p>
</section>

<section class="pb"><h2>Error by distance and remaining bias</h2>
<p class="sub">Left: median relative error per range bin; the dashed gray line is PnP on the exact keypoints (the error the fixed
cone model alone contributes). Right: signed range bias by box height. Raw model D places small cones 5–8% too far; the correction
flattens this to about zero across sizes.</p>
<img class="mid" src="file://{figdir}/by.png">
<div class="two"><div><img src="file://{figdir}/cdf.png"></div>
<div><h3>Reading this</h3><ul>
<li>Near cones (2–10 m) were already good; the gain is at 10–30 m, where cones are under ~40 px tall.</li>
<li>Correction at 960 px gives most of the benefit for no inference cost. Running at 1920 adds about
{d19['latency'] - d9['latency']:.1f} ms per frame, finds more far cones and lowers error further.</li>
<li>The remaining error is scatter, not bias: about 0.5–0.7 px per keypoint at 960 and 0.35–0.6 px at 1920.</li>
</ul></div></div>
</section>

<section class="pb"><h2>Why the raw keypoints are biased</h2>
<p class="sub">Model D on the synthetic frames, matched blue/yellow cones, keypoints measured against the model's own predicted box.
Below about 25 px box height at network input, the model draws a narrow box that stops short of the base, then places the
keypoints at fixed proportions inside it, regardless of the actual cone shape. Right: the real human labels (all 18 sources)
show the same box convention, and it runs opposite to the geometry.</p>
<img class="mid" src="file://{figdir}/template.png">
<table><tr><th>box height at input</th><th>n</th><th>predicted box w/h</th><th>true box w/h</th><th>box bottom too high by</th>
<th>model foot / h</th><th>true foot / h</th><th>model top width / h</th><th>true top width / h</th></tr>{tt}</table>
<h3>Evidence chain</h3>
<ul>
<li><b>Template, not edges:</b> model foot height stays at about 0.17 of its box at every small size, while the true foot moves from
0.08 to 0.17. Top width stays at about 0.25 h. The foot ends up about 0.1 h too high and the widths 2–3% narrow, so the cone looks
smaller and PnP puts it too far away.</li>
<li><b>It comes from the labels:</b> on real val+test, model D matches the human labels at every size (foot / h, label / model:
{rft}). On real images there is nothing to see; the bias is only visible against exact geometry.</li>
<li><b>Labels are size-biased:</b> small cones are boxed tight (w/h ≈ 0.73 vs ≈ 0.85 for large) in every source, while exact geometry
gives the opposite (0.90 → 0.84). Small-cone keypoints also vary less between labels (foot spread 0.022 h vs 0.056 h for large).</li>
<li><b>Ruled out:</b> simulated camera defects (no change in bias from mild to strong), a general shrink tendency of the model
(absent on real data), and resolution alone (it helps only by moving cones above the ~25 px threshold).</li>
</ul>
</section>

<section class="pb"><h2>The correction</h2>
<p class="sub">Fitted offsets for model D @960, averaged over each keypoint pair: the median position error of each predicted keypoint
relative to the exact one, in units of predicted box height, per log-spaced box-height bin (even frames). At run time the offset is
interpolated in log(box height) and subtracted; a second table rescales the solved range by box height.</p>
<img src="file://{figdir}/offsets.png">
<h3>Limits</h3>
<ul>
<li>Fitted on synthetic cones. It assumes the synthetic cone geometry (rings at 0.045 / 0.14 / 0.22 m, radii 80 / 61 / 44.5 mm,
height 0.325 m) matches the real cones, and that the real-image bias equals the synthetic one.</li>
<li>Real-world range accuracy is not proven: the real labels share the convention, so real val cannot show it. A short drive or
static capture with tape-measured cone distances would confirm (or refit) the correction.</li>
<li>It is specific to a model and input size; refit after retraining (a few seconds on the cached predictions).</li>
<li>A model-side fix is possible: exact keypoints for small cones (synthetic training data, or real small-cone labels re-derived
from geometry), or a higher input size so fewer cones fall under the template threshold. A crop refiner trained on real labels would
learn the same template.</li>
</ul>
</section>

<section class="pb"><h2>Example frame {stem}</h2>
<p class="sub">{ex['car_context'].replace('_', ' ')}, defect severity {frames[fi][1]['defects']['severity']:.2f}. Top-down camera
frame (camera at the origin): dots are true cones, crosses are PnP estimates, lines join each estimate to its cone.</p>
<img class="tall" src="file://{figdir}/topdown.png"></section>
<section class="pb"><h2>Keypoint close-ups, frame {stem}</h2>
<p class="sub">Cones nearest to farthest. White rings are the exact keypoints; dots are the predictions. Rows: old nano raw, D raw,
D corrected. Labels give range and native box height.</p>
<img class="tall" src="file://{figdir}/zoom.jpg"></section>
</body></html>"""
    with tempfile.TemporaryDirectory() as td:
        hp = Path(td) / 'report_v2.html'
        hp.write_text(page)
        (figdir / 'report_v2.html').write_text(page)
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
    (root / 'eval' / 'report_v2.json').write_text(json.dumps(dict(n_total=n_total, common_n=len(common), rows=rows,
                                                                  floor=fl, template=tstats, frame=stem), indent=1))
    print('frame', stem, '->', args.out)


if __name__ == '__main__':
    main()
