"""Static PDF version of the PnP report, plus per-model detection / top-down examples on one frame.

uv run --locked synth/report_pdf.py synth_data [--frame 000123] [--out ~/Downloads/cone-pnp-report.pdf]

Needs Chromium for printing: uses the Playwright-cached Chromium (~/.cache/ms-playwright).
"""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pnp as ev  # noqa: E402

MODELS = ['nano', 'small', 'medium']
MODEL_COLORS = {'nano': '#2a78d6', 'small': '#eb6834', 'medium': '#1baf7a'}
CONE_COLORS = {'BLUE': '#2a78d6', 'YELLOW': '#d9a400', 'ORANGE': '#eb6834', 'ORANGE_BIG': '#c2410c'}
KP_COLORS = [(0, 0, 255), (0, 128, 255), (0, 220, 255), (0, 200, 0), (255, 200, 0), (255, 0, 0), (255, 0, 255), (255, 255, 255)]


def pick_frame(root):
    """A turn frame with several in-range cones, a big orange cone and moderate defects."""
    best = None
    for p in sorted((root / 'annotations').glob('*.json')):
        a = json.loads(p.read_text())
        cones = [c for c in a['cones'] if c['in_eval_range'] and ev.detectable(c)]
        classes = {c['class_name'] for c in cones}
        if a['capture']['extras']['car_context'] == 'straight' or not 0.35 <= a['defects']['severity'] <= 0.7:
            continue
        score = len(cones) + 6 * ('ORANGE_BIG' in classes) + 3 * ('ORANGE' in classes)
        if best is None or score > best[0]:
            best = (score, p.stem)
    return best[1]


def frame_results(root, stem, imgsz):
    stems = sorted(p.stem for p in (root / 'annotations').glob('*.json'))
    i = stems.index(stem)
    ann = json.loads((root / 'annotations' / f'{stem}.json').read_text())
    meta = json.loads((root / 'dataset.json').read_text())
    K, dist = np.array(meta['calibration']['K']), np.array(meta['calibration']['dist'])
    out = {}
    for m in MODELS:
        p = pickle.loads((root / 'eval' / 'predictions' / f'{m}_{imgsz}.pkl').read_bytes())[i]
        gb = np.array([c['bbox_xyxy_clipped'] for c in ann['cones']])
        iou = ev.iou_matrix(p['boxes'], gb)
        from scipy.optimize import linear_sum_assignment
        pi, gi = linear_sum_assignment(-iou)
        match = {d: g for d, g in zip(pi, gi) if iou[d, g] >= ev.IOU_MATCH}
        dets = []
        for d in range(len(p['boxes'])):
            cls = int(p['cls'][d])
            n = 8 if cls == 1 else 6
            kp = p['kpts'][d]
            use = np.where(kp[:n, 2] >= ev.KP_CONF)[0]
            sol = ev.solve_pnp(ev.planar_model(cls)[use], kp[use, :2].astype(np.float64), K, dist) if len(use) >= 4 else None
            dets.append(dict(box=p['boxes'][d], cls=cls, conf=float(p['conf'][d]), kp=kp, gt=match.get(d),
                             t=None if sol is None else sol[1]))
        out[m] = dets
    return ann, out


def draw_detections(img, ann, dets, model):
    v = img.copy()
    for c in ann['cones']:  # ground-truth keypoints: small hollow white rings
        for x, y, s in c['keypoints_2d']:
            if s:
                cv2.circle(v, (round(x), round(y)), 4, (255, 255, 255), 1, cv2.LINE_AA)
    col = tuple(int(MODEL_COLORS[model][i:i + 2], 16) for i in (5, 3, 1))
    for d in dets:
        x0, y0, x1, y1 = map(int, d['box'])
        cv2.rectangle(v, (x0, y0), (x1, y1), col if d['gt'] is not None else (0, 0, 255), 2)
        for k, (x, y, cf) in enumerate(d['kp']):
            if cf >= ev.KP_CONF and (k < 6 or d['cls'] == 1):
                cv2.circle(v, (round(x), round(y)), 3, KP_COLORS[k], -1, cv2.LINE_AA)
        if d['t'] is not None and d['gt'] is not None:
            g = ann['cones'][d['gt']]
            err = np.linalg.norm(d['t'] - np.array(g['base_center_cam']))
            cv2.putText(v, f"{err:.2f}m err", (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return v


def zoom_strip(img, ann, all_dets, n=4):
    """Close-ups of the nearest in-range cones: GT keypoints (white rings) vs each model's points."""
    cones = sorted([(i, c) for i, c in enumerate(ann['cones']) if c['in_eval_range'] and ev.detectable(c)
                    and not c['truncated']], key=lambda t: t[1]['ground_range_m'])
    picks = [cones[j] for j in np.linspace(0, len(cones) - 1, min(n, len(cones))).astype(int)]
    rows = []
    for m in MODELS:
        row = []
        for gi, c in picks:
            x0, y0, x1, y1 = c['bbox_xyxy_clipped']
            pad = 0.25 * (y1 - y0)
            X0, Y0, X1, Y1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(img.shape[1], x1 + pad)), int(min(img.shape[0], y1 + pad))
            s = 260 / (Y1 - Y0)
            crop = cv2.resize(img[Y0:Y1, X0:X1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            P = lambda x, y: (round((x - X0) * s), round((y - Y0) * s))
            for x, y, v in c['keypoints_2d']:
                if v:
                    cv2.circle(crop, P(x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
            d = next((d for d in all_dets[m] if d['gt'] == gi), None)
            if d is not None:
                for k, (x, y, cf) in enumerate(d['kp']):
                    if cf >= ev.KP_CONF and (k < 6 or d['cls'] == 1):
                        cv2.circle(crop, P(x, y), 4, KP_COLORS[k], -1, cv2.LINE_AA)
            label = f"{m} - {c['ground_range_m']:.1f} m" + ('' if d is not None else ' MISSED')
            cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            row.append(cv2.copyMakeBorder(crop, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        h = max(r.shape[0] for r in row)
        rows.append(np.hstack([cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in row]))
    w = max(r.shape[1] for r in rows)
    return np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])


def top_down(ann, all_dets, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    hfov = np.degrees(2 * np.arctan(ann['width'] / 2 / ann['camera']['K'][0][0]))
    for ax, m in zip(axes, MODELS):
        ax.set_facecolor('white')
        for a in (-hfov / 2, hfov / 2):
            ax.plot([0, 40 * np.tan(np.radians(a))], [0, 40], color='#c9c8c3', lw=1, zorder=0)
        errs = []
        for gi, c in enumerate(ann['cones']):
            x, _, z = c['base_center_cam']
            ax.scatter(x, z, s=36, color=CONE_COLORS[c['class_name']], edgecolor='white', linewidth=1, zorder=3)
        for d in all_dets[m]:
            if d['t'] is None:
                continue
            ex, _, ez = d['t']
            if d['gt'] is not None:
                gx, _, gz = ann['cones'][d['gt']]['base_center_cam']
                ax.plot([gx, ex], [gz, ez], color='#52514e', lw=1, zorder=2)
                g = ann['cones'][d['gt']]
                if g['in_eval_range']:
                    errs.append(np.linalg.norm(d['t'] - np.array(g['base_center_cam'])))
            ax.scatter(ex, ez, s=34, marker='x', color=MODEL_COLORS[m], linewidth=1.8, zorder=4)
        n_gt = sum(c['in_eval_range'] and ev.detectable(c) for c in ann['cones'])
        ax.set_title(f"{m}: {len(errs)}/{n_gt} solved (2–30 m), median error {np.median(errs):.2f} m"
                     if errs else m, fontsize=10.5, loc='left')
        ax.set_xlim(-17, 17); ax.set_ylim(0, 40); ax.set_aspect('equal')
        ax.grid(color='#ecebe7', lw=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#d6d5d0')
        ax.set_xlabel('lateral x (m)')
    axes[0].set_ylabel('forward z (m)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=v, label=f'true {k.lower().replace("_", " ")}') for k, v in CONE_COLORS.items()]
    handles += [plt.Line2D([], [], marker='x', ls='', color=MODEL_COLORS[m], label=f'{m} PnP estimate') for m in MODELS]
    handles += [plt.Line2D([], [], color='#52514e', label='estimate → true cone')]
    fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.2, wspace=0.08)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--frame')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--out', default=str(Path.home() / 'Downloads' / 'cone-pnp-report.pdf'))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    figdir = root / 'eval' / 'pdf_figures'
    figdir.mkdir(exist_ok=True)
    stem = args.frame or pick_frame(root)
    ann, dets = frame_results(root, stem, args.imgsz)
    img = cv2.imread(str(root / ann['image']))
    for m in MODELS:
        cv2.imwrite(str(figdir / f'det_{m}.jpg'), draw_detections(img, ann, dets[m], m), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(figdir / 'zoom.jpg'), zoom_strip(img, ann, dets), [cv2.IMWRITE_JPEG_QUALITY, 92])
    top_down(ann, dets, figdir / 'topdown.png')

    ex = ann['capture']['extras']
    info = (f"Frame {stem}: {ex['car_context'].replace('_', ' ')}"
            + (f" ({ex['turn_direction']}, radius {ex['turn_radius_m']:.0f} m)" if ex.get('turn_radius_m') else '')
            + f", defect severity {ann['defects']['severity']:.2f}, {ann['capture']['hdri'].split('__')[1].removesuffix('.hdr').replace('_', ' ')}."
            f" Input size {args.imgsz} px.")
    examples = f"""
<section class="pb"><h1>Examples on one synthetic frame</h1>
<p class="sub">{info} Boxes in the model's color are matched to a true cone; red boxes are unmatched detections.
Filled dots are predicted keypoints (confidence ≥ 0.5), colored by index; hollow white rings are the true keypoints.
Labels give each solved cone's translation error.</p>
""" + '</section>'.join(f"""{'' if i == 0 else '<section class="pb">'}<h2>{m} @{args.imgsz}</h2><img class="full det" src="file://{figdir}/det_{m}.jpg">""" for i, m in enumerate(MODELS)) + '</section>' + f"""
<section class="pb"><h2>Keypoint close-ups</h2><p class="sub">Four in-range cones, nearest to farthest. Rows: nano, small, medium.
White rings: true keypoints. Dots: predicted.</p><img class="full" src="file://{figdir}/zoom.jpg"></section>
<section class="pb"><h2>Top-down PnP estimates</h2><p class="sub">Camera frame seen from above (camera at the origin, gray wedge = field of view).
Dots are true cone positions (bottom-face centre), crosses are PnP estimates, lines join each estimate to its cone.
Errors are mostly along the viewing ray (range), as in the aggregate results.</p><img class="full" src="file://{figdir}/topdown.png"></section>
"""
    tpl = (root / 'eval' / 'report.html').read_text()
    print_css = """<style>
:root{--background:#ffffff;--foreground:#0b0b0b;--muted-foreground:#52514e;--border:#e4e3df;--popover:#fff;--popover-foreground:#111}
html,body{background:#fff}.filters,.tip{display:none!important}
@page{size:A4 landscape;margin:10mm}
.viz-root{max-width:none}.grid2>div,.tile,table,img{break-inside:avoid}
.pb{break-before:page}h1{font:600 18px var(--font-sans,system-ui);margin:0 0 4px}
img.full{max-width:100%;max-height:165mm;width:auto;display:block;margin:6px auto 0}img.det{max-height:140mm}
.cover h1{font-size:22px;margin:0 0 4px}.cover p{color:#52514e;margin:0 0 10px;font-size:13px}
</style>"""
    cover = """<div class="cover"><h1>YOLO26 cone pose models + PnP on a synthetic test set</h1>
<p>1000 Blender frames (1920×1200, 80° HFOV, FS track layouts, randomized camera defects). Nano, small and medium BRT
pose checkpoints at four input sizes; planar IPPE PnP on predicted keypoints versus the true cone poses. Generated 2026-10-13.</p></div>"""
    html = (tpl.replace('</head>', print_css + '</head>')
               .replace('<div class="viz-root" id="root">', '<div class="viz-root" id="root">' + cover)
               .replace('<h2>All configurations</h2>', '<h2 class="pb">All configurations</h2>')
               .replace('<h2>Error by distance</h2>', '<h2 class="pb">Error by distance</h2>')
               .replace('<h2>By condition</h2>', '<h2 class="pb">By condition</h2>')
               .replace('<p class="note" id="def"></p>', '<p class="note" id="def"></p>' + examples))
    with tempfile.TemporaryDirectory() as td:
        hp = Path(td) / 'report_print.html'
        hp.write_text(html)
        exe = sorted(Path.home().glob('.cache/ms-playwright/chromium-*/chrome-linux64/chrome'))[-1]
        script = f"""
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path={str(exe)!r})
    pg = b.new_page(viewport={{'width': 1046, 'height': 900}}, color_scheme='light')
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto({hp.as_uri()!r}); pg.wait_for_timeout(800)
    pg.pdf(path={args.out!r}, format='A4', landscape=True, print_background=True,
           margin={{'top': '10mm', 'bottom': '10mm', 'left': '10mm', 'right': '10mm'}})
    b.close()
    print('page errors:', errs)
"""
        subprocess.run(['uv', 'run', '--no-project', '--with', 'playwright', 'python', '-c', script], check=True)
    print('frame', stem, '->', args.out)


if __name__ == '__main__':
    main()
