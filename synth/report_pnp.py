"""Summarise eval_pnp.py output into eval/summary.json and a self-contained eval/report.html.

uv run --locked synth/report_pnp.py synth_data
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODELS = ['nano', 'small', 'medium']
BINS = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60)]
SEV = [(0, 0.33, 'mild'), (0.33, 0.66, 'moderate'), (0.66, 1.01, 'strong')]


def f(v):
    return float(v) if v not in ('', None) else np.nan


def stats(rows, n_total=None):
    ok = [r for r in rows if r.get('pnp_ok') == '1']
    n = len(rows) if n_total is None else n_total
    col = lambda k: np.array([f(r[k]) for r in ok]) if ok else np.array([np.nan])
    t, rel = col('t_err'), col('rel_err')
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else None
    return dict(
        n=n, matched=sum(r['matched'] == '1' for r in rows), solved=len(ok),
        recall=sum(r['matched'] == '1' for r in rows) / n if n else None,
        success=len(ok) / n if n else None,
        t_err_median_m=med(t), t_err_p90_m=float(np.nanpercentile(t, 90)) if ok else None,
        rel_err_median_pct=None if med(rel) is None else 100 * med(rel),
        rel_err_p90_pct=float(100 * np.nanpercentile(rel, 90)) if ok else None,
        outlier_rate=float(np.mean(rel > 0.10)) if ok else None,
        range_err_median_abs_m=med(np.abs(col('range_err'))), range_bias_median_m=med(col('range_err')),
        lat_err_median_m=med(col('lat_err')), ground_err_median_m=med(col('ground_err')),
        axis_err_median_deg=med(col('axis_err_deg')),
        kp_err_median_px=med(np.array([f(r.get('kp_err_px')) for r in rows if r['matched'] == '1'] or [np.nan])),
    )


KEEP = {'n', 'recall', 'success', 't_err_median_m', 't_err_p90_m', 'rel_err_median_pct', 'rel_err_p90_pct', 'outlier_rate',
        'range_err_median_abs_m', 'range_bias_median_m', 'lat_err_median_m', 'ground_err_median_m', 'axis_err_median_deg',
        'kp_err_median_px', 'model', 'imgsz', 'latency_median_ms', 'latency_p95_ms', 'pnp_median_ms', 'common_rel_err_median_pct',
        'common_rel_err_ci95', 'rel_err_cdf_pct', 'by_range', 'by_class', 'by_context', 'by_severity', 'by_occlusion',
        'configs', 'baselines', 'gt_kp_planar', 'gt_kp_exact', 'eval_definition'}


SUB = {'n', 'recall', 'success', 't_err_median_m', 'rel_err_median_pct'}


def slim(o, depth=0):
    """Only the fields report_template.html reads, rounded, to keep the page small."""
    if isinstance(o, dict):
        if depth == 4:  # per-bin / per-condition stats
            return {k: slim(v, depth + 1) for k, v in o.items() if k in SUB}
        return {k: slim(v, depth + 1) for k, v in o.items() if k in KEEP or depth in (2, 3) and isinstance(v, dict)}
    if isinstance(o, list):
        return [slim(v, depth + 1) for v in (o[::4] + o[-1:] if len(o) > 50 else o)]
    if isinstance(o, float):
        return float(f'{o:.4g}')
    return o


def main():
    root = Path(sys.argv[1])
    ev = root / 'eval'
    rows = list(csv.DictReader(open(ev / 'cones.csv')))
    configs = json.loads((ev / 'configs.json').read_text())
    evalset = [r for r in rows if r['detectable'] == '1' and 2 <= f(r['range_m']) <= 30]
    by = {}
    for r in evalset:
        by.setdefault((r['model'], int(r['imgsz'])), []).append(r)
    allrows = {}
    for r in rows:
        if r['detectable'] == '1':
            allrows.setdefault((r['model'], int(r['imgsz'])), []).append(r)

    # Common set: cones solved by every configuration, so error medians compare like with like.
    solved = [{(r['frame'], r['gt']) for r in by[(c['model'], c['imgsz'])] if r.get('pnp_ok') == '1'} for c in configs]
    common = set.intersection(*solved)
    rng = np.random.default_rng(0)

    def common_stats(rs):
        rel = np.array([f(r['rel_err']) for r in rs if (r['frame'], r['gt']) in common]) * 100
        t = np.array([f(r['t_err']) for r in rs if (r['frame'], r['gt']) in common])
        boots = np.median(rng.choice(rel, (300, len(rel))), axis=1)
        return dict(common_n=len(rel), common_rel_err_median_pct=float(np.median(rel)),
                    common_rel_err_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    common_t_err_median_m=float(np.median(t)), common_rel_err_p90_pct=float(np.percentile(rel, 90)),
                    common_outlier_rate=float(np.mean(rel > 10)))

    out = dict(configs=[], baselines={}, eval_definition=(
        'Detectable cones (>=12 visible px, >=10% visible) at 2-30 m ground range. Success = matched '
        '(IoU>=0.5) and PnP solved (>=4 keypoints with conf>=0.5). Errors over solved cones; '
        'relative error = translation error / true distance; outlier = relative error > 10%. '
        'Common set = cones solved by every model/size; its CI is a cone-level bootstrap of the median.'))
    for c in configs:
        key = (c['model'], c['imgsz'])
        rs = by[key]
        s = dict(c, **stats(rs), **common_stats(rs))
        s['by_range'] = {f'{lo}-{hi}': stats([r for r in allrows[key] if lo <= f(r['range_m']) < hi]) for lo, hi in BINS}
        s['by_class'] = {k: stats([r for r in rs if r['cls'] == k]) for k in ('BLUE', 'YELLOW', 'ORANGE', 'ORANGE_BIG')}
        s['by_context'] = {k: stats([r for r in rs if r['context'] == k]) for k in ('straight', 'approaching_turn', 'in_turn')}
        s['by_severity'] = {n: stats([r for r in rs if lo <= f(r['severity']) < hi]) for lo, hi, n in SEV}
        s['by_occlusion'] = {k: stats([r for r in rs if r['occluded'] == v]) for k, v in (('clear', '0'), ('occluded', '1'))}
        rel = np.sort([f(r['rel_err']) for r in rs if r.get('pnp_ok') == '1'])
        s['rel_err_cdf_pct'] = [[float(100 * np.percentile(rel, q)), q / 100] for q in range(0, 100)] if len(rel) else []
        out['configs'].append(s)
    for b in ('gt_kp_planar', 'gt_kp_exact'):
        key = (b, 0)
        out['baselines'][b] = dict(stats(by[key]), by_range={f'{lo}-{hi}': stats([r for r in allrows[key] if lo <= f(r['range_m']) < hi]) for lo, hi in BINS})
    (ev / 'summary.json').write_text(json.dumps(out, indent=1))
    html = (Path(__file__).parent / 'report_template.html').read_text().replace('/*DATA*/null', json.dumps(slim(out), separators=(',', ':')))
    (ev / 'report.html').write_text(html)
    for s in out['configs']:
        print(f"{s['model']:>6} {s['imgsz']:>4}  {s['latency_median_ms']:5.1f} ms  success {100*s['success']:5.1f}%  "
              f"t_err {s['t_err_median_m']:.3f} m  rel {s['rel_err_median_pct']:.2f}%  p90 {s['rel_err_p90_pct']:.1f}%  "
              f"outl {100*s['outlier_rate']:.1f}%  kp {s['kp_err_median_px']:.2f}px  common {s['common_rel_err_median_pct']:.2f}% "
              f"[{s['common_rel_err_ci95'][0]:.2f},{s['common_rel_err_ci95'][1]:.2f}] p90 {s['common_rel_err_p90_pct']:.1f}% n={s['common_n']}")
    for b, s in out['baselines'].items():
        print(b, f"t_err {s['t_err_median_m']:.4f} m rel {s['rel_err_median_pct']:.3f}%")


if __name__ == '__main__':
    main()
