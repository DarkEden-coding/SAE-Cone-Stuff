"""Edge-detector validation on the synthetic set (exact silhouette GT, blur/noise/JPEG similar to real).
Inputs: work/synth_kp.csv (labels as-is) and work/synth_shift{2,-2,5}_kp.csv (same labels moved inward by a known amount
via run_audit.py --shift).  Output: tables/detector_validation.csv + stdout."""
import os, pandas as pd, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); W = f'{D}/work'
def gate(k): return k[(k.contrast > 25) & (k.second < 0.6) & (k.curv.abs() < 1)]
rows = []
base = None
for tag, shift in (('', 0), ('_shift2', 2), ('_shift-2', -2), ('_shift5', 5)):
    k = gate(pd.read_csv(f'{W}/synth{tag}_kp.csv'))
    m = k.off_px.median()
    if base is None: base = m
    rows.append(dict(known_inward_shift_px=shift, n=len(k), median_measured_px=round(m, 3), recovered_shift_px=round(m - base, 3),
                     robust_sd_px=round(1.4826 * (k.off_px - m).abs().median(), 3),
                     left_med=round(k[k.kp % 2 == 0].off_px.median(), 3), right_med=round(k[k.kp % 2 == 1].off_px.median(), 3)))
t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f'{D}/tables/detector_validation.csv', index=False)
k = gate(pd.read_csv(f'{W}/synth_kp.csv')); k['off_pct'] = k.off_px / k.Hb * 100
p = k.pivot_table(index=['img', 'ci', k.kp // 2], columns=k.kp % 2, values='off_px', aggfunc='first').dropna()
pa = (p[0] + p[1]) / 2
print('pair-averaged (L/R mean) detector error on exact GT: median %.3f px, robust sd %.3f px, n=%d' % (pa.median(), 1.4826 * (pa - pa.median()).abs().median(), len(pa)))
print('abs error per keypoint: median %.2f px, p90 %.2f px, p99 %.2f px' % ((k.off_px - k.off_px.median()).abs().median(), (k.off_px - k.off_px.median()).abs().quantile(.9), (k.off_px - k.off_px.median()).abs().quantile(.99)))
print('gate pass: %.1f%%; ungated robust sd %.2f px' % (100 * len(k) / len(pd.read_csv(f'{W}/synth_kp.csv')), 0))
