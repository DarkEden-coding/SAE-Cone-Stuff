"""Aggregate the per-keypoint measurements into the tables/figures used in REPORT.md.
Inputs : work/real_kp.csv real_cone.csv synth_kp.csv synth_cone.csv (from run_audit.py)
Outputs: keypoint_measurements.csv, cone_summary.csv, tables/*.csv, fig_*.png, analysis_stdout.txt"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom_flags import geom_metrics

D = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(D, 'work'); T = os.path.join(D, 'tables'); os.makedirs(T, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 200)
CLS = {0: 'BLUE', 1: 'ORANGE_BIG', 2: 'ORANGE', 4: 'YELLOW'}


def mad(x):
    x = x.dropna()
    return 1.4826 * (x - x.median()).abs().median() if len(x) else np.nan


def summ(x):
    x = x.dropna()
    return pd.Series(dict(n=len(x), median=x.median(), mean=x.mean(), rsd=mad(x), p5=x.quantile(.05), p25=x.quantile(.25),
                          p75=x.quantile(.75), p95=x.quantile(.95)))


def load(name):
    k = pd.read_csv(f'{W}/{name}_kp.csv'); c = pd.read_csv(f'{W}/{name}_cone.csv')
    k['off_pct'] = k.off_px / k.Hb * 100
    k['ew_rel'] = k.ew / k.Hb * 100
    k['gate'] = (k.contrast > 25) & (k.second < 0.6) & (k.curv.abs() < 1)
    k['vgate'] = (k.vcon > 15) & (k.vsec < 0.6)
    k['v_off_pct'] = k.v_off_px / k.Hb * 100
    k['lr'] = np.where(k.kp % 2 == 0, 'L', 'R'); k['pair'] = k.kp // 2
    k['hbin'] = pd.cut(k.Hb, [60, 80, 100, 130, 180, 2000])
    return k, c


def save(df, name, **kw):
    df.to_csv(f'{T}/{name}.csv', **kw)


kr, cr = load('real'); ks, cs = load('synth')
out = open(f'{D}/analysis_stdout.txt', 'w')


def P(*a, **kw):
    print(*a, **kw); print(*a, **kw, file=out)


P('real cones', len(cr), 'usable (H>=60,full,not trunc,no overlap)', int(cr.usable.sum()), 'kp rows', len(kr), 'gated', int(kr.gate.sum()))
P('synth cones', len(cs), 'usable', int(cs.usable.sum()), 'kp rows', len(ks), 'gated', int(ks.gate.sum()))

# ---------------------------------------------------------------- 1. horizontal offsets
gr = kr[kr.gate]; gs = ks[ks.gate]
P('\n== Overall inward offset (gated). positive = label inside silhouette')
for nm, g in (('REAL', gr), ('SYNTH-GT (detector baseline)', gs)):
    P(nm); P(pd.DataFrame({'px': summ(g.off_px), '%H': summ(g.off_pct)}).round(3).T)
P('gate pass rate real', kr.gate.mean().round(3), 'synth', ks.gate.mean().round(3))

# pair level (convention free): mean of L and R
def pair_table(g):
    p = g.pivot_table(index=['img', 'ci', 'pair'], columns='lr', values=['off_px', 'off_pct'], aggfunc='first').dropna()
    p.columns = [f'{a}_{b}' for a, b in p.columns]
    p['off_px'] = (p.off_px_L + p.off_px_R) / 2; p['off_pct'] = (p.off_pct_L + p.off_pct_R) / 2
    return p.reset_index()
pr = pair_table(gr); ps = pair_table(gs)
P('\n== Pair-averaged inward offset (mean of L and R of a pair; immune to a global x-shift convention)')
for nm, g in (('REAL', pr), ('SYNTH', ps)):
    P(nm); P(pd.DataFrame({'px': summ(g.off_px), '%H': summ(g.off_pct)}).round(3).T)
P('REAL fraction of pairs with mean inward offset > 1% H:', (pr.off_pct > 1).mean().round(4), ' > 2%:', (pr.off_pct > 2).mean().round(4),
  ' < -1%:', (pr.off_pct < -1).mean().round(4), '| SYNTH >1%:', (ps.off_pct > 1).mean().round(4), '<-1%:', (ps.off_pct < -1).mean().round(4))

P('\n== by keypoint index and class (real, %H)  [synth in second table]')
for nm, g in (('real', gr), ('synth', gs)):
    t = g.groupby(['cls', 'kp']).apply(lambda d: pd.Series(dict(n=len(d), med_px=d.off_px.median(), med_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct), rsd_px=mad(d.off_px)))).round(2)
    P(nm); P(t); save(t, f'offset_by_class_kp_{nm}')
P('\n== by keypoint pair (pair-averaged), class')
for nm, g in (('real', pr.merge(cr[['img', 'ci', 'cls', 'prefix', 'Hb']], on=['img', 'ci'])), ('synth', ps.merge(cs[['img', 'ci', 'cls', 'prefix', 'Hb']], on=['img', 'ci']))):
    t = g.groupby(['cls', 'pair']).apply(lambda d: pd.Series(dict(n=len(d), med_px=d.off_px.median(), med_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct)))).round(2)
    P(nm); P(t); save(t, f'offset_by_class_pair_{nm}')
P('\n== by source prefix (real; all keypoints; pair-avg median %H; L/R asym)')
t = gr.groupby('prefix').apply(lambda d: pd.Series(dict(
    n=len(d), cones=d.groupby(['img', 'ci']).ngroups, medH_px=d.Hb.median(), med_px=d.off_px.median(), med_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct),
    pct_in_gt1=(d.off_pct > 1).mean() * 100, pct_out_lt_m1=(d.off_pct < -1).mean() * 100,
    L_px=d[d.lr == 'L'].off_px.median(), R_px=d[d.lr == 'R'].off_px.median(), ew_px=d.ew.median(), ew_pctH=d.ew_rel.median()))).round(2)
P(t); save(t, 'offset_by_prefix')
P('\n== by cone size (box height px)')
t = pd.concat([g.groupby('hbin').apply(lambda d: pd.Series(dict(n=len(d), med_px=d.off_px.median(), med_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct), rsd_px=mad(d.off_px)))).round(2).add_prefix(nm + '_') for nm, g in (('real', gr), ('synth', gs))], axis=1)
P(t); save(t, 'offset_by_size')
P('\n== by split'); P(gr.groupby('split').off_pct.agg(['count', 'median']).round(3))

# restricted to well-conditioned edges (high contrast, sharp-ish) as a robustness check
for nm, sel in (('contrast>40', gr.contrast > 40), ('second<0.3', gr.second < 0.3), ('ew<5px', gr.ew < 5), ('no gate, all', gr.contrast > -1)):
    P('robustness', nm, 'n', int(sel.sum()), 'median px %.3f  %%H %.3f  rsd %%H %.2f' % (gr[sel].off_px.median(), gr[sel].off_pct.median(), mad(gr[sel].off_pct)))
P('ungated (all measured kps, incl. low contrast): median %%H %.3f px %.3f n=%d' % (kr.off_pct.median(), kr.off_px.median(), kr.off_pct.notna().sum()))

# ---------------------------------------------------------------- 2. vertical
P('\n== Vertical: label_y - detected colour-transition row (just inside the silhouette), %H; positive = label BELOW the transition')
vr = kr[kr.vgate & kr.cls.isin([0, 2, 4, 1])]; vs = ks[ks.vgate]
rows = []
for cl in [0, 4, 2, 1]:
    for kp in range(8):
        a = vr[(vr.cls == cl) & (vr.kp == kp)]; b = vs[(vs.cls == cl) & (vs.kp == kp)]
        if len(a) > 20 and len(b) > 20:
            rows.append(dict(cls=CLS[cl], kp=kp, n_real=len(a), real_med=a.v_off_pct.median(), real_rsd=mad(a.v_off_pct), n_syn=len(b), syn_med=b.v_off_pct.median(), syn_rsd=mad(b.v_off_pct),
                             real_minus_syn=a.v_off_pct.median() - b.v_off_pct.median()))
vt = pd.DataFrame(rows).round(2); P(vt); save(vt, 'vertical_by_class_kp', index=False)
# combine small cones kp0-3
v03r = vr[vr.kp < 4]; v03s = vs[vs.kp < 4]
P('stripe-edge kps 0-3, all classes: real median %.2f (rsd %.2f, n=%d) vs synth GT %.2f (rsd %.2f, n=%d) -> real labels sit %.2f %%H lower than GT-relative' % (
    v03r.v_off_pct.median(), mad(v03r.v_off_pct), len(v03r), v03s.v_off_pct.median(), mad(v03s.v_off_pct), len(v03s), v03r.v_off_pct.median() - v03s.v_off_pct.median()))
for p_ in range(2):
    a = v03r[v03r.pair == p_]; b = v03s[v03s.pair == p_]
    P('  pair %d (%s): real-synth = %.2f %%H' % (p_, 'stripe top' if p_ == 0 else 'stripe bottom', a.v_off_pct.median() - b.v_off_pct.median()))
P('\nvertical by prefix (stripe-edge kps 0-3, small cones+big, real minus overall synth median):')
t = v03r.groupby('prefix').apply(lambda d: pd.Series(dict(n=len(d), med_pct=d.v_off_pct.median(), rsd_pct=mad(d.v_off_pct)))).round(2)
t['minus_synth'] = (t.med_pct - v03s.v_off_pct.median()).round(2); P(t); save(t, 'vertical_by_prefix')
# L vs R vertical difference of the label itself
P('\n== Foot points (4/5) and layout consistency: geometry-only, all cones with H>=30 px')
gm_r = geom_metrics(cr[(cr.Hb >= 30) & (~cr.trunc)]); gm_s = geom_metrics(cs[(cs.Hb >= 30) & (~cs.trunc) & (cs.overlap < 0.02)])
cc = cr.loc[gm_r.index].join(gm_r)
cc_s = cs.loc[gm_s.index].join(gm_s)
rows = []
for nm, d in (('real', cc), ('synth', cc_s)):
    for cl in [0, 4, 2, 1]:
        e = d[d.cls == cl]
        for p_ in range(4 if cl == 1 else 3):
            rows.append(dict(src=nm, cls=CLS[cl], pair=p_, n=int(e[f'ypos{p_}'].notna().sum()), ypos_med=e[f'ypos{p_}'].median(), ypos_rsd=mad(e[f'ypos{p_}']), wH_med=e[f'wH{p_}'].median(), wH_rsd=mad(e[f'wH{p_}']),
                             asym_med=e[f'asym{p_}'].median(), asym_p99=e[f'asym{p_}'].quantile(.99)))
lt = pd.DataFrame(rows).round(3); P(lt); save(lt, 'layout_by_class_pair', index=False)
P('\nlayout by prefix (small cones; ypos = pair height above box bottom, in fraction of box height; wH = pair width / H)')
d = cc[cc.cls.isin([0, 2, 4])]
t = d.groupby('prefix').apply(lambda e: pd.Series(dict(n=len(e), ypos0=e.ypos0.median(), ypos1=e.ypos1.median(), ypos2=e.ypos2.median(), wH0=e.wH0.median(), wH1=e.wH1.median(), wH2=e.wH2.median(),
                                                      ypos2_rsd=mad(e.ypos2), ypos0_rsd=mad(e.ypos0)))).round(3)
P(t); save(t, 'layout_by_prefix')
P('synth GT small cones: ', cc_s[cc_s.cls.isin([0, 2, 4])][['ypos0', 'ypos1', 'ypos2', 'wH0', 'wH1', 'wH2']].median().round(3).to_dict())

# ---------------------------------------------------------------- 3. blur
P('\n== Blur / sharpness')
imgs = cr.drop_duplicates('img')
P('per-image variance of Laplacian at 1280 px width (real): '); P(imgs.vol1280.describe().round(1))
t = imgs.groupby('prefix').vol1280.describe().round(1); P(t); save(t, 'blur_vol_by_prefix')
P('synth per image vol1280:'); P(cs.drop_duplicates('img').vol1280.describe().round(1))
P('10-90 edge rise width (px) per keypoint, real vs synth'); P(pd.DataFrame({'real': summ(gr.ew), 'synth': summ(gs.ew)}).round(2))
P('rise width as %H'); P(pd.DataFrame({'real': summ(gr.ew_rel), 'synth': summ(gs.ew_rel)}).round(2))
t = gr.groupby('prefix').apply(lambda d: pd.Series(dict(n=len(d), ew_px=d.ew.median(), ew_p90=d.ew.quantile(.9), ew_pctH=d.ew_rel.median(), ew_pctH_p90=d.ew_rel.quantile(.9), vol1280=d.vol1280.median()))).round(2)
P(t); save(t, 'blur_edgewidth_by_prefix')
# sharpness floor: edges < X
P('fraction of real kps with rise width >8 px: %.3f ; >12 px: %.3f ; synth >8: %.3f' % ((gr.ew > 8).mean(), (gr.ew > 12).mean(), (gs.ew > 8).mean()))

# correlation of blur with offset and scatter
def blur_bins(g, nm, col='ew_rel', q=6):
    g = g[g[col].notna()].copy()
    g['b'] = pd.qcut(g[col], q)
    mu = g.groupby('prefix').off_pct.transform('median')           # remove per-source median
    g['res'] = g.off_pct - mu
    t = g.groupby('b').apply(lambda d: pd.Series(dict(n=len(d), ew_med=d[col].median(), off_px=d.off_px.median(), off_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct), rsd_px=mad(d.off_px),
                                                    res_rsd_pct=mad(d.res), frac_in_gt1=(d.off_pct > 1).mean() * 100, frac_in_gt2=(d.off_pct > 2).mean() * 100,
                                                    frac_out_lt_m1=(d.off_pct < -1).mean() * 100, absdev_med=(d.off_pct - d.off_pct.median()).abs().median()))).round(3)
    P(f'-- {nm}: bins of {col}'); P(t); return t
tb_r = blur_bins(gr, 'real'); tb_s = blur_bins(gs, 'synth')
save(tb_r, 'blur_bins_real'); save(tb_s, 'blur_bins_synth')
tb_r2 = blur_bins(gr, 'real', 'ew', 6); save(tb_r2, 'blur_bins_real_px')
from scipy.stats import spearmanr
for nm, g in (('real', gr), ('synth', gs)):
    h = g[g.ew_rel.notna()]
    r1 = spearmanr(h.ew_rel, h.off_pct)
    r2 = spearmanr(h.ew_rel, (h.off_pct - h.groupby('prefix').off_pct.transform('median')).abs())
    r3 = spearmanr(h.ew_rel, h.off_px)
    P(f'{nm}: spearman(blur %H, offset %H)={r1[0]:+.3f}  spearman(blur %H, |offset - source median|)={r2[0]:+.3f}  spearman(blur %H, off px)={r3[0]:+.3f}  n={len(h)}')
    # robust linear slope (theil-sen on subsample)
    from scipy.stats import theilslopes
    s = h.sample(min(6000, len(h)), random_state=0)
    sl = theilslopes(s.off_px, s.ew)
    P(f'    Theil-Sen: offset_px = {sl[1]:+.2f} + {sl[0]:+.3f} * rise_width_px')
# core subset: stripe-edge keypoints 0-3 of small cones (cleanest geometry, no big-orange/foot confounds)
core_r = gr[gr.cls.isin([0, 2, 4]) & (gr.kp < 4)]; core_s = gs[gs.cls.isin([0, 2, 4]) & (gs.kp < 4)]
P('\n== CORE subset (small cones, stripe-edge kps 0-3): real n=%d  synth n=%d' % (len(core_r), len(core_s)))
P(pd.DataFrame({'real_px': summ(core_r.off_px), 'real_pctH': summ(core_r.off_pct), 'synth_px': summ(core_s.off_px), 'synth_pctH': summ(core_s.off_pct)}).round(3).T)
tbc_r = blur_bins(core_r, 'core real'); tbc_s = blur_bins(core_s, 'core synth'); save(tbc_r, 'blur_bins_core_real'); save(tbc_s, 'blur_bins_core_synth')
tbc_r2 = blur_bins(core_r, 'core real', 'ew', 6); save(tbc_r2, 'blur_bins_core_real_px')
for nm, g in (('core real', core_r), ('core synth', core_s)):
    h = g[g.ew.notna()]
    sl = theilslopes(h.off_px.sample(min(6000, len(h)), random_state=0), h.ew.loc[h.off_px.sample(min(6000, len(h)), random_state=0).index])
    r = spearmanr(h.ew_rel, h.off_pct)
    P(f'{nm}: Theil-Sen offset_px = {sl[1]:+.2f} {sl[0]:+.3f}*rise_px ; spearman(rise %H, offset %H) = {r[0]:+.3f} (n={len(h)})')
# blur threshold test: how often are labels inward (>1 %H, >2 %H) when edge is blurred vs sharp
for nm, g in (('core real', core_r), ('core synth', core_s)):
    h = g[g.ew_rel.notna()]
    lo = h[h.ew_rel < h.ew_rel.quantile(.25)]; hi = h[h.ew_rel > h.ew_rel.quantile(.75)]
    P(f'{nm}: sharpest quartile med {lo.off_pct.median():+.2f} rsd {mad(lo.off_pct):.2f} inward>1%H {100*(lo.off_pct>1).mean():.1f}% | blurriest quartile med {hi.off_pct.median():+.2f} rsd {mad(hi.off_pct):.2f} inward>1%H {100*(hi.off_pct>1).mean():.1f}%')
# cone vol as alternative blur measure
P('\nalt blur measure: per-cone Laplacian variance (vol_cone) quartile bins, real')
h = gr[gr.vol_cone.notna()].copy(); h['b'] = pd.qcut(h.vol_cone, 5)
P(h.groupby('b').apply(lambda d: pd.Series(dict(n=len(d), off_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct), ew=d.ew.median()))).round(3))
h = gr.copy(); h['b'] = pd.qcut(h.vol1280, 5)
P('image-level Laplacian variance (1280w) quintiles:'); P(h.groupby('b').apply(lambda d: pd.Series(dict(n=len(d), off_pct=d.off_pct.median(), rsd_pct=mad(d.off_pct), ew=d.ew.median(), H=d.Hb.median()))).round(3))
# within-prefix check for the biggest sources
P('\nwithin-source tercile blur effect on offset (%H): median offset in sharp / mid / blurred tercile of rise width %H')
rows = []
for pf, d in gr.groupby('prefix'):
    d = d[d.ew_rel.notna()]
    if len(d) < 300: continue
    d = d.assign(b=pd.qcut(d.ew_rel, 3, labels=['sharp', 'mid', 'blur']))
    m = d.groupby('b').off_pct.median(); s_ = d.groupby('b').off_pct.apply(mad)
    rows.append(dict(prefix=pf, n=len(d), sharp=m['sharp'], mid=m['mid'], blur=m['blur'], sharp_rsd=s_['sharp'], blur_rsd=s_['blur']))
t = pd.DataFrame(rows).round(2); P(t); save(t, 'blur_within_prefix', index=False)

# ---------------------------------------------------------------- 4. gross errors
P('\n== Gross-error checks (geometry-only, thresholds set from synth GT extremes)')
THR = dict(asym=0.15, order=0.08, wgrow=-0.03)


def flags(d):
    f = pd.DataFrame(index=d.index)
    npair = np.where(d.cls == 1, 4, 3)
    sw = np.zeros(len(d), bool); asy = np.zeros(len(d), bool); od = np.zeros(len(d), bool); wg = np.zeros(len(d), bool); ob = np.zeros(len(d), bool)
    for p_ in range(4):
        m = p_ < npair
        sw |= m & d[f'swap{p_}'].fillna(False).values
        asy |= m & (d[f'asym{p_}'].fillna(0) > THR['asym']).values
        ob |= m & d[f'outbox{p_}'].fillna(False).values
        if p_ < 3:
            m2 = (p_ + 1) < npair
            od |= m2 & (d[f'order{p_}'].fillna(1) < THR['order']).values
            wg |= m2 & (d[f'wgrow{p_}'].fillna(1) < THR['wgrow']).values
    f['swap'] = sw; f['asym'] = asy; f['order'] = od; f['narrowing'] = wg; f['outbox'] = ob; f['miscount'] = d.miscount.values
    f['any'] = f[['swap', 'asym', 'order', 'narrowing', 'outbox', 'miscount']].any(axis=1)
    f['hard'] = f[['swap', 'order', 'outbox', 'miscount']].any(axis=1)
    return f
fr = flags(cc); fs = flags(cc_s)
cc = cc.join(fr.drop(columns='miscount'))
rows = []
for nm, d, f in (('real H>=30', cc, fr), ('real H>=60', cc[cc.Hb >= 60], fr[cc.Hb >= 60]), ('real H>=100', cc[cc.Hb >= 100], fr[cc.Hb >= 100]),
                 ('synth H>=30', cc_s, fs), ('synth H>=60', cc_s[cc_s.Hb >= 60], fs[cc_s.Hb >= 60])):
    rows.append(dict(subset=nm, n=len(d), **{c_: round(100 * f[c_].mean(), 3) for c_ in f.columns}))
gt = pd.DataFrame(rows); P(gt.to_string(index=False)); save(gt, 'gross_geometry_rates', index=False)
P('\nby prefix (real, H>=30), % of cones flagged:')
t = cc.groupby('prefix').apply(lambda d: pd.Series(dict(n=len(d), **{c_: 100 * d[c_].mean() for c_ in ['swap', 'order', 'asym', 'narrowing', 'outbox', 'any', 'hard']}))).round(2); P(t); save(t, 'gross_by_prefix')
t = cc.groupby('cls').apply(lambda d: pd.Series(dict(n=len(d), **{c_: 100 * d[c_].mean() for c_ in ['swap', 'order', 'asym', 'narrowing', 'outbox', 'any', 'hard']}))).round(2); P(t)
t = cc.groupby(pd.cut(cc.Hb, [30, 45, 60, 100, 2000])).apply(lambda d: pd.Series(dict(n=len(d), **{c_: 100 * d[c_].mean() for c_ in ['swap', 'order', 'asym', 'narrowing', 'outbox', 'any', 'hard']}))).round(2); P(t)
# edge-based gross: stable strong edge but label far from it
big = gr[(gr.contrast > 30) & (gr.second < 0.4)]
for th in (3, 5, 8):
    cone_bad = big.assign(bad=big.off_pct.abs() > th).groupby(['img', 'ci']).bad.any()
    P(f'edge-based: cones (H>=60, strong clean edges) with any |offset| > {th}% H: {cone_bad.mean()*100:.2f}% of {len(cone_bad)}')
_bs = gs[(gs.contrast > 30) & (gs.second < 0.4)]
P('kp-level share with |offset| > 3/5/8 %H: real ' + ' / '.join('%.2f%%' % (100 * (big.off_pct.abs() > t_).mean()) for t_ in (3, 5, 8)) +
  '   synth GT ' + ' / '.join('%.2f%%' % (100 * (_bs.off_pct.abs() > t_).mean()) for t_ in (3, 5, 8)))
cr_ = core_r[(core_r.contrast > 30) & (core_r.second < 0.4)]; cs__ = core_s[(core_s.contrast > 30) & (core_s.second < 0.4)]
P('core kps (small cones, stripe-edge pts, strong clean edges) share |offset| > 2/3/5/8 %H: real ' + ' / '.join('%.2f%%' % (100 * (cr_.off_pct.abs() > t_).mean()) for t_ in (2, 3, 5, 8)) +
  ' (n=%d)   synth GT ' % len(cr_) + ' / '.join('%.2f%%' % (100 * (cs__.off_pct.abs() > t_).mean()) for t_ in (2, 3, 5, 8)) + ' (n=%d)' % len(cs__))
cb = cr_.assign(bad=cr_.off_pct.abs() > 5).groupby(['img', 'ci']).bad.any()
P('core cones with any stripe-edge kp |offset| > 5 %H: {:.2f}% of {} cones (strong-edge subset)'.format(100 * cb.mean(), len(cb)))
# scatter of kp-level offsets per source: label consistency
P('\nLabel scatter: synth baseline robust sd of offset = %.2f pctH' % mad(gs.off_pct))

# ---------------------------------------------------------------- outputs: per-kp CSV, cone summary
keep = ['img', 'split', 'prefix', 'ci', 'cls', 'kp', 'Hb', 'Wb', 'pair_w', 'lab_x', 'lab_y', 'edge_x', 'off_px', 'off_pct', 'contrast', 'second', 'curv', 'slope', 'ew', 'ew_rel', 'x10', 'x90',
        'vy', 'v_off_px', 'v_off_pct', 'vcon', 'vsec', 'vol_cone', 'vol1280', 'gate', 'vgate']
kr[keep].round(4).to_csv(f'{D}/keypoint_measurements.csv', index=False)
cs_ = gr.groupby(['img', 'ci']).agg(prefix=('prefix', 'first'), cls=('cls', 'first'), split=('split', 'first'), Hb=('Hb', 'first'), n_gated=('kp', 'count'), off_px=('off_px', 'median'), off_pct=('off_pct', 'median'),
                                    ew=('ew', 'median'), ew_rel=('ew_rel', 'median'), vol_cone=('vol_cone', 'first'), vol1280=('vol1280', 'first')).reset_index()
cs_ = cs_.merge(cc[['img', 'ci', 'swap', 'asym', 'order', 'narrowing', 'outbox', 'miscount', 'any', 'hard']], on=['img', 'ci'], how='outer')
cs_.to_csv(f'{D}/cone_summary.csv', index=False)
cc.to_csv(f'{W}/real_cone_flags.csv', index=False)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
bins = np.linspace(-6, 6, 97)
ax[0].hist(gr.off_pct.clip(-6, 6), bins, density=True, alpha=.6, label=f'real labels (n={len(gr)})', color='#d95f02')
ax[0].hist(gs.off_pct.clip(-6, 6), bins, density=True, alpha=.5, label=f'synthetic GT (detector baseline, n={len(gs)})', color='#1b9e77')
ax[0].axvline(0, c='k', lw=.8); ax[0].set_xlabel('inward offset of keypoint from detected silhouette edge  [% box height]'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=8)
ax[0].set_title('Horizontal label offset (positive = inside)')
bx = np.linspace(-6, 6, 97)
for cl, col in ((0, '#1f78b4'), (4, '#e6ab02'), (2, '#e7298a'), (1, '#d95f02')):
    d = gr[gr.cls == cl]; ax[1].hist(d.off_pct.clip(-6, 6), bx, density=True, histtype='step', lw=1.6, color=col, label=f'{CLS[cl]} (med {d.off_pct.median():+.2f})')
ax[1].axvline(0, c='k', lw=.8); ax[1].legend(fontsize=8); ax[1].set_xlabel('% box height'); ax[1].set_title('Real, by class')
tp = t = gr.groupby('prefix').off_pct.agg(['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)]); tp.columns = ['m', 'q1', 'q3']; tp = tp.sort_values('m')
ax[2].errorbar(tp.m, range(len(tp)), xerr=[tp.m - tp.q1, tp.q3 - tp.m], fmt='o', c='#d95f02', capsize=2)
ax[2].set_yticks(range(len(tp))); ax[2].set_yticklabels(tp.index); ax[2].axvline(0, c='k', lw=.8); ax[2].set_xlabel('median (IQR) inward offset, % box height'); ax[2].set_title('Real, by source')
plt.tight_layout(); plt.savefig(f'{D}/fig_offsets.png', dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].hist(gr.ew.clip(0, 18), np.linspace(0, 18, 73), density=True, alpha=.6, color='#d95f02', label='real'); ax[0].hist(gs.ew.clip(0, 18), np.linspace(0, 18, 73), density=True, alpha=.5, color='#1b9e77', label='synthetic')
ax[0].set_xlabel('10-90% edge rise width [px]'); ax[0].legend(); ax[0].set_title('Edge blur (silhouette edges, cones >=60 px)')
o = imgs.groupby('prefix').vol1280.median().sort_values().index
ax[1].boxplot([imgs[imgs.prefix == p].vol1280.clip(upper=imgs.vol1280.quantile(.99)) for p in o], tick_labels=o, vert=False, showfliers=False)
ax[1].set_xlabel('variance of Laplacian (image rescaled to 1280 px wide)'); ax[1].set_title('Per-image sharpness by source (low = blurrier)')
for nm, tb, col in (('real', tb_r, '#d95f02'), ('synthetic GT', tb_s, '#1b9e77')):
    ax[2].errorbar(tb.ew_med, tb.off_pct, yerr=tb.rsd_pct, fmt='o-', c=col, capsize=3, label=nm)
ax[2].axhline(0, c='k', lw=.8); ax[2].set_xlabel('edge rise width [% box height] (sextile medians)'); ax[2].set_ylabel('median inward offset [% H]  (bars = robust sd)'); ax[2].legend(); ax[2].set_title('Does blur push labels inward?')
plt.tight_layout(); plt.savefig(f'{D}/fig_blur.png', dpi=110); plt.close()
out.close()
