"""Contact sheets of cone crops with labels (red dots) and detected silhouette edges (cyan ticks).
Yellow ticks = detected colour transition (stripe boundary) near the keypoint.  Run after analyze.py."""
import os, sys
import cv2, numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = '/mnt/Primary/Projects/SAE-Object--Detection'
REAL = f'{R}/datasets/brt-cone-pose/brt-cone-pose-11k/images'
SYN = f'{R}/synth_data/images'
CLS = {0: 'BLUE', 1: 'BIG', 2: 'ORNG', 4: 'YEL'}
PW, PH = 240, 318
rng = np.random.default_rng(1)


def panel(path, c, kps, caption, chains=False):
    img = cv2.imread(path)
    Hb = c.by1 - c.by0; Wb = c.bx1 - c.bx0
    m = 0.18 * max(Hb, Wb)
    x0 = int(max(c.bx0 - m, 0)); y0 = int(max(c.by0 - m, 0)); x1 = int(min(c.bx1 + m, img.shape[1])); y1 = int(min(c.by1 + m, img.shape[0]))
    crop = img[y0:y1, x0:x1]
    s = min((PH - 30) / crop.shape[0], PW / crop.shape[1])
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    def T(x, y): return int(round((x - x0) * s)), int(round((y - y0) * s))
    if chains:
        for side, col in ((0, (0, 220, 0)), (1, (255, 0, 255))):
            pts = [T(c[f'x{k}'], c[f'y{k}']) for k in range(side, 8, 2) if c[f'v{k}'] > 0]
            if len(pts) > 1:
                cv2.polylines(big, [np.array(pts)], False, col, 1, cv2.LINE_AA)
    for _, r in kps.iterrows():
        if not np.isnan(r.edge_x):
            ex, ey = T(r.edge_x, r.lab_y)
            cv2.line(big, (ex, ey - 6), (ex, ey + 6), (255, 255, 0), 1, cv2.LINE_AA)    # BGR cyan
        if r.get('vgate', False) and not np.isnan(r.get('vy', np.nan)):
            s_ = 1 if int(r.kp) % 2 == 0 else -1
            vx, vy = T(r.edge_x + s_ * max(0.12 * r.pair_w, 2.5), r.vy)
            cv2.line(big, (vx - 4, vy), (vx + 4, vy), (0, 255, 255), 1, cv2.LINE_AA)    # yellow
    for k in range(8):
        if c[f'v{k}'] > 0:
            cv2.circle(big, T(c[f'x{k}'], c[f'y{k}']), 2, (0, 0, 255), -1, cv2.LINE_AA)
            if chains:
                cv2.putText(big, str(k), (T(c[f'x{k}'], c[f'y{k}'])[0] + 3, T(c[f'x{k}'], c[f'y{k}'])[1] - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    canvas = np.full((PH, PW, 3), 30, np.uint8)
    h, w = big.shape[:2]
    canvas[:h, :w] = big[:PH - 30, :PW]
    cv2.putText(canvas, caption[0], (3, PH - 17), cv2.FONT_HERSHEY_PLAIN, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, caption[1], (3, PH - 4), cv2.FONT_HERSHEY_PLAIN, 0.85, (200, 200, 255), 1, cv2.LINE_AA)
    return canvas


def sheet(items, out, cols=6, title=''):
    ps = [panel(*it) for it in items]
    while len(ps) % cols: ps.append(np.zeros((PH, PW, 3), np.uint8))
    rows = [np.hstack(ps[i:i + cols]) for i in range(0, len(ps), cols)]
    img = np.vstack(rows)
    if title:
        bar = np.zeros((22, img.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([bar, img])
    cv2.imwrite(f'{D}/{out}', img, [cv2.IMWRITE_JPEG_QUALITY, 90] if out.endswith('.jpg') else [])
    print(out, img.shape)


def items_from(cone_df, kp_df, base, n, caps, seed=0, chains=False, kpfilter=None):
    g = kp_df.groupby(['img', 'ci'])
    sel = cone_df.sample(min(n, len(cone_df)), random_state=seed) if n else cone_df
    its = []
    for _, c in sel.iterrows():
        try:
            k = g.get_group((c.img, c.ci))
        except KeyError:
            k = kp_df.iloc[0:0]
        if kpfilter is not None:
            k = k[kpfilter(k)]
        split = c.split if base == REAL else ''
        path = f'{base}/{split}/{c.img}' if split else f'{base}/{c.img}'
        its.append((path, c, k, caps(c, k), chains))
    return its


def main():
    kr = pd.read_csv(f'{D}/work/real_kp.csv'); cf = pd.read_csv(f'{D}/work/real_cone_flags.csv')
    kr['gate'] = (kr.contrast > 25) & (kr.second < 0.6) & (kr.curv.abs() < 1)
    kr['vgate'] = (kr.vcon > 15) & (kr.vsec < 0.6)
    kr['off_pct'] = kr.off_px / kr.Hb * 100; kr['ew_rel'] = kr.ew / kr.Hb * 100
    kg = kr[kr.gate]
    # cone table with coordinates
    cx = cf.copy()
    # per-cone summary stats from gated kps, small-cone stripe points 0-3 and all lr points
    st = kg.groupby(['img', 'ci']).agg(n=('kp', 'count'), off_pct=('off_pct', 'median'), off_px=('off_px', 'median'), ew=('ew', 'median'), ew_rel=('ew_rel', 'median'),
                                       maxabs=('off_px', lambda x: x.abs().max()))
    st03 = kg[kg.kp < 4].groupby(['img', 'ci']).agg(n03=('kp', 'count'), off03=('off_pct', 'median'))
    cx = cx.merge(st, on=['img', 'ci'], how='left').merge(st03, on=['img', 'ci'], how='left')
    cap = lambda c, k: (f'{c.prefix} {CLS[c.cls]} H{c.Hb:.0f} ew' + (f'{c.ew:.1f}px' if c.ew == c.ew else 'n/a'), f'off(med)={c.off_pct if c.off_pct == c.off_pct else 0:+.1f}%H')
    usable = cx[cx.usable & (cx.n >= 5)]
    small = usable[usable.cls.isin([0, 2, 4])]
    # A good examples
    good = small[(small.off_px.abs() < 0.3) & (small.maxabs < 1.5) & (small.Hb > 80)]
    sheet(items_from(good, kr, REAL, 18, cap, 1), 'sheet_A_on_the_line.jpg', title='A. labels on the silhouette edge (|median offset|<0.3 px): red=label, cyan=detected edge, yellow=detected stripe transition')
    inward = small[(small.off03 > 1.8) & (small.n03 >= 4)]
    sheet(items_from(inward, kr, REAL, 18, cap, 2), 'sheet_B_inward.jpg', title=f'B. labels clearly INSIDE the silhouette (stripe pts 0-3 median > 1.8 %H); {len(inward)} such cones of {len(small)}')
    outward = small[(small.off03 < -1.8) & (small.n03 >= 4)]
    sheet(items_from(outward, kr, REAL, 18, cap, 3), 'sheet_C_outward.jpg', title=f'C. labels OUTSIDE the silhouette (median < -1.8 %H); {len(outward)} cones')
    blur = small[(small.ew > 8.5) & (small.Hb >= 70)]
    sheet(items_from(blur, kr, REAL, 18, cap, 4), 'sheet_D_blurred.jpg', title=f'D. blurred edges (cone median 10-90% rise > 8.5 px, H>=70); {len(blur)} cones')
    sharp = small[(small.ew < 3.5) & (small.Hb >= 80)]
    sheet(items_from(sharp, kr, REAL, 18, cap, 5), 'sheet_D2_sharp.jpg', title=f'D2. sharp edges (cone median rise < 3.5 px, H>=80); {len(sharp)} cones')
    bigc = usable[usable.cls == 1]
    sheet(items_from(bigc, kr, REAL, 18, cap, 6), 'sheet_E_big_orange.jpg', title=f'E. big orange cones (random): labels drift inside going down; {len(bigc)} cones')
    # gross
    capg = lambda c, k: (f'{c.prefix} {CLS[c.cls]} H{c.Hb:.0f}', ' '.join(n for n in ['swap', 'order', 'asym', 'narrowing'] if c[n]))
    g = cx[(cx.Hb >= 60)]
    for nm, ttl in (('swap', 'left/right swapped pair (x_left >= x_right)'), ('order', 'pair ordering violation (lower pair above upper pair)'), ('asym', '|yL-yR| > 0.15 H (asymmetric pair)'), ('narrowing', 'cone narrows going down')):
        d = g[g[nm]]
        sheet(items_from(d, kr, REAL, 12, capg, 7, chains=True), f'sheet_F_gross_{nm}.jpg', cols=6, title=f'F. {ttl}: green chain = even (left) kps, magenta = odd (right); {len(d)} cones H>=60')
    # synthetic GT validation sheet
    ks = pd.read_csv(f'{D}/work/synth_kp.csv'); cs_ = pd.read_csv(f'{D}/work/synth_cone.csv')
    ks['gate'] = (ks.contrast > 25) & (ks.second < 0.6) & (ks.curv.abs() < 1); ks['vgate'] = (ks.vcon > 15) & (ks.vsec < 0.6)
    ks['off_pct'] = ks.off_px / ks.Hb * 100
    cs_ = cs_[cs_.usable & cs_.cls.isin([0, 4, 1])]
    capS = lambda c, k: (f'SYNTH-GT {CLS[c.cls]} H{c.Hb:.0f}', f'off(med)={k.off_px.median():+.2f}px' if len(k) else '')
    sheet(items_from(cs_, ks, SYN, 18, capS, 8), 'sheet_V_synthetic_validation.jpg', title='V. detector validation on synthetic GT (labels are exact silhouette tangent points): cyan should sit on red')


if __name__ == '__main__':
    main()
