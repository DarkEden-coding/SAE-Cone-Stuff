"""Geometry-only sanity checks on labels (all cones, any size).  Imported by analyze.py."""
import numpy as np, pandas as pd


def geom_metrics(c):
    """c: cone dataframe (columns x0..x7,y0..y7,v0..v7,Hb,Wb,cls,bx0..by1)."""
    out = pd.DataFrame(index=c.index)
    big = c.cls == 1
    npair = np.where(big, 4, 3)
    H = c.Hb
    W = {}; Ym = {}
    for p in range(4):
        xl, xr, yl, yr = c[f'x{2*p}'], c[f'x{2*p+1}'], c[f'y{2*p}'], c[f'y{2*p+1}']
        vis = (c[f'v{2*p}'] > 0) & (c[f'v{2*p+1}'] > 0)
        W[p] = (xr - xl).where(vis); Ym[p] = ((yl + yr) / 2).where(vis)
        out[f'swap{p}'] = (xl >= xr) & vis
        out[f'asym{p}'] = ((yl - yr).abs() / H).where(vis)       # fraction of box height
        out[f'wH{p}'] = (W[p] / H)
        out[f'ypos{p}'] = ((c.by1 - Ym[p]) / H)                  # height of the pair above box bottom
        # outside the box
        m = 0.05
        o = ((c[f'x{2*p}'] < c.bx0 - m * H) | (c[f'x{2*p}'] > c.bx1 + m * H) | (c[f'x{2*p+1}'] < c.bx0 - m * H) |
             (c[f'x{2*p+1}'] > c.bx1 + m * H) | (yl < c.by0 - m * H) | (yl > c.by1 + m * H) | (yr < c.by0 - m * H) | (yr > c.by1 + m * H))
        out[f'outbox{p}'] = o & vis
    for p in range(3):
        out[f'order{p}'] = (Ym[p + 1] - Ym[p]) / H                  # must be > 0
        out[f'wgrow{p}'] = (W[p + 1] - W[p]) / H                    # cone widens downward -> should be > 0
    out['miscount'] = np.where(big, c.nvis != 8, c.nvis != 6)
    return out
