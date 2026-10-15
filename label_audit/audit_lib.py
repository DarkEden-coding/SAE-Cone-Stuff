"""Edge measurement library for the cone keypoint label audit (read-only on datasets/).

Conventions: coordinates are continuous pixel coordinates (pixel i spans [i, i+1), centre i+0.5),
i.e. label_x_px = x_norm * W.  Offsets are 'inside positive': for a left keypoint (even index)
offset = label_x - edge_x; for a right keypoint (odd index) offset = edge_x - label_x.
"""
import cv2
import numpy as np

MIN_H_PX = 60
SIGMA = 0.8          # px, Gaussian pre-smoothing of the row profile
NROWS = 3            # rows averaged for the horizontal profile


def load_labels(path):
    out = []
    for l in open(path):
        p = l.split()
        if len(p) < 29:
            continue
        v = np.array(p[1:], float)
        k = v[4:].reshape(8, 3)
        out.append(dict(cls=int(p[0]), cx=v[0], cy=v[1], w=v[2], h=v[3], kp=k))
    return out


def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _row_profile(lab, y, nr=NROWS):
    iy = int(np.floor(y))
    a = max(iy - nr // 2, 0)
    b = min(iy + nr // 2 + 1, lab.shape[0])
    if b <= a:
        return None
    prof = lab[a:b].mean(0).astype(np.float32)[None]  # 1,W,3
    return cv2.GaussianBlur(prof, (0, 0), SIGMA, borderType=cv2.BORDER_REPLICATE)[0]


def _grad_mag(prof):
    g = np.zeros(len(prof), np.float32)
    d = (prof[2:] - prof[:-2]) * 0.5
    g[1:-1] = np.sqrt((d ** 2).sum(1))
    return g


def find_edge(lab, y, lo, hi, nr=NROWS):
    """Strongest horizontal colour edge on row y inside [lo, hi] (continuous x).
    Returns (edge_x, contrast, second_ratio, prof) or None."""
    prof = _row_profile(lab, y, nr)
    if prof is None:
        return None
    W = len(prof)
    a = int(max(np.floor(lo - 0.5), 1))
    b = int(min(np.ceil(hi - 0.5), W - 2))
    if b - a < 3:
        return None
    g = _grad_mag(prof)
    seg = g[a:b + 1]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not pk:
        return None
    pk = sorted(pk, key=lambda i: -seg[i])
    m = pk[0]
    y0, y1, y2 = seg[m - 1], seg[m], seg[m + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    d = float(np.clip(d, -1, 1))
    ex = a + m + d + 0.5
    mi = a + m
    contrast = float(g[max(mi - 3, 0):mi + 4].sum())
    sec = 0.0
    for i in pk[1:]:
        if abs(i - m) > 2:
            sec = float(seg[i] / max(seg[m], 1e-6))
            break
    return ex, contrast, sec, prof


def edge_width(prof, ex, s, pw):
    """10-90% rise distance of the edge at ex; s=+1 if cone interior is to the right of the edge.
    Returns (width, x10, x90) in px (x relative to ex, positive = toward interior) or None."""
    if pw < 24:
        return None
    dd = min(10.0, 0.3 * pw)
    W = len(prof)

    def samp(x0, x1):
        i0 = int(np.floor(min(x0, x1) - 0.5)); i1 = int(np.ceil(max(x0, x1) - 0.5))
        i0 = max(i0, 0); i1 = min(i1, W - 1)
        if i1 < i0:
            return None
        return prof[i0:i1 + 1]

    out = samp(ex - s * dd, ex - s * dd * 0.6)
    inn = samp(ex + s * dd * 0.6, ex + s * dd)
    if out is None or inn is None:
        return None
    lo = np.median(out, 0); hi = np.median(inn, 0)
    u = hi - lo
    n = np.linalg.norm(u)
    if n < 6:
        return None
    u /= n
    t = np.arange(-dd, dd + 0.01, 0.25)           # offset toward interior
    xs = ex + s * t
    idx = xs - 0.5
    if idx.min() < 0 or idx.max() > W - 1:
        return None
    pj = prof @ u
    q = np.interp(idx, np.arange(W), pj)
    q = (q - lo @ u) / (hi @ u - lo @ u)
    q = np.maximum.accumulate(q)                  # enforce monotone
    if q[0] > 0.1 or q[-1] < 0.9:
        return None
    i10 = np.argmax(q >= 0.1); i90 = np.argmax(q >= 0.9)
    return float(t[i90] - t[i10]), float(t[i10]), float(t[i90])


def vert_edge(lab, x_in, hw, y, H):
    """Strongest vertical colour edge in column strip around x_in, window +-0.07H around y.
    Returns (edge_y, contrast, second_ratio) or None."""
    h, w = lab.shape[:2]
    xa = int(max(np.floor(x_in - hw), 0)); xb = int(min(np.ceil(x_in + hw), w))
    if xb - xa < 1:
        return None
    ya = int(max(np.floor(y - 0.07 * H) - 3, 1)); yb = int(min(np.ceil(y + 0.07 * H) + 3, h - 2))
    if yb - ya < 6:
        return None
    col = lab[:, xa:xb].mean(1).astype(np.float32)[:, None]  # h,1,3
    col = cv2.GaussianBlur(col, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)[:, 0]
    g = np.zeros(h, np.float32)
    d = (col[2:] - col[:-2]) * 0.5
    g[1:-1] = np.sqrt((d ** 2).sum(1))
    lo = int(max(np.floor(y - 0.07 * H - 0.5), 1)); hi = int(min(np.ceil(y + 0.07 * H - 0.5), h - 2))
    seg = g[lo:hi + 1]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not pk:
        return None
    pk = sorted(pk, key=lambda i: -seg[i])
    m = pk[0]
    y0, y1, y2 = seg[m - 1], seg[m], seg[m + 1]
    den = y0 - 2 * y1 + y2
    d_ = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den < 0 else 0.0
    ey = lo + m + d_ + 0.5
    sec = 0.0
    for i in pk[1:]:
        if abs(i - m) > 2:
            sec = float(seg[i] / max(seg[m], 1e-6)); break
    return ey, float(g[lo + m - 3:lo + m + 4].sum()), sec


def measure_cone(lab, ox, oy, kp, Wimg, Himg, Hbox, ncls_kp):
    """lab: Lab crop with origin (ox,oy) in image px. kp: (8,3) in image px.
    Returns list of per-keypoint dicts for the left/right keypoints."""
    rows = []
    nk = ncls_kp
    for k in range(0, nk, 2):
        L = kp[k]; R = kp[k + 1]
        pw = R[0] - L[0]
        if pw < 6:   # swapped or degenerate: skip measurement
            continue
        for j, (P, s) in enumerate(((L, +1), (R, -1))):
            idx = k + j
            x = P[0] - ox; y = P[1] - oy
            outr = max(0.25 * pw, 3.0); inr = max(0.30 * pw, 3.0)
            lo, hi = (x - outr, x + inr) if s > 0 else (x - inr, x + outr)
            r = find_edge(lab, y, lo, hi)
            row = dict(kp=idx, pair_w=pw, lab_x=P[0], lab_y=P[1])
            if r is None:
                row.update(edge_x=np.nan)
                rows.append(row); continue
            ex, con, sec, prof = r
            row.update(edge_x=ex + ox, contrast=con, second=sec)
            row['off_px'] = s * (x - ex)
            # stability: edge at +-3 rows
            e2 = []
            for dy in (-3, 3):
                rr = find_edge(lab, y + dy, ex - 3, ex + 3)
                e2.append(rr[0] if rr else np.nan)
            row['curv'] = e2[0] + e2[1] - 2 * ex
            row['slope'] = (e2[1] - e2[0]) / 6.0        # dx/dy of silhouette
            ew = edge_width(prof, ex, s, pw)
            if ew:
                row['ew'], row['x10'], row['x90'] = ew
            # vertical stripe-boundary check just inside the silhouette
            xin = ex + s * max(0.12 * pw, 2.5)
            vr = vert_edge(lab, xin, max(0.04 * pw, 1.0), y, Hbox)
            if vr:
                row['vy'] = vr[0] + oy; row['vcon'] = vr[1]; row['vsec'] = vr[2]
                row['v_off_px'] = P[1] - (vr[0] + oy)    # + = label below transition
            rows.append(row)
    return rows
