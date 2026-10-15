"""Measure label-vs-edge offsets.  usage: run_audit.py OUT_PREFIX img_dir:label_dir [img_dir:label_dir ...] [--shift DX]
Writes OUT_PREFIX_kp.csv (per keypoint, cones >= 60 px), OUT_PREFIX_cone.csv (all cones, geometry + image sharpness).
--shift DX: validation only, move every label inward by DX px before measuring."""
import sys, os, glob, csv
import cv2, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as A

SHIFT = 0.0


def prefix_of(name):
    return name.split('_')[0] if '_' in name else 'synth'


def process(args):
    ip, lp, split = args
    img = cv2.imread(ip)
    if img is None:
        return [], []
    Himg, Wimg = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vol = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sc = 1280.0 / Wimg
    g2 = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    vol1280 = float(cv2.Laplacian(g2, cv2.CV_64F).var())
    # letterbox / padding detection: valid content rectangle
    mk = gray > 6
    xs = np.where(mk.mean(0) > 0.3)[0]; ys = np.where(mk.mean(1) > 0.3)[0]
    cx0, cx1 = (int(xs[0]), int(xs[-1]) + 1) if len(xs) else (0, Wimg)
    cy0, cy1 = (int(ys[0]), int(ys[-1]) + 1) if len(ys) else (0, Himg)
    name = os.path.basename(ip)
    pre = prefix_of(name)
    labs = A.load_labels(lp)
    boxes = []
    for l in labs:
        boxes.append((l['cx'] * Wimg - l['w'] * Wimg / 2, l['cy'] * Himg - l['h'] * Himg / 2,
                      l['cx'] * Wimg + l['w'] * Wimg / 2, l['cy'] * Himg + l['h'] * Himg / 2))
    kprows, conerows = [], []
    for ci, l in enumerate(labs):
        kp = l['kp'].copy()
        kp[:, 0] *= Wimg; kp[:, 1] *= Himg
        bx0, by0, bx1, by1 = boxes[ci]
        Hb = by1 - by0; Wb = bx1 - bx0
        vis = kp[:, 2] > 0
        nk = 8 if l['cls'] == 1 else 6
        full = bool(vis[:nk].all())
        trunc = bx0 < cx0 + 3 or by0 < cy0 + 3 or bx1 > cx1 - 3 or by1 > cy1 - 3
        # overlap with other boxes
        ov = 0.0
        for cj, b in enumerate(boxes):
            if cj == ci:
                continue
            iw = min(bx1, b[2]) - max(bx0, b[0]); ih = min(by1, b[3]) - max(by0, b[1])
            if iw > 0 and ih > 0:
                ov = max(ov, iw * ih / (Hb * Wb))
        crow = dict(img=name, split=split, prefix=pre, ci=ci, cls=l['cls'], Himg=Himg, Wimg=Wimg,
                    Hb=Hb, Wb=Wb, nvis=int(vis.sum()), full=full, trunc=trunc, overlap=ov,
                    vol_img=vol, vol1280=vol1280, cx0=cx0, cy0=cy0, cx1=cx1, cy1=cy1)
        for k in range(8):
            crow[f'x{k}'] = kp[k, 0]; crow[f'y{k}'] = kp[k, 1]; crow[f'v{k}'] = kp[k, 2]
        crow['bx0'], crow['by0'], crow['bx1'], crow['by1'] = bx0, by0, bx1, by1
        usable = Hb >= A.MIN_H_PX and full and not trunc and ov < 0.02
        if usable:
            pad = int(0.4 * max(Hb, Wb)) + 12
            x0 = int(max(bx0 - pad, 0)); y0 = int(max(by0 - pad, 0))
            x1 = int(min(bx1 + pad, Wimg)); y1 = int(min(by1 + pad, Himg))
            crop = img[y0:y1, x0:x1]
            lab = A.to_lab(crop)
            gc = gray[int(max(by0, 0)):int(by1), int(max(bx0, 0)):int(bx1)]
            crow['vol_cone'] = float(cv2.Laplacian(gc, cv2.CV_64F).var())
            kps = kp.copy()
            if SHIFT:
                for k in range(0, nk, 2):
                    kps[k, 0] += SHIFT; kps[k + 1, 0] -= SHIFT
            for r in A.measure_cone(lab, x0, y0, kps, Wimg, Himg, Hb, nk):
                r.update(img=name, split=split, prefix=pre, ci=ci, cls=l['cls'], Hb=Hb, Wb=Wb,
                         vol_cone=crow['vol_cone'], vol1280=vol1280)
                kprows.append(r)
        crow['usable'] = usable
        conerows.append(crow)
    return kprows, conerows


def main():
    global SHIFT
    argv = sys.argv[1:]
    if '--shift' in argv:
        i = argv.index('--shift'); SHIFT = float(argv[i + 1]); del argv[i:i + 2]
    out = argv[0]
    tasks = []
    for spec in argv[1:]:
        idir, ldir = spec.split(':')
        split = os.path.basename(idir.rstrip('/'))
        for ip in sorted(glob.glob(os.path.join(idir, '*'))):
            lp = os.path.join(ldir, os.path.splitext(os.path.basename(ip))[0] + '.txt')
            if os.path.exists(lp):
                tasks.append((ip, lp, split))
    print(len(tasks), 'images', flush=True)
    kr, cr = [], []
    with Pool(14) as p:
        for i, (a, b) in enumerate(p.imap(process, tasks, chunksize=8)):
            kr += a; cr += b
            if i % 1000 == 0:
                print(i, flush=True)
    import pandas as pd
    pd.DataFrame(kr).to_csv(out + '_kp.csv', index=False)
    pd.DataFrame(cr).to_csv(out + '_cone.csv', index=False)
    print('done', len(kr), len(cr))


if __name__ == '__main__':
    main()
