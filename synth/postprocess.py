# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "opencv-python-headless", "openexr"]
# ///
"""Turn raw Blender renders into the final synthetic test set.

uv run synth/postprocess.py synth/output/test [--viz]

Per frame: undistorted linear render -> lens distortion -> lateral chromatic aberration,
lens softness, vignetting -> auto exposure -> RGGB sensor with shot/read/row noise and
12-bit quantisation -> demosaic -> white balance, tone curve, sharpening, grain -> JPEG.
Writes images/, labels/ (YOLO pose, BRT class/keypoint order), annotations/ (full ground
truth JSON), plus dataset.json with calibration, cone models and conventions.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cone_model as cm  # noqa: E402

KP_NAMES = {
    'small': ['stripe_top_left', 'stripe_top_right', 'stripe_bottom_left', 'stripe_bottom_right',
              'body_foot_left', 'body_foot_right', None, None],
    'big': ['upper_stripe_top_left', 'upper_stripe_top_right', 'upper_stripe_bottom_left',
            'upper_stripe_bottom_right', 'lower_stripe_top_left', 'lower_stripe_top_right',
            'lower_stripe_bottom_left', 'lower_stripe_bottom_right'],
}


def load_exr(path):
    f = OpenEXR.File(str(path))
    ch = f.parts[0].channels
    rgb = ch['ViewLayer.Combined'].pixels[..., :3].astype(np.float32)
    idx = np.rint(ch['ViewLayer.IndexOB.X'].pixels).astype(np.int32)
    return rgb, idx, len(ch) > 2


def strip_exr(path, rgb, idx):
    """Rewrite the raw EXR with only the beauty and index passes (drops Blender's noisy copy)."""
    hdr = {'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage}
    chans = {'RGB': rgb.astype(np.float16), 'IndexOB': idx.astype(np.float32)}
    OpenEXR.File(hdr, chans).write(str(path))


def load_exr_stripped(path):
    ch = OpenEXR.File(str(path)).parts[0].channels
    if 'ViewLayer.Combined' in ch:
        return load_exr(path)
    rgb = np.stack([ch[f'RGB.{c}'].pixels for c in 'RGB'], -1) if 'RGB.R' in ch else ch['RGB'].pixels
    return rgb.astype(np.float32), np.rint(ch['IndexOB'].pixels).astype(np.int32), False


def distortion_maps(K, dist, W, H, margins, cache):
    if cache.exists():
        m = np.load(cache)
        return m['x'], m['y']
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xyd = np.stack([(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1]], -1)
    xy = cm.undistort_normalized(xyd, dist)
    mx = (xy[..., 0] * K[0][0] + K[0][2] + margins[0]).astype(np.float32)
    my = (xy[..., 1] * K[1][1] + K[1][2] + margins[1]).astype(np.float32)
    np.savez(cache, x=mx, y=my)
    return mx, my


def sample_defects(rng, sev):
    u = lambda: float(rng.uniform(0.3, 1.0))
    return dict(
        severity=sev,
        chromatic_aberration=0.0002 + 0.0012 * sev * u(),
        softness_sigma_center_px=0.15 + 0.6 * sev * u(),
        softness_sigma_corner_px=0.35 + 1.2 * sev * u(),
        vignetting=float(rng.uniform(0.12, 0.4)),
        exposure_target=float(0.16 * 2 ** rng.normal(0, 0.35)),
        white_balance_gains=[float(rng.uniform(1.7, 2.1)), 1.0, float(rng.uniform(1.4, 1.8))],
        full_well_electrons=float(12000 * 2 ** (-4.0 * sev * u())),
        read_noise_electrons=float(2 + 6 * sev * u()),
        row_noise_electrons=float(1.5 * sev * u()),
        tone_contrast=float(rng.uniform(0.1, 0.35)),
        saturation=float(rng.uniform(0.9, 1.15)),
        sharpen_amount=float(rng.uniform(0.15, 0.7)),
        grain_amplitude=0.003 + 0.018 * sev * u(),
        jpeg_quality=int(round(95 - 28 * sev * u())),
    )


def srgb_oetf(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def camera_pipeline(rgb, K, d, rng):
    H, W = rgb.shape[:2]
    cx, cy = K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    rn2 = ((u - cx) ** 2 + (v - cy) ** 2) / (cx ** 2 + cy ** 2)

    # Lateral chromatic aberration: red magnified, blue shrunk about the principal point.
    out = rgb.copy()
    for c, s in ((0, 1 + d['chromatic_aberration']), (2, 1 - d['chromatic_aberration'])):
        out[..., c] = cv2.remap(rgb[..., c], (cx + (u - cx) / s), (cy + (v - cy) / s),
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Lens softness, worse towards the corners.
    b0 = cv2.GaussianBlur(out, (0, 0), d['softness_sigma_center_px'])
    b1 = cv2.GaussianBlur(out, (0, 0), d['softness_sigma_corner_px'])
    w = np.clip(rn2, 0, 1)[..., None]
    out = b0 * (1 - w) + b1 * w
    out *= (1 / (1 + d['vignetting'] * rn2) ** 2)[..., None]

    # Auto exposure on log-average luminance.
    Y = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    out *= d['exposure_target'] / math.exp(float(np.mean(np.log(Y + 1e-4))))

    # Sensor: RGGB mosaic, Poisson shot noise, read noise, row noise, 12-bit ADC.
    g = np.array(d['white_balance_gains'], np.float32)
    raw = out / g
    m = np.empty((H, W), np.float32)
    m[0::2, 0::2], m[0::2, 1::2] = raw[0::2, 0::2, 0], raw[0::2, 1::2, 1]
    m[1::2, 0::2], m[1::2, 1::2] = raw[1::2, 0::2, 1], raw[1::2, 1::2, 2]
    fw = d['full_well_electrons']
    e = rng.poisson(np.clip(m, 0, 1.2) * fw).astype(np.float32)
    e += rng.normal(0, d['read_noise_electrons'], e.shape).astype(np.float32)
    e += rng.normal(0, d['row_noise_electrons'], (H, 1)).astype(np.float32)
    adc = np.clip(np.rint(e / fw * 4095), 0, 4095).astype(np.uint16) * 16
    rgb_d = cv2.cvtColor(adc, cv2.COLOR_BayerRGGB2RGB_EA).astype(np.float32) / 65520

    # ISP: white balance, saturation, tone curve, sharpening, grain.
    x = np.clip(rgb_d * g, 0, 1)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    x = np.clip(lum + d['saturation'] * (x - lum), 0, 1)
    y = srgb_oetf(x)
    y = (1 - d['tone_contrast']) * y + d['tone_contrast'] * (3 * y * y - 2 * y ** 3)
    y = y + d['sharpen_amount'] * (y - cv2.GaussianBlur(y, (0, 0), 1.0))
    n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.6)
    nc = cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32), (0, 0), 0.8)
    y = y + d['grain_amplitude'] * 2.2 * (n[..., None] + 0.3 * nc)
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)


def hull_points(class_id):
    s = cm.cone_spec(class_id)
    half, rc, t = s['plate_size'] / 2, s['plate_corner'], s['plate_thickness']
    pts = []
    for cx, cy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for a in np.linspace(a0, a0 + 90, 7):
            x = cx * (half - rc) + rc * math.cos(math.radians(a))
            y = cy * (half - rc) + rc * math.sin(math.radians(a))
            pts += [(x, y, 0.0), (x, y, t)]
    r_top = max(r for r, z in s['profile'] if z >= s['height'] - 0.006)
    for a in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        pts.append((r_top * math.cos(a), r_top * math.sin(a), s['height']))
        pts.append((s['profile'][0][0] * math.cos(a), s['profile'][0][0] * math.sin(a), s['profile'][0][1]))
    return np.array(pts)


def annotate(meta, idx_img, K, dist, W, H):
    T_wc = np.array(meta['T_world_cam'])
    T_cw = np.linalg.inv(T_wc)
    cam_w = T_wc[:3, 3]
    cones, yolo = [], []
    for c in meta['cones']:
        cls = c['class_id']
        T_wo = np.array(c['T_world_cone'])
        T_co = T_cw @ T_wo
        kp_cone = np.array([p if p is not None else [np.nan] * 3 for p in c['keypoints_cone_unordered']])
        occ = list(c['keypoint_occluded_unordered'])
        kp_cam = (T_co[:3, :3] @ kp_cone.T).T + T_co[:3, 3]
        uv = np.full((8, 2), np.nan)
        ok = ~np.isnan(kp_cam[:, 0])
        if ok.any():
            uv[ok] = cm.project(kp_cam[ok], K, dist)
        for k in range(0, 8, 2):  # even index = image-left
            if ok[k] and uv[k, 0] > uv[k + 1, 0]:
                for arr in (kp_cone, kp_cam, uv):
                    arr[[k, k + 1]] = arr[[k + 1, k]]
                occ[k], occ[k + 1] = occ[k + 1], occ[k]
        kps = []
        for k in range(8):
            if not ok[k]:
                kps.append([0.0, 0.0, 0])
                continue
            inside = 0 <= uv[k, 0] <= W - 1 and 0 <= uv[k, 1] <= H - 1 and kp_cam[k, 2] > 0
            kps.append([float(uv[k, 0]), float(uv[k, 1]), 0 if not inside else (1 if occ[k] else 2)])

        hp = (T_co[:3, :3] @ hull_points(cls).T).T + T_co[:3, 3]
        huv = cm.project(hp, K, dist)
        hull = cv2.convexHull(huv.astype(np.float32))
        amodal_area = float(cv2.contourArea(hull))
        x0, y0 = huv.min(0)
        x1, y1 = huv.max(0)
        clip = [max(0.0, x0), max(0.0, y0), min(W - 1.0, x1), min(H - 1.0, y1)]
        vis = idx_img == c['pass_index']
        n_vis = int(vis.sum())
        vbox = None
        if n_vis:
            ys, xs = np.nonzero(vis)
            vbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        ground_range = float(np.linalg.norm((T_wo[:3, 3] - cam_w)[:2]))
        name = 'big' if cls == 1 else 'small'
        rec = dict(
            class_id=cls, class_name=cm.CLASS_NAMES[cls],
            T_cam_cone=T_co.tolist(),
            base_center_cam=T_co[:3, 3].tolist(),
            distance_m=float(np.linalg.norm(T_co[:3, 3])),
            ground_range_m=ground_range,
            in_eval_range=bool(2.0 <= ground_range <= 30.0),
            bbox_xyxy_amodal=[float(x0), float(y0), float(x1), float(y1)],
            bbox_xyxy_clipped=clip,
            bbox_xyxy_visible=vbox,
            visible_pixels=n_vis,
            visible_fraction=float(n_vis / amodal_area) if amodal_area > 0 else 0.0,
            truncated=bool(x0 < 0 or y0 < 0 or x1 > W - 1 or y1 > H - 1),
            keypoint_names=KP_NAMES[name],
            keypoints_2d=kps,
            keypoints_3d_cone=[None if np.isnan(p[0]) else p.tolist() for p in kp_cone],
            keypoints_3d_cam=[None if np.isnan(p[0]) else p.tolist() for p in kp_cam],
            keypoint_occluded=occ,
            appearance={k: c[k] for k in ('dirt', 'wear', 'scuff', 'fade')},
        )
        cones.append(rec)
        if n_vis >= 12 and rec['visible_fraction'] >= 0.1 and clip[2] > clip[0] and clip[3] > clip[1]:
            bx = [(clip[0] + clip[2]) / 2 / W, (clip[1] + clip[3]) / 2 / H,
                  (clip[2] - clip[0]) / W, (clip[3] - clip[1]) / H]
            kv = []
            for x, y, v in kps:
                kv += [x / W, y / H, 2 if v else 0] if v else [0.0, 0.0, 0]
            yolo.append(' '.join([str(cls)] + [f'{a:.6f}' for a in bx] +
                                 [f'{a:.6f}' if i % 3 != 2 else str(int(a)) for i, a in enumerate(kv)]))
    return cones, yolo


def draw_viz(img, cones):
    v = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cols = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 255)]
    for c in cones:
        if c['visible_pixels'] < 12:
            continue
        x0, y0, x1, y1 = map(int, c['bbox_xyxy_clipped'])
        cv2.rectangle(v, (x0, y0), (x1, y1), (0, 255, 0) if c['in_eval_range'] else (128, 128, 128), 1)
        for k, (x, y, s) in enumerate(c['keypoints_2d']):
            if s:
                cv2.circle(v, (round(x), round(y)), 2, cols[k], -1 if s == 2 else 1)
        cv2.putText(v, f"{c['ground_range_m']:.1f}m", (x0, y0 - 3), 0, 0.4, (255, 255, 255), 1)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--viz', action='store_true', help='also write viz/ overlays')
    ap.add_argument('--keep-noisy', action='store_true', help="don't strip Blender's noisy pass from raw EXRs")
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    root = Path(args.root)
    for d in ('images', 'labels', 'annotations') + (('viz',) if args.viz else ()):
        (root / d).mkdir(exist_ok=True)
    cam = cm.CAMERA
    K, dist, W, H = cam['K'], cam['dist'], cam['width'], cam['height']

    for jp in sorted((root / 'raw').glob('*.json')):
        stem = jp.stem
        if not args.force and (root / 'annotations' / f'{stem}.json').exists():
            continue
        meta = json.loads(jp.read_text())
        exr = jp.with_suffix('.exr')
        rgb, idx, has_noisy = load_exr_stripped(exr)
        if has_noisy and not args.keep_noisy:
            strip_exr(exr, rgb, idx)
        mx, my = distortion_maps(K, dist, W, H, meta['margins_ltrb'], root / 'raw' / 'distortion_map.npz')
        lin = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        idx_d = cv2.remap(idx.astype(np.float32), mx, my, cv2.INTER_NEAREST).astype(np.int32)
        rng = np.random.default_rng(meta['seed'] + 7_777_777)
        d = sample_defects(rng, meta['scene']['severity'])
        img = camera_pipeline(lin, K, d, rng)
        cv2.imwrite(str(root / 'images' / f'{stem}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, d['jpeg_quality']])
        cones, yolo = annotate(meta, idx_d, K, dist, W, H)
        (root / 'labels' / f'{stem}.txt').write_text('\n'.join(yolo) + ('\n' if yolo else ''))
        sc = meta['scene']
        ann = dict(
            image=f'images/{stem}.jpg', width=W, height=H,
            camera=dict(K=K, dist=dist, T_world_cam=meta['T_world_cam'], height_m=sc['camera']['height'],
                        pitch_down_deg=math.degrees(sc['camera']['pitch']), roll_deg=math.degrees(sc['camera']['roll'])),
            capture=dict(hdri=sc['hdri'], hdri_rotation=sc['hdri_rotation'], ground=sc['ground'],
                         optics=sc['optics'], motion=sc['motion'], extras=sc['extras'], track_width_m=sc['track_width']),
            defects=d, cones=cones, seed=meta['seed'], render_seconds=meta['render_seconds'], samples=meta['samples'])
        (root / 'annotations' / f'{stem}.json').write_text(json.dumps(ann, indent=1))
        if args.viz:
            cv2.imwrite(str(root / 'viz' / f'{stem}.jpg'), draw_viz(img, cones))
        print(f'{stem}: {len(cones)} cones, {len(yolo)} labelled, severity {d["severity"]:.2f}')

    write_dataset_json(root)


def write_dataset_json(root):
    models = {}
    for cls in (0, 1, 2, 4):
        s = cm.cone_spec(cls)
        rings = cm.canonical_rings(cls)
        models[cm.CLASS_NAMES[cls]] = dict(
            height_m=s['height'], base_plate_m=s['plate_size'], plate_thickness_m=s['plate_thickness'],
            body_frustum=dict(z0=s['body_z0'], r0=s['body_r0'], z1=s['body_z1'], r1=s['body_r1']),
            stripes_z_m=s['stripes'],
            keypoint_rings_z_r_m=rings,
            planar_model_xyz_m=[[sgn * r, 0.0, z] for z, r in rings for sgn in (-1, 1)],
        )
    (root / 'dataset.json').write_text(json.dumps(dict(
        description='Synthetic FS driverless cone keypoint/PnP test set (Blender Cycles).',
        calibration=dict(model='opencv_radial_tangential', **cm.CAMERA),
        classes=cm.CLASS_NAMES, keypoint_names=KP_NAMES, cone_models=models,
        conventions=dict(
            camera_frame='OpenCV: x right, y down, z forward; metres',
            pixels='OpenCV: pixel centres at integer coordinates, distorted image',
            cone_frame='origin at centre of the cone bottom face on the ground; z up the axis; x,y along base plate edges',
            T_cam_cone='4x4, maps cone-frame points into the camera frame; translation = base_center_cam',
            T_world_cam='4x4, camera (OpenCV axes) to world; world z up, ground plane z=0',
            keypoints='BRT order; even index = image-left silhouette point, odd = image-right. '
                      'keypoints_3d_* are the true view-dependent silhouette points for this frame. '
                      'planar_model_xyz_m is the fixed approximation (x lateral, z up, y=0).',
            visibility='keypoints_2d v: 2 visible, 1 in frame but occluded by another cone, 0 out of frame/undefined. '
                       'YOLO labels use 2 for any in-frame point (BRT convention), 0 otherwise.',
            yolo_labels='cones with >=12 visible pixels and visible_fraction>=0.1; box = amodal box clipped to image',
            eval_range='in_eval_range: 2 m <= ground range <= 30 m; farther cones are rendered and annotated',
        ),
        licenses='HDRIs and asphalt textures: Poly Haven, CC0.',
    ), indent=1))
    (root / 'data.yaml').write_text(
        f"path: {root.resolve()}\ntrain: images\nval: images\ntest: images\n"
        "kpt_shape: [8, 3]\nnames:\n" + ''.join(f'  {k}: {v}\n' for k, v in cm.CLASS_NAMES.items()))


if __name__ == '__main__':
    main()
