"""Render synthetic FS driverless cone frames with Blender Cycles.

blender -b -P synth/render_blender.py -- --out synth/output/test --start 0 --count 1

Each frame writes raw/<id>.exr (linear multilayer EXR with an object-index pass, rendered
undistorted with overscan) and raw/<id>.json (scene, camera and cone ground truth).
postprocess.py then applies lens distortion and camera defects and writes final labels.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cone_model as cm  # noqa: E402

ASSETS = HERE / 'assets'
CLASS_COLORS = {  # linear RGB
    0: (0.004, 0.105, 0.42), 1: (0.86, 0.13, 0.012), 2: (0.86, 0.13, 0.012), 4: (0.82, 0.56, 0.012)}
STRIPE_COLORS = {0: (0.78, 0.78, 0.76), 1: (0.78, 0.78, 0.76), 2: (0.78, 0.78, 0.76), 4: (0.022, 0.022, 0.024)}


# ---------------------------------------------------------------- scene sampling

def centerline(rng, length, ds=0.25):
    """Random FS-like track centreline of straights and constant-radius arcs."""
    pts, heads, curv = [], [], []
    p, h = np.zeros(2), 0.0
    while len(pts) * ds < length:
        if rng.random() < 0.4:
            n, k = int(rng.uniform(6, 30) / ds), 0.0
        else:
            r = 9 + 31 * rng.uniform(0, 1) ** 1.6  # biased towards tight FS turns
            n, k = int(r * math.radians(rng.uniform(20, 160)) / ds), rng.choice([-1, 1]) / r
        for _ in range(max(n, 1)):
            pts.append(p.copy()); heads.append(h); curv.append(k)
            h += k * ds
            p = p + ds * np.array([math.cos(h), math.sin(h)])
    return np.array(pts), np.array(heads), np.array(curv), ds


def boundary_cones(rng, track, width, s_from=0.0):
    pts, heads, curv, ds = track
    cones = []
    for side, cls in ((1, 0), (-1, 4)):  # left blue, right yellow
        s = s_from + rng.uniform(0, 2)
        while s < len(pts) * ds:
            i = int(s / ds)
            n = np.array([-math.sin(heads[i]), math.cos(heads[i])])
            xy = pts[i] + side * n * width / 2 + rng.normal(0, 0.04, 2)
            cones.append(dict(cls=cls, xy=xy, s=s, side=side))
            r = 1 / max(abs(curv[i]), 1e-6)
            s += 4.8 if r > 60 else float(np.clip(2.0 + (r - 9) / 31 * 2.5, 2.0, 4.5))
            if rng.random() < 0.015:
                s += 4  # occasional missing cone
    return cones


def transform_track(track, angle, offset):
    pts, heads, curv, ds = track
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + offset, heads + angle, curv, ds


def dist_to_polyline(xy, pts):
    return np.min(np.linalg.norm(pts[None, :, :] - xy[:, None, :], axis=2), axis=1)


def place_car(rng, curv, ds):
    """Pick the car's arc length: on a straight, 2-14 m before a turn, or inside a turn."""
    mode = str(rng.choice(['straight', 'approaching_turn', 'in_turn'], p=[0.3, 0.35, 0.35]))
    i = np.arange(len(curv))
    lo, hi = int(6 / ds), int(60 / ds)
    if mode == 'in_turn':
        # Inside the first 60% of an arc, so the rest of the bend is ahead of the car.
        starts = i[1:][(curv[1:] != 0) & (curv[:-1] == 0)]
        starts = starts[(starts >= lo) & (starts <= hi)]
        if len(starts):
            a = int(rng.choice(starts))
            b = a + int(np.argmax(curv[a:] != curv[a])) if (curv[a:] != curv[a]).any() else len(curv)
            return mode, float((a + rng.uniform(0.05, 0.6) * (b - a)) * ds)
    elif mode == 'approaching_turn':
        starts = i[1:][(curv[1:] != 0) & (curv[:-1] == 0)]
        starts = starts[(starts * ds >= 10) & (starts * ds <= 70)]
        if len(starts):
            return mode, float(rng.choice(starts) * ds - rng.uniform(2, 14))
    else:
        cand = i[(curv == 0) & (i >= lo) & (i <= hi)]
        if len(cand):
            return mode, float(rng.choice(cand) * ds)
    return None, None


def sample_scene(rng, severity=None):
    for _ in range(50):  # resample the track until the requested car placement exists
        track = centerline(rng, 170)
        pts, heads, curv, ds = track
        mode, s_car = place_car(rng, curv, ds)
        if mode:
            break
    width = rng.uniform(3.0, 4.5)
    cones = boundary_cones(rng, track, width)

    # Remove cones that land on the track where it crosses itself.
    keep = []
    for c in cones:
        i = int(c['s'] / ds)
        far = np.abs(np.arange(len(pts)) - i) * ds > 12
        if not far.any() or dist_to_polyline(c['xy'][None], pts[far])[0] > width / 2 + 0.6:
            keep.append(c)
    cones = keep

    ic = int(s_car / ds)
    ahead = curv[ic:ic + int(15 / ds)]
    turn_k = float(curv[ic] if curv[ic] else (ahead[ahead != 0][0] if (ahead != 0).any() else 0.0))
    extras = dict(start_line=False, orange_lane=False, second_track=False, car_context=mode,
                  turn_radius_m=None if turn_k == 0 else float(1 / abs(turn_k)),
                  turn_direction=None if turn_k == 0 else ('left' if turn_k > 0 else 'right'))

    if rng.random() < 0.3:  # start / finish / timekeeping line with big orange cones
        extras['start_line'] = True
        s_line = s_car + rng.uniform(5, 25)
        cones = [c for c in cones if abs(c['s'] - s_line) > 1.6]
        i = int(s_line / ds)
        for side in (1, -1):
            for ds_ in (-0.6, 0.6):
                j = int((s_line + ds_) / ds)
                n = np.array([-math.sin(heads[j]), math.cos(heads[j])])
                cones.append(dict(cls=1, xy=pts[j] + side * n * (width / 2 + rng.uniform(0, 0.15)), s=s_line + ds_, side=side))

    if rng.random() < 0.2:  # entry/exit lane marked with small orange cones
        extras['orange_lane'] = True
        side = rng.choice([1, -1])
        s_b = s_car + rng.uniform(6, 25)
        cones = [c for c in cones if not (c['side'] == side and s_b - 1 < c['s'] < s_b + 6)]
        i = int(s_b / ds)
        n = np.array([-math.sin(heads[i]), math.cos(heads[i])])
        a = heads[i] + side * math.radians(rng.uniform(15, 35))
        d = np.array([math.cos(a), math.sin(a)])
        nl = np.array([-d[1], d[0]])
        start = pts[i] + side * n * (width / 2 + 0.8)
        for lat in (-1.75, 1.75):
            for t in np.arange(0, rng.uniform(18, 35), 3.0):
                cones.append(dict(cls=2, xy=start + d * t + nl * lat + rng.normal(0, 0.04, 2), s=-1, side=0))

    if rng.random() < 0.45:  # another part of the track, visible further away
        extras['second_track'] = True
        t2 = centerline(rng, 140)
        side = rng.choice([1, -1])
        h0 = heads[ic]
        fwd, nrm = np.array([math.cos(h0), math.sin(h0)]), np.array([-math.sin(h0), math.cos(h0)])
        origin = pts[ic] + fwd * rng.uniform(-10, 30) + nrm * side * rng.uniform(9, 25)
        t2 = transform_track(t2, h0 + math.pi + rng.normal(0, 0.5), origin - fwd * 60)
        c2 = boundary_cones(rng, t2, rng.uniform(3.0, 4.5))
        if c2:
            d2 = dist_to_polyline(np.array([c['xy'] for c in c2]), pts)
            cones += [c for c, d in zip(c2, d2) if d > width / 2 + 2.0]

    # Camera on the car at s_car.
    h = heads[ic] + math.radians(float(np.clip(rng.normal(0, 4), -10, 10)))
    n = np.array([-math.sin(heads[ic]), math.cos(heads[ic])])
    cam_xy = pts[ic] + n * rng.uniform(-0.6, 0.6)
    cam = dict(xy=cam_xy.tolist(), height=float(rng.uniform(0.95, 1.15)), yaw=float(h),
               pitch=float(math.radians(rng.uniform(3, 8))), roll=float(math.radians(rng.normal(0, 0.8))))

    # Keep cones in a widened horizontal frustum between 2 and 60 m.
    f = np.array([math.cos(h), math.sin(h)])
    out = []
    for c in cones:
        v = c['xy'] - cam_xy
        rng_m = float(np.linalg.norm(v))
        az = math.degrees(math.atan2(f[0] * v[1] - f[1] * v[0], f @ v))
        if 2.0 <= rng_m <= 60 and abs(az) < 52:
            out.append(c)

    sev = float(rng.uniform(0, 1)) if severity is None else severity
    rand = lambda lo, hi: float(rng.uniform(lo, hi))
    cone_list = []
    for c in out:
        cone_list.append(dict(
            class_id=int(c['cls']), xy=[float(c['xy'][0]), float(c['xy'][1])], yaw=rand(0, math.pi / 2),
            dirt=float(rng.uniform(0, 1) ** 1.3), wear=float(rng.uniform(0, 1) ** 2), scuff=rand(0, 1),
            fade=float(rng.uniform(0, 0.35) * rng.uniform(0, 1)), value=float(rng.normal(1, 0.07)),
            hue=float(rng.normal(0, 0.012)), seed=rand(0, 1000)))

    hdris = sorted((ASSETS / 'hdri').glob('*.hdr'))
    grounds = sorted(p.name for p in (ASSETS / 'tex').iterdir())
    return dict(
        cones=cone_list, camera=cam, track_width=width, extras=extras,
        hdri=hdris[rng.integers(len(hdris))].name, hdri_rotation=rand(0, 2 * math.pi),
        ground=grounds[rng.integers(len(grounds))], ground_tile_m=rand(2.0, 4.5),
        ground_value=rand(0.75, 1.2), ground_rotation=rand(0, math.pi),
        ground_patch=rand(0.8, 1.15), ground_streak_dir=rand(0, math.pi),
        severity=sev,
        optics=dict(
            fstop=float(4.0 * (1.2 / 4.0) ** (sev * rng.uniform(0.5, 1.0))),
            focus_distance=float(math.exp(rng.uniform(math.log(1.5), math.log(30)))),
        ),
        motion=dict(
            exposure_s=float((0.5 + 3.5 * sev * rng.uniform(0.3, 1.0)) / 1000),
            speed_mps=rand(3, 15),
            angular_rate_rps=rng.normal(0, 0.1 + 0.5 * sev, 3).tolist(),
        ),
    )


# ---------------------------------------------------------------- geometry & materials

def build_cone_mesh(name, spec, segments=96):
    bm = bmesh.new()
    rings = []
    for r, z in spec['profile']:
        if r == 0:
            rings.append([bm.verts.new((0, 0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(a), r * math.sin(a), z))
                          for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)])
    for lo, hi in zip(rings, rings[1:]):
        for j in range(segments):
            k = (j + 1) % segments
            if len(hi) == 1:
                f = bm.faces.new((lo[j], lo[k], hi[0]))
            else:
                f = bm.faces.new((lo[j], lo[k], hi[k], hi[j]))
            f.smooth = True
    # Base plate: rounded square slab.
    half, rc, t = spec['plate_size'] / 2, spec['plate_corner'], spec['plate_thickness']
    outline = []
    for cx, cy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for a in np.linspace(a0, a0 + 90, 7):
            outline.append((cx * (half - rc) + rc * math.cos(math.radians(a)),
                            cy * (half - rc) + rc * math.sin(math.radians(a))))
    bot = [bm.verts.new((x, y, 0)) for x, y in outline]
    top = [bm.verts.new((x, y, t)) for x, y in outline]
    bm.faces.new(top)
    bm.faces.new(list(reversed(bot)))
    for j in range(len(outline)):
        k = (j + 1) % len(outline)
        bm.faces.new((bot[j], bot[k], top[k], top[j]))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


class Nodes:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def _in(self, sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            self.link(v, sock)

    def math(self, op, a, b=0.0, clamp=False):
        n = self.new('ShaderNodeMath', operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs[0]

    def attr(self, name):
        return self.new('ShaderNodeAttribute', attribute_type='OBJECT', attribute_name=name).outputs['Fac']

    def noise(self, vec, scale, detail=4.0, rough=0.55, dims='3D'):
        n = self.new('ShaderNodeTexNoise', noise_dimensions=dims)
        self.link(vec, n.inputs['Vector'])
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        return n.outputs['Fac']

    def maprange(self, v, a, b, c=0.0, d=1.0):
        n = self.new('ShaderNodeMapRange', clamp=True)
        self._in(n.inputs['Value'], v)
        for sock, val in zip(('From Min', 'From Max', 'To Min', 'To Max'), (a, b, c, d)):
            self._in(n.inputs[sock], val)
        return n.outputs['Result']

    def mix(self, fac, a, b):
        n = self.new('ShaderNodeMix', data_type='RGBA', clamp_factor=True)
        self._in(n.inputs['Factor'], fac)
        for sock, v in ((n.inputs[6], a), (n.inputs[7], b)):
            if isinstance(v, tuple):
                sock.default_value = (*v, 1.0)
            else:
                self.link(v, sock)
        return n.outputs[2]


def cone_material(class_id):
    spec = cm.cone_spec(class_id)
    mat = bpy.data.materials.new(f'cone_{cm.CLASS_NAMES[class_id]}')
    N = Nodes(mat)
    out = N.new('ShaderNodeOutputMaterial')
    bsdf = N.new('ShaderNodeBsdfPrincipled')
    N.link(bsdf.outputs[0], out.inputs[0])
    tc = N.new('ShaderNodeTexCoord')
    sep = N.new('ShaderNodeSeparateXYZ')
    N.link(tc.outputs['Object'], sep.inputs[0])
    z = sep.outputs['Z']
    seed = N.new('ShaderNodeVectorMath', operation='ADD')
    N.link(tc.outputs['Object'], seed.inputs[0])
    comb = N.new('ShaderNodeCombineXYZ')
    for i in range(3):
        N.link(N.math('MULTIPLY', N.attr('seed'), 1.0 + 0.37 * i), comb.inputs[i])
    N.link(comb.outputs[0], seed.inputs[1])
    pos = seed.outputs[0]
    dirt, wear, scuff = N.attr('dirt'), N.attr('wear'), N.attr('scuff')

    # Stripes from object-space height; worn stripes peel in patches and fade.
    stripe = None
    for lo, hi in spec['stripes']:
        m = N.math('MULTIPLY', N.math('GREATER_THAN', z, lo), N.math('LESS_THAN', z, hi))
        stripe = m if stripe is None else N.math('MAXIMUM', stripe, m)
    peel_lo = N.math('SUBTRACT', 1.0, N.math('MULTIPLY', wear, 0.42))
    peel = N.maprange(N.noise(pos, 22.0, 5.0, 0.6), peel_lo, N.math('ADD', peel_lo, 0.04))
    stripe = N.math('MULTIPLY', stripe, N.math('SUBTRACT', 1.0, peel))

    hsv = N.new('ShaderNodeHueSaturation')
    hsv.inputs['Color'].default_value = (*CLASS_COLORS[class_id], 1)
    N.link(N.math('ADD', N.attr('hue'), 0.5), hsv.inputs['Hue'])
    N.link(N.math('SUBTRACT', 1.0, N.attr('fade')), hsv.inputs['Saturation'])
    N.link(N.attr('value'), hsv.inputs['Value'])
    base = N.mix(N.math('MULTIPLY', N.attr('fade'), 0.35), hsv.outputs[0], (0.55, 0.52, 0.5))
    stripe_col = N.mix(N.math('MULTIPLY', wear, 0.25), STRIPE_COLORS[class_id], base)
    col = N.mix(stripe, base, stripe_col)

    # Dirt: blotchy grime concentrated low on the cone, plus fine speckle.
    low = N.maprange(z, 0.0, spec['height'] * 0.45, 1.0, 0.25)
    blot = N.maprange(N.noise(pos, 7.0, 8.0, 0.62), 0.38, 0.72)
    dirt_amt = N.math('MULTIPLY', N.math('MULTIPLY', dirt, low), blot, clamp=True)
    col = N.mix(N.math('MULTIPLY', dirt_amt, 0.85), col, (0.085, 0.072, 0.058))
    speck = N.maprange(N.noise(pos, 190.0, 2.0, 0.5), 0.55, 0.7)
    col = N.mix(N.math('MULTIPLY', speck, N.math('MULTIPLY', dirt, 0.35)), col, (0.05, 0.045, 0.04))

    # Scuffs: thin horizontal scrape marks that lighten the plastic.
    mp = N.new('ShaderNodeMapping')
    N.link(pos, mp.inputs['Vector'])
    mp.inputs['Scale'].default_value = (9.0, 9.0, 110.0)
    scr = N.maprange(N.noise(mp.outputs[0], 1.0, 3.0, 0.5), 0.66, 0.74)
    scr = N.math('MULTIPLY', scr, N.math('MULTIPLY', scuff, 0.45))
    col = N.mix(scr, col, (0.6, 0.58, 0.55))

    # Dust film: a thin matte layer over the whole cone.
    dust = N.math('MULTIPLY', N.math('ADD', 0.03, N.math('MULTIPLY', dirt, 0.2)),
                  N.maprange(N.noise(pos, 3.0, 2.0, 0.5), 0.3, 0.7, 0.6, 1.0))
    col = N.mix(dust, col, (0.26, 0.25, 0.23))

    N.link(col, bsdf.inputs['Base Color'])
    rough = N.math('ADD', N.math('ADD', 0.42, N.math('MULTIPLY', dust, 0.6)), N.math('ADD', N.math('MULTIPLY', dirt_amt, 0.4), N.math('MULTIPLY', scr, 0.6)))
    N.link(rough, bsdf.inputs['Roughness'])
    bump = N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.15
    bump.inputs['Distance'].default_value = 0.0005
    N.link(N.noise(pos, 140.0, 3.0, 0.5), bump.inputs['Height'])
    N.link(bump.outputs[0], bsdf.inputs['Normal'])
    return mat


def ground_material(scene):
    mat = bpy.data.materials.new('ground')
    N = Nodes(mat)
    out = N.new('ShaderNodeOutputMaterial')
    bsdf = N.new('ShaderNodeBsdfPrincipled')
    N.link(bsdf.outputs[0], out.inputs[0])
    tc = N.new('ShaderNodeTexCoord')
    mp = N.new('ShaderNodeMapping')
    N.link(tc.outputs['Object'], mp.inputs['Vector'])
    s = 1 / scene['ground_tile_m']
    mp.inputs['Scale'].default_value = (s, s, s)
    mp.inputs['Rotation'].default_value = (0, 0, scene['ground_rotation'])
    tex = ASSETS / 'tex' / scene['ground']

    def img(name, colorspace):
        n = N.new('ShaderNodeTexImage')
        n.image = bpy.data.images.load(str(tex / f'{name}.jpg'), check_existing=True)
        n.image.colorspace_settings.name = colorspace
        N.link(mp.outputs[0], n.inputs['Vector'])
        return n.outputs['Color']

    # Brightness variation at several scales breaks up tiling: broad patches, resurfaced
    # repairs, and dark rubber streaks along a random direction.
    obj = tc.outputs['Object']
    macro = N.maprange(N.noise(obj, 0.06, 3.0, 0.5), 0.3, 0.7, 0.8, 1.18)
    patch = N.maprange(N.noise(obj, 0.45, 2.0, 0.5), 0.55, 0.62, 1.0, scene['ground_patch'])
    sm = N.new('ShaderNodeMapping')
    N.link(obj, sm.inputs['Vector'])
    sm.inputs['Rotation'].default_value = (0, 0, scene['ground_streak_dir'])
    sm.inputs['Scale'].default_value = (0.08, 1.6, 1.0)
    streak = N.maprange(N.noise(sm.outputs[0], 1.0, 3.0, 0.55), 0.6, 0.75, 1.0, 0.72)
    val = N.math('MULTIPLY', N.math('MULTIPLY', macro, patch), N.math('MULTIPLY', streak, scene['ground_value']))
    col = N.new('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    col.inputs['Factor'].default_value = 1.0
    N.link(img('diff', 'sRGB'), col.inputs[6])
    cv = N.new('ShaderNodeCombineColor')
    for i in range(3):
        N.link(val, cv.inputs[i])
    N.link(cv.outputs[0], col.inputs[7])
    N.link(col.outputs[2], bsdf.inputs['Base Color'])
    N.link(img('rough', 'Non-Color'), bsdf.inputs['Roughness'])
    nm = N.new('ShaderNodeNormalMap')
    nm.inputs['Strength'].default_value = 0.8
    N.link(img('nor', 'Non-Color'), nm.inputs['Color'])
    N.link(nm.outputs[0], bsdf.inputs['Normal'])

    # Far away, camera rays see the HDRI's own ground instead of the plane, hiding the seam
    # between the synthetic ground and the photographed horizon.
    geo = N.new('ShaderNodeNewGeometry')
    view = N.new('ShaderNodeVectorMath', operation='SCALE')
    N.link(geo.outputs['Incoming'], view.inputs[0])
    view.inputs['Scale'].default_value = -1.0
    mp2 = N.new('ShaderNodeMapping')
    mp2.inputs['Rotation'].default_value = (0, 0, scene['hdri_rotation'])
    N.link(view.outputs[0], mp2.inputs['Vector'])
    env = N.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(str(ASSETS / 'hdri' / scene['hdri']), check_existing=True)
    N.link(mp2.outputs[0], env.inputs['Vector'])
    emis = N.new('ShaderNodeEmission')
    N.link(env.outputs[0], emis.inputs['Color'])
    far = N.maprange(N.new('ShaderNodeCameraData').outputs['View Distance'], 70.0, 220.0)
    fac = N.math('MULTIPLY', far, N.new('ShaderNodeLightPath').outputs['Is Camera Ray'])
    mix = N.new('ShaderNodeMixShader')
    N.link(fac, mix.inputs[0])
    N.link(bsdf.outputs[0], mix.inputs[1])
    N.link(emis.outputs[0], mix.inputs[2])
    N.link(mix.outputs[0], out.inputs[0])
    return mat


# ---------------------------------------------------------------- camera

def camera_rotation_cv(cam):
    """World-from-camera rotation, OpenCV axes (columns: right, down, forward)."""
    yaw, pitch, roll = cam['yaw'], cam['pitch'], cam['roll']
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    d = np.cross(f, r)
    c, s = math.cos(roll), math.sin(roll)
    r, d = c * r + s * d, -s * r + c * d
    return np.stack([r, d, f], 1)


def small_rotation(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx


def blender_matrix(R_cv, t):
    R_bl = R_cv @ np.diag([1.0, -1.0, -1.0])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R_bl, t
    return Matrix(M.tolist())


# ---------------------------------------------------------------- frame

def setup_render(args):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == 'OPTIX' or (args.cpu and d.type == 'CPU')
    cy = sc.cycles
    cy.device = 'GPU'
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = args.adaptive_threshold
    cy.use_denoising = True
    cy.denoiser = 'OPTIX'
    cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 8, 3, 3
    cy.transmission_bounces, cy.volume_bounces, cy.transparent_max_bounces = 0, 0, 4
    cy.sample_clamp_indirect = 10.0
    sc.render.film_transparent = False
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 1.0
    sc.render.motion_blur_position = 'CENTER'
    sc.view_layers[0].use_pass_object_index = True
    sc.view_settings.view_transform = 'Standard'
    im = sc.render.image_settings
    im.file_format = 'OPEN_EXR_MULTILAYER'
    im.color_depth = '16'
    im.exr_codec = 'ZIP'
    sc.render.resolution_percentage = 100
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.worlds, bpy.data.lights):
        for d in list(coll):
            coll.remove(d)
    for im in list(bpy.data.images):
        if im.type == 'IMAGE':
            bpy.data.images.remove(im)


def render_frame(idx, scene, out_dir, args):
    clear_scene()
    sc = bpy.context.scene
    K, dist = cm.CAMERA['K'], cm.CAMERA['dist']
    W, H = cm.CAMERA['width'], cm.CAMERA['height']
    ml, mt, mr, mb = cm.render_margins(K, dist, W, H)
    Wr, Hr = W + ml + mr, H + mt + mb
    sc.render.resolution_x, sc.render.resolution_y = Wr, Hr

    # World
    world = bpy.data.worlds.new('world')
    sc.world = world
    N = Nodes(world)
    bg = N.new('ShaderNodeBackground')
    wo = N.new('ShaderNodeOutputWorld')
    env = N.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(str(ASSETS / 'hdri' / scene['hdri']))
    tc = N.new('ShaderNodeTexCoord')
    mp = N.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, scene['hdri_rotation'])
    N.link(tc.outputs['Generated'], mp.inputs['Vector'])
    N.link(mp.outputs[0], env.inputs['Vector'])
    N.link(env.outputs[0], bg.inputs[0])
    N.link(bg.outputs[0], wo.inputs[0])

    # Ground
    bpy.ops.mesh.primitive_plane_add(size=1000)
    ground = bpy.context.object
    ground.name = 'ground'
    ground.data.materials.append(ground_material(scene))

    # Cones
    # Shared meshes with one object-linked material slot, so each cone gets its class material.
    meshes = {False: build_cone_mesh('cone_small', cm.SMALL), True: build_cone_mesh('cone_big', cm.BIG)}
    for m in meshes.values():
        m.materials.append(None)
    mats = {}
    cone_objs = []
    for i, c in enumerate(scene['cones']):
        cls = c['class_id']
        if cls not in mats:
            mats[cls] = cone_material(cls)
        o = bpy.data.objects.new(f'cone_{i:03d}', meshes[cls == 1])
        sc.collection.objects.link(o)
        o.material_slots[0].link = 'OBJECT'
        o.material_slots[0].material = mats[cls]
        o.location = (c['xy'][0], c['xy'][1], 0.0)
        o.rotation_euler = (0.0, 0.0, c['yaw'])
        o.pass_index = i + 1
        for k in ('dirt', 'wear', 'scuff', 'fade', 'value', 'hue', 'seed'):
            o[k] = c[k]
        cone_objs.append(o)
    bpy.context.view_layer.update()

    # Camera (ground truth pose = mid-exposure)
    cam = scene['camera']
    R = camera_rotation_cv(cam)
    t = np.array([cam['xy'][0], cam['xy'][1], cam['height']])
    cdata = bpy.data.cameras.new('cam')
    cobj = bpy.data.objects.new('cam', cdata)
    sc.collection.objects.link(cobj)
    sc.camera = cobj
    cdata.sensor_fit = 'HORIZONTAL'
    cdata.lens = cm.CAMERA['focal_length_mm']
    cdata.sensor_width = cdata.lens * Wr / K[0][0]
    cdata.shift_x = (Wr / 2 - (K[0][2] + ml + 0.5)) / Wr
    cdata.shift_y = ((K[1][2] + mt + 0.5) - Hr / 2) / Wr
    cdata.clip_start, cdata.clip_end = 0.05, 500
    cdata.dof.use_dof = True
    cdata.dof.focus_distance = scene['optics']['focus_distance']
    cdata.dof.aperture_fstop = scene['optics']['fstop']
    cdata.dof.aperture_blades = 6

    # Motion during exposure: forward travel plus vibration rotation (camera axes).
    mo = scene['motion']
    dt = mo['exposure_s']
    dR = small_rotation(R @ (np.array(mo['angular_rate_rps']) * dt))
    dtv = R[:, 2] * np.array([1, 1, 0]) * mo['speed_mps'] * dt
    cobj.rotation_mode = 'XYZ'
    cobj.matrix_world = blender_matrix(R, t)
    e_mid, l_mid = Vector(cobj.rotation_euler), Vector(cobj.location)
    cobj.matrix_world = blender_matrix(dR @ R, t + dtv)
    e_end = Euler(cobj.rotation_euler).copy()
    e_end.make_compatible(Euler(e_mid))
    d_e, d_l = Vector(e_end) - e_mid, Vector(cobj.location) - l_mid
    for frame, s in ((0, -1), (2, 1)):
        cobj.rotation_euler = e_mid + s * d_e
        cobj.location = l_mid + s * d_l
        cobj.keyframe_insert('rotation_euler', frame=frame)
        cobj.keyframe_insert('location', frame=frame)
    sc.frame_set(1)

    # Keypoint occlusion by other cones, ray cast from the mid-exposure camera centre.
    dg = bpy.context.evaluated_depsgraph_get()
    T_wc = np.eye(4)
    T_wc[:3, :3], T_wc[:3, 3] = R, t
    cones_gt = []
    for o, c in zip(cone_objs, scene['cones']):
        T_wo = np.array(o.matrix_world)
        cam_in_cone = (np.linalg.inv(T_wo) @ np.append(t, 1))[:3]
        kp = cm.silhouette_points(c['class_id'], cam_in_cone)
        occ = []
        for p in kp:
            if np.isnan(p[0]):
                occ.append(None)
                continue
            q = p.copy()
            q[:2] *= 1 + 0.002 / max(np.linalg.norm(q[:2]), 1e-6)  # nudge just outside the surface
            pw = (T_wo @ np.append(q, 1))[:3]
            v = pw - t
            dlen = float(np.linalg.norm(v))
            hit, _, _, _, hobj, _ = sc.ray_cast(dg, Vector(t), Vector(v / dlen), distance=dlen - 0.003)
            occ.append(bool(hit and hobj is not None and hobj.name != 'ground'))
        cones_gt.append(dict(c, T_world_cone=T_wo.tolist(), pass_index=o.pass_index,
                             keypoints_cone_unordered=[None if np.isnan(p[0]) else p.tolist() for p in kp],
                             keypoint_occluded_unordered=occ))

    stem = f'{idx:06d}'
    sc.render.filepath = str(out_dir / 'raw' / f'{stem}.exr')
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    meta = dict(
        frame=idx, seed=args.seed + idx, render_seconds=time.time() - t0,
        render_size=[Wr, Hr], margins_ltrb=[ml, mt, mr, mb], camera_model=cm.CAMERA,
        T_world_cam=T_wc.tolist(), samples=args.samples,
        scene={k: v for k, v in scene.items() if k != 'cones'}, cones=cones_gt)
    (out_dir / 'raw' / f'{stem}.json').write_text(json.dumps(meta, indent=1))
    print(f'[synth] frame {stem}: {len(cones_gt)} cones, {meta["render_seconds"]:.1f}s', flush=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=1)
    ap.add_argument('--seed', type=int, default=20261013)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--adaptive-threshold', type=float, default=0.015)
    ap.add_argument('--cpu', action='store_true', help='render on the CPU alongside the OptiX GPU')
    ap.add_argument('--severity', type=float, default=None, help='force defect severity 0..1')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    setup_render(args)
    for idx in range(args.start, args.start + args.count):
        if args.skip_existing and (out / 'raw' / f'{idx:06d}.json').exists():
            continue
        rng = np.random.default_rng(args.seed + idx)
        render_frame(idx, sample_scene(rng, args.severity), out, args)


main()
