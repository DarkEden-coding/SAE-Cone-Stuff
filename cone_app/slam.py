"""Cone-only metric SLAM; no image features, inertial data, or supplied poses.

Camera poses are always full SE(3) (hand-held phones pitch, roll and change
height); ``flat_ground`` instead constrains the *map*: landmarks live on the
z = 0 ground plane. The world is levelled once from the first frame's cone
ground plane, so +z is up in both modes.

Measurements are cone silhouette keypoints. Every residual splits a left/right
ring pair into its midpoint (the cone axis, robust to motion blur and keypoint
width bias) and a down-weighted half-width, normalised by a per-cone pixel
sigma that grows with apparent size. Association is done in pixel space with
scale-free gates (fractions of the cone's apparent size).

Tracking per frame: hypotheses from constant velocity and the last pose are
associated (Hungarian), refined with a robust Gauss-Newton on keypoints and
re-associated. When they explain too few detections, association-free
relocalisation generates similarity-aligned hypotheses from class-compatible
pairs of cones (horizontal pair distances, the camera's up axis from the
frame's own cone plane or the prediction), scored by bearing/range agreement
over all detections, and the best few are refined the same way. Tentative
landmarks need three sightings; keyframe bundle adjustment runs in bounded
windows and globally; nearby, never-coobserved duplicates are consolidated.
After mapping, every non-keyframe is re-localised forwards and backwards
against the final map, which recovers frames lost online.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .geometry import (CLASS_NAMES, ConeMeasurement, _world_transform, fit_ground_plane,
                       n_keypoints, ring_model, solve_cone)

_UP = np.array([0., 0., 1.])
_RINGS = np.zeros((5, 8, 2))
for _cls in CLASS_NAMES:
    _RINGS[_cls, :n_keypoints(_cls)] = np.repeat(ring_model(_cls), 2, axis=0)
_SIGNS = np.tile([-1., 1.], 4)

_HALF_WEIGHT = float(os.environ.get('HW', .35))      # half-width residuals are blur/bias prone
_SINGLE_WEIGHT = .7     # lone keypoint of a ring pair
_ROBUST = 2.5           # Cauchy scale in sigma units
_MIN_INLIERS = 2       # only with a consistent motion prior; otherwise 3+
_FLOOR = .02          # association pixel floor as a fraction of focal length


@dataclass
class _Landmark:
    id: int
    class_id: int
    p: np.ndarray
    covariance: np.ndarray
    first: int
    last: int
    n: int = 1
    information: np.ndarray = field(default_factory=lambda: np.eye(3))
    rhs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: bool = False


@dataclass
class _Observation:
    frame: int
    landmark: int
    class_id: int
    pixels: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    sigma: float = 1.


@dataclass
class _Detections:
    """Per-frame detections usable for tracking, as arrays."""
    ids: list
    classes: np.ndarray
    pixels: np.ndarray      # (n, 8, 2)
    valid: np.ndarray       # (n, 8)
    sigma: np.ndarray       # (n,)
    scale: np.ndarray       # (n,) apparent size in pixels
    truncated: np.ndarray   # (n,) box touches the image border


def _compatible(a, b):
    return (a == b) | (a == 3) | (b == 3)


def _pixels(poses: np.ndarray, points: np.ndarray, classes: np.ndarray, K: np.ndarray, up: np.ndarray = _UP) -> np.ndarray:
    """Vectorized version of geometry.project_cone_keypoints for optimizer batches."""
    ring = _RINGS[classes]
    ray = points - poses[:, :3, 3]
    lateral = np.cross(up, ray)
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-10)
    # First tangent must be image-left even under large roll/yaw.
    x_direction = np.einsum('ni,ni->n', lateral, poses[:, :3, 0])
    lateral *= np.where(x_direction >= 0, 1., -1.)[:, None]
    world = (points[:, None, :] + ring[:, :, :1] * up +
             ring[:, :, 1:] * _SIGNS[None, :, None] * lateral[:, None, :])
    cam = np.einsum('nki,nij->nkj', world - poses[:, None, :3, 3], poses[:, :3, :3])
    homogeneous = np.einsum('nki,nji->nkj', cam, K)
    return homogeneous[:, :, :2] / np.maximum(homogeneous[:, :, 2:], .05)


def _terms(pred, pixels, valid, sigma):
    """(n, 16) normalised residuals: per ring pair midpoint and half-width."""
    d = pred - pixels
    left, right = d[:, 0::2], d[:, 1::2]
    vl, vr = valid[:, 0::2, None], valid[:, 1::2, None]
    both = vl & vr
    first = np.where(both, (left + right) / 2, np.where(vl, left, right) * _SINGLE_WEIGHT)
    second = np.where(both, (right - left) / 2 * _HALF_WEIGHT, 0.)
    return np.concatenate([first, second], axis=2).reshape(len(d), 16) / sigma[:, None]


def _term_mask(valid):
    vl, vr = valid[:, 0::2], valid[:, 1::2]
    either, both = vl | vr, vl & vr
    return np.stack([either, either, both, both], axis=2).reshape(len(valid), 16)


def _parameters(T: np.ndarray) -> np.ndarray:
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def _pose(parameters: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = parameters[:3]
    T[:3, :3] = Rotation.from_rotvec(parameters[3:]).as_matrix()
    return T


def _perturb(T, deltas):
    """Right perturbation in the camera frame: (B, 6) -> (B, 4, 4)."""
    out = np.repeat(T[None], len(deltas), axis=0)
    out[:, :3, :3] = T[:3, :3] @ Rotation.from_rotvec(deltas[:, 3:]).as_matrix()
    out[:, :3, 3] = T[:3, 3] + deltas[:, :3] @ T[:3, :3].T
    return out


def _cauchy(r, c=_ROBUST):
    return float(np.sum(np.log1p((r / c) ** 2)))


def _measurement_world(m: ConeMeasurement, T: np.ndarray, flat: bool, K: np.ndarray):
    R = T[:3, :3]
    p = R @ m.p_cam + T[:3, 3]
    covariance = R @ m.cov_cam @ R.T
    if flat:
        ray = R @ m.p_cam
        if ray[2] < -.02 and T[2, 3] > .2:
            p = T[:3, 3] - ray * T[2, 3] / ray[2]
            distance = np.linalg.norm(p - T[:3, 3])
            direction = p[:2] - T[:2, 3]
            direction /= max(np.linalg.norm(direction), 1e-9)
            axial = max(.03, distance ** 2 * max(1., m.reproj_rms_px) / (K[0, 0] * T[2, 3]))
            lateral = max(.015, distance * max(1., m.reproj_rms_px) / K[0, 0])
            covariance = np.eye(3) * lateral ** 2
            covariance[:2, :2] += axial ** 2 * np.outer(direction, direction)
        p[2] = 0.
        covariance[2, 2] = .005 ** 2
    return p, covariance


def _prepare(frame, kp_conf):
    K = np.asarray(frame['K'], float)
    width, height = 2 * K[0, 2], 2 * K[1, 2]
    ids, classes, pixels, valid, scale, truncated = [], [], [], [], [], []
    for det in _deduplicated(frame['detections']):
        cls = int(det['class_id'])
        kp = np.asarray(det['keypoints'], float)[:n_keypoints(cls)]
        px, ok = np.zeros((8, 2)), np.zeros(8, bool)
        px[:len(kp)] = np.nan_to_num(kp[:, :2])
        ok[:len(kp)] = (kp[:, 2] >= kp_conf) & np.isfinite(kp[:, :2]).all(axis=1)
        if ok.sum() < 3:
            continue
        extent = float(np.linalg.norm(np.ptp(px[ok], axis=0)))
        box = det.get('box_xyxy')
        edge = False
        if box is not None:
            margin = 4.
            edge = bool(box[0] < margin or box[1] < margin or box[2] > width - margin or box[3] > height - margin)
        ids.append(det['det_id'])
        classes.append(cls)
        pixels.append(px)
        valid.append(ok)
        scale.append(max(extent, 8.))
        truncated.append(edge)
    scale = np.array(scale)
    sigma = 1.5 + float(os.environ.get('SIGK', .02)) * scale
    truncated = np.array(truncated, bool)
    sigma = np.where(truncated, sigma * 3., sigma)
    return _Detections(ids, np.array(classes, int), np.array(pixels).reshape(-1, 8, 2),
                       np.array(valid, bool).reshape(-1, 8), sigma, scale, truncated)


def _deduplicated(detections, iou=.3):
    """Drop overlapping boxes of compatible classes (detector duplicates)."""
    order = sorted(detections, key=lambda d: -float(d.get('conf', 1.)))
    kept = []
    for det in order:
        box = det.get('box_xyxy')
        duplicate = False
        if box is not None:
            for other in kept:
                ob = other.get('box_xyxy')
                if ob is None or not _compatible(det['class_id'], other['class_id']):
                    continue
                w = min(box[2], ob[2]) - max(box[0], ob[0])
                h = min(box[3], ob[3]) - max(box[1], ob[1])
                inter = max(w, 0.) * max(h, 0.)
                union = (box[2] - box[0]) * (box[3] - box[1]) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter
                if union > 0 and inter / union > iou:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return [d for d in detections if any(d is k for k in kept)]


class _Problem:
    """Motion-only refinement problem over a fixed association."""

    def __init__(self, dets, rows, points, classes, K, prior=None, inflation=None):
        self.prior = self.mean_height = None
        if prior is not None:
            reference, dt, self.mean_height = prior
            dt = min(abs(dt), 5.)
            # Hand-held tilt and height change smoothly; soft priors keep
            # few-cone or collinear frames well posed.
            self.prior = (reference[2, :3].copy(), float(reference[2, 3]),
                          _TILT_SIGMA[0] + _TILT_SIGMA[1] * dt, _HEIGHT_SIGMA[0] + _HEIGHT_SIGMA[1] * dt)
        self.points, self.classes = points, classes
        self.K = np.broadcast_to(K, (len(points), 3, 3))
        self.pixels, self.valid, self.sigma = dets.pixels[rows], dets.valid[rows], dets.sigma[rows]
        if inflation is not None:
            self.sigma = self.sigma * inflation
        self.mask = _term_mask(self.valid)
        self.n = len(points)

    def residuals(self, poses):
        B, n = len(poses), self.n
        pred = _pixels(np.repeat(poses, n, axis=0), np.tile(self.points, (B, 1)), np.tile(self.classes, B),
                       np.tile(self.K, (B, 1, 1)))
        terms = _terms(pred, np.tile(self.pixels, (B, 1, 1)), np.tile(self.valid, (B, 1)),
                       np.tile(self.sigma, B)).reshape(B, n, 16)
        out = terms[:, self.mask]
        if self.prior is not None:
            up, height, tilt_sigma, height_sigma = self.prior
            out = np.concatenate([out, (poses[:, 2, :3] - up) / tilt_sigma,
                                  ((poses[:, 2, 3] - height) / height_sigma)[:, None]], axis=1)
            if self.mean_height is not None:
                out = np.concatenate([out, ((poses[:, 2, 3] - self.mean_height) / _MEAN_HEIGHT_SIGMA)[:, None]], axis=1)
        return out

    def per_cone_error(self, T):
        pred = _pixels(np.broadcast_to(T, (self.n, 4, 4)), self.points, self.classes, self.K)
        terms = _terms(pred, self.pixels, self.valid, self.sigma).reshape(self.n, 4, 4)
        mid = np.sum(terms[:, :, :2] ** 2, axis=2)
        either = self.valid[:, 0::2] | self.valid[:, 1::2]
        return np.sqrt(np.sum(mid * either, axis=1) / np.maximum(1, either.sum(axis=1)))


_EPS = np.r_[np.full(3, 1e-4), np.full(3, 1e-5)]
_EPS_BA = _EPS


def _refine(T, problem, iterations=15):
    """Levenberg-Marquardt with Cauchy IRLS weights on a right SE(3) perturbation."""
    r = problem.residuals(T[None])[0]
    cost = _cauchy(r)
    lam = 1e-3
    steps = np.vstack([np.zeros(6), np.diag(_EPS)])
    for _ in range(iterations):
        rs = problem.residuals(_perturb(T, steps))
        r = rs[0]
        J = (rs[1:] - r) / _EPS[:, None]
        w = 1. / (1. + (r / _ROBUST) ** 2)
        H = (J * w) @ J.T
        g = (J * w) @ r
        improved = False
        for _ in range(6):
            dx = -np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-9 * np.eye(6), g)
            candidate = _perturb(T, dx[None])[0]
            new_cost = _cauchy(problem.residuals(candidate[None])[0])
            if new_cost < cost:
                T, improved = candidate, True
                lam = max(lam / 3, 1e-7)
                break
            lam *= 4
        if not improved or (abs(cost - new_cost) < 1e-4 * max(cost, 1.) and np.abs(dx).max() < 1e-4):
            if improved:
                cost = new_cost
            break
        cost = new_cost
    return T, cost


class _MapView:
    def __init__(self, landmarks, minimum_n=0):
        selected = [lm for lm in landmarks.values() if lm.n >= minimum_n]
        self.ids = [lm.id for lm in selected]
        self.points = np.array([lm.p for lm in selected]).reshape(-1, 3)
        self.classes = np.array([lm.class_id for lm in selected], int)
        self.n = np.array([lm.n for lm in selected], int)
        # Young points carry PnP range bias; let established ones dominate.
        self.inflation = np.where((self.n >= 3) | np.array([lm.ba for lm in selected], bool), 1., _TENTATIVE_INFLATION)


def _subset(view, keep):
    out = _MapView({})
    out.ids = [view.ids[i] for i in keep]
    out.points, out.classes = view.points[keep], view.classes[keep]
    out.n, out.inflation = view.n[keep], view.inflation[keep]
    return out


def _associate(T, dets, view, K, gate, shift=False):
    """Hungarian association in pixel space; returns (rows, cols, cost).

    With ``shift`` the dominant common image offset (a small camera rotation,
    the typical hand-held prediction error) is found by voting first.
    """
    N, M = len(dets.ids), len(view.ids)
    empty = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    if not N or not M:
        return empty
    cam = (view.points - T[:3, 3]) @ T[:3, :3]
    # Cheap field-of-view prefilter (generous: the pose may be off).
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]
        v = cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]
    width, height = 2 * K[0, 2], 2 * K[1, 2]
    near = ((cam[:, 2] > .3) & (u > -.5 * width) & (u < 1.5 * width) & (v > -.5 * height) & (v < 1.5 * height))
    if not near.any():
        return empty
    if not near.all():
        index = np.nonzero(near)[0]
        rows, cols, cost = _associate(T, dets, _subset(view, index), K, gate, shift)
        return rows, index[cols], cost
    depth = cam[:, 2]
    pred = _pixels(np.broadcast_to(T, (M, 4, 4)), view.points, view.classes, np.broadcast_to(K, (M, 3, 3)))
    d = pred[None] - dets.pixels[:, None]                      # (N, M, 8, 2)
    vl, vr = dets.valid[:, None, 0::2, None], dets.valid[:, None, 1::2, None]
    first = np.where(vl & vr, (d[:, :, 0::2] + d[:, :, 1::2]) / 2, np.where(vl, d[:, :, 0::2], d[:, :, 1::2]))
    either = (dets.valid[:, 0::2] | dets.valid[:, 1::2])[:, None]
    normaliser = dets.scale[:, None] + _FLOOR * K[0, 0]
    if shift:
        offsets = np.sum(first * either[..., None], axis=2) / np.maximum(1, either.sum(axis=2))[..., None]
        plausible = _compatible(dets.classes[:, None], view.classes[None]) & (depth[None] > .3)
        ci, cj = np.nonzero(plausible)
        if len(ci):
            candidates = np.vstack([np.zeros((1, 2)), offsets[ci, cj]])
            # support[c, k] = det k has a landmark within tolerance under candidate c
            residual = np.linalg.norm(offsets[None] - candidates[:, None, None], axis=3) / normaliser[None]
            residual = np.where(plausible[None], residual, np.inf)
            support = np.sum(np.maximum(0, 1 - residual.min(axis=2) / .6), axis=1)
            best = int(np.argmax(support))
            if support[best] > support[0] + .5:
                first = first - candidates[best]
    error = np.sqrt(np.sum(np.sum(first ** 2, axis=3) * either, axis=2) / np.maximum(1, either.sum(axis=2)))
    # Normalise by apparent size plus a focal-relative floor: rotation errors
    # shift every cone by the same number of pixels.
    cost = error / normaliser
    # Apparent-size consistency is a scale-free depth gate.
    masked = np.where(dets.valid[:, None, :, None], pred[None], np.nan)
    extent = np.linalg.norm(np.nanmax(masked, axis=2) - np.nanmin(masked, axis=2), axis=2)
    ratio = extent / dets.scale[:, None]
    allowed = (_compatible(dets.classes[:, None], view.classes[None]) & (depth[None] > .3) &
               (cost < gate) & (ratio > .5) & (ratio < 2.))
    total = np.where(allowed, cost + .1 * np.abs(np.log(np.maximum(ratio, 1e-3))), 1e6)
    rows, cols = linear_sum_assignment(total)
    keep = total[rows, cols] < 1e6
    return rows[keep], cols[keep], cost[rows[keep], cols[keep]]


def _localize(T0, dets, view, K, prior=None, gates=(1., .45, .3)):
    """Associate, refine and re-associate from one hypothesis."""
    T = T0
    rows = cols = None
    for step, gate in enumerate(gates):
        r, c, _ = _associate(T, dets, view, K, gate, shift=step == 0)
        if len(r) < _MIN_INLIERS:
            return None
        if rows is not None and np.array_equal(r, rows) and np.array_equal(c, cols):
            break
        rows, cols = r, c
        problem = _Problem(dets, rows, view.points[cols], view.classes[cols], K, prior, view.inflation[cols])
        T, _ = _refine(T, problem)
    rows, cols, _ = _associate(T, dets, view, K, gates[-1])
    if len(rows) < _MIN_INLIERS:
        return None
    problem = _Problem(dets, rows, view.points[cols], view.classes[cols], K, prior, view.inflation[cols])
    errors = problem.per_cone_error(T)
    good = errors < 3.
    if good.sum() < _MIN_INLIERS:
        return None
    if not good.all():
        rows, cols = rows[good], cols[good]
        problem = _Problem(dets, rows, view.points[cols], view.classes[cols], K, prior, view.inflation[cols])
        T, _ = _refine(T, problem, 8)
        errors = problem.per_cone_error(T)
    visible = _visible(T, view, K)
    missing = int(np.sum(visible & (view.n >= 3))) - int(np.sum(visible[cols] & (view.n[cols] >= 3)))
    return dict(T=T, rows=rows, cols=cols, n=len(rows), error=float(np.median(errors)),
                truncated=int(dets.truncated[rows].sum()), missing=missing,
                visible=int(np.sum(visible & (view.n >= 3))))


def _visible(T, view, K):
    """Landmarks a detector should clearly see from T (negative evidence)."""
    if not len(view.ids):
        return np.zeros(0, bool)
    cam = (view.points - T[:3, 3]) @ T[:3, :3]
    depth = cam[:, 2]
    width, height = 2 * K[0, 2], 2 * K[1, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * cam[:, 0] / depth + K[0, 2]
        v = K[1, 1] * cam[:, 1] / depth + K[1, 2]
        size = K[1, 1] * .2 / depth
    margin = .05 * width
    return ((depth > .5) & (u > margin) & (u < width - margin) & (v > margin) & (v < height - margin) &
            (size > _VISIBLE_PX))


_TILT_SIGMA = (.02, .5)      # rad, rad/s
_HEIGHT_SIGMA = (.03, .5)    # m, m/s
_FUSE_UNTIL = int(os.environ.get('FUSE', 3))
_BA_EVERY = int(os.environ.get('BAE', 3))
_KEYFRAME_S = float(os.environ.get('KFS', .5))
_KEYFRAME_M = float(os.environ.get('KFM', .5))
_TENTATIVE_INFLATION = float(os.environ.get('TINF', 2.))
_JUMP_FRACTION = float(os.environ.get('JUMPF', .7))
_PROBATION = 3
_VISIBLE_PX = 30.         # apparent ring span that is reliably detected
_MEAN_HEIGHT_SIGMA = float(os.environ.get('MHS', .15))  # flat mode: operator height about its ~3 s mean
_HEIGHT_TAU = 3.
_SAME_CONE_M = .15
_CONFIRM_DENSITY = .05
_GAP_S = .35             # longest neighbour gap bridged by _fill_gaps
_GLOBAL_GROWTH = float(os.environ.get('GG', 2.))
_RELOC_SEEDS = 7
_RELOC_MAX_HYPOTHESES = 3000
_RELOC_MAX_LANDMARKS = 40
_STRONG = 5             # inliers that justify a pose far from the motion prior


def _consistent(T, prior):
    """Is T within the hand-held motion bound of prior = (pose, seconds since it)?"""
    if prior is None:
        return False
    reference, dt = prior[:2]
    return bool(np.linalg.norm(T[:3, 3] - reference[:3, 3]) < float(os.environ.get("CONS_M", .6)) + 1.5 * dt and
                _rotation_angle(T[:3, :3] @ reference[:3, :3].T) < .2 + 2.5 * dt)


def _score(result):
    if result is None:
        return -1e9
    return (result['n'] + .75 * result['consistent'] - result['truncated'] * .5 - .1 * result['error'] -
            .5 * result['missing'])


def _admissible(result, usable):
    if result is None:
        return False
    if result['n'] < _STRONG and result['missing'] > result['n']:
        # Too few cones to outvote the many mapped cones that should be seen.
        return False
    if result['consistent']:
        return result['n'] >= 3 or (result['n'] >= 2 and result['error'] < 1.5)
    # A jump away from the motion prior needs broad support.
    return (result['n'] >= max(_STRONG, int(np.ceil(_JUMP_FRACTION * usable))) and
            result['missing'] <= .25 * max(result['visible'], 1))


def _level(u):
    """Rotation taking unit vector u onto +z (minimal)."""
    u = u / np.linalg.norm(u)
    axis = np.cross(u, _UP)
    s, c = np.linalg.norm(axis), u[2]
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def _relocalize_hypotheses(ms, dets, view, ups, flat, top=6):
    """Association-free pose hypotheses from class-compatible pairs of cones."""
    index = [i for i, det_id in enumerate(dets.ids) if det_id in ms and not dets.truncated[i]]
    if len(index) < 2 or len(view.ids) < 2:
        return []
    q = np.array([ms[dets.ids[i]].p_cam for i in index])
    qc = dets.classes[index]
    rng_q = np.linalg.norm(q, axis=1)
    P, Pc = view.points, view.classes
    # Seed pairs come from the nearest well-fitted cones; all cones score.
    quality = rng_q + np.array([5. if ms[dets.ids[i]].reproj_rms_px > 4. else 0. for i in index])
    seeds = np.sort(np.argsort(quality)[:_RELOC_SEEDS])
    I, J = np.triu_indices(len(seeds), 1)
    I, J = seeds[I], seeds[J]
    A, B = np.nonzero(~np.eye(len(P), dtype=bool))
    pair_ok = (_compatible(qc[I][:, None], Pc[A][None]) & _compatible(qc[J][:, None], Pc[B][None]))
    hypotheses = []
    for u in ups:
        R1 = _level(u)
        ql = q @ R1.T
        dq = ql[J] - ql[I]
        dP = P[B] - P[A]
        hq, hP = np.linalg.norm(dq[:, :2], axis=1), np.linalg.norm(dP[:, :2], axis=1)
        scale = hP[None] / np.maximum(hq[:, None], 1e-6)
        ok = pair_ok & (hq[:, None] > .3) & (hP[None] > .3) & (scale > .6) & (scale < 1.6)
        if not flat:
            ok &= np.abs(dq[:, None, 2] * scale - dP[None, :, 2]) < .3 + .2 * hP[None]
        pi, pa = np.nonzero(ok)
        if not len(pi):
            continue
        if len(pi) > _RELOC_MAX_HYPOTHESES:
            pick = np.random.default_rng(len(pi)).choice(len(pi), _RELOC_MAX_HYPOTHESES, replace=False)
            pi, pa = pi[pick], pa[pick]
        yaw = (np.arctan2(dP[pa, 1], dP[pa, 0]) - np.arctan2(dq[pi, 1], dq[pi, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.zeros((len(pi), 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.
        R = Rz @ R1
        k = scale[pi, pa]
        centre_q = (q[I[pi]] + q[J[pi]]) / 2
        centre_P = (P[A[pa]] + P[B[pa]]) / 2
        t = centre_P - k[:, None] * np.einsum('hij,hj->hi', R, centre_q)
        if flat:
            # Camera height follows from the levelled cone heights.
            t[:, 2] = -k * np.mean(ql[:, 2])
        world = k[:, None, None] * np.einsum('hij,nj->hni', R, q) + t[:, None]
        delta = world[:, :, None] - P[None, None]
        dims = 2 if flat else 3
        distance = np.linalg.norm(delta[..., :dims], axis=3)
        tolerance = (.2 * rng_q + .2)[None, :, None]
        compatible = _compatible(qc[:, None], Pc[None])[None]
        close = np.where(compatible, distance / tolerance, 1e3).min(axis=2)
        score = np.sum(np.maximum(0, 1 - close ** 2), axis=1)
        order = np.argsort(-score)[:200]
        for h in order:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R[h], t[h]
            hypotheses.append((score[h], T))
    hypotheses.sort(key=lambda x: -x[0])
    chosen = []
    for value, T in hypotheses:
        if all(np.linalg.norm(T[:3, 3] - other[:3, 3]) > .3 or
               np.linalg.norm(Rotation.from_matrix(T[:3, :3].T @ other[:3, :3]).as_rotvec()) > .1
               for _, other in chosen):
            chosen.append((value, T))
        if len(chosen) >= top:
            break
    return [T for _, T in chosen]


def _track(fi, frame, prep, ms, landmarks, hypotheses, flat, prior=None, force_reloc=False):
    K = np.asarray(frame['K'], float)
    view = _MapView(landmarks)
    best = None
    usable = int((~prep.truncated).sum())

    seen = []

    def consider(T0, source):
        nonlocal best
        candidate = _localize(T0, prep, view, K, prior)
        if candidate is None:
            return
        seen.append(candidate)
        candidate['consistent'] = _consistent(candidate['T'], prior)
        candidate['source'] = source
        if os.environ.get('CANDDBG') and fi in eval(os.environ['CANDDBG']):
            print('CAND', fi, source, candidate['n'], len(prep.ids), usable, round(candidate['error'], 2), candidate['consistent'],
                  candidate['missing'], candidate['visible'], _admissible(candidate, usable), candidate['T'][:3, 3].round(2),
                  None if prior is None else prior[0][:3, 3].round(2))
        if _admissible(candidate, usable) and _score(candidate) > _score(best):
            best = candidate

    expected = max(_MIN_INLIERS, int(.7 * min(len(prep.ids), len(view.ids))))
    for T0 in hypotheses:
        if T0 is not None:
            consider(T0, 'motion')
            if best is not None and best['n'] >= max(expected, len(prep.ids) - 1) and best['error'] < 1.:
                break
    if force_reloc or best is None or best['n'] < expected or best['error'] > 1.5:
        confirmed = _MapView(landmarks, 3)
        reloc_view = confirmed if len(confirmed.ids) >= 3 else view
        if prior is not None and len(reloc_view.ids) > _RELOC_MAX_LANDMARKS:
            # Only the neighbourhood the camera can have reached is searched.
            distance = np.linalg.norm(reloc_view.points - prior[0][:3, 3], axis=1)
            reach = max([np.linalg.norm(m.p_cam) for m in ms.values()] + [0.]) + 2. + 1.5 * min(abs(prior[1]), 20.)
            keep = np.argsort(distance)[:_RELOC_MAX_LANDMARKS]
            keep = keep[distance[keep] < reach]
            reloc_view = _subset(reloc_view, keep)
        ups = []
        good = [m for m in ms.values() if m.reproj_rms_px < 6.]
        plane = fit_ground_plane([m.p_cam for m in good], [1 / max(.01, np.trace(m.cov_cam)) for m in good]) \
            if len(good) >= 3 else None
        if plane is not None:
            ups.append(plane[0])
        for T0 in hypotheses:
            if T0 is not None:
                ups.append(T0[2, :3].copy())
                break
        if not ups:
            ups.append(np.array([0., -1., 0.]))
        for T0 in _relocalize_hypotheses(ms, prep, reloc_view, ups, flat):
            consider(T0, 'reloc')
    if best is None:
        return None
    if not best['consistent']:
        # Cone layouts are often periodic (track boundaries): a jump is only
        # trusted when no clearly different pose explains nearly as much.
        for other in seen:
            if (other['n'] >= best['n'] - 2 and
                    (np.linalg.norm(other['T'][:3, 3] - best['T'][:3, 3]) > 1. or
                     _rotation_angle(other['T'][:3, :3] @ best['T'][:3, :3].T) > .35)):
                return None
    best['assoc'] = {prep.ids[r]: view.ids[c] for r, c in zip(best['rows'], best['cols'])}
    return best


def _observations(frame_number, frame, assoc, prep):
    out = []
    K = np.asarray(frame['K'], float)
    for i, det_id in enumerate(prep.ids):
        if det_id in assoc:
            out.append(_Observation(frame_number, assoc[det_id], int(prep.classes[i]), prep.pixels[i],
                                    prep.valid[i], K, float(prep.sigma[i])))
    return out


def _soft_l1(r):
    z = (r / _ROBUST) ** 2
    return float(np.sum(2 * (np.sqrt(1 + z) - 1))) * _ROBUST ** 2


def _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=False, max_nfev=None):
    """Sparse Levenberg-Marquardt BA with the camera blocks eliminated.

    Poses use a world-frame perturbation (R <- exp(w) R, t <- t + v). Each
    camera block is inverted independently, leaving a small dense Schur system
    over the landmarks (the map has tens of cones, the trajectory hundreds of
    keyframes). The first keyframe fixes the gauge; in flat global passes only
    its ground position and yaw are pinned, so its tilt and height adapt.
    """
    selected = set(keyframes)
    obs = [o for o in observations if o.frame in selected and o.landmark in landmarks]
    if len(obs) < 6 or len(keyframes) < 2:
        return
    counts = {}
    for o in obs:
        counts[o.landmark] = counts.get(o.landmark, 0) + 1
    # A single-view point is not constrained enough to belong in BA.
    obs = [o for o in obs if counts[o.landmark] >= 2]
    if not obs:
        return
    anchor = keyframes[0]
    partial_anchor = flat and global_pass and not os.environ.get('NOPART')
    frames_used = sorted({o.frame for o in obs})
    free_frames = [f for f in frames_used if f != anchor or partial_anchor]
    if not free_frames:
        return
    pose_index = {f: i for i, f in enumerate(free_frames)}
    landmark_ids = sorted({o.landmark for o in obs})
    lm_index = {j: i for i, j in enumerate(landmark_ids)}
    ld = 2 if flat else 3
    P, L, n = len(free_frames), len(landmark_ids), len(obs)
    poses = np.array([results[f]['T_world_cam'] for f in free_frames])
    fixed = {f: results[f]['T_world_cam'] for f in frames_used if f not in pose_index}
    points = np.array([landmarks[j].p for j in landmark_ids], float)
    if flat:
        points[:, 2] = 0.
    oi = np.array([pose_index.get(o.frame, -1) for o in obs])
    oj = np.array([lm_index[o.landmark] for o in obs])
    fixed_poses = np.array([fixed.get(o.frame, np.eye(4)) for o in obs])
    classes, Ks = np.array([o.class_id for o in obs]), np.array([o.K for o in obs])
    pixels, valid = np.array([o.pixels for o in obs]), np.array([o.valid for o in obs])
    sigma = np.array([o.sigma for o in obs])
    mask = _term_mask(valid).astype(float)
    free = oi >= 0
    # Pairs of observations sharing a free camera feed the Schur complement.
    by_pose = [np.nonzero(oi == i)[0] for i in range(P)]
    pa = np.concatenate([np.repeat(g, len(g)) for g in by_pose])
    pb = np.concatenate([np.tile(g, len(g)) for g in by_pose])

    def observation_poses(poses_):
        out = fixed_poses.copy()
        out[free] = poses_[oi[free]]
        return out

    def residuals(poses_, points_):
        return _terms(_pixels(observation_poses(poses_), points_[oj], classes, Ks), pixels, valid, sigma) * mask

    def perturb(poses_, dp):
        out = poses_.copy()
        out[:, :3, :3] = Rotation.from_rotvec(dp[:, 3:]).as_matrix() @ poses_[:, :3, :3]
        out[:, :3, 3] = poses_[:, :3, 3] + dp[:, :3]
        return out

    gauge = np.zeros(6)
    if partial_anchor and anchor in pose_index:
        gauge[[0, 1, 5]] = 1e8   # first keyframe: ground position and yaw
    r = residuals(poses, points)
    cost = _soft_l1(r)
    lam = 1e-3
    iterations = max_nfev or (int(os.environ.get('GIT', 15)) if global_pass else int(os.environ.get('LIT', 6)))
    H_ll_final = None
    for _ in range(iterations):
        w = 1. / np.sqrt(1. + (r / _ROBUST) ** 2) * mask
        # Batched forward-difference Jacobians (n, 16, k).
        Jp = np.empty((n, 16, 6))
        for k in range(6):
            dp = np.zeros((P, 6))
            dp[:, k] = _EPS_BA[k]
            Jp[:, :, k] = (residuals(perturb(poses, dp), points) - r) / _EPS_BA[k]
        Jp[~free] = 0.
        Jl = np.empty((n, 16, ld))
        for k in range(ld):
            moved = points.copy()
            moved[:, k] += 1e-4
            Jl[:, :, k] = (residuals(poses, moved) - r) / 1e-4
        WJp, WJl = Jp * w[..., None], Jl * w[..., None]
        Hpp = np.zeros((P, 6, 6))
        gp = np.zeros((P, 6))
        np.add.at(Hpp, oi[free], np.einsum('nri,nrj->nij', WJp[free], Jp[free]))
        np.add.at(gp, oi[free], np.einsum('nri,nr->ni', WJp[free], r[free]))
        Hll = np.zeros((L, ld, ld))
        gl = np.zeros((L, ld))
        np.add.at(Hll, oj, np.einsum('nri,nrj->nij', WJl, Jl))
        np.add.at(gl, oj, np.einsum('nri,nr->ni', WJl, r))
        Hpl = np.einsum('nri,nrj->nij', WJp, Jl)            # per observation
        if partial_anchor and anchor in pose_index:
            Hpp[pose_index[anchor], range(6), range(6)] += gauge
        while True:
            Dp = Hpp + lam * Hpp[:, range(6), range(6)][:, :, None] * np.eye(6) + 1e-9 * np.eye(6)
            Dl = Hll + lam * Hll[:, range(ld), range(ld)][:, :, None] * np.eye(ld) + 1e-9 * np.eye(ld)
            Dp_inv = np.linalg.inv(Dp)
            S = np.zeros((L * ld, L * ld))
            for j in range(L):
                S[j * ld:(j + 1) * ld, j * ld:(j + 1) * ld] = Dl[j]
            if len(pa):
                Y = np.einsum('nij,njk->nik', Dp_inv[oi[pa]], Hpl[pb])          # (pairs, 6, ld)
                blocks = np.einsum('nji,njk->nik', Hpl[pa], Y)                   # (pairs, ld, ld)
                rows = (oj[pa][:, None, None] * ld + np.arange(ld)[None, :, None])
                cols = (oj[pb][:, None, None] * ld + np.arange(ld)[None, None, :])
                np.add.at(S, (np.broadcast_to(rows, blocks.shape), np.broadcast_to(cols, blocks.shape)), -blocks)
            rhs = gl.ravel().copy()
            correction = np.einsum('nji,nj->ni', Hpl[free], np.einsum('nij,nj->ni', Dp_inv[oi[free]], gp[oi[free]]))
            np.add.at(rhs.reshape(L, ld), oj[free], -correction)
            try:
                dl = np.linalg.solve(S, -rhs).reshape(L, ld)
            except np.linalg.LinAlgError:
                lam *= 10
                if lam > 1e6:
                    break
                continue
            coupling = np.zeros((P, 6))
            np.add.at(coupling, oi[free], np.einsum('nij,nj->ni', Hpl[free], dl[oj[free]]))
            dp = np.einsum('nij,nj->ni', Dp_inv, -gp - coupling)
            new_poses = perturb(poses, dp)
            new_points = points.copy()
            new_points[:, :ld] += dl
            new_r = residuals(new_poses, new_points)
            new_cost = _soft_l1(new_r)
            if new_cost < cost:
                break
            lam *= 10
            if lam > 1e6:
                break
        if lam > 1e6 or new_cost >= cost:
            break
        improvement = (cost - new_cost) / max(cost, 1e-9)
        if os.environ.get('BAITER'):
            print('  it', round(new_cost, 2), lam, np.abs(dp).max().round(5), np.abs(dl).max().round(5))
        poses, points, r, cost = new_poses, new_points, new_r, new_cost
        lam = max(lam / 5, 1e-7)
        H_ll_final = S
        if improvement < 1e-4:
            break
    if os.environ.get('BADEBUG'):
        print('BA', global_pass, P, L, n, 'cost', round(_soft_l1(residuals(np.array([results[f]['T_world_cam'] for f in free_frames]), np.array([landmarks[j].p for j in landmark_ids]))), 1), '->', round(cost, 1), 'lam', lam)
    for f, i in pose_index.items():
        results[f]['T_world_cam'] = poses[i]
    for lm_id, i in lm_index.items():
        lm = landmarks[lm_id]
        lm.p = points[i].copy()
        lm.rhs = lm.information @ lm.p
        lm.ba = True
    if H_ll_final is not None:
        # Marginal landmark covariance from the Schur complement.
        try:
            covariance = np.linalg.inv(H_ll_final)
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is not None:
            sigma_px = float(np.median(sigma))
            for lm_id, i in lm_index.items():
                block = covariance[i * ld:(i + 1) * ld, i * ld:(i + 1) * ld]
                landmarks[lm_id].covariance[:ld, :ld] = np.diag(np.maximum(np.diag(block), .01 ** 2))


def _merge_landmarks(results, landmarks, observations, distance=1.5):
    """Consolidate nearby compatible points unless they were ever seen together."""
    if len(landmarks) < 2:
        return False
    coobserved = {i: set() for i in landmarks}
    for row in results:
        ids = set(row['assoc'].values()) & landmarks.keys()
        for i in ids:
            coobserved[i].update(ids - {i})
    remap = {i: i for i in landmarks}
    members = {i: [lm.p.copy()] for i, lm in landmarks.items()}
    ids = sorted(landmarks, key=lambda i: (landmarks[i].first, -landmarks[i].n))
    merged = False
    for i in ids:
        if remap[i] != i:
            continue
        a = landmarks[i]
        for j in ids:
            if i == j or remap[j] != j:
                continue
            b = landmarks[j]
            # Two physical cones cannot stand closer than about a base width.
            if j in coobserved[i] and np.linalg.norm(a.p - b.p) > _SAME_CONE_M:
                continue
            if not _compatible(a.class_id, b.class_id) or np.linalg.norm(a.p - b.p) > distance:
                continue
            # Complete-link clustering prevents a chain of individually nearby
            # duplicates from swallowing distinct same-colour neighbours.
            if max(np.linalg.norm(pa - pb) for pa in members[i] for pb in members[j]) > distance:
                continue
            members[i].extend(members[j])
            a.p = (a.p * a.n + b.p * b.n) / (a.n + b.n)
            a.n += b.n
            a.first, a.last = min(a.first, b.first), max(a.last, b.last)
            a.information += b.information
            a.rhs = a.information @ a.p
            a.covariance = np.linalg.inv(a.information) + np.eye(3) * .02 ** 2
            if a.class_id == 3:
                a.class_id = b.class_id
            remap[j] = i
            coobserved[i].update(coobserved[j])
            for other in coobserved[j]:
                coobserved[other].add(i)
            merged = True
    if not merged:
        return False
    for j in remap:
        root = remap[j]
        while remap[root] != root:
            root = remap[root]
        remap[j] = root
    for row in results:
        row['assoc'] = {d: remap.get(i, i) for d, i in row['assoc'].items()}
    for o in observations:
        o.landmark = remap.get(o.landmark, o.landmark)
    for j, i in remap.items():
        if j != i:
            del landmarks[j]
    return True


def _initial_ground(frame, ms, plane, prep):
    """Refine the first ground estimate in pixels instead of noisy PnP depths."""
    T = _world_transform(plane)
    rows = [i for i, det_id in enumerate(prep.ids) if det_id in ms]
    if len(rows) < 3:
        return plane
    K = np.asarray(frame['K'], float)
    points = np.array([_measurement_world(ms[prep.ids[i]], T, True, K)[0] for i in rows])
    normal, height = plane
    x0 = np.r_[normal[0] / -normal[1], normal[2] / -normal[1], height, points[:, :2].ravel()]
    classes = prep.classes[rows]
    pixels, valid, sigma = prep.pixels[rows], prep.valid[rows], prep.sigma[rows]
    mask = _term_mask(valid)
    Ks = np.broadcast_to(K, (len(rows), 3, 3))

    def residual(x):
        up = np.array([x[0], -1., x[1]])
        up /= np.linalg.norm(up)
        pose = _world_transform((up, x[2]))
        positions = np.c_[x[3:].reshape(-1, 2), np.zeros(len(rows))]
        return _terms(_pixels(np.broadcast_to(pose, (len(rows), 4, 4)), positions, classes, Ks),
                      pixels, valid, sigma)[mask]
    sparse = lil_matrix((int(mask.sum()), len(x0)), dtype=int)
    row = 0
    for i in range(len(rows)):
        size = int(mask[i].sum())
        sparse[row:row + size, :3] = 1
        sparse[row:row + size, 3 + 2 * i:5 + 2 * i] = 1
        row += size
    opt = least_squares(residual, x0, jac_sparsity=sparse.tocsr(), loss='cauchy', f_scale=_ROBUST,
                        max_nfev=40, ftol=1e-5, x_scale='jac')
    up = np.array([opt.x[0], -1., opt.x[1]])
    up /= np.linalg.norm(up)
    if .2 < opt.x[2] < 5 and up[1] < -np.cos(np.deg2rad(35)):
        return up, float(opt.x[2])
    return plane


def _rotation_angle(R):
    return float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def run_slam(frames: list[dict], options: dict, progress: Callable | None = None) -> dict:
    start = time.perf_counter()
    flat_requested = bool(options.get('flat_ground', True))
    kp_conf = float(options.get('kp_conf', .5))
    max_range = float(options.get('max_range_m', 40.))
    results, landmarks, keyframes, observations = [], {}, [], []
    preps, goods = [], []
    next_id, initialized, flat = 0, False, False
    plane = None
    previous = None          # (frame number, pose) of the last tracked frame
    velocity = None          # per-second twist (rotvec, translation) between tracked frames
    lost_count = 0
    mean_height = None
    next_global = 15
    probation = 0            # frames to confirm a relocalisation before mapping
    seen_time = {}
    if progress:
        progress(0., 'slam')

    def report(fi, low=0., high=.7):
        if progress:
            progress(low + (high - low) * (fi + 1) / max(1, len(frames)), 'slam')

    def frame_time(fi):
        return float(frames[fi].get('t', fi * .1))

    for fi, frame in enumerate(frames):
        K = np.asarray(frame['K'], float)
        ms = {}
        for det in frame['detections']:
            measurement = solve_cone(det['class_id'], det['keypoints'], K, kp_conf,
                                     det.get('range_scale', 1.))
            if measurement is not None:
                ms[det['det_id']] = measurement
        prep = _prepare(frame, kp_conf)
        preps.append(prep)
        truncated = {d for d, t in zip(prep.ids, prep.truncated) if t}
        kept = set(prep.ids)
        good = {i: m for i, m in ms.items() if i in kept and np.linalg.norm(m.p_cam) <= max_range and m.reproj_rms_px < 6.}
        goods.append(good)
        row = dict(index=frame['index'], T_world_cam=None, status='lost', assoc={}, measurements=ms)
        results.append(row)
        if not initialized:
            points = np.array([m.p_cam for m in good.values()])
            spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False) if len(points) >= 3 else []
            if len(spread) < 2 or spread[1] <= .4:
                report(fi)
                continue
            current = fit_ground_plane([m.p_cam for m in good.values()],
                                       [1 / max(.01, np.trace(m.cov_cam)) for m in good.values()])
            flat = flat_requested and current is not None
            if current is not None:
                current = _initial_ground(frame, good, current, prep)
            T = _world_transform(current)
            initialized = True
            row.update(T_world_cam=T, status='init')
            keyframes.append(fi)
        else:
            hypotheses = []
            if previous is not None:
                last_fi, last_T = previous
                dt = frame_time(fi) - frame_time(last_fi)
                if velocity is not None and lost_count == 0 and 0 < dt < .5:
                    increment = np.eye(4)
                    increment[:3, :3] = Rotation.from_rotvec(velocity[0] * dt).as_matrix()
                    T = last_T.copy()
                    T[:3, :3] = increment[:3, :3] @ last_T[:3, :3]
                    T[:3, 3] = last_T[:3, 3] + velocity[1] * dt
                    hypotheses.append(T)
                hypotheses.append(last_T)
            prior = None if previous is None else (hypotheses[0], frame_time(fi) - frame_time(previous[0]),
                                                   mean_height if flat else None)
            tracked = _track(fi, frame, prep, good, landmarks, hypotheses, flat, prior)
            if tracked is None:
                lost_count += 1
                row['T_world_cam'] = previous[1].copy() if previous is not None and lost_count <= 10 else None
                report(fi)
                continue
            T = tracked['T']
            if not tracked['consistent']:
                probation = _PROBATION
            elif probation and tracked['n'] >= 4:
                probation -= 1
            if os.environ.get('TRKDBG'):
                print('TRK', fi, tracked['n'], len(prep.ids), round(tracked['error'], 2), tracked['consistent'], tracked['missing'], tracked['visible'], tracked.get('source'), len(landmarks), T[:3, 3].round(2))
            row.update(T_world_cam=T, status='tracked', assoc=dict(tracked['assoc']), _quality=tracked)
            if previous is not None and lost_count == 0:
                dt = frame_time(fi) - frame_time(previous[0])
                if dt > 0:
                    rotation = Rotation.from_matrix(T[:3, :3] @ previous[1][:3, :3].T).as_rotvec() / dt
                    translation = (T[:3, 3] - previous[1][:3, 3]) / dt
                    if velocity is None:
                        velocity = (rotation, translation)
                    else:
                        velocity = (.5 * velocity[0] + .5 * rotation, .5 * velocity[1] + .5 * translation)
            else:
                velocity = None
            lost_count = 0
        # Mapping: fuse associated detections and seed tentative landmarks.
        reliable = row['status'] == 'init' or (not probation and (
            row['_quality']['n'] >= 4 or (row['_quality']['n'] >= 3 and row['_quality']['consistent'])))
        if reliable:
            if mean_height is None or row['status'] == 'init':
                mean_height = float(T[2, 3])
            else:
                step = frame_time(fi) - frame_time(previous[0]) if previous is not None else .1
                alpha = min(1., max(step, 0.) / _HEIGHT_TAU)
                mean_height = (1 - alpha) * mean_height + alpha * float(T[2, 3])
        view = _MapView(landmarks)
        occupied = set()
        if len(view.ids):
            r, c, _ = _associate(T, prep, view, K, 1.2)
            occupied = {prep.ids[i] for i in r}
        for det_id, m in good.items():
            if not reliable or det_id in truncated:
                continue
            p, covariance = _measurement_world(m, T, flat, K)
            if not np.isfinite(p).all() or np.linalg.norm(p - T[:3, 3]) > max_range * 1.25:
                continue
            lm_id = row['assoc'].get(det_id)
            if lm_id is None:
                if (det_id in occupied or det_id not in prep.ids or
                        np.linalg.norm(m.p_cam) > min(max_range, 25.) or m.reproj_rms_px > 4.):
                    continue
                lm_id, next_id = next_id, next_id + 1
                info = np.linalg.inv(covariance + np.eye(3) * .02 ** 2)
                landmarks[lm_id] = _Landmark(lm_id, m.class_id, p, covariance, frame['index'], frame['index'],
                                             information=info, rhs=info @ p)
                row['assoc'][det_id] = lm_id
            else:
                lm = landmarks[lm_id]
                if lm.n < _FUSE_UNTIL or not lm.ba:
                    # PnP ranges are biased by a few percent, so only young
                    # points are fused; bundle adjustment owns the rest.
                    # Huber-like attenuation limits erroneous PnP fits.
                    error = np.linalg.norm(p - lm.p)
                    info = np.linalg.inv(covariance + np.eye(3) * .03 ** 2) * min(1., .5 / max(error, 1e-9))
                    lm.information += info
                    lm.rhs += info @ p
                    lm.p = np.linalg.solve(lm.information, lm.rhs)
                    if flat:
                        lm.p[2] = 0.
                    lm.covariance = np.linalg.inv(lm.information) + np.eye(3) * .02 ** 2
                lm.n += 1
                lm.last = frame['index']
                if lm.class_id == 3 and m.class_id != 3:
                    lm.class_id = m.class_id
        for lm_id in row['assoc'].values():
            if lm_id in landmarks and landmarks[lm_id].last != frame['index']:
                landmarks[lm_id].n += 1
                landmarks[lm_id].last = frame['index']
        # Stale tentative points disappear (time based, independent of stride).
        for lm_id in row['assoc'].values():
            seen_time[lm_id] = frame_time(fi)
        stale = [i for i, lm in landmarks.items() if lm.n < 3 and frame_time(fi) - seen_time.get(i, frame_time(fi)) > 1.]
        for i in stale:
            del landmarks[i]
            row['assoc'] = {d: j for d, j in row['assoc'].items() if j != i}
        if row['status'] != 'init' and reliable:
            last_key = results[keyframes[-1]]['T_world_cam']
            if (frame_time(fi) - frame_time(keyframes[-1]) >= _KEYFRAME_S or
                    np.linalg.norm(T[:3, 3] - last_key[:3, 3]) > _KEYFRAME_M or
                    _rotation_angle(T[:3, :3] @ last_key[:3, :3].T) > .3):
                keyframes.append(fi)
        if fi in keyframes:
            observations.extend(_observations(fi, frame, row['assoc'], prep))
            if len(keyframes) >= 4 and len(keyframes) % _BA_EVERY == 0:
                global_update = len(keyframes) >= next_global
                _bundle_adjust(results, landmarks, keyframes if global_update else keyframes[-8:],
                               observations, flat, global_pass=global_update)
                if global_update:
                    # Geometric spacing bounds the total cost of growing global passes.
                    next_global = int(len(keyframes) * _GLOBAL_GROWTH)
                    _merge_landmarks(results, landmarks, observations)
                T = row['T_world_cam']
        previous = (fi, T.copy())
        report(fi)

    if progress:
        progress(.71, 'bundle_adjustment')
    _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=True)
    for _ in range(2):
        if not _merge_landmarks(results, landmarks, observations):
            break
        _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=True)
    if progress:
        progress(.8, 'refining_trajectory')
    if keyframes and not os.environ.get('SKIP2'):
        _relocalize_all(frames, results, preps, goods, landmarks, keyframes, flat, progress)
    _finish_lost(results)
    for lm in landmarks.values():
        sightings = [row['index'] for row in results if lm.id in row['assoc'].values()]
        lm.n = len(sightings)
        if sightings:
            lm.first, lm.last = min(sightings), max(sightings)
    position = {row['index']: i for i, row in enumerate(results)}
    confirmed = set()
    for i, lm in landmarks.items():
        span = position.get(lm.last, 0) - position.get(lm.first, 0) + 1
        # Sporadic false detections of a static cone-like object are sparse.
        if lm.n >= 3 and lm.n >= _CONFIRM_DENSITY * span:
            confirmed.add(i)
    output = []
    for i in sorted(confirmed):
        lm = landmarks[i]
        output.append(dict(id=i, class_id=lm.class_id, class_name=CLASS_NAMES[lm.class_id],
                           p_world=lm.p.tolist(), sigma_m=np.sqrt(np.maximum(np.diag(lm.covariance), 0)).tolist(),
                           n_obs=lm.n, first_frame=lm.first, last_frame=lm.last))
    for row in results:
        row['assoc'] = {d: i for d, i in row['assoc'].items() if i in confirmed}
        row.pop('_quality', None)
    tracked_rows = [r for r in results if r['status'] in ('init', 'tracked')]
    stats = dict(n_frames_total=len(frames), n_frames_processed=len(frames),
                 n_tracked=len(tracked_rows), n_lost=sum(r['status'] == 'lost' for r in results),
                 n_landmarks=len(output), n_keyframes=len(keyframes), runtime_s=time.perf_counter() - start)
    ground = None
    if flat and tracked_rows:
        ground = dict(camera_height_m=float(np.median([r['T_world_cam'][2, 3] for r in tracked_rows])),
                      normal_world=[0., 0., 1.])
    if progress:
        progress(1., 'slam')
    return dict(frames=results, landmarks=output, ground_plane=ground, stats=stats)


def _relocalize_all(frames, results, preps, goods, landmarks, keyframes, flat, progress):
    """Re-localise every non-keyframe against the final map, forwards then backwards."""
    keyset = set(keyframes)
    order = list(range(len(frames)))
    key_times = np.array([float(frames[k].get('t', k)) for k in keyframes])
    key_heights = np.array([results[k]['T_world_cam'][2, 3] for k in keyframes])

    def _local_height(fi):
        near = np.abs(key_times - float(frames[fi].get('t', fi))) <= _HEIGHT_TAU / 2
        return float(np.median(key_heights[near])) if near.any() else float(np.median(key_heights))

    for direction, sweep in enumerate((order, order[::-1])):
        neighbour = None
        for step, fi in enumerate(sweep):
            row = results[fi]
            if fi in keyset:
                neighbour = (fi, row['T_world_cam'])
                continue
            if direction == 1 and row['status'] == 'tracked' and row.get('_final'):
                neighbour = (fi, row['T_world_cam'])
                continue
            hypotheses = []
            if neighbour is not None:
                hypotheses.append(neighbour[1])
            if row['status'] == 'tracked':
                hypotheses.append(row['T_world_cam'])
            prior = None
            if neighbour is not None:
                prior = (neighbour[1], abs(float(frames[fi].get('t', fi)) - float(frames[neighbour[0]].get('t', neighbour[0]))),
                         _local_height(fi) if flat else None)
            tracked = _track(fi, frames[fi], preps[fi], goods[fi], landmarks, hypotheses, flat, prior)
            if tracked is not None:
                row.update(T_world_cam=tracked['T'], status='tracked', assoc=tracked['assoc'], _final=True)
                neighbour = (fi, tracked['T'])
            elif direction == 1:
                row.update(status='lost', assoc={})
            if progress:
                progress(.8 + .19 * (direction * len(frames) + step + 1) / (2 * max(1, len(frames))),
                         'refining_trajectory')
    _fill_gaps(frames, results, preps, landmarks, flat, _local_height)
    for row in results:
        row.pop('_final', None)


def _interpolate(Ta, Tb, fraction):
    T = Ta.copy()
    T[:3, 3] = (1 - fraction) * Ta[:3, 3] + fraction * Tb[:3, 3]
    relative = Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()
    T[:3, :3] = Ta[:3, :3] @ Rotation.from_rotvec(relative * fraction).as_matrix()
    return T


def _fill_gaps(frames, results, preps, landmarks, flat, local_height):
    """Low-support frames between two tracked neighbours: accept a pose that
    stays close to the interpolation of its neighbours (a fixed-lag smoother
    check that a single frame with two or three blurred cones cannot pass by
    itself)."""
    view = _MapView(landmarks)
    tracked = [i for i, row in enumerate(results) if row['status'] in ('init', 'tracked')]
    if len(tracked) < 2 or not len(view.ids):
        return
    times = [float(frame.get('t', i)) for i, frame in enumerate(frames)]
    for fi, row in enumerate(results):
        if row['status'] != 'lost' or len(preps[fi].ids) < 2:
            continue
        right = int(np.searchsorted(tracked, fi))
        if right == 0 or right == len(tracked):
            continue
        a, b = tracked[right - 1], tracked[right]
        gap = times[b] - times[a]
        if gap > _GAP_S or gap <= 0:
            continue
        fraction = (times[fi] - times[a]) / gap
        seed = _interpolate(results[a]['T_world_cam'], results[b]['T_world_cam'], fraction)
        K = np.asarray(frames[fi]['K'], float)
        candidate = _localize(seed, preps[fi], view, K, (seed, gap / 2, local_height(fi) if flat else None))
        if candidate is None:
            continue
        T = candidate['T']
        if (np.linalg.norm(T[:3, 3] - seed[:3, 3]) < .1 + .5 * gap and
                _rotation_angle(T[:3, :3] @ seed[:3, :3].T) < .05 + gap):
            row.update(T_world_cam=T, status='tracked',
                       assoc={preps[fi].ids[r]: view.ids[c] for r, c in zip(candidate['rows'], candidate['cols'])})


def _finish_lost(results):
    """Lost frames keep the last good pose for up to 10 frames, then None."""
    last, count = None, 0
    for row in results:
        if row['status'] in ('init', 'tracked'):
            last, count = row['T_world_cam'], 0
            continue
        count += 1
        row['T_world_cam'] = last.copy() if last is not None and count <= 10 else None
