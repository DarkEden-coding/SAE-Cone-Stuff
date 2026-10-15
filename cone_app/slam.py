"""Cone-only metric SLAM; no image features, inertial data, or supplied poses.

Tracking uses constant velocity, class-compatible Hungarian/Mahalanobis matching,
RANSAC rigid alignment, and robust silhouette-keypoint motion refinement. Flat
mode smooths measured ground normals/heights and estimates SE(2); otherwise poses
are SE(3). Tentative landmarks need three sightings. Sparse keyframe bundle
adjustment runs in bounded windows, periodically over the connected map, and globally, followed by motion-only
refinement of other frames. Ground ray intersections improve flat-mode mapping;
metric scale comes from cone dimensions and the measured camera height. Runtime
is bounded by keyframe spacing, finite optimizer budgets, and sampled relocalization
hypotheses. New landmarks are seeded within 25 m (or the configured lower maximum)
to avoid unstable distant depth. Global flat-mode BA also refines camera height and
tilt with the first horizontal pose anchored. Nearby, never-coobserved duplicates
are consolidated. Full mode estimates a fixed upright axis once in the fallback
world gauge, without constraining landmark heights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class _Observation:
    frame: int
    landmark: int
    class_id: int
    pixels: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    up: np.ndarray = field(default_factory=lambda: _UP.copy())


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


def _parameters(T: np.ndarray, flat: bool) -> np.ndarray:
    if flat:
        return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 2], T[0, 2])])
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def _pose(parameters: np.ndarray, reference: np.ndarray, flat: bool) -> np.ndarray:
    T = np.array(reference, copy=True)
    if flat:
        x, y, yaw = parameters
        original_yaw = np.arctan2(reference[1, 2], reference[0, 2])
        angle = yaw - original_yaw
        c, s = np.cos(angle), np.sin(angle)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ reference[:3, :3]
        T[:2, 3] = [x, y]
    else:
        T[:3, 3] = parameters[:3]
        T[:3, :3] = Rotation.from_rotvec(parameters[3:]).as_matrix()
    return T


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


def _associate(ms, landmarks, T, flat, K, uncertainty=.2, detections=None, up=_UP):
    if not ms or not landmarks:
        return {}
    ids = list(ms)
    lms = list(landmarks.values())
    ps, covs = zip(*[_measurement_world(ms[i], T, flat, K) for i in ids])
    ps, covs = np.array(ps), np.array(covs)
    target = np.array([lm.p for lm in lms])
    delta = ps[:, None] - target[None]
    covariance = covs[:, None] + np.array([lm.covariance for lm in lms])[None] + np.eye(3) * uncertainty ** 2
    cost = np.einsum('nmi,nmij,nmj->nm', delta, np.linalg.inv(covariance), delta)
    compatible = _compatible(np.array([ms[i].class_id for i in ids])[:, None],
                             np.array([lm.class_id for lm in lms])[None])
    distances = np.linalg.norm(delta, axis=2)
    allowed = compatible & (cost < 20) & (distances < max(3.5, uncertainty * 4))
    if detections is not None:
        # Pixel-space Mahalanobis gating avoids duplicate far landmarks when
        # depth inferred from a tiny noisy cone is weakly constrained.
        by_id = {d['det_id']: d for d in detections}
        pair_i, pair_j = np.nonzero(allowed)
        if len(pair_i):
            observed = np.zeros((len(ids), 8, 2))
            valid = np.zeros((len(ids), 8), dtype=bool)
            for i, det_id in enumerate(ids):
                kp = np.asarray(by_id[det_id]['keypoints'], float)
                n = min(n_keypoints(ms[det_id].class_id), len(kp))
                observed[i, :n] = np.nan_to_num(kp[:n, :2])
                valid[i, :n] = np.isfinite(kp[:n]).all(axis=1) & (kp[:n, 2] >= .5)
            ci = np.array([ms[ids[i]].class_id for i in pair_i])
            predictions = _pixels(np.broadcast_to(T, (len(pair_i), 4, 4)), target[pair_j], ci,
                                  np.broadcast_to(K, (len(pair_i), 3, 3)), up)
            diff = predictions - observed[pair_i]
            depth = ((target[pair_j] - T[:3, 3]) @ T[:3, :3])[:, 2]
            sigma_sq = 3. ** 2 + (K[0, 0] * uncertainty / np.maximum(2., depth)) ** 2
            pixel_cost = (np.sum(np.sum(diff ** 2, axis=2) * valid[pair_i], axis=1) /
                          np.maximum(1, valid[pair_i].sum(axis=1)) / sigma_sq)
            allowed[pair_i, pair_j] &= (pixel_cost < 12.) & (depth > .2)
            cost[pair_i, pair_j] += pixel_cost
    cost = np.where(allowed, cost + .05 * distances ** 2, 1e8)
    rows, cols = linear_sum_assignment(cost)
    return {ids[i]: lms[j].id for i, j in zip(rows, cols) if cost[i, j] < 1e8}


def _alignment(source: np.ndarray, target: np.ndarray, weights: np.ndarray, flat: bool):
    dimensions = 2 if flat else 3
    a, b = source[:, :dimensions], target[:, :dimensions]
    weights = weights / weights.sum()
    ac, bc = weights @ a, weights @ b
    u, _, vt = np.linalg.svd((a - ac).T @ ((b - bc) * weights[:, None]))
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    rotation = np.eye(3)
    rotation[:dimensions, :dimensions] = R
    translation = np.zeros(3)
    translation[:dimensions] = bc - R @ ac
    return rotation, translation


def _rigid_pose(ms, assoc, landmarks, predicted, flat, K, rng):
    minimum = 2 if flat else 3
    if len(assoc) < minimum:
        return None
    ids = list(assoc)
    source = np.array([_measurement_world(ms[i], predicted, flat, K)[0] for i in ids])
    target = np.array([landmarks[assoc[i]].p for i in ids])
    weights = np.array([1 / max(.01, np.trace(ms[i].cov_cam)) for i in ids])
    best = np.ones(len(ids), dtype=bool)
    best_score = -1
    for iteration in range(35 if len(ids) > minimum else 1):
        sample = np.arange(len(ids)) if iteration == 0 else rng.choice(len(ids), minimum, replace=False)
        if np.linalg.norm(source[sample[0]] - source[sample[-1]]) < .3:
            continue
        R, t = _alignment(source[sample], target[sample], weights[sample], flat)
        residual = np.linalg.norm(source @ R.T + t - target, axis=1)
        good = residual < np.clip(.25 + np.array([np.sqrt(np.trace(ms[i].cov_cam)) for i in ids]), .4, 1.5)
        score = good.sum() + .05 * weights[good].sum() / max(weights.sum(), 1e-9)
        if good.sum() >= minimum and score > best_score:
            best, best_score = good, score
    if best_score < 0:
        return None
    R, t = _alignment(source[best], target[best], weights[best], flat)
    T = np.array(predicted, copy=True)
    T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = R @ T[:3, 3] + t
    return T


def _observations(frame_number, frame, assoc, kp_conf, up=_UP):
    out = []
    for detection in frame['detections']:
        det_id = detection['det_id']
        if det_id not in assoc:
            continue
        cls = detection['class_id']
        kp = np.asarray(detection['keypoints'], float)[:n_keypoints(cls)]
        pixels, valid = np.zeros((8, 2)), np.zeros(8, bool)
        pixels[:len(kp)] = np.nan_to_num(kp[:, :2])
        valid[:len(kp)] = (kp[:, 2] >= kp_conf) & np.isfinite(kp[:, :2]).all(axis=1)
        out.append(_Observation(frame_number, assoc[det_id], cls, pixels, valid, np.asarray(frame['K'], float), up))
    return out


def _motion_refine(T, observations, landmarks, flat, max_nfev=10):
    observations = [o for o in observations if o.landmark in landmarks]
    if len(observations) < (2 if flat else 3):
        return T, float('inf')
    points = np.array([landmarks[o.landmark].p for o in observations])
    classes = np.array([o.class_id for o in observations])
    Ks = np.array([o.K for o in observations])
    pixels = np.array([o.pixels for o in observations])
    valid = np.array([o.valid for o in observations])
    def residual(x):
        poses = np.broadcast_to(_pose(x, T, flat), (len(points), 4, 4))
        return (_pixels(poses, points, classes, Ks, observations[0].up) - pixels)[valid].ravel()
    initial = _parameters(T, flat)
    result = least_squares(residual, initial, loss='soft_l1', f_scale=2., max_nfev=max_nfev,
                           ftol=1e-4, xtol=1e-5, gtol=1e-3)
    output = _pose(result.x, T, flat)
    errors = residual(result.x).reshape(-1, 2)
    rms = float(np.sqrt(np.median(np.sum(errors ** 2, axis=1))))
    return output, rms


def _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=False, max_nfev=None):
    selected = set(keyframes)
    obs = [o for o in observations if o.frame in selected and o.landmark in landmarks]
    if len(obs) < 6 or len(keyframes) < 2:
        return
    # A single-view point is not constrained enough to belong in BA.
    counts = {}
    for o in obs:
        counts[o.landmark] = counts.get(o.landmark, 0) + 1
    obs = [o for o in obs if counts[o.landmark] >= 2]
    if not obs:
        return
    landmark_ids = sorted({o.landmark for o in obs})
    pose_ids = [i for i in keyframes[1:] if any(o.frame == i for o in obs)]
    pd, ld = (6 if global_pass or not flat else 3), (2 if flat else 3)
    pose_flat = flat and not global_pass
    pose_index = {f: i for i, f in enumerate(pose_ids)}
    lm_index = {j: i for i, j in enumerate(landmark_ids)}
    npose = len(pose_ids) * pd
    x0 = np.r_[np.array([_parameters(results[f]['T_world_cam'], pose_flat) for f in pose_ids]).ravel(),
               np.array([landmarks[j].p[:ld] for j in landmark_ids]).ravel()]
    ground_offset = len(x0)
    optimize_anchor = flat and global_pass
    anchor_frame = keyframes[0]
    if optimize_anchor:
        anchor = results[anchor_frame]['T_world_cam']
        normal = anchor[2, :3]
        x0 = np.r_[x0, normal[0] / -normal[1], normal[2] / -normal[1], anchor[2, 3]]
    references = np.array([results[o.frame]['T_world_cam'] for o in obs])
    classes, Ks = np.array([o.class_id for o in obs]), np.array([o.K for o in obs])
    pixels, valid = np.array([o.pixels for o in obs]), np.array([o.valid for o in obs])
    point_indices = np.array([lm_index[o.landmark] for o in obs])
    frame_groups = {f: np.array([i for i, o in enumerate(obs) if o.frame == f]) for f in pose_ids}
    def residual(x):
        poses = references.copy()
        for f, group in frame_groups.items():
            offset = pose_index[f] * pd
            poses[group] = _pose(x[offset:offset + pd], references[group[0]], pose_flat)
        if optimize_anchor:
            normal = np.array([x[ground_offset], -1., x[ground_offset + 1]])
            normal /= np.linalg.norm(normal)
            poses[[i for i, o in enumerate(obs) if o.frame == anchor_frame]] = _world_transform((normal, x[ground_offset + 2]))
        points = np.zeros((len(landmark_ids), 3))
        points[:, :ld] = x[npose:npose + len(landmark_ids) * ld].reshape(-1, ld)
        return (_pixels(poses, points[point_indices], classes, Ks, obs[0].up) - pixels)[valid].ravel()
    sparsity = lil_matrix((int(valid.sum()) * 2, len(x0)), dtype=int)
    row = 0
    for o in obs:
        size = int(o.valid.sum()) * 2
        if o.frame in pose_index:
            col = pose_index[o.frame] * pd
            sparsity[row:row + size, col:col + pd] = 1
        if optimize_anchor and o.frame == anchor_frame:
            sparsity[row:row + size, ground_offset:ground_offset + 3] = 1
        col = npose + lm_index[o.landmark] * ld
        sparsity[row:row + size, col:col + ld] = 1
        row += size
    opt = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(), loss='soft_l1', f_scale=2.,
                        max_nfev=max_nfev or (35 if global_pass else 12), ftol=1e-3, xtol=1e-4,
                        tr_solver='lsmr', x_scale='jac')
    for f, i in pose_index.items():
        results[f]['T_world_cam'] = _pose(opt.x[i * pd:(i + 1) * pd], results[f]['T_world_cam'], pose_flat)
    if optimize_anchor:
        normal = np.array([opt.x[ground_offset], -1., opt.x[ground_offset + 1]])
        normal /= np.linalg.norm(normal)
        results[anchor_frame]['T_world_cam'] = _world_transform((normal, opt.x[ground_offset + 2]))
    for lm_id, i in lm_index.items():
        lm = landmarks[lm_id]
        lm.p[:ld] = opt.x[npose + i * ld:npose + (i + 1) * ld]
        # Preserve accumulated information while moving its centre to the BA estimate.
        lm.rhs = lm.information @ lm.p
    # Diagonal of J^T J is an inexpensive conditional uncertainty approximation.
    diagonal = np.asarray(opt.jac.power(2).sum(axis=0)).ravel()
    variance = max(1., float(np.median(opt.fun ** 2)))
    for lm_id, i in lm_index.items():
        diag = variance / np.maximum(diagonal[npose + i * ld:npose + (i + 1) * ld], 1e-8)
        landmarks[lm_id].covariance[:ld, :ld] = np.diag(np.maximum(diag, .01 ** 2))


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
            if i == j or remap[j] != j or j in coobserved[i]:
                continue
            b = landmarks[j]
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


def _initial_ground(frame, ms, plane, kp_conf):
    """Refine the first ground estimate in pixels instead of noisy PnP depths."""
    T = _world_transform(plane)
    ids = list(ms)
    obs = _observations(0, frame, {i: j for j, i in enumerate(ids)}, kp_conf)
    if len(obs) < 3:
        return plane
    # Preserve observation order even if detector IDs are not sorted.
    ids = [ids[o.landmark] for o in obs]
    points = np.array([_measurement_world(ms[i], T, True, np.asarray(frame['K']))[0] for i in ids])
    normal, height = plane
    x0 = np.r_[normal[0] / -normal[1], normal[2] / -normal[1], height, points[:, :2].ravel()]
    classes = np.array([o.class_id for o in obs])
    pixels, valid, Ks = np.array([o.pixels for o in obs]), np.array([o.valid for o in obs]), np.array([o.K for o in obs])
    def residual(x):
        up = np.array([x[0], -1., x[1]])
        up /= np.linalg.norm(up)
        pose = _world_transform((up, x[2]))
        positions = np.c_[x[3:].reshape(-1, 2), np.zeros(len(obs))]
        return (_pixels(np.broadcast_to(pose, (len(obs), 4, 4)), positions, classes, Ks) - pixels)[valid].ravel()
    sparse = lil_matrix((int(valid.sum()) * 2, len(x0)), dtype=int)
    row = 0
    for i, o in enumerate(obs):
        size = int(o.valid.sum()) * 2
        sparse[row:row + size, :3] = 1
        sparse[row:row + size, 3 + 2 * i:5 + 2 * i] = 1
        row += size
    opt = least_squares(residual, x0, jac_sparsity=sparse.tocsr(), loss='soft_l1', f_scale=2.,
                        max_nfev=40, ftol=1e-5, x_scale='jac')
    up = np.array([opt.x[0], -1., opt.x[1]])
    up /= np.linalg.norm(up)
    if .2 < opt.x[2] < 5 and up[1] < -np.cos(np.deg2rad(35)):
        return up, float(opt.x[2])
    return plane


def _relocalize(ms, landmarks, predicted, flat, K, rng):
    minimum = 2 if flat else 3
    ids = list(ms)
    if len(ids) < minimum or len(landmarks) < minimum:
        return None, {}
    # Prefer the nearby map, then allow all mapped cones. Hypotheses never use IDs
    # from the detector or temporal image features.
    for nearby in (True, False):
        candidates = [lm for lm in landmarks.values() if lm.n >= 3 and
                      (not nearby or np.linalg.norm(lm.p - predicted[:3, 3]) < 50)]
        if len(candidates) < minimum:
            continue
        source = np.array([_measurement_world(ms[i], predicted, flat, K)[0] for i in ids])
        pools = [[lm for lm in candidates if _compatible(ms[i].class_id, lm.class_id)] for i in ids]
        best = None
        for _ in range(120):
            sample = rng.choice(len(ids), minimum, replace=False)
            if any(not pools[j] for j in sample):
                continue
            chosen = [pools[j][rng.integers(len(pools[j]))] for j in sample]
            if len({lm.id for lm in chosen}) < minimum:
                continue
            target = np.array([lm.p for lm in chosen])
            if abs(np.linalg.norm(source[sample[0]] - source[sample[1]]) - np.linalg.norm(target[0] - target[1])) > 1.5:
                continue
            R, t = _alignment(source[sample], target, np.ones(minimum), flat)
            T = predicted.copy()
            T[:3, :3], T[:3, 3] = R @ T[:3, :3], R @ T[:3, 3] + t
            assoc = _associate(ms, landmarks, T, flat, K, .35)
            if len(assoc) >= max(minimum + 1, int(.55 * len(ids))) and (best is None or len(assoc) > len(best[1])):
                best = (T, assoc)
        if best is not None:
            return best
    return None, {}


def run_slam(frames: list[dict], options: dict, progress: Callable | None = None) -> dict:
    start = time.perf_counter()
    flat_requested = bool(options.get('flat_ground', True))
    kp_conf = float(options.get('kp_conf', .5))
    max_range = float(options.get('max_range_m', 40.))
    results, landmarks, keyframes, observations = [], {}, [], []
    next_id, initialized, flat = 0, False, False
    plane = None
    world_up = _UP.copy()
    previous, velocity = None, np.eye(4)
    last_time, last_dt = None, .1
    lost_count = 0
    rng = np.random.default_rng(19)
    if progress:
        progress(0., 'slam')
    for fi, frame in enumerate(frames):
        K = np.asarray(frame['K'], float)
        ms = {}
        for det in frame['detections']:
            measurement = solve_cone(det['class_id'], det['keypoints'], K, kp_conf,
                                     det.get('range_scale', 1.))
            if measurement is not None:
                ms[det['det_id']] = measurement
        good = {i: m for i, m in ms.items() if np.linalg.norm(m.p_cam) <= max_range and m.reproj_rms_px < 6.}
        current_plane = None
        if flat_requested and len(good) >= 3:
            current_plane = fit_ground_plane([m.p_cam for m in good.values()],
                                            [1 / max(.01, np.trace(m.cov_cam)) for m in good.values()])
        if current_plane is not None:
            if plane is None:
                plane = current_plane
            else:
                alpha = .025
                normal = (1 - alpha) * plane[0] + alpha * current_plane[0]
                normal /= np.linalg.norm(normal)
                plane = normal, float((1 - alpha) * plane[1] + alpha * current_plane[1])
        row = dict(index=frame['index'], T_world_cam=None, status='lost', assoc={}, measurements=ms)
        results.append(row)
        if not initialized:
            points = np.array([m.p_cam for m in good.values()])
            spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False) if len(points) >= 3 else []
            if len(spread) >= 2 and spread[1] > .4:
                flat = flat_requested and plane is not None
                if flat:
                    plane = _initial_ground(frame, good, plane, kp_conf)
                T = _world_transform(plane if flat else None)
                if not flat:
                    # A one-time upright-axis estimate sets cone orientation in
                    # the fallback gauge; subsequent map points remain fully 3D.
                    axis_plane = fit_ground_plane([m.p_cam for m in good.values()],
                                                  [1 / max(.01, np.trace(m.cov_cam)) for m in good.values()])
                    if axis_plane is not None:
                        axis_plane = _initial_ground(frame, good, axis_plane, kp_conf)
                        world_up = T[:3, :3] @ axis_plane[0]
                initialized = True
                row.update(T_world_cam=T, status='init')
                keyframes.append(fi)
            else:
                if progress:
                    progress((fi + 1) / max(1, len(frames)) * .8, 'slam')
                continue
        else:
            dt = max(.001, float(frame.get('t', fi * .1)) - last_time)
            ratio = np.clip(dt / last_dt, .1, 5.)
            increment = np.eye(4)
            increment[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(velocity[:3, :3]).as_rotvec() * ratio).as_matrix()
            increment[:3, 3] = velocity[:3, 3] * ratio
            T = increment @ previous
            if flat and plane is not None:
                level = _world_transform(plane)
                level[:2, 3] = T[:2, 3]
                parameters = _parameters(T, True)
                T = _pose(parameters, level, True)
            assoc = _associate(good, landmarks, T, flat, K, .15 + min(lost_count * .2, 1.5), frame['detections'], world_up)
            estimated = _rigid_pose(good, assoc, landmarks, T, flat, K, rng)
            if estimated is None:
                estimated, assoc = _relocalize(good, landmarks, T, flat, K, rng)
            if estimated is not None:
                obs = _observations(fi, frame, assoc, kp_conf, world_up)
                refined, rms = _motion_refine(estimated, obs, landmarks, flat)
                if rms < 9. and np.linalg.norm(refined[:3, 3] - T[:3, 3]) < 8. + lost_count * 2:
                    T = refined
                    # Re-associate after pose refinement before mapping.
                    assoc = _associate(good, landmarks, T, flat, K, .1, frame['detections'], world_up)
                    row.update(T_world_cam=T, status='tracked', assoc=assoc)
                    lost_count = 0
                else:
                    lost_count += 1
            else:
                lost_count += 1
            if row['status'] == 'lost':
                row['T_world_cam'] = T if lost_count <= 10 else None
                if progress:
                    progress((fi + 1) / max(1, len(frames)) * .8, 'slam')
                previous, last_time, last_dt = T, float(frame.get('t', fi * .1)), dt
                continue
        # Assimilate a detection once per frame; stale tentative points disappear.
        for det_id, m in good.items():
            p, covariance = _measurement_world(m, T, flat, K)
            if not np.isfinite(p).all() or np.linalg.norm(p - T[:3, 3]) > max_range * 1.25:
                continue
            lm_id = row['assoc'].get(det_id)
            if lm_id is None:
                if np.linalg.norm(m.p_cam) > min(max_range, 25.) or m.reproj_rms_px > 4.:
                    continue
                lm_id, next_id = next_id, next_id + 1
                info = np.linalg.inv(covariance + np.eye(3) * .02 ** 2)
                landmarks[lm_id] = _Landmark(lm_id, m.class_id, p, covariance, frame['index'], frame['index'],
                                             information=info, rhs=info @ p)
                row['assoc'][det_id] = lm_id
            else:
                lm = landmarks[lm_id]
                # Huber-like attenuation prevents occasional erroneous PnP fits
                # from pulling a well-supported map point.
                error = np.linalg.norm(p - lm.p)
                info = np.linalg.inv(covariance + np.eye(3) * .03 ** 2) * min(1., .5 / max(error, 1e-9))
                lm.information += info
                lm.rhs += info @ p
                lm.p = np.linalg.solve(lm.information, lm.rhs)
                lm.covariance = np.linalg.inv(lm.information) + np.eye(3) * .02 ** 2
                lm.n += 1
                lm.last = frame['index']
                if lm.class_id == 3 and m.class_id != 3:
                    lm.class_id = m.class_id
        # Time is based on processed-frame positions, not possibly strided indices.
        stale = [i for i, lm in landmarks.items() if lm.n < 3 and frame['index'] - lm.last >
                 max(30, 20 * (frame['index'] - frames[max(0, fi - 1)]['index']))]
        for i in stale:
            del landmarks[i]
        if row['status'] != 'init':
            last_key_pose = results[keyframes[-1]]['T_world_cam']
            if fi - keyframes[-1] >= 10 or np.linalg.norm(T[:3, 3] - last_key_pose[:3, 3]) > 2.:
                keyframes.append(fi)
        if fi in keyframes:
            observations.extend(_observations(fi, frame, row['assoc'], kp_conf, world_up))
            if len(keyframes) >= 6 and len(keyframes) % 5 == 0:
                before_ba = row['T_world_cam'].copy()
                global_update = len(keyframes) % 10 == 0
                _bundle_adjust(results, landmarks, keyframes if global_update else keyframes[-6:],
                               observations, flat, global_pass=global_update)
                if global_update:
                    _merge_landmarks(results, landmarks, observations)
                T = row['T_world_cam']
                if previous is not None:
                    previous = T @ np.linalg.inv(before_ba) @ previous
                if flat:
                    plane = T[2, :3].copy(), float(T[2, 3])
        if previous is not None:
            velocity = T @ np.linalg.inv(previous)
        previous, last_time = T.copy(), float(frame.get('t', fi * .1))
        if fi:
            last_dt = max(.001, last_time - float(frames[fi - 1].get('t', (fi - 1) * .1)))
        if progress:
            progress((fi + 1) / max(1, len(frames)) * .8, 'slam')
    if progress:
        progress(.81, 'bundle_adjustment')
    _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=True)
    for _ in range(2):
        if not _merge_landmarks(results, landmarks, observations):
            break
        _bundle_adjust(results, landmarks, keyframes, observations, flat, global_pass=True)
    if progress:
        progress(.92, 'refining_trajectory')
    if flat and keyframes:
        heights = [results[i]['T_world_cam'][2, 3] for i in keyframes]
        plane = results[keyframes[0]]['T_world_cam'][2, :3].copy(), float(np.median(heights))
    # Seed motion-only refinement from optimized neighbouring keyframes. The
    # online poses predate BA; using their old coordinates can leave refinement
    # outside the local optimizer's basin. Keep the established associations and
    # retain lost-frame extrapolation whenever refinement has no support.
    selected = set(keyframes)
    for fi, row in enumerate(results):
        if fi in selected or not keyframes or fi < keyframes[0]:
            continue
        right = int(np.searchsorted(keyframes, fi))
        a, b = keyframes[max(0, right - 1)], keyframes[min(len(keyframes) - 1, right)]
        fraction = (fi - a) / max(1, b - a)
        Ta, Tb = results[a]['T_world_cam'], results[b]['T_world_cam']
        seed = Ta.copy()
        seed[:3, 3] = (1 - fraction) * Ta[:3, 3] + fraction * Tb[:3, 3]
        relative = Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()
        seed[:3, :3] = Ta[:3, :3] @ Rotation.from_rotvec(relative * fraction).as_matrix()
        assoc = row['assoc']
        if row['status'] == 'lost':
            usable = {j: m for j, m in row['measurements'].items()
                      if np.linalg.norm(m.p_cam) <= max_range and m.reproj_rms_px < 6.}
            assoc = _associate(usable, landmarks, seed, flat, np.asarray(frames[fi]['K']),
                               .3, frames[fi]['detections'], world_up)
        refined, rms = _motion_refine(seed,
                                     _observations(fi, frames[fi], assoc, kp_conf, world_up),
                                     landmarks, False)
        if rms < 9.:
            row.update(T_world_cam=refined, status='tracked', assoc=assoc)
    for lm in landmarks.values():
        sightings = [row['index'] for row in results if lm.id in row['assoc'].values()]
        lm.n = len(sightings)
        if sightings:
            lm.first, lm.last = min(sightings), max(sightings)
    confirmed = {i for i, lm in landmarks.items() if lm.n >= 3}
    output = []
    for i in sorted(confirmed):
        lm = landmarks[i]
        output.append(dict(id=i, class_id=lm.class_id, class_name=CLASS_NAMES[lm.class_id],
                           p_world=lm.p.tolist(), sigma_m=np.sqrt(np.maximum(np.diag(lm.covariance), 0)).tolist(),
                           n_obs=lm.n, first_frame=lm.first, last_frame=lm.last))
    for row in results:
        row['assoc'] = {d: i for d, i in row['assoc'].items() if i in confirmed}
    stats = dict(n_frames_total=len(frames), n_frames_processed=len(frames),
                 n_tracked=sum(r['status'] in ('init', 'tracked') for r in results),
                 n_lost=sum(r['status'] == 'lost' for r in results), n_landmarks=len(output),
                 n_keyframes=len(keyframes), runtime_s=time.perf_counter() - start)
    if progress:
        progress(1., 'slam')
    return dict(frames=results, landmarks=output,
                ground_plane=dict(camera_height_m=float(plane[1]), normal_world=[0., 0., 1.]) if flat else None,
                stats=stats)
