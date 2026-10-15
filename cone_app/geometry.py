"""Metric cone geometry in OpenCV camera and z-up world coordinates.

PnP uses the repository's camera-facing planar ring model and IPPE/SQPnP/
ITERATIVE fallback, copied here to avoid importing the evaluation CLI's GPU stack.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import ArrayLike

from synth.cone_model import CLASS_NAMES as _CLASS_NAMES, canonical_rings, cone_spec

CLASS_NAMES: dict[int, str] = dict(_CLASS_NAMES)


def cone_dims(class_id: int) -> dict[str, float]:
    spec = cone_spec(class_id)
    return {"height": spec["height"], "base": spec["plate_size"]}


def ring_model(class_id: int) -> list[tuple[float, float]]:
    return canonical_rings(class_id)


def n_keypoints(class_id: int) -> int:
    return 8 if class_id == 1 else 6


class KeypointCorrection:
    """Interpolate fitted bias tables in log box height; null bins are ignored."""

    def __init__(self, table: dict):
        self.edges = np.asarray(table["edges_px"], dtype=float)
        self.offsets = np.asarray(table["offsets_inward_below_frac_h"], dtype=float)
        self.bias = np.asarray(table.get("range_bias_frac", []), dtype=float)
        self.scales = np.asarray(table.get("range_scale", table.get("range_scales", [])), dtype=float)
        if (self.edges.ndim != 1 or len(self.edges) < 2 or
                np.any(self.edges <= 0) or np.any(np.diff(self.edges) <= 0)):
            raise ValueError("correction edges must be positive and increasing")
        if self.offsets.ndim != 3 or self.offsets.shape[0] != len(self.edges) - 1 or self.offsets.shape[2] != 2:
            raise ValueError("invalid keypoint correction table shape")
        self.centres = np.log(np.sqrt(self.edges[:-1] * self.edges[1:]))

    @classmethod
    def load(cls, path: str | Path) -> "KeypointCorrection":
        data = json.loads(Path(path).read_text())
        if "edges_px" not in data:
            tables = [v for v in data.values() if isinstance(v, dict) and "edges_px" in v]
            if len(tables) != 1:
                raise ValueError("expected one model correction table")
            data = tables[0]
        return cls(data)

    def _interp(self, values: np.ndarray, height: float, default: float) -> float:
        if values.shape != self.centres.shape:
            return default
        good = np.isfinite(values)
        return float(np.interp(np.log(max(height, 1e-6)), self.centres[good], values[good])) if good.any() else default

    def apply(self, kpts: np.ndarray, box_xyxy: Sequence[float]) -> np.ndarray:
        out = np.array(kpts, dtype=float, copy=True)
        height = float(box_xyxy[3] - box_xyxy[1])
        if not np.isfinite(height) or height <= 0:
            return out
        for k in range(min(len(out), self.offsets.shape[1])):
            out[k, 0] -= height * (1 if k % 2 == 0 else -1) * self._interp(self.offsets[:, k, 0], height, 0)
            out[k, 1] -= height * self._interp(self.offsets[:, k, 1], height, 0)
        return out

    def range_scale(self, box_h_px: float) -> float:
        if not np.isfinite(box_h_px) or box_h_px <= 0:
            return 1.0
        if self.scales.size:
            scale = self._interp(self.scales, box_h_px, 1.0)
        else:
            bias = self._interp(self.bias, box_h_px, 0.0)
            scale = 1.0 / (1.0 + bias) if bias > -1 else 1.0
        return scale if np.isfinite(scale) and scale > 0 else 1.0


@dataclass
class ConeMeasurement:
    class_id: int
    p_cam: np.ndarray
    cov_cam: np.ndarray
    reproj_rms_px: float
    n_kp: int


def solve_cone(class_id: int, kpts: ArrayLike, K: ArrayLike, kp_conf: float = 0.5,
               range_scale: float = 1.0) -> ConeMeasurement | None:
    kp, K = np.asarray(kpts, float), np.asarray(K, float)
    if kp.ndim != 2 or kp.shape[1] < 3 or K.shape != (3, 3) or not np.isfinite(K).all():
        return None
    if not np.isfinite(range_scale) or range_scale <= 0 or min(K[0, 0], K[1, 1]) <= 0:
        return None
    count = min(len(kp), n_keypoints(class_id))
    valid = np.isfinite(kp[:count]).all(axis=1) & (kp[:count, 2] >= kp_conf)
    if valid.sum() < 4:
        return None
    obj = np.array([[s * r, -z, 0.] for z, r in ring_model(class_id) for s in (-1, 1)])[:count][valid]
    img = np.ascontiguousarray(kp[:count, :2][valid])
    best = None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            _, rvs, tvs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=flag)
        except cv2.error:
            continue
        for rv, tv in zip(rvs or (), tvs or ()):
            if not np.isfinite(rv).all() or not np.isfinite(tv).all() or tv[2, 0] <= 0:
                continue
            projected = cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))
            if best is None or rms < best[1]:
                best = (tv.ravel(), rms)
        if best is not None:
            break
    if best is None:
        return None
    p, rms = best[0] * range_scale, best[1]
    distance = float(np.linalg.norm(p))
    if distance < 0.05 or distance > 1000:
        return None
    # Pixel uncertainty yields lateral r*sigma/f and axial r²*sigma/(f*H).
    # A 2% axial floor accounts for model/calibration error. All units are metres.
    sigma_px = max(1., rms)
    f = float(np.sqrt(K[0, 0] * K[1, 1]))
    lateral = max(0.005, distance * sigma_px / f)
    axial = distance ** 2 * sigma_px / (f * cone_dims(class_id)["height"]) + .02 * distance
    ray = p / distance
    cov = lateral ** 2 * np.eye(3) + (axial ** 2 - lateral ** 2) * np.outer(ray, ray)
    return ConeMeasurement(int(class_id), p, cov, rms, int(valid.sum()))


def project_cone_keypoints(class_id: int, p_world: ArrayLike, up_world: ArrayLike,
                           T_world_cam: ArrayLike, K: ArrayLike) -> np.ndarray:
    p, up, T, K = (np.asarray(x, float) for x in (p_world, up_world, T_world_cam, K))
    up = up / np.linalg.norm(up)
    lateral = np.cross(up, p - T[:3, 3])
    norm = np.linalg.norm(lateral)
    if norm < 1e-10:
        return np.full((n_keypoints(class_id), 2), np.nan)
    lateral /= norm
    points = np.array([p + z * up + sign * radius * lateral
                       for z, radius in ring_model(class_id) for sign in (-1, 1)])
    cam = (points - T[:3, 3]) @ T[:3, :3]
    pixels_h = cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    pixels[cam[:, 2] <= 0] = np.nan
    for k in range(0, len(pixels), 2):
        if pixels[k, 0] > pixels[k + 1, 0]:
            pixels[[k, k + 1]] = pixels[[k + 1, k]]
    return pixels


def fit_ground_plane(points_cam: ArrayLike, weights: ArrayLike | None = None) -> tuple[np.ndarray, float] | None:
    points = np.asarray(points_cam, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return None
    w = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    good = np.isfinite(points).all(axis=1) & np.isfinite(w) & (w > 0)
    points, w = points[good], w[good]
    if len(points) < 3:
        return None
    w = w / w.max()
    rng = np.random.default_rng(7)
    best, score = None, -1.
    for _ in range(min(160, 12 * len(points))):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        size = np.linalg.norm(normal)
        if size < 1e-5:
            continue
        normal /= size
        if normal[1] > 0:
            normal = -normal
        h = -float(normal @ a)
        if normal[1] > -np.cos(np.deg2rad(35)) or not .2 <= h <= 5:
            continue
        inliers = np.abs(points @ normal + h) < .12
        value = float(w[inliers].sum())
        if inliers.sum() >= 3 and value > score:
            best, score = inliers, value
    if best is None:
        return None
    for _ in range(3):
        q, v = points[best], w[best]
        centre = np.average(q, axis=0, weights=v)
        _, singular, vt = np.linalg.svd((q - centre) * np.sqrt(v[:, None]), full_matrices=False)
        if len(singular) < 2 or singular[1] < 1e-4:
            return None
        normal = vt[-1]
        if normal[1] > 0:
            normal = -normal
        height = -float(normal @ centre)
        candidate = np.abs(points @ normal + height) < .12
        if candidate.sum() < 3:
            return None
        best = candidate
    if normal[1] > -np.cos(np.deg2rad(35)) or not .2 <= height <= 5:
        return None
    return normal, height


def _world_transform(plane: tuple[np.ndarray, float] | None) -> np.ndarray:
    up, height = (np.array([0., -1., 0.]), 0.) if plane is None else plane
    forward = np.array([0., 0., 1.]) - up * up[2]
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    T = np.eye(4)
    T[:3, :3] = np.stack([forward, left, up])
    T[2, 3] = height
    return T


def single_frame_scene(detections: list[dict], K: ArrayLike, flat_ground: bool) -> dict:
    measurements = {}
    for det in detections:
        m = solve_cone(det["class_id"], det["keypoints"], K, range_scale=det.get("range_scale", 1.))
        if m is not None:
            measurements[det["det_id"]] = m
    ms = list(measurements.values())
    plane = fit_ground_plane([m.p_cam for m in ms], [1 / max(.01, np.trace(m.cov_cam)) for m in ms]) if flat_ground else None
    T = _world_transform(plane)
    landmarks = []
    for det_id, m in measurements.items():
        p = T[:3, :3] @ m.p_cam + T[:3, 3]
        if plane is not None:
            p[2] = 0.
        covariance = T[:3, :3] @ m.cov_cam @ T[:3, :3].T
        landmarks.append(dict(id=int(det_id), class_id=m.class_id, class_name=CLASS_NAMES[m.class_id],
                              p_world=p.tolist(), sigma_m=np.sqrt(np.diag(covariance)).tolist(),
                              n_obs=1, first_frame=0, last_frame=0))
    return dict(T_world_cam=T, ground_plane=None if plane is None else
                dict(camera_height_m=float(plane[1]), normal_world=[0., 0., 1.]),
                measurements=measurements, landmarks=landmarks)
