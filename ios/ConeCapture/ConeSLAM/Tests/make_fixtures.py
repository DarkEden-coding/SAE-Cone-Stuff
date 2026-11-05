"""Regenerate the ConeSLAM test fixtures from the Python reference implementation.

Run from the repo root:  PYTHONPATH=. uv run --locked python ios/ConeCapture/ConeSLAM/Tests/make_fixtures.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cone_app.geometry import project_cone_keypoints
from cone_app.tests.test_slam import K, simulate

OUT = Path(__file__).parent / "ConeSLAMTests" / "Fixtures"


def r(x, n=3):
    return np.round(np.asarray(x, float), n).tolist()


def projection_cases():
    rng = np.random.default_rng(3)
    cases = []
    for _ in range(40):
        cls = int(rng.choice([0, 1, 2, 4]))
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi), rng.uniform(-.3, .1), rng.uniform(-.1, .1)
        R = Rotation.from_euler('z', yaw).as_matrix() @ np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]]) @ \
            Rotation.from_euler('xy', [pitch, roll]).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(.5, 2)]
        cam = np.array([rng.uniform(-3, 3), rng.uniform(0, 1), rng.uniform(3, 25)])
        p = R @ cam + T[:3, 3]
        p[2] = 0.
        kp = project_cone_keypoints(cls, p, [0, 0, 1], T, K)
        if not np.isfinite(kp).all():
            continue
        cases.append(dict(class_id=cls, p_world=r(p, 6), T=r(T, 9), keypoints=r(kp, 4),
                          p_cam=r(R.T @ (p - T[:3, 3]), 6), up_cam=r(R.T @ [0, 0, 1], 9)))
    return dict(K=K.tolist(), cases=cases)


def sequence(n_frames, flat, seed=22, gravity_noise_deg=.3):
    frames, truth, points, classes, seen = simulate(n_frames, flat=flat, seed=seed)
    rng = np.random.default_rng(seed + 1)
    out = []
    for frame, T in zip(frames, truth):
        up = T[:3, :3].T @ [0, 0, 1]
        noise = Rotation.from_rotvec(rng.normal(0, np.deg2rad(gravity_noise_deg) / np.sqrt(2), 3)).as_matrix()
        up = noise @ up
        dets = []
        for d in frame['detections']:
            kp = np.asarray(d['keypoints'], float)
            box = d.get('box_xyxy') or [kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]
            full = np.zeros((8, 3))
            full[:len(kp)] = kp
            dets.append(dict(c=d['class_id'], s=d['conf'], b=r(box, 2), k=r(full, 2)))
        out.append(dict(t=frame['t'], up=r(up / np.linalg.norm(up), 6), d=dets))
    return dict(K=K.tolist(), size=[1920, 1080], frames=out, truth=[r(T, 5) for T in truth],
                landmarks=r(points, 4), classes=classes.tolist(), seen=sorted(int(i) for i in seen))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "projection.json").write_text(json.dumps(projection_cases()))
    (OUT / "loop_flat.json").write_text(json.dumps(sequence(600, True), separators=(',', ':')))
    (OUT / "short_se3.json").write_text(json.dumps(sequence(90, False), separators=(',', ':')))
    for f in sorted(OUT.glob("*.json")):
        print(f.name, f.stat().st_size // 1024, "KiB")
