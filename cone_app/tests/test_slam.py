"""Deterministic cone-only simulation; also runnable without pytest.

uv run --locked python -m cone_app.tests.test_slam
"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from cone_app.geometry import project_cone_keypoints
from cone_app.slam import _Landmark, _merge_landmarks, run_slam

K = np.array([[1500., 0, 960], [0, 1500., 540], [0, 0, 1.]])
R0 = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])


def simulate(n_frames=600, flat=True, seed=22):
    rng = np.random.default_rng(seed)
    a, b = 26., 16.
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 70, endpoint=False)
    landmarks, classes = [], []
    # ~4 m separation on each boundary, 3.5 m track width.
    for j, angle in enumerate(angles[::2]):
        centre = np.array([a * np.cos(angle), b * np.sin(angle), 0.])
        tangent = np.array([-a * np.sin(angle), b * np.cos(angle), 0.])
        tangent /= np.linalg.norm(tangent)
        left = np.cross([0, 0, 1], tangent)
        for side, cls in ((1, 0), (-1, 4)):
            landmarks.append(centre + side * 1.75 * left)
            classes.append(cls)
    for angle in (-1.4, .4, 2.):
        centre = np.array([a * np.cos(angle), b * np.sin(angle), 0.])
        tangent = np.array([-a * np.sin(angle), b * np.cos(angle), 0.])
        tangent /= np.linalg.norm(tangent)
        landmarks.append(centre + 3.1 * np.cross([0, 0, 1], tangent))
        classes.append(1 if angle < 0 else 2)
    landmarks, classes = np.array(landmarks), np.array(classes)
    frames, truth, seen = [], [], set()
    for fi in range(n_frames):
        angle = -np.pi / 2 + 2 * np.pi * fi / 600
        centre = np.array([a * np.cos(angle), b * np.sin(angle), 1.1])
        yaw = np.arctan2(b * np.cos(angle), -a * np.sin(angle))
        T = np.eye(4)
        wobble = .003 * np.sin(fi / 30) if not flat else 0.
        T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix() @ R0 @ Rotation.from_euler('xy', [-.045 + wobble, wobble]).as_matrix()
        T[:3, 3] = centre
        detections = []
        for li, (point, cls) in enumerate(zip(landmarks, classes)):
            camera = T[:3, :3].T @ (point - centre)
            if camera[2] < 2. or np.linalg.norm(camera) > 40:
                continue
            kp = project_cone_keypoints(int(cls), point, [0, 0, 1], T, K)
            full_height = K[1, 1] * (.505 if cls == 1 else .325) / camera[2]
            if (full_height < 12 or not np.isfinite(kp).all() or kp[:, 0].min() < 5 or
                    kp[:, 0].max() > 1915 or kp[:, 1].min() < 5 or kp[:, 1].max() > 1075):
                continue
            seen.add(li)
            if rng.random() < .12:
                continue
            noise = 1. + .6 * min(1., 20 / full_height)
            kp += rng.normal(0., noise, kp.shape)
            detections.append(dict(det_id=len(detections), class_id=int(cls), conf=.95,
                                   keypoints=np.c_[kp, np.ones(len(kp))], range_scale=1.,
                                   box_xyxy=[kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]))
        # A false cone has coherent geometry but no persistent track location.
        if rng.random() < .07:
            point = centre + T[:3, :3] @ [rng.uniform(-3, 3), 1.1, rng.uniform(6, 20)]
            point[2] = 0.
            kp = project_cone_keypoints(2, point, [0, 0, 1], T, K)
            detections.append(dict(det_id=len(detections), class_id=2, conf=.4,
                                   keypoints=np.c_[kp, np.ones(6)], range_scale=1.))
        frames.append(dict(index=fi, t=fi / 10., K=K, detections=detections))
        truth.append(T)
    return frames, np.array(truth), landmarks, classes, seen


def assess(result, truth, points, classes, seen):
    rows = result['frames']
    initialized = next(i for i, row in enumerate(rows) if row['status'] == 'init')
    estimated_start = rows[initialized]['T_world_cam']
    alignment = estimated_start @ np.linalg.inv(truth[initialized])
    aligned = np.einsum('ij,njk->nik', alignment, truth)
    selected = [i for i, row in enumerate(rows) if row['status'] in ('tracked', 'init')]
    errors = [np.linalg.norm(rows[i]['T_world_cam'][:3, 3] - aligned[i, :3, 3]) for i in selected]
    ate = float(np.sqrt(np.mean(np.square(errors))))
    aligned_points = points @ alignment[:3, :3].T + alignment[:3, 3]
    predicted = np.array([lm['p_world'] for lm in result['landmarks']])
    predicted_classes = np.array([lm['class_id'] for lm in result['landmarks']])
    eligible = sorted(seen)
    distance = np.linalg.norm(predicted[:, None] - aligned_points[eligible][None], axis=2)
    distance[predicted_classes[:, None] != classes[eligible][None]] = 1e6
    a, b = linear_sum_assignment(distance)
    matched = distance[a, b]
    mean_error = float(np.mean(matched[matched < 1e5])) if np.any(matched < 1e5) else float('inf')
    path = float(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum())
    return dict(ate_m=ate, path_m=path, landmark_error_m=mean_error, true_landmarks=len(seen),
                landmarks=len(predicted), tracked_fraction=len(selected) / len(rows),
                runtime_s=result['stats']['runtime_s'])


def test_loop_flat_ground():
    frames, truth, points, classes, seen = simulate()
    callbacks = []
    result = run_slam(frames, dict(flat_ground=True, max_range_m=40, kp_conf=.5),
                      lambda frac, stage: callbacks.append((frac, stage)))
    metrics = assess(result, truth, points, classes, seen)
    print('flat loop:', metrics, flush=True)
    assert metrics['ate_m'] < .04 * metrics['path_m']
    assert metrics['tracked_fraction'] > .95
    assert abs(metrics['landmarks'] - metrics['true_landmarks']) < .2 * metrics['true_landmarks']
    assert metrics['landmark_error_m'] < .5
    assert result['ground_plane'] is not None
    assert callbacks[0][0] == 0 and callbacks[-1][0] == 1
    assert all(a[0] <= b[0] for a, b in zip(callbacks, callbacks[1:]))
    ids = {lm['id'] for lm in result['landmarks']}
    assert all(set(row['assoc'].values()) <= ids for row in result['frames'])
    return metrics


def test_short_se3():
    frames, truth, points, classes, seen = simulate(90, flat=False)
    result = run_slam(frames, dict(flat_ground=False))
    metrics = assess(result, truth, points, classes, seen)
    print('SE3 short:', metrics, flush=True)
    assert metrics['tracked_fraction'] > .9
    assert metrics['ate_m'] < .1
    assert metrics['landmark_error_m'] < 1.
    assert result['ground_plane'] is None
    return metrics


def test_empty_and_uninitialized():
    assert run_slam([], {})['stats']['n_frames_processed'] == 0
    result = run_slam([dict(index=7, t=0., K=K, detections=[])], {})
    assert result['frames'][0]['status'] == 'lost'
    assert result['frames'][0]['T_world_cam'] is None
    assert result['landmarks'] == []


def test_tracking_gap():
    frames, _, _, _, _ = simulate(55)
    for frame in frames[15:35]:
        frame['detections'] = []
    result = run_slam(frames, dict(flat_ground=True))
    assert all(row['status'] == 'lost' for row in result['frames'][15:35])
    assert result['frames'][24]['T_world_cam'] is not None
    assert result['frames'][25]['T_world_cam'] is None
    assert result['frames'][-1]['status'] == 'tracked'


def test_thousand_frame_budget():
    frames, truth, points, classes, seen = simulate(1000)
    result = run_slam(frames, dict(flat_ground=True))
    metrics = assess(result, truth, points, classes, seen)
    print('1000 frames:', metrics, flush=True)
    assert metrics['runtime_s'] < 40.
    assert metrics['tracked_fraction'] > .95
    assert metrics['ate_m'] < .04 * metrics['path_m']
    return metrics


def _test_landmark(i, p, cls=0):
    p = np.asarray(p, float)
    return _Landmark(i, cls, p, np.eye(3)*.01, 0, 0, n=10,
                     information=np.eye(3)*100, rhs=p*100)


def test_merge_preserves_neighbouring_cones():
    # No co-observation evidence: the spatial/complete-link guard must suffice.
    landmarks = {i: _test_landmark(i, [x, 0, 0])
                 for i, x in enumerate((0., 1.2, 2.4, 3.6, 7.2))}
    rows = [dict(assoc={0: i}) for i in landmarks]
    assert _merge_landmarks(rows, landmarks, [])
    assert len(landmarks) == 3
    assert min(np.linalg.norm(a.p-b.p) for i, a in landmarks.items()
               for j, b in landmarks.items() if i != j) > 2.
    # Even very close cones are distinct when they appear in the same frame.
    landmarks = {i: _test_landmark(i, [i*.4, 0, 0]) for i in range(2)}
    assert not _merge_landmarks([dict(assoc={0: 0, 1: 1})], landmarks, [])


if __name__ == '__main__':
    start = time.perf_counter()
    test_empty_and_uninitialized()
    test_merge_preserves_neighbouring_cones()
    test_loop_flat_ground()
    test_short_se3()
    test_tracking_gap()
    test_thousand_frame_budget()
    print(f'SLAM tests passed in {time.perf_counter() - start:.2f} s')
