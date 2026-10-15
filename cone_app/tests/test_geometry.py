"""Run with pytest or: uv run --locked python -m cone_app.tests.test_geometry."""
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from cone_app.geometry import (KeypointCorrection, fit_ground_plane, n_keypoints,
                               project_cone_keypoints, single_frame_scene, solve_cone)

K = np.array([[1500., 0, 960], [0, 1500., 540], [0, 0, 1.]])
R0 = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])


def test_round_trip():
    maximum = 0.
    for cls in range(5):
        for distance in (3., 8., 20., 40.):
            for azimuth in (-.45, 0., .45):
                T = np.eye(4)
                T[:3, :3] = R0 @ Rotation.from_euler('x', -.045).as_matrix()
                T[2, 3] = 1.1
                p = np.array([distance * np.cos(azimuth), distance * np.sin(azimuth), 0])
                pixels = project_cone_keypoints(cls, p, [0, 0, 1], T, K)
                assert pixels.shape == (n_keypoints(cls), 2)
                assert np.all(pixels[::2, 0] < pixels[1::2, 0])
                m = solve_cone(cls, np.c_[pixels, np.ones(len(pixels))], K)
                assert m is not None
                expected = T[:3, :3].T @ (p - T[:3, 3])
                maximum = max(maximum, float(np.linalg.norm(m.p_cam - expected)))
                assert np.linalg.norm(m.p_cam - expected) < .003
                assert np.linalg.eigvalsh(m.cov_cam).min() > 0
                scaled = solve_cone(cls, np.c_[pixels, np.ones(len(pixels))], K, range_scale=1.1)
                assert np.allclose(scaled.p_cam, m.p_cam * 1.1)
    print(f'geometry maximum round-trip error: {maximum:.8f} m')


def test_ground_plane_and_scene():
    rng = np.random.default_rng(2)
    T = np.eye(4)
    T[:3, :3] = R0 @ Rotation.from_euler('xy', [-.06, .025]).as_matrix()
    T[2, 3] = 1.1
    points = np.c_[rng.uniform(3, 25, 35), rng.uniform(-6, 6, 35), np.zeros(35)]
    camera_points = (points - T[:3, 3]) @ T[:3, :3]
    noisy = camera_points + rng.normal(0, .012, camera_points.shape)
    noisy[:5] += [0, .7, 0]
    plane = fit_ground_plane(noisy)
    assert plane is not None
    normal, height = plane
    assert abs(height - 1.1) < .03
    assert normal @ (T[:3, :3].T @ [0, 0, 1]) > .999
    assert fit_ground_plane(np.array([[0, 1, 3], [0, 1, 4], [0, 1, 5]])) is None
    assert fit_ground_plane(np.array([[0, .05, 3], [1, .05, 3], [0, .05, 5]])) is None
    detections = []
    for i, p in enumerate(points[:8]):
        kp = project_cone_keypoints(0, p, [0, 0, 1], T, K)
        detections.append(dict(det_id=i, class_id=0, keypoints=np.c_[kp, np.ones(6)], range_scale=1.))
    scene = single_frame_scene(detections, K, True)
    assert scene['ground_plane'] is not None
    assert abs(scene['T_world_cam'][2, 3] - 1.1) < .003
    assert np.allclose([p['p_world'][2] for p in scene['landmarks']], 0)
    assert np.allclose(scene['T_world_cam'][:3, :3] @ normal, [0, 0, 1], atol=.004)
    forward = scene['T_world_cam'][:3, :3] @ [0, 0, 1]
    assert abs(forward[1]) < 1e-10 and forward[0] > .99
    fallback = single_frame_scene(detections, K, False)
    assert np.array_equal(fallback['T_world_cam'][:3, :3], R0)
    assert np.array_equal(fallback['T_world_cam'][:3, 3], np.zeros(3))


def test_correction_and_invalid():
    table = dict(edges_px=[10, 40, 160], offsets_inward_below_frac_h=[[[.1, .2]] * 6] * 2,
                 range_bias_frac=[.25, None])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'table.json'
        path.write_text(json.dumps({'E@1920': table}))
        correction = KeypointCorrection.load(path)
        kp = np.ones((8, 3))
        out = correction.apply(kp, [0, 0, 10, 20])
        assert np.allclose(out[:6:2, 0], -1)
        assert np.allclose(out[1:6:2, 0], 3)
        assert np.allclose(out[:6, 1], -3)
        assert np.array_equal(out[6:], kp[6:]) and np.array_equal(kp, np.ones((8, 3)))
        assert correction.range_scale(25) == .8
        del table['range_bias_frac']
        path.write_text(json.dumps(table))
        assert KeypointCorrection.load(path).range_scale(30) == 1.
    assert solve_cone(0, np.ones((8, 3)), K, kp_conf=2) is None
    bad = np.full((8, 3), np.nan)
    assert solve_cone(0, bad, K) is None


if __name__ == '__main__':
    test_round_trip()
    test_ground_plane_and_scene()
    test_correction_and_invalid()
    print('geometry tests passed')
