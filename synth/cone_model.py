"""Cone geometry, keypoint definitions, and camera model shared by Blender and post-processing.

numpy only, so it imports inside Blender's bundled Python as well as the project env.

Conventions
-----------
Cone frame: origin at the centre of the cone's bottom face (ground contact), +z up the
cone axis, +x/+y parallel to the square base plate edges. Units are metres.
Camera frame: OpenCV (x right, y down, z forward). Pixel coordinates are OpenCV
(pixel centres at integer coordinates).

Keypoints follow the BRT dataset layout: pairs of left/right points (even = image-left,
odd = image-right) on the cone's silhouette at stripe edges. They are view dependent:
for a given camera position the 3D point is where the visible outline (the silhouette
generator line of the conical body) crosses the stripe-edge circle.
"""
import math
import numpy as np

CLASS_NAMES = {0: 'BLUE', 1: 'ORANGE_BIG', 2: 'ORANGE', 3: 'UNDEFINED', 4: 'YELLOW'}

# Dimensions follow the FSG/FSAE driverless cone spec (WEMAS): small 228x228x325 mm,
# big 285x285x505 mm. Stripe and foot heights were fitted to the BRT label statistics.
SMALL = dict(
    height=0.325, plate_size=0.228, plate_thickness=0.012, plate_corner=0.022,
    body_z0=0.045, body_r0=0.080, body_z1=0.316, body_r1=0.025,
    # (r, z) profile from the base flare up over the rim and into the top opening.
    profile=[(0.100, 0.012), (0.092, 0.020), (0.085, 0.030), (0.0815, 0.039), (0.080, 0.045),
             (0.025, 0.316), (0.0245, 0.321), (0.022, 0.325), (0.0185, 0.3235), (0.0170, 0.318),
             (0.0160, 0.285), (0.0, 0.285)],
    stripes=[(0.140, 0.220)],
    keypoint_z=[0.220, 0.140, 0.045],
    keypoint_names=['stripe_top', 'stripe_bottom', 'body_foot'],
)
BIG = dict(
    height=0.505, plate_size=0.285, plate_thickness=0.015, plate_corner=0.028,
    body_z0=0.055, body_r0=0.100, body_z1=0.492, body_r1=0.030,
    profile=[(0.125, 0.015), (0.114, 0.025), (0.106, 0.037), (0.1015, 0.048), (0.100, 0.055),
             (0.030, 0.492), (0.0295, 0.499), (0.027, 0.505), (0.0230, 0.5035), (0.0215, 0.497),
             (0.0205, 0.465), (0.0, 0.465)],
    stripes=[(0.285, 0.370), (0.115, 0.200)],
    keypoint_z=[0.370, 0.285, 0.200, 0.115],
    keypoint_names=['upper_stripe_top', 'upper_stripe_bottom', 'lower_stripe_top', 'lower_stripe_bottom'],
)


def cone_spec(class_id):
    return BIG if class_id == 1 else SMALL


def body_radius(spec, z):
    t = (z - spec['body_z0']) / (spec['body_z1'] - spec['body_z0'])
    return spec['body_r0'] + t * (spec['body_r1'] - spec['body_r0'])


def canonical_rings(class_id):
    """(z, r) of every keypoint ring, in BRT pair order."""
    s = cone_spec(class_id)
    return [(z, body_radius(s, z)) for z in s['keypoint_z']]


def silhouette_points(class_id, cam_pos_cone):
    """True view-dependent keypoints in the cone frame, shape (8, 3); NaN rows are undefined.

    Body surface: r(z) = a - s*z. Its surface normal is ~(cos p, sin p, s), so the outline
    seen from camera centre c satisfies cx*cos p + cy*sin p = a - s*cz, which is independent
    of z: the outline is a straight generator line. Each ring point lies on it.
    Returned unordered left/right: index 2k and 2k+1 are assigned later by image x.
    """
    s = cone_spec(class_id)
    slope = (s['body_r0'] - s['body_r1']) / (s['body_z1'] - s['body_z0'])
    a = s['body_r0'] + slope * s['body_z0']
    cx, cy, cz = cam_pos_cone
    rho = math.hypot(cx, cy)
    out = np.full((8, 3), np.nan)
    c = (a - slope * cz) / rho
    if rho < 1e-6 or abs(c) >= 1:
        return out  # camera inside the cone's tangent cone (looking down its axis)
    th, d = math.atan2(cy, cx), math.acos(c)
    for k, z in enumerate(s['keypoint_z']):
        r = body_radius(s, z)
        for j, p in enumerate((th + d, th - d)):
            out[2 * k + j] = (r * math.cos(p), r * math.sin(p), z)
    return out


# ---------------------------------------------------------------- camera

def distort_normalized(xy, dist):
    """OpenCV radial-tangential model: undistorted normalized coords -> distorted."""
    k1, k2, p1, p2, k3 = dist
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], -1)


def undistort_normalized(xyd, dist, iters=20):
    """Inverse of distort_normalized by fixed-point iteration (as cv2.undistortPoints)."""
    k1, k2, p1, p2, k3 = dist
    xy = xyd.copy()
    for _ in range(iters):
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        xy = np.stack([(xyd[..., 0] - dx) / radial, (xyd[..., 1] - dy) / radial], -1)
    return xy


def project(points_cam, K, dist):
    """Camera-frame points (N, 3) -> distorted OpenCV pixels (N, 2)."""
    p = np.asarray(points_cam, float)
    xy = p[:, :2] / p[:, 2:3]
    d = distort_normalized(xy, dist)
    return np.stack([K[0][0] * d[:, 0] + K[0][2], K[1][1] * d[:, 1] + K[1][2]], -1)


def render_margins(K, dist, width, height, pad=8):
    """Undistorted overscan (left, top, right, bottom) in pixels needed to fill the distorted image."""
    u = np.concatenate([np.linspace(0, width - 1, 64), np.full(64, width - 1.0),
                        np.linspace(0, width - 1, 64), np.zeros(64)])
    v = np.concatenate([np.zeros(64), np.linspace(0, height - 1, 64),
                        np.full(64, height - 1.0), np.linspace(0, height - 1, 64)])
    xyd = np.stack([(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1]], -1)
    xy = undistort_normalized(xyd, dist)
    ur, vr = xy[:, 0] * K[0][0] + K[0][2], xy[:, 1] * K[1][1] + K[1][2]
    return (int(math.ceil(max(0, -ur.min()))) + pad, int(math.ceil(max(0, -vr.min()))) + pad,
            int(math.ceil(max(0, ur.max() - (width - 1)))) + pad,
            int(math.ceil(max(0, vr.max() - (height - 1)))) + pad)


# Generic FS car camera: 1920x1200 global shutter (IMX174-class, 11.25 mm wide sensor)
# behind a 6.7 mm lens -> 80 deg horizontal FOV. One fixed calibration for the dataset.
WIDTH, HEIGHT = 1920, 1200
SENSOR_WIDTH_MM = 11.25
FX = WIDTH / 2 / math.tan(math.radians(40))
CAMERA = dict(
    width=WIDTH, height=HEIGHT,
    K=[[FX, 0.0, 963.4], [0.0, FX, 596.2], [0.0, 0.0, 1.0]],
    dist=[-0.0820, 0.0115, 0.00035, -0.00028, 0.0],
    focal_length_mm=SENSOR_WIDTH_MM * FX / WIDTH,
    sensor_width_mm=SENSOR_WIDTH_MM,
    pixel_size_um=SENSOR_WIDTH_MM / WIDTH * 1000,
)
