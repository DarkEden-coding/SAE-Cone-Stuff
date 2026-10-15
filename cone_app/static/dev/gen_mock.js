#!/usr/bin/env node
// Generates mock_jobs.json, mock_result_video.json, mock_result_photo.json for ?mock=1.
// Usage: node gen_mock.js   (writes next to this script)
'use strict';
const fs = require('fs');
const path = require('path');

let seed = 20261015;
function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }
const r2 = (x) => Math.round(x * 100) / 100;
const r3 = (x) => Math.round(x * 1000) / 1000;

const W = 1920, H = 1080, FX = 1500, FY = 1500, CX = 960, CY = 540;
const K = [[FX, 0, CX], [0, FY, CY], [0, 0, 1]];
const CLASS_NAMES = { 0: 'BLUE', 1: 'ORANGE_BIG', 2: 'ORANGE', 3: 'UNDEFINED', 4: 'YELLOW' };
const CONE_DIMS = {
  0: { height: 0.325, base: 0.228 }, 1: { height: 0.505, base: 0.285 }, 2: { height: 0.325, base: 0.228 },
  3: { height: 0.325, base: 0.228 }, 4: { height: 0.325, base: 0.228 },
};
const RING_FRAC = { small: [0.80, 0.50, 0.15], big: [0.85, 0.62, 0.38, 0.12] };

// ---- vec helpers
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

// ---- centreline (arc-length param, dense samples)
const DS = 0.05, L = 250;
const theta = (s) => 0.55 * Math.sin(s / 38) + 0.25 * Math.sin(s / 14.5) - 0.0;
const line = [];
{ let x = 0, y = 0; for (let s = 0; s <= L + 1e-9; s += DS) { line.push({ s, x, y, th: theta(s) - theta(0) }); x += Math.cos(theta(s) - theta(0)) * DS; y += Math.sin(theta(s) - theta(0)) * DS; } }
const at = (s) => line[Math.min(line.length - 1, Math.max(0, Math.round(s / DS)))];

// ---- cones
const cones = []; // {cls, p:[x,y,0]}
for (let s = 6; s < L - 4; s += 4) {
  const c = at(s), nx = -Math.sin(c.th), ny = Math.cos(c.th);
  const big = (Math.abs(s - 102) < 3) || (Math.abs(s - 106) < 3);
  const small = Math.abs(s - 182) < 2;
  for (const side of [1, -1]) {
    let cls = side > 0 ? 0 : 4;
    if (big) cls = 1; else if (small) cls = 2;
    const off = side * (1.75 + (big ? 0.5 : 0) + 0.08 * gauss());
    cones.push({ cls, p: [c.x + nx * off + 0.08 * gauss(), c.y + ny * off + 0.08 * gauss(), 0] });
  }
}

// ---- camera pose
function rotFor(yaw, pitch) {
  const f = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), -Math.sin(pitch)];
  const r = [Math.sin(yaw), -Math.cos(yaw), 0];
  const d = cross(f, r);
  return { x: r, y: d, z: f }; // camera axes in world
}
function camPoseAt(t) {
  const s = 4.0 * t;
  const c = at(s);
  const lat = 0.2 * Math.sin(t * 0.9) * Math.min(1, t / 2);
  const yaw = c.th + 0.03 * Math.sin(t * 1.3) * Math.min(1, t / 2);
  const nx = -Math.sin(c.th), ny = Math.cos(c.th);
  const pos = [c.x + nx * lat, c.y + ny * lat, 1.1 + 0.02 * Math.sin(t * 3.1)];
  const pitch = 0.07 + 0.01 * Math.sin(t * 0.7);
  return { pos, axes: rotFor(yaw, pitch) };
}
function toMatrix(pose) {
  const { x, y, z } = pose.axes, p = pose.pos;
  return [[r3(x[0]), r3(y[0]), r3(z[0]), r3(p[0])], [r3(x[1]), r3(y[1]), r3(z[1]), r3(p[1])],
          [r3(x[2]), r3(y[2]), r3(z[2]), r3(p[2])], [0, 0, 0, 1]];
}
const toCam = (pose, pw) => { const d = sub(pw, pose.pos); return [dot(d, pose.axes.x), dot(d, pose.axes.y), dot(d, pose.axes.z)]; };
const proj = (pc) => [FX * pc[0] / pc[2] + CX, FY * pc[1] / pc[2] + CY];

// ---- detections for one pose; cones given in the pose's world frame
function detect(pose, coneList, maxRange) {
  const out = [];
  coneList.forEach((cone, ci) => {
    const base = toCam(pose, cone.p);
    const range = norm(base);
    if (base[2] < 1.5 || range > maxRange) return;
    if (rnd() > Math.min(1, 1.15 - range / 45)) return; // dropouts rise with range
    const dims = CONE_DIMS[cone.cls], h = dims.height;
    const fr = cone.cls === 1 ? RING_FRAC.big : RING_FRAC.small;
    const upc = [dot([0, 0, 1], pose.axes.x), dot([0, 0, 1], pose.axes.y), dot([0, 0, 1], pose.axes.z)];
    const kps = [], allPts = [];
    for (const f of fr) {
      const z = h * f, r = (dims.base / 2 * 0.9) * (1 - f) + 0.02 * f;
      const c = [base[0] + upc[0] * z, base[1] + upc[1] * z, base[2] + upc[2] * z];
      const perp = unit(cross(upc, unit(c)));
      const a = proj([c[0] + perp[0] * r, c[1] + perp[1] * r, c[2] + perp[2] * r]);
      const b = proj([c[0] - perp[0] * r, c[1] - perp[1] * r, c[2] - perp[2] * r]);
      const [l, rr] = a[0] < b[0] ? [a, b] : [b, a];
      kps.push(l, rr);
    }
    const sig = 0.4 + range * 0.03;
    const noisy = kps.map((p) => [p[0] + sig * gauss(), p[1] + sig * gauss()]);
    const apex = proj([base[0] + upc[0] * h, base[1] + upc[1] * h, base[2] + upc[2] * h]);
    const bl = proj([base[0] - dims.base / 2, base[1], base[2]]), br = proj([base[0] + dims.base / 2, base[1], base[2]]);
    const bp = proj(base);
    const xs = [...noisy.map((p) => p[0]), apex[0], bl[0], br[0]], ys = [...noisy.map((p) => p[1]), apex[1], bp[1] + 3];
    const box = [Math.min(...xs) - 2, Math.min(...ys) - 2, Math.max(...xs) + 2, Math.max(...ys) + 2];
    if (box[0] < 4 || box[1] < 4 || box[2] > W - 4 || box[3] > H - 4) return;
    if (box[3] - box[1] < 9) return;
    const conf = Math.max(0.3, Math.min(0.99, 0.96 - range * 0.011 + 0.04 * gauss()));
    const kp = [], kpRaw = [];
    for (let i = 0; i < 8; i++) {
      if (i < noisy.length) {
        const kc = Math.max(0.2, Math.min(0.99, 0.95 - range * 0.006 + 0.05 * gauss()));
        kp.push([r2(noisy[i][0]), r2(noisy[i][1]), r2(kc)]);
        kpRaw.push([r2(noisy[i][0] + 0.7), r2(noisy[i][1] + 1.1 + 0.02 * (box[3] - box[1])), r2(kc)]);
      } else { kp.push([0, 0, 0]); kpRaw.push([0, 0, 0]); }
    }
    const e = 0.0015 * range * range;
    const pcam = [base[0] + e * gauss(), base[1] + e * gauss(), base[2] + e * gauss()];
    out.push({
      ci, class_id: cone.cls, class_name: CLASS_NAMES[cone.cls], conf: r2(conf), box_xyxy: box.map(r2),
      keypoints: kp, keypoints_raw: kpRaw, p_cam: pcam.map(r3), range_m: r2(norm(pcam)),
      reproj_rms_px: r2(0.3 + 0.9 * rnd()),
    });
  });
  return out;
}

// ---- video
const FPS = 30, STRIDE = 3, DUR = 60, N_PROC = Math.floor(DUR * FPS / STRIDE);
const LOST = [300, 319];
const lmByCone = new Map(), lmList = [];
const videoFrames = [];
for (let k = 0; k < N_PROC; k++) {
  const idx = k * STRIDE, t = idx / FPS;
  const pose = camPoseAt(t);
  const lost = k >= LOST[0] && k <= LOST[1];
  const dets = detect(pose, cones, 32);
  const frame = {
    index: idx, t: r3(t), K,
    T_world_cam: lost ? null : toMatrix(pose),
    status: lost ? 'lost' : (k === 0 ? 'init' : 'tracked'),
    detections: dets.map((d, i) => {
      let lid = null;
      if (!lost) {
        if (!lmByCone.has(d.ci)) { lmByCone.set(d.ci, lmList.length); lmList.push({ ci: d.ci, cls: d.class_id, n: 0, first: idx, last: idx, acc: [0, 0, 0] }); }
        lid = lmByCone.get(d.ci);
        const lm = lmList[lid]; lm.n++; lm.last = idx;
        for (let j = 0; j < 3; j++) lm.acc[j] += cones[d.ci].p[j];
      }
      const { ci, ...rest } = d;
      return { det_id: i, ...rest, landmark_id: lid };
    }),
  };
  videoFrames.push(frame);
}
const videoLandmarks = lmList.map((l, id) => {
  const p = cones[l.ci].p, s = 0.5 / Math.sqrt(l.n);
  return {
    id, class_id: l.cls, class_name: CLASS_NAMES[l.cls],
    p_world: [r3(p[0] + 0.1 * gauss() * Math.sqrt(1 / l.n)), r3(p[1] + 0.1 * gauss() * Math.sqrt(1 / l.n)), 0],
    sigma_m: [r3(s * 0.6), r3(s * 0.6), r3(0.02)], n_obs: l.n, first_frame: l.first, last_frame: l.last,
  };
});
const nLost = videoFrames.filter((f) => f.status === 'lost').length;
const videoResult = {
  schema: 'cone-result/1', job_id: 'mock-video', kind: 'video', image_size: [W, H],
  model: { weights: 'runs/by-n-E-far/weights/best.pt', imgsz: 1920, kp_correction: 'assets/kp_correction_E.json' },
  options: { stride: null, target_fps: 10, imgsz: 1920, conf: 0.25, kp_conf: 0.5, kp_correction: true, flat_ground: true, max_range_m: 40 },
  class_names: CLASS_NAMES, cone_dims: CONE_DIMS, frames: videoFrames, landmarks: videoLandmarks,
  ground_plane: { camera_height_m: 1.1, normal_world: [0, 0, 1] },
  stats: { n_frames_total: DUR * FPS, n_frames_processed: N_PROC, n_tracked: N_PROC - nLost, n_lost: nLost,
           n_landmarks: videoLandmarks.length, runtime_s: 41.7, detector_ms_per_frame: 24.8 },
};

// ---- photo: frame at t=14 s re-expressed in a photo world (camera at origin, x forward)
const tp = 14, pp = camPoseAt(tp);
const yaw0 = Math.atan2(pp.axes.z[1], pp.axes.z[0]);
const cy0 = Math.cos(-yaw0), sy0 = Math.sin(-yaw0);
const conesP = cones.map((c) => {
  const dx = c.p[0] - pp.pos[0], dy = c.p[1] - pp.pos[1];
  return { cls: c.cls, p: [dx * cy0 - dy * sy0, dx * sy0 + dy * cy0, 0] };
});
const photoPose = { pos: [0, 0, 1.1], axes: rotFor(0, 0.07) };
const pdets = detect(photoPose, conesP, 40);
const photoLandmarks = pdets.map((d, i) => {
  const p = conesP[d.ci].p, s = 0.02 * d.range_m;
  return { id: i, class_id: d.class_id, class_name: d.class_name,
           p_world: [r3(p[0] + 0.03 * gauss()), r3(p[1] + 0.03 * gauss()), 0],
           sigma_m: [r3(s), r3(s * 0.5), r3(0.02)], n_obs: 1, first_frame: 0, last_frame: 0 };
});
const photoResult = {
  schema: 'cone-result/1', job_id: 'mock-photo', kind: 'photo', image_size: [W, H],
  model: videoResult.model, options: videoResult.options, class_names: CLASS_NAMES, cone_dims: CONE_DIMS,
  frames: [{ index: 0, t: 0, K, T_world_cam: toMatrix(photoPose), status: 'single',
             detections: pdets.map((d, i) => { const { ci, ...rest } = d; return { det_id: i, ...rest, landmark_id: i }; }) }],
  landmarks: photoLandmarks, ground_plane: { camera_height_m: 1.1, normal_world: [0, 0, 1] },
  stats: { n_frames_total: 1, n_frames_processed: 1, n_tracked: 0, n_lost: 0, n_landmarks: photoLandmarks.length,
           runtime_s: 0.8, detector_ms_per_frame: 31.0 },
};

// ---- jobs
const now = Date.now();
const iso = (ms) => new Date(now - ms).toISOString();
const opt = videoResult.options;
const mk = (o) => ({ error: null, options: opt, image_size: [W, H], n_frames: null, duration_s: null,
                     media_url: null, result_url: null, source: 'ios', ...o });
const jobs = [
  mk({ id: 'mock-running', name: 'Parking lot run 3', kind: 'video', status: 'running', stage: 'detecting', progress: 0.42,
       created_at: iso(90e3), updated_at: iso(2e3), n_frames: 1800, duration_s: 60, media_url: '/api/jobs/mock-running/media' }),
  mk({ id: 'mock-video', name: 'Autocross lap (mock)', kind: 'video', status: 'done', stage: 'writing', progress: 1,
       created_at: iso(3600e3), updated_at: iso(3500e3), n_frames: DUR * FPS, duration_s: DUR,
       media_url: '/api/jobs/mock-video/media', result_url: '/api/jobs/mock-video/result' }),
  mk({ id: 'mock-photo', name: 'Straight, 5-30 m (mock)', kind: 'photo', status: 'done', stage: 'writing', progress: 1,
       created_at: iso(7200e3), updated_at: iso(7190e3), n_frames: 1, source: 'web',
       media_url: '/api/jobs/mock-photo/media', result_url: '/api/jobs/mock-photo/result' }),
  mk({ id: 'mock-failed', name: 'Broken upload', kind: 'video', status: 'failed', stage: 'decoding', progress: 0.05,
       error: 'Could not decode video: moov atom not found', created_at: iso(86400e3), updated_at: iso(86300e3), source: 'web' }),
];

const dir = __dirname;
fs.writeFileSync(path.join(dir, 'mock_jobs.json'), JSON.stringify(jobs, null, 1));
fs.writeFileSync(path.join(dir, 'mock_result_video.json'), JSON.stringify(videoResult));
fs.writeFileSync(path.join(dir, 'mock_result_photo.json'), JSON.stringify(photoResult));
console.log(`video: ${N_PROC} frames, ${videoLandmarks.length} landmarks; photo: ${pdets.length} detections; cones total ${cones.length}`);
