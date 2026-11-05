// Offline data for ?mock=1: a synthetic cone-live/1 stream (a camera driving down a lane of cones).
const W = 1280, H = 720, FX = 1000, K = [[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]];
const HZ = 10, DURATION = 30, CAM_Z = 1.2, SPEED = 1.5, CONE_H = 0.325;
const CONES = [];
for (let i = 0; i < 12; i++) {
  CONES.push({ id: 2 * i, class_id: 0, p: [3 + i * 4, 2, 0] });
  CONES.push({ id: 2 * i + 1, class_id: 4, p: [3 + i * 4, -2, 0] });
}
const r3 = (x) => Math.round(x * 1000) / 1000;

function pose(t) {
  const x = SPEED * t, y = 0.3 * Math.sin(t * 0.7);
  return [[0, 0, 1, x], [-1, 0, 0, y], [0, -1, 0, CAM_Z], [0, 0, 0, 1]];
}
// camera looks along +x: cam x = -world y, cam y = -world z, cam z = world x
function project(T, p) {
  const d = [p[0] - T[0][3], p[1] - T[1][3], p[2] - T[2][3]];
  const c = [-d[1], -d[2], d[0]];
  return { c, uv: c[2] > 0.1 ? [FX * c[0] / c[2] + W / 2, FX * c[1] / c[2] + H / 2] : null };
}

function detection(T, cone) {
  const base = project(T, cone.p), top = project(T, [cone.p[0], cone.p[1], CONE_H]);
  if (!base.uv || !top.uv) return null;
  const z = base.c[2], hw = 0.114 * FX / z;
  const box = [base.uv[0] - hw, top.uv[1], base.uv[0] + hw, base.uv[1]];
  if (box[2] < 0 || box[0] > W || box[3] < 0 || box[1] > H || z > 14) return null;
  const kp = [];
  for (const f of [0.15, 0.5, 0.85]) {
    const r = 0.114 * (1 - f) * FX / z, y = base.uv[1] + (top.uv[1] - base.uv[1]) * f;
    kp.push([r3(base.uv[0] - r), r3(y), 0.9], [r3(base.uv[0] + r), r3(y), 0.9]);
  }
  kp.push([0, 0, 0], [0, 0, 0]);
  return { class_id: cone.class_id, conf: 0.9, box_xyxy: box.map(r3), keypoints: kp, p_cam: base.c.map(r3), landmark_id: cone.id };
}

let cache = null;
function messages() {
  if (cache) return cache;
  const out = [{ type: 'hello', schema: 'cone-live/1', session_id: 'mock-live', t: 0, started_at: new Date().toISOString(), name: 'Mock live drive',
    device: { model: 'mock', system: 'mock', app_version: '0' }, image_size: [W, H], K, model: { name: 'mock', imgsz: [736, 1280] },
    options: { conf: 0.25, kp_conf: 0.5, kp_correction: true, flat_ground: true, gravity_prior: true } }];
  const kfs = [];
  let nextMap = 0.5;
  const n = DURATION * HZ;
  out.push({ type: 'recording', t: 4, event: 'start', recording_id: 'mock-rec', name: 'mock recording' });
  for (let i = 0; i < n; i++) {
    const t = i / HZ, T = pose(t);
    const status = t < 1 ? 'waiting' : t < 1.2 ? 'init' : (t > 20 && t < 21 ? 'lost' : 'tracked');
    const dets = status === 'waiting' ? [] : CONES.map((c) => detection(T, c)).filter(Boolean);
    const rec = t >= 4 && t < 24 ? { recording_id: 'mock-rec', t: r3(t - 4) } : null;
    out.push({ type: 'frame', i, t: r3(t), status, T_world_cam: status === 'waiting' ? null : T, K, image_size: [W, H], detections: dets,
      n_inliers: dets.length, rec, stats: { det_ms: 18, slam_ms: 6, fps: HZ } });
    if (status !== 'waiting' && i % 5 === 0) kfs.push({ i, t: r3(t), T_world_cam: T });
    if (t >= nextMap && status !== 'waiting') {
      nextMap += 0.5;
      const lms = CONES.filter((c) => c.p[0] < SPEED * t + 14).map((c) => ({ id: c.id, class_id: c.class_id, p_world: c.p, sigma_m: [0.05, 0.05, 0.02], n_obs: 10, confirmed: true }));
      out.push({ type: 'map', t: r3(t + 0.01), landmarks: lms, keyframes: kfs.slice() });
    }
    if (t === 24) out.push({ type: 'recording', t, event: 'stop', recording_id: 'mock-rec', name: 'mock recording' });
  }
  out.push({ type: 'bye', t: DURATION });
  return (cache = out);
}

export function mockSessions() {
  const now = Date.now();
  return [
    { id: 'mock-live', name: 'Mock live drive', device: { model: 'mock' }, started_at: new Date(now - 5000).toISOString(), updated_at: new Date(now).toISOString(),
      active: true, duration_s: 5, n_frames: 50, n_landmarks: 6, recordings: [] },
    { id: 'mock-done', name: 'Finished drive', device: { model: 'mock' }, started_at: new Date(now - 3600e3).toISOString(), updated_at: new Date(now - 3500e3).toISOString(),
      active: false, duration_s: DURATION, n_frames: DURATION * HZ, n_landmarks: 24,
      recordings: [{ recording_id: 'mock-rec', name: 'mock recording', t_start: 4, t_end: 24, job_id: null, has_log: true }] },
  ];
}

// NDJSON text; a recording log is the slice between its start/stop (plus hello and the last map before)
export function mockLog(recordingId) {
  let msgs = messages();
  if (recordingId) {
    const hello = msgs[0], inRec = msgs.filter((m) => m.t >= 4 && m.t <= 24 && m.type !== 'hello');
    const prevMap = msgs.filter((m) => m.type === 'map' && m.t < 4).pop();
    msgs = [hello, ...(prevMap ? [prevMap] : []), ...inRec];
  }
  return msgs.map((m) => JSON.stringify(m)).join('\n') + '\n';
}

// Minimal WebSocket stand-in: replays the stream in real time, ends with "bye".
export class MockSocket {
  constructor() {
    this.readyState = 0; this.timers = []; this.t0 = performance.now() + 100;
    const msgs = messages();
    setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 50);
    msgs.forEach((m) => this.timers.push(setTimeout(() => this.onmessage?.({ data: JSON.stringify(m) }), 100 + m.t * 1000)));
    this.timers.push(setTimeout(() => this.close(), 100 + DURATION * 1000 + 100));
    this.timers.push(setInterval(() => this.sendImage(), 200));
  }
  // binary preview message: JSON header, newline, JPEG (a gradient with the projected cones as dots)
  sendImage() {
    if (this.readyState !== 1) return;
    const t = (performance.now() - this.t0) / 1000, T = pose(t);
    const c = document.createElement('canvas'); c.width = 480; c.height = 270;
    const g = c.getContext('2d'), k = c.width / W;
    const bg = g.createLinearGradient(0, 0, 0, c.height);
    bg.addColorStop(0, '#46586a'); bg.addColorStop(0.5, '#222'); bg.addColorStop(1, '#14181c');
    g.fillStyle = bg; g.fillRect(0, 0, c.width, c.height);
    for (const cone of CONES) {
      const d = detection(T, cone);
      if (!d) continue;
      g.fillStyle = cone.class_id === 0 ? '#2f80ff' : '#ffd400';
      g.fillRect(d.box_xyxy[0] * k, d.box_xyxy[1] * k, (d.box_xyxy[2] - d.box_xyxy[0]) * k, (d.box_xyxy[3] - d.box_xyxy[1]) * k);
    }
    c.toBlob(async (blob) => {
      if (!blob || this.readyState !== 1) return;
      const head = new TextEncoder().encode(JSON.stringify({ type: 'image', t: r3(t), size: [c.width, c.height], image_size: [W, H] }) + '\n');
      const body = new Uint8Array(await blob.arrayBuffer());
      const out = new Uint8Array(head.length + body.length);
      out.set(head); out.set(body, head.length);
      this.onmessage?.({ data: out.buffer });
    }, 'image/jpeg', 0.5);
  }
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3; this.timers.forEach((id) => { clearTimeout(id); clearInterval(id); });
    this.onclose?.({ code: 1000 });
  }
}
