// Adapter: turns cone-live/1 messages into the result-like object the Viewer understands.
// Scene = keyframes of the latest map + the live frames since that map.
export const CLASS_NAMES = { 0: 'BLUE', 1: 'ORANGE_BIG', 2: 'ORANGE', 3: 'UNDEFINED', 4: 'YELLOW' };
const CONE_DIMS = { 1: { height: 0.505, base: 0.285 } };
const TAIL_MAX = 400;

// frame message -> frame the Overlay / Viewer read (class_name, det_id, range_m, index)
export function adaptFrame(f) {
  return {
    ...f, index: f.i,
    detections: (f.detections || []).map((d, k) => ({
      ...d, det_id: k, class_name: CLASS_NAMES[d.class_id] || 'UNDEFINED',
      range_m: d.p_cam ? Math.hypot(...d.p_cam) : null,
    })),
  };
}

export class LiveScene {
  constructor(viewer) {
    this.viewer = viewer;
    this.imageSize = [1280, 720];
    this.K = [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]];
    this.map = null; this.landmarks = []; this.built = false; this.fitted = false;
  }

  setHello(h) {
    if (h.image_size) this.imageSize = h.image_size;
    if (h.K) this.K = h.K;
  }

  // map: latest map message or null; tail: adapted frames after map.t (time order); cur: latest adapted frame or null
  update(map, tail, cur) {
    if (tail.length > TAIL_MAX) tail = tail.slice(-TAIL_MAX);
    const K = this.K;
    const frames = ((map && map.keyframes) || []).map((k) => ({ index: k.i, t: k.t, status: 'tracked', T_world_cam: k.T_world_cam, K, detections: [] }));
    frames.push(...tail);
    if (!tail.length && cur) frames.push(cur); // a pose-less cur shows the last pose, greyed
    const mapChanged = map !== this.map || !this.built;
    if (mapChanged) {
      this.map = map;
      this.landmarks = ((map && map.landmarks) || []).map((l) => ({ ...l, class_name: CLASS_NAMES[l.class_id] || 'UNDEFINED' }));
      this.viewer.setResult({ frames, landmarks: this.landmarks, image_size: this.imageSize, cone_dims: CONE_DIMS }, { keepCamera: true });
      this.built = true;
    } else {
      this.viewer.updateFrames(frames);
    }
    if (!this.fitted && (this.landmarks.length || frames.some((f) => f.T_world_cam))) { this.viewer.fit(); this.fitted = true; }
  }

  reset() { this.map = null; this.built = false; this.fitted = false; this.landmarks = []; }
}
