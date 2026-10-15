// Three.js 3D viewport. World frame is z-up (camera.up = (0,0,1)); camera frames are OpenCV (x right, y down, z fwd).
import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { classColor, STATUS_COLORS } from './util.js';

const BG = 0x0b0e14;
const FRUSTUM_DEPTH = 0.6;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(BG, 1);
    this.renderer.autoClear = false;
    container.append(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BG);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 1500);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-8, -8, 6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => { this.dirty = true; });

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dl = new THREE.DirectionalLight(0xffffff, 1.4);
    dl.position.set(6, -9, 14);
    this.scene.add(dl);

    this.world = new THREE.Group(); this.scene.add(this.world);
    this.opts = { ids: false, ellipses: false, lines: false };
    this.mode = 'orbit';
    this.result = null; this.cur = 0;
    this.dirty = true; this.follow = null; this.saved = null;
    this.lmObjs = new Map();

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    this.last = performance.now();
    const loop = (t) => { this.raf = requestAnimationFrame(loop); this.tick(t); };
    this.raf = requestAnimationFrame(loop);
  }

  destroy() {
    cancelAnimationFrame(this.raf); this.ro.disconnect(); this.controls.dispose();
    this.world.traverse((o) => { o.geometry?.dispose?.(); if (o.material) [].concat(o.material).forEach((m) => { m.map?.dispose?.(); m.dispose(); }); });
    this.renderer.dispose(); this.renderer.forceContextLoss?.();
    this.renderer.domElement.remove();
  }

  // ---------------- data ----------------
  setResult(result) {
    this.result = result;
    this.world.traverse((o) => { o.geometry?.dispose?.(); });
    this.world.clear();
    this.lmObjs.clear();
    const frames = result.frames;
    const [W, H] = result.image_size;
    this.imgSize = [W, H];

    // poses
    this.poses = frames.map((f) => {
      if (!f.T_world_cam) return null;
      const m = new THREE.Matrix4(); m.set(...f.T_world_cam.flat());
      return m;
    });
    this.camPos = this.poses.map((m) => (m ? new THREE.Vector3().setFromMatrixPosition(m) : null));
    this.lastValid = [];
    let lv = -1;
    for (let i = 0; i < frames.length; i++) { if (this.poses[i]) lv = i; this.lastValid.push(lv); }
    const firstValid = this.poses.findIndex(Boolean);
    if (firstValid > 0) for (let i = 0; i < firstValid; i++) this.lastValid[i] = firstValid;

    // bounds
    const bb = new THREE.Box3();
    for (const l of result.landmarks || []) bb.expandByPoint(new THREE.Vector3(...l.p_world));
    for (const p of this.camPos) if (p) bb.expandByPoint(p);
    if (bb.isEmpty()) bb.set(new THREE.Vector3(-5, -5, 0), new THREE.Vector3(5, 5, 1));
    bb.expandByPoint(new THREE.Vector3(0, 0, 0));
    this.bounds = bb;

    this.buildGrid(bb);
    const axes = new THREE.AxesHelper(2); this.world.add(axes);
    this.buildLandmarks(result);
    this.buildTrajectory(frames);
    this.buildFrustum();
    this.buildLinks();
    this.fit(true);
    this.setFrame(0);
  }

  buildGrid(bb) {
    const pad = 10;
    const x0 = Math.floor((bb.min.x - pad) / 5) * 5, x1 = Math.ceil((bb.max.x + pad) / 5) * 5;
    const y0 = Math.floor((bb.min.y - pad) / 5) * 5, y1 = Math.ceil((bb.max.y + pad) / 5) * 5;
    const minor = [], major = [];
    const small = (x1 - x0) * (y1 - y0) < 150000;
    for (let x = x0; x <= x1; x++) { const a = x % 5 === 0 ? major : minor; if (a === minor && !small) continue; a.push(x, y0, 0, x, y1, 0); }
    for (let y = y0; y <= y1; y++) { const a = y % 5 === 0 ? major : minor; if (a === minor && !small) continue; a.push(x0, y, 0, x1, y, 0); }
    const mk = (arr, color) => {
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
      return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
    };
    this.world.add(mk(minor, 0x1c2230), mk(major, 0x3a455c));
  }

  buildLandmarks(result) {
    const dims = result.cone_dims || {};
    this.ellipseGeo = new THREE.BufferGeometry();
    const pts = [];
    for (let i = 0; i < 48; i++) pts.push(Math.cos(i / 48 * Math.PI * 2), Math.sin(i / 48 * Math.PI * 2), 0);
    this.ellipseGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.lmGroup = new THREE.Group(); this.world.add(this.lmGroup);
    for (const lm of result.landmarks || []) {
      const d = dims[lm.class_id] || { height: 0.325, base: 0.228 };
      const col = new THREE.Color(classColor(lm.class_name));
      const g = new THREE.Group();
      g.position.set(...lm.p_world);
      const r0 = d.base / 2 * 0.9, rT = 0.02;
      const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.55, metalness: 0 });
      const body = new THREE.Mesh(new THREE.CylinderGeometry(rT, r0, d.height, 28).rotateX(Math.PI / 2).translate(0, 0, d.height / 2), mat);
      const plate = new THREE.Mesh(new THREE.BoxGeometry(d.base, d.base, 0.012).translate(0, 0, 0.006),
        new THREE.MeshStandardMaterial({ color: 0x20242c, roughness: 0.8 }));
      g.add(body, plate);
      if (lm.class_name === 'BLUE' || lm.class_name === 'YELLOW') {
        const z1 = d.height * 0.42, z2 = d.height * 0.68;
        const rr = (z) => (r0 + (rT - r0) * z / d.height) * 1.012;
        const band = new THREE.Mesh(new THREE.CylinderGeometry(rr(z2), rr(z1), z2 - z1, 28, 1, true).rotateX(Math.PI / 2).translate(0, 0, (z1 + z2) / 2),
          new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.6, side: THREE.DoubleSide }));
        g.add(band);
      } else if (lm.class_id === 1) {
        for (const [a, b] of [[0.3, 0.45], [0.62, 0.77]]) {
          const z1 = d.height * a, z2 = d.height * b, rr = (z) => (r0 + (rT - r0) * z / d.height) * 1.012;
          g.add(new THREE.Mesh(new THREE.CylinderGeometry(rr(z2), rr(z1), z2 - z1, 28, 1, true).rotateX(Math.PI / 2).translate(0, 0, (z1 + z2) / 2),
            new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.6, side: THREE.DoubleSide })));
        }
      }
      const s = lm.sigma_m || [0, 0, 0];
      const ell = new THREE.LineLoop(this.ellipseGeo, new THREE.LineBasicMaterial({ color: col }));
      ell.scale.set(Math.max(2 * s[0], 0.02), Math.max(2 * s[1], 0.02), 1);
      ell.position.z = 0.015; ell.visible = this.opts.ellipses;
      g.add(ell);
      this.lmGroup.add(g);
      this.lmObjs.set(lm.id, { group: g, mat, color: col, ell, height: d.height, lm, sprite: null, lit: false });
    }
    this.updateIds();
  }

  idSprite(id) {
    const c = document.createElement('canvas'); c.width = 128; c.height = 64;
    const g = c.getContext('2d');
    g.fillStyle = 'rgba(10,12,18,0.8)'; g.beginPath(); g.roundRect(4, 8, 120, 48, 12); g.fill();
    g.fillStyle = '#fff'; g.font = '600 34px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('#' + id, 64, 33);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(0.6, 0.3, 1); sp.renderOrder = 20;
    return sp;
  }

  updateIds() {
    for (const o of this.lmObjs.values()) {
      if (this.opts.ids && !o.sprite) { o.sprite = this.idSprite(o.lm.id); o.sprite.position.z = o.height + 0.25; o.group.add(o.sprite); }
      if (o.sprite) o.sprite.visible = this.opts.ids;
    }
  }

  buildTrajectory(frames) {
    const pos = [], cnt = new Array(frames.length).fill(0);
    let n = 0;
    for (let i = 0; i < frames.length; i++) {
      if (i > 0 && this.camPos[i] && this.camPos[i - 1] && frames[i].status !== 'lost' && frames[i - 1].status !== 'lost') {
        const a = this.camPos[i - 1], b = this.camPos[i];
        pos.push(a.x, a.y, a.z, b.x, b.y, b.z); n++;
      }
      cnt[i] = n;
    }
    this.segCount = cnt;
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const dim = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x4a5a7a, transparent: true, opacity: 0.8 }));
    // separate geometry instance so draw ranges are independent
    const bright = new THREE.LineSegments(g.clone(), new THREE.LineBasicMaterial({ color: 0x5eead4 }));
    bright.frustumCulled = false; bright.geometry.setDrawRange(0, 0);
    dim.frustumCulled = false;
    this.trajDim = dim; this.trajBright = bright;
    this.world.add(dim, bright);
    this.hasTraj = n > 0;
  }

  buildFrustum() {
    this.frustum = new THREE.Group();
    this.frustum.matrixAutoUpdate = false;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 2 * 16), 3));
    this.frustumLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: STATUS_COLORS.tracked }));
    this.frustumLines.frustumCulled = false;
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 4), 3));
    pg.setIndex([0, 1, 2, 0, 2, 3]);
    this.frustumPlane = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ color: STATUS_COLORS.tracked, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
    this.frustumPlane.frustumCulled = false;
    const axes = new THREE.AxesHelper(0.35);
    this.frustum.add(this.frustumLines, this.frustumPlane, axes);
    this.world.add(this.frustum);
    this.frustumKey = '';
  }

  setFrustumGeometry(K, W, H) {
    const key = [K[0][0], K[1][1], K[0][2], K[1][2], W, H].join();
    if (key === this.frustumKey) return;
    this.frustumKey = key;
    const d = FRUSTUM_DEPTH;
    const c = (u, v) => [(u - K[0][2]) / K[0][0] * d, (v - K[1][2]) / K[1][1] * d, d];
    const p = [c(0, 0), c(W, 0), c(W, H), c(0, H)];
    const o = [0, 0, 0];
    const segs = [];
    for (const q of p) segs.push(o, q);
    for (let i = 0; i < 4; i++) segs.push(p[i], p[(i + 1) % 4]);
    // "up" marker triangle above the top edge (image up = -y in the camera frame)
    const tm = [(p[0][0] + p[1][0]) / 2, p[0][1], d], w = (p[1][0] - p[0][0]) * 0.07, hh = (p[3][1] - p[0][1]) * 0.18;
    const a = [tm[0] - w, tm[1] - 0.01 * d, d], b = [tm[0] + w, tm[1] - 0.01 * d, d], t = [tm[0], tm[1] - hh, d];
    segs.push(a, b, b, t, t, a);
    const arr = this.frustumLines.geometry.attributes.position.array;
    segs.forEach((s, i) => arr.set(s, i * 3));
    this.frustumLines.geometry.setDrawRange(0, segs.length);
    this.frustumLines.geometry.attributes.position.needsUpdate = true;
    const pa = this.frustumPlane.geometry.attributes.position.array;
    p.forEach((s, i) => pa.set(s, i * 3));
    this.frustumPlane.geometry.attributes.position.needsUpdate = true;
  }

  buildLinks() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(3 * 2 * 512), 3));
    g.setDrawRange(0, 0);
    this.links = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 }));
    this.links.frustumCulled = false; this.links.visible = this.opts.lines;
    this.world.add(this.links);
  }

  // ---------------- per-frame ----------------
  setFrame(i) {
    if (!this.result) return;
    const frames = this.result.frames;
    i = Math.max(0, Math.min(frames.length - 1, i));
    this.cur = i;
    const f = frames[i];
    const vi = this.lastValid[i];
    const M = vi >= 0 ? this.poses[vi] : null;
    // frustum
    this.frustum.visible = !!M && this.mode !== 'camera';
    if (M) {
      // enlarge the frustum glyph for whole-scene views so it stays visible; real size (0.6 m) when close
      const R = this.bounds.getSize(new THREE.Vector3()).length() / 2;
      const k = this.mode === 'follow' ? 1 : Math.min(10, Math.max(1, R / 12));
      this.frustum.matrix.copy(M).scale(new THREE.Vector3(k, k, k)); this.frustum.matrixWorldNeedsUpdate = true;
      this.setFrustumGeometry(f.K, ...this.imgSize);
      const stale = f.status === 'lost' || !this.poses[i];
      const col = new THREE.Color(stale ? STATUS_COLORS.lost : STATUS_COLORS[f.status] || STATUS_COLORS.tracked);
      this.frustumLines.material.color.copy(col); this.frustumPlane.material.color.copy(col);
      this.frustumLines.material.opacity = stale ? 0.5 : 1; this.frustumLines.material.transparent = stale;
    }
    // trajectory
    this.trajBright.geometry.setDrawRange(0, this.segCount[i] * 2);
    // observed landmarks
    const seen = new Set();
    for (const d of f.detections || []) if (d.landmark_id != null) seen.add(d.landmark_id);
    for (const [id, o] of this.lmObjs) {
      const lit = seen.has(id);
      if (lit !== o.lit) { o.lit = lit; o.mat.emissive.copy(lit ? o.color : new THREE.Color(0)); o.mat.emissiveIntensity = lit ? 0.55 : 0; }
    }
    // camera -> landmark lines
    if (M) {
      const arr = this.links.geometry.attributes.position.array;
      const c = this.camPos[vi]; let n = 0;
      for (const id of seen) {
        const o = this.lmObjs.get(id); if (!o || n >= 512) continue;
        const p = o.group.position;
        arr.set([c.x, c.y, c.z, p.x, p.y, p.z + o.height * 0.5], n * 6); n++;
      }
      this.links.geometry.setDrawRange(0, n * 2);
      this.links.geometry.attributes.position.needsUpdate = true;
    }
    this.placeForMode(false);
    this.dirty = true;
  }

  setOption(k, v) {
    this.opts[k] = v;
    if (k === 'ids') this.updateIds();
    if (k === 'ellipses') for (const o of this.lmObjs.values()) o.ell.visible = v;
    if (k === 'lines' && this.links) this.links.visible = v;
    this.dirty = true;
  }

  // ---------------- camera modes ----------------
  setMode(mode) {
    if (mode === 'fit') { this.mode = 'orbit'; this.leaveSpecial(); this.fit(); return; }
    if (mode === this.mode) return;
    if (this.mode === 'camera') this.leaveSpecial();
    this.mode = mode;
    if (mode === 'top') { this.controls.enabled = true; this.topDown(); }
    else if (mode === 'follow') { this.controls.enabled = false; this.follow = null; this.placeForMode(true); }
    else if (mode === 'camera') { this.controls.enabled = false; this.camera.matrixAutoUpdate = false; }
    else this.controls.enabled = true;
    this.setFrame(this.cur);
    this.dirty = true;
    this.onMode?.(this.mode);
  }

  leaveSpecial() {
    this.camera.matrixAutoUpdate = true; this.controls.enabled = true;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.onMode?.(this.mode);
  }

  chaseTarget() {
    const vi = this.lastValid[this.cur]; if (vi < 0) return null;
    const M = this.poses[vi], p = this.camPos[vi];
    const e = M.elements;
    let fx = e[8], fy = e[9]; const n = Math.hypot(fx, fy) || 1; fx /= n; fy /= n;
    return {
      pos: new THREE.Vector3(p.x - fx * 4.5, p.y - fy * 4.5, p.z + 2.4),
      look: new THREE.Vector3(p.x + fx * 4, p.y + fy * 4, p.z * 0.4),
    };
  }

  placeForMode(snap) {
    if (this.mode === 'follow') {
      const t = this.chaseTarget(); if (!t) return;
      this.follow = t;
      if (snap) { this.camera.position.copy(t.pos); this.controls.target.copy(t.look); this.controls.update(); }
    } else if (this.mode === 'camera') {
      this.applyThroughCamera();
    }
  }

  applyThroughCamera() {
    const vi = this.lastValid[this.cur]; if (vi < 0) return;
    const f = this.result.frames[this.cur], K = f.K, [W, H] = this.imgSize;
    const flip = new THREE.Matrix4().makeScale(1, -1, -1);
    this.camera.matrix.copy(this.poses[vi]).multiply(flip);
    this.camera.matrixWorldNeedsUpdate = true;
    const n = 0.05, fr = 1500;
    this.camera.projectionMatrix.set(
      2 * K[0][0] / W, 0, 1 - 2 * K[0][2] / W, 0,
      0, 2 * K[1][1] / H, 2 * K[1][2] / H - 1, 0,
      0, 0, -(fr + n) / (fr - n), -2 * fr * n / (fr - n),
      0, 0, -1, 0);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }

  fit(initial = false) {
    const bb = this.bounds, c = bb.getCenter(new THREE.Vector3()), sz = bb.getSize(new THREE.Vector3());
    const R = Math.max(sz.length() / 2, 3);
    const dist = R / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 0.8;
    const dir = new THREE.Vector3(-0.75, -0.55, 0.5).normalize();
    this.camera.position.copy(c).addScaledVector(dir, dist);
    this.controls.target.copy(c);
    this.camera.near = Math.max(0.05, dist / 3000); this.camera.far = dist * 20 + 200;
    this.camera.updateProjectionMatrix();
    this.controls.update(); this.dirty = true;
    if (!initial) this.onMode?.(this.mode);
  }

  topDown() {
    const bb = this.bounds, c = bb.getCenter(new THREE.Vector3()), sz = bb.getSize(new THREE.Vector3());
    const aspect = this.container.clientWidth / Math.max(1, this.container.clientHeight);
    const t = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    // camera sits a hair behind (-x) so world +x is screen-up, avoiding the lookAt singularity with up = +z
    const Hh = Math.max((sz.x / 2 + 3) / t, (sz.y / 2 + 3) / (t * aspect), 8);
    this.camera.position.set(c.x - 0.03 * Hh, c.y, c.z + Hh);
    this.controls.target.set(c.x, c.y, 0);
    this.camera.far = Math.max(this.camera.far, Hh * 4);
    this.camera.updateProjectionMatrix();
    this.controls.update(); this.dirty = true;
  }

  // ---------------- render ----------------
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    if (this.mode !== 'camera') this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  tick(t) {
    const dt = Math.min(0.1, (t - this.last) / 1000); this.last = t;
    if (this.mode === 'follow' && this.follow) {
      const a = 1 - Math.exp(-dt * 7);
      const dp = this.follow.pos.distanceTo(this.camera.position), dl = this.follow.look.distanceTo(this.controls.target);
      if (dp > 1e-3 || dl > 1e-3) {
        this.camera.position.lerp(this.follow.pos, dp > 30 ? 1 : a);
        this.controls.target.lerp(this.follow.look, dl > 30 ? 1 : a);
        this.controls.update(); this.dirty = true;
      }
    }
    if (!this.dirty) return;
    this.dirty = false;
    this.render();
  }

  render() {
    const r = this.renderer, w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    r.setScissorTest(false); r.setViewport(0, 0, w, h); r.clear();
    if (this.mode === 'camera' && this.lastValid[this.cur] >= 0) {
      const [iw, ih] = this.imgSize, s = Math.min(w / iw, h / ih);
      const vw = Math.round(iw * s), vh = Math.round(ih * s), vx = Math.round((w - vw) / 2), vy = Math.round((h - vh) / 2);
      r.setScissorTest(true); r.setViewport(vx, vy, vw, vh); r.setScissor(vx, vy, vw, vh);
      this.camera.updateMatrixWorld(true);
      r.render(this.scene, this.camera);
      r.setScissorTest(false);
    } else {
      r.render(this.scene, this.camera);
    }
  }

  // the rectangle (CSS px inside the container) used by the through-camera view
  cameraViewRect() {
    const w = this.container.clientWidth, h = this.container.clientHeight, [iw, ih] = this.imgSize || [16, 9];
    const s = Math.min(w / iw, h / ih);
    return { x: (w - iw * s) / 2, y: (h - ih * s) / 2, w: iw * s, h: ih * s };
  }
}
