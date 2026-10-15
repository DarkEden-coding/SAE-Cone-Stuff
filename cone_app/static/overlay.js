// 2D overlay: boxes / keypoints / labels drawn on a canvas that covers the media box.
// The media is displayed with object-fit: contain, so the image occupies a letterboxed sub-rectangle.
import { classColor } from './util.js';

export class Overlay {
  constructor(stage, canvas, imageSize) {
    this.stage = stage; this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.iw = imageSize[0]; this.ih = imageSize[1];
    this.frame = null; this.opts = { boxes: true, keypoints: true, labels: true, ids: true };
    this.sel = null; this.dim = false; this.kpConf = 0.5;
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(stage);
  }
  destroy() { this.ro.disconnect(); }

  // Map from image pixels to canvas CSS pixels.
  rect() {
    const cw = this.stage.clientWidth, ch = this.stage.clientHeight;
    const s = Math.min(cw / this.iw, ch / this.ih);
    return { s, ox: (cw - this.iw * s) / 2, oy: (ch - this.ih * s) / 2, cw, ch };
  }

  set(frame, { dim = false, sel = null } = {}) { this.frame = frame; this.dim = dim; this.sel = sel; this.draw(); }

  draw() {
    const { s, ox, oy, cw, ch } = this.rect();
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) {
      c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr);
    }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cw, ch);
    if (!this.frame || this.dim) return;
    const o = this.opts;
    // line widths / fonts in CSS px, independent of image scale
    const lw = Math.max(1.5, Math.min(3, s * 6));
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.font = '600 12px system-ui, sans-serif'; g.textBaseline = 'top';
    const X = (x) => ox + x * s, Y = (y) => oy + y * s;
    for (const d of this.frame.detections || []) {
      const col = classColor(d.class_name);
      const selected = this.sel === d.det_id;
      const b = d.box_xyxy;
      if (o.boxes && b) {
        g.strokeStyle = col; g.lineWidth = selected ? lw + 1.5 : lw;
        g.strokeRect(X(b[0]), Y(b[1]), (b[2] - b[0]) * s, (b[3] - b[1]) * s);
      }
      if (o.keypoints && d.keypoints) {
        const kp = d.keypoints;
        const n = d.class_id === 1 ? 8 : 6;
        g.lineWidth = Math.max(1, lw - 0.5);
        // ring pairs (left-right) and side chains
        g.strokeStyle = '#fff'; g.globalAlpha = 0.9;
        for (let i = 0; i + 1 < n; i += 2) seg(g, kp[i], kp[i + 1], X, Y);
        g.strokeStyle = col; g.globalAlpha = 0.7;
        for (let i = 0; i + 2 < n; i++) seg(g, kp[i], kp[i + 2], X, Y);
        g.globalAlpha = 1;
        for (let i = 0; i < n; i++) {
          const k = kp[i];
          if (!k || k[2] <= 0) continue;
          const low = k[2] < this.kpConf;
          g.beginPath(); g.arc(X(k[0]), Y(k[1]), 3, 0, 6.2832);
          g.fillStyle = low ? 'rgba(0,0,0,0.5)' : (i % 2 ? '#ff5fa2' : '#38e8ff');
          g.fill(); g.lineWidth = 1; g.strokeStyle = low ? '#999' : '#000'; g.stroke();
        }
      }
      if (o.labels && b) {
        // shrink labels for small boxes so distant cones don't produce a wall of text
        const bh = (b[3] - b[1]) * s;
        const range = d.range_m != null ? d.range_m.toFixed(1) + ' m' : null;
        const lid = o.ids && d.landmark_id != null ? '#' + d.landmark_id : null;
        const parts = (bh < 22 && !selected ? [lid || range] : bh < 45 && !selected ? [range, lid]
          : [d.class_name, d.conf != null ? d.conf.toFixed(2) : null, range, lid]).filter(Boolean);
        if (!parts.length) continue;
        const text = parts.join(' · ');
        const w = g.measureText(text).width + 8;
        let tx = X(b[0]), ty = Y(b[1]) - 17;
        if (ty < 0) ty = Y(b[3]) + 2;
        tx = Math.min(Math.max(0, tx), cw - w);
        g.fillStyle = 'rgba(10,12,18,0.78)'; g.fillRect(tx, ty, w, 16);
        g.fillStyle = col; g.fillRect(tx, ty, 3, 16);
        g.fillStyle = '#fff'; g.fillText(text, tx + 6, ty + 2);
      }
    }
  }
}

function seg(g, a, b, X, Y) {
  if (!a || !b || a[2] <= 0 || b[2] <= 0) return;
  g.beginPath(); g.moveTo(X(a[0]), Y(a[1])); g.lineTo(X(b[0]), Y(b[1])); g.stroke();
}
