// Live camera preview: binary WebSocket messages are `<json header>\n<jpeg>`; the newest decoded image is drawn
// onto a background canvas (the overlay canvas above it stays in image_size coordinates).
const STATS_WINDOW_S = 2;
export const NO_CAMERA_AFTER_S = 2;

export class Preview {
  constructor(canvas) {
    this.canvas = canvas;
    this.g = canvas.getContext('2d');
    this.arrivals = []; // [performance.now() seconds, bytes]
    this.pending = null; this.busy = false; this.generation = 0;
    canvas.hidden = true;
  }

  // ArrayBuffer of one binary message
  push(buffer) {
    const bytes = new Uint8Array(buffer);
    const split = bytes.indexOf(0x0a);
    if (split < 0 || split === bytes.length - 1) return;
    this.arrivals.push([performance.now() / 1000, bytes.length]);
    this.pending = new Blob([bytes.subarray(split + 1)], { type: 'image/jpeg' });
    if (!this.busy) this.pump();
  }

  async pump() {
    this.busy = true;
    const generation = this.generation;
    while (this.pending) {
      const blob = this.pending; this.pending = null;
      try {
        const bitmap = await createImageBitmap(blob);
        if (generation === this.generation) {
          if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
            this.canvas.width = bitmap.width; this.canvas.height = bitmap.height;
          }
          this.g.drawImage(bitmap, 0, 0);
          this.canvas.hidden = false;
        }
        bitmap.close();
      } catch (e) { /* undecodable image: keep the previous one */ }
    }
    this.busy = false;
  }

  // text for the status strip; clears the picture once images stop arriving
  status() {
    const now = performance.now() / 1000;
    while (this.arrivals.length && now - this.arrivals[0][0] > STATS_WINDOW_S) this.arrivals.shift();
    const last = this.arrivals.length ? this.arrivals[this.arrivals.length - 1][0] : -Infinity;
    if (now - last > NO_CAMERA_AFTER_S) { this.canvas.hidden = true; return 'no camera'; }
    const span = Math.max(now - this.arrivals[0][0], 0.5);
    const kb = this.arrivals.reduce((sum, a) => sum + a[1], 0) / 1024;
    const intervals = this.arrivals.length - 1;
    return `cam ${(intervals > 0 ? intervals / Math.max(last - this.arrivals[0][0], 0.1) : 0).toFixed(1)} fps · ${Math.round(kb / span)} KB/s`;
  }

  reset() {
    this.generation++; this.pending = null; this.arrivals = [];
    this.canvas.hidden = true;
  }
}
