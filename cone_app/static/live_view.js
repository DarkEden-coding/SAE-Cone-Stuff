// Live session view: watch an active session over WebSocket, or replay its stored log (optionally over the recording's video).
import { getLive, deleteLive, getLiveLog, liveWatchSocket, jobMediaUrl } from './api.js';
import { el, toast, fmtNum, fmtTime, fmtAgo, classColor, STATUS_COLORS } from './util.js';
import { Overlay } from './overlay.js';
import { Viewer } from './viewer3d.js';
import { LiveScene, adaptFrame } from './live_scene.js';
import { Preview } from './live_preview.js';

const TAIL = 400;
const SPEEDS = [0.5, 1, 2];

// first index in sorted array a (by key fn) with key(a[i]) > x
function upperBound(a, x, key = (v) => v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (key(a[m]) <= x) lo = m + 1; else hi = m; }
  return lo;
}

export function mountLiveView(root, id) {
  let dead = false, session = null, mode = null;
  const box = el('div', { class: 'page wide' });
  root.append(box);
  const cleanups = [];

  // ---------------------------------------------------------------- UI skeleton
  const nameH = el('h1', {}, id);
  const modeBadge = el('span', { class: 'status' });
  const liveBtn = el('button', { class: 'btn small', type: 'button', onclick: () => startLive() }, 'Live');
  const replayBtn = el('button', { class: 'btn small', type: 'button', onclick: () => startReplay(recSel.value || null) }, 'Replay');
  const delBtn = el('button', { class: 'btn small danger', type: 'button', onclick: async () => {
    if (!confirm('Delete this live session?')) return;
    try { await deleteLive(id); location.hash = '#/live'; } catch (e) { toast('Delete failed: ' + e.message); }
  } }, 'Delete');
  const head = el('div', { class: 'jv-head' }, el('a', { class: 'btn small', href: '#/live' }, '← Live'), nameH, modeBadge,
    el('span', { class: 'spacer' }), liveBtn, replayBtn, delBtn);

  // status strip
  const chips = {};
  const strip = el('div', { class: 'lv-status card' });
  for (const [k, label] of [['conn', 'Connection'], ['status', 'Frame'], ['fps', 'fps'], ['det', 'det ms'], ['slam', 'slam ms'], ['inl', 'inliers'], ['cones', 'Cones'], ['rec', 'Recording'], ['cam', 'Camera']]) {
    const v = el('b', { class: 'mono' }, '-');
    chips[k] = v;
    strip.append(el('div', { class: 'lv-chip' }, el('span', { class: 'muted small' }, label), v));
  }

  // camera view (live: low-rate preview images under the overlays; replay: the recording's video, else black)
  const canvas = el('canvas', { class: 'overlay' });
  const previewCanvas = el('canvas', { class: 'preview' });
  const preview = new Preview(previewCanvas);
  const stage = el('div', { class: 'stage', style: { '--ar': '1280 / 720' } });
  const noMedia = el('div', { class: 'nomedia' }, 'no video is streamed: overlay on black');
  stage.append(previewCanvas, noMedia, canvas);
  const overlay = new Overlay(stage, canvas, [1280, 720]);
  cleanups.push(() => overlay.destroy());
  const toggles = el('div', { class: 'toggles' }, ...[['boxes', 'Boxes'], ['keypoints', 'Keypoints'], ['labels', 'Labels'], ['ids', 'Landmark ids']].map(([k, label]) => {
    const c = el('input', { type: 'checkbox', checked: true, onchange: () => { overlay.opts[k] = c.checked; overlay.draw(); } });
    return el('label', { class: 'chk' }, c, label);
  }));
  const mediaPane = el('section', { class: 'pane media-pane' }, el('div', { class: 'pane-title' }, 'Camera view'), stage, toggles);

  // 3D
  const vp = el('div', { class: 'viewport' });
  const viewer = new Viewer(vp);
  cleanups.push(() => viewer.destroy());
  const scene = new LiveScene(viewer);
  const modeBtns = {};
  const mkBtn = (m, label, title) => { const b = el('button', { class: 'btn small', type: 'button', title, onclick: () => viewer.setMode(m) }, label); modeBtns[m] = b; return b; };
  const vtoggles = ['ids:Ids', 'ellipses:Uncertainty', 'lines:Cam rays'].map((s) => {
    const [k, label] = s.split(':');
    const c = el('input', { type: 'checkbox', onchange: () => viewer.setOption(k, c.checked) });
    return el('label', { class: 'chk' }, c, label);
  });
  viewer.onMode = (m) => { for (const [k, b] of Object.entries(modeBtns)) b.classList.toggle('active', k === m); };
  const vbar = el('div', { class: 'vbar' }, mkBtn('fit', 'Fit', 'Fit view to the scene'), mkBtn('orbit', 'Orbit', 'Free orbit'),
    mkBtn('follow', 'Follow', 'Chase camera'), mkBtn('camera', 'Through camera', 'View through the phone camera'),
    mkBtn('top', 'Top', 'Top-down'), el('span', { class: 'spacer' }), ...vtoggles);
  const legend = el('div', { class: 'legend' }, ...['BLUE', 'YELLOW', 'ORANGE', 'ORANGE_BIG'].map((n) => el('span', {}, el('i', { style: { background: classColor(n) } }), n)));
  const viewPane = el('section', { class: 'pane view3d-pane' }, el('div', { class: 'pane-title' }, '3D map (z up, metres)'), vp, vbar, legend);
  viewer.onMode('orbit');

  // replay timeline
  const recSel = el('select', { 'aria-label': 'log', onchange: () => startReplay(recSel.value || null) });
  const playBtn = el('button', { class: 'btn small', type: 'button', onclick: () => togglePlay() }, 'Play');
  const speedSel = el('select', { 'aria-label': 'speed', onchange: () => { speed = Number(speedSel.value); if (video) video.playbackRate = speed; } },
    SPEEDS.map((s) => el('option', { value: s, selected: s === 1 }, s + '×')));
  const slider = el('input', { type: 'range', min: 0, max: 0, step: 1, value: 0, 'aria-label': 'time' });
  const tlStrip = el('canvas', { class: 'strip', width: 1, height: 1 });
  const counter = el('span', { class: 'counter mono' });
  const timeline = el('section', { class: 'timeline', hidden: true },
    el('div', { class: 'tl-row' }, playBtn, speedSel, el('div', { class: 'slider-wrap' }, slider, tlStrip), counter, recSel));

  const side = el('aside', { class: 'side' });
  const sideCard = el('div', { class: 'card' });
  side.append(sideCard);

  box.append(head, strip, el('div', { class: 'jv-grid' }, mediaPane, viewPane, timeline, side));

  // ---------------------------------------------------------------- shared rendering
  function setImageSize(sz) {
    if (!sz || (sz[0] === overlay.iw && sz[1] === overlay.ih)) return;
    overlay.iw = sz[0]; overlay.ih = sz[1];
    stage.style.setProperty('--ar', `${sz[0]} / ${sz[1]}`);
    overlay.draw();
  }
  function onHello(h) {
    scene.setHello(h);
    setImageSize(h.image_size);
    overlay.kpConf = (h.options && h.options.kp_conf) ?? 0.5;
  }
  function setStatusChips(f, nMapped, extra = {}) {
    const st = (f && f.stats) || {};
    const status = f ? f.status : null;
    chips.status.textContent = status || '-';
    chips.status.className = 'mono' + (status ? ' st-' + status : '');
    chips.fps.textContent = fmtNum(st.fps, 1);
    chips.det.textContent = fmtNum(st.det_ms, 1);
    chips.slam.textContent = fmtNum(st.slam_ms, 1);
    chips.inl.textContent = f && f.n_inliers != null ? f.n_inliers : '-';
    chips.cones.textContent = f ? `${nMapped} mapped · ${(f.detections || []).length} in view` : String(nMapped);
    const rec = f && f.rec;
    chips.rec.textContent = rec ? '● REC ' + fmtTime(rec.t) : 'off';
    chips.rec.className = 'mono' + (rec ? ' rec-on' : '');
    if (extra.conn != null) setConnText(extra.conn);
  }
  function setConnText(text, cls = '') { chips.conn.textContent = text; chips.conn.className = 'mono ' + cls; }

  function renderSide() {
    const s = session;
    const rows = [['Device', s.device && (s.device.model || '') + (s.device.system ? ' · ' + s.device.system : '')], ['Started', s.started_at && fmtAgo(s.started_at)],
      ['Duration', fmtNum(s.duration_s, 1) + ' s'], ['Frames', s.n_frames], ['Cones', s.n_landmarks], ['Id', s.id]].filter((r) => r[1]);
    const recs = s.recordings || [];
    sideCard.replaceChildren(el('h3', {}, 'Session'), el('dl', { class: 'stats' }, rows.map(([k, v]) => [el('dt', {}, k), el('dd', {}, String(v))])),
      el('h3', {}, 'Recordings'),
      recs.length ? recs.map((r) => el('div', { class: 'rec-item' },
        el('div', {}, el('i', { class: 'rec-dot' }), r.name || r.recording_id,
          el('div', { class: 'muted small' }, `${fmtTime(r.t_start)} - ${r.t_end != null ? fmtTime(r.t_end) : 'ongoing'}`)),
        el('div', { class: 'row wrap' },
          r.has_log ? el('button', { class: 'btn small', type: 'button', onclick: () => { recSel.value = r.recording_id; startReplay(r.recording_id); } }, 'Replay') : null,
          r.job_id ? el('a', { class: 'btn small', href: '#/job/' + encodeURIComponent(r.job_id) }, 'Job') : null)))
        : el('div', { class: 'muted small' }, 'None yet.'));
    nameH.textContent = s.name || s.id;
    recSel.replaceChildren(el('option', { value: '' }, 'Whole session'),
      ...recs.filter((r) => r.has_log).map((r) => el('option', { value: r.recording_id }, 'Recording: ' + (r.name || r.recording_id))));
  }
  async function refreshSession() {
    try { const keep = recSel.value; session = await getLive(id); if (!dead) { renderSide(); recSel.value = keep; setModeUI(); } } catch (e) { /* keep the old info */ }
  }

  function setModeUI() {
    const isEnded = mode === 'live' && ended;
    modeBadge.textContent = isEnded ? 'ended' : mode === 'live' ? 'live' : mode === 'replay' ? 'replay' : '';
    modeBadge.className = 'status ' + (isEnded ? 'gap' : mode === 'live' ? 'tracked' : 'init');
    liveBtn.hidden = !(session && session.active) || mode === 'live';
    replayBtn.classList.toggle('active', mode === 'replay');
    timeline.hidden = mode !== 'replay';
  }

  // ---------------------------------------------------------------- live
  let ws = null, imgWs = null, imgRetry = null, retry = null, backoff = 1000, ended = false, phone = 'waiting';
  let liveMap = null, liveTail = [], latest = null, flushRaf = 0, sessionTimer = null, camTimer = 0;

  function connLabel() {
    if (ended) return setConnText('ended', 'warn');
    if (!ws || ws.readyState !== 1) return setConnText(retry ? 'reconnecting…' : 'connecting…', 'warn');
    if (phone === 'disconnected') return setConnText('phone disconnected', 'bad');
    setConnText(phone === 'streaming' ? 'live' : 'waiting for phone', phone === 'streaming' ? 'ok' : 'warn');
  }
  async function connect() {
    clearTimeout(retry); retry = null;
    connLabel();
    let sock;
    try { sock = await liveWatchSocket(id); } catch (e) { return scheduleRetry(); }
    if (dead || mode !== 'live') { sock.close(); return; }
    ws = sock;
    sock.onopen = () => { backoff = 1000; connLabel(); };
    sock.binaryType = 'arraybuffer';
    sock.onmessage = (ev) => {
      if (typeof ev.data !== 'string') { preview.push(ev.data); return; }
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      onLiveMsg(m);
    };
    sock.onclose = () => { if (ws !== sock) return; ws = null; if (dead || mode !== 'live' || ended) return; scheduleRetry(); };
    sock.onerror = () => { /* onclose follows */ };
    openImages();
  }
  // camera preview on its own socket, so images never delay metadata; follows the metadata socket's lifetime
  function openImages() {
    closeImages();
    const sock = liveWatchSocket(id, 'images');
    if (!sock) return;
    imgWs = sock;
    sock.binaryType = 'arraybuffer';
    sock.onmessage = (ev) => { if (typeof ev.data !== 'string') preview.push(ev.data); };
    sock.onclose = () => {
      if (imgWs !== sock) return;
      imgWs = null;
      if (!dead && mode === 'live' && !ended && ws) imgRetry = setTimeout(() => { if (ws && !imgWs) openImages(); }, 2000);
    };
  }
  function closeImages() {
    clearTimeout(imgRetry);
    if (imgWs) { const s = imgWs; imgWs = null; s.close(); }
  }
  function scheduleRetry() {
    clearTimeout(retry);
    retry = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 1.7, 10000);
    connLabel();
  }
  function onLiveMsg(m) {
    switch (m.type) {
      case 'hello': onHello(m); phone = phone === 'disconnected' ? 'waiting' : phone; break;
      case 'map': liveMap = m; liveTail = liveTail.filter((f) => f.t > m.t); break;
      case 'frame': {
        const f = adaptFrame(m);
        setImageSize(f.image_size);
        liveTail.push(f); if (liveTail.length > TAIL) liveTail.shift();
        latest = f; phone = 'streaming';
        break;
      }
      case 'recording': refreshSession(); break;
      case 'disconnected': phone = 'disconnected'; break;
      case 'bye': ended = true; phone = 'ended'; if (ws) { const s = ws; ws = null; s.close(); } closeImages(); refreshSession(); break;
      default: return; // ack and unknown types
    }
    connLabel();
    if (m.type === 'bye') setModeUI();
    if (!flushRaf) flushRaf = requestAnimationFrame(flushLive);
  }
  function flushLive() {
    flushRaf = 0;
    if (mode !== 'live') return;
    overlay.set(latest, { dim: phone === 'disconnected' });
    scene.update(liveMap, liveTail, latest);
    setStatusChips(latest, scene.landmarks.length);
  }
  function stopLive() {
    clearTimeout(retry); retry = null; clearTimeout(sessionTimer); cancelAnimationFrame(flushRaf); flushRaf = 0;
    clearInterval(camTimer); camTimer = 0; preview.reset(); chips.cam.textContent = '-';
    if (ws) { const s = ws; ws = null; s.close(); }
    closeImages();
  }
  function startLive() {
    stopReplay();
    mode = 'live'; ended = false; phone = 'waiting'; backoff = 1000;
    liveMap = null; liveTail = []; latest = null; scene.reset();
    overlay.set(null); setStatusChips(null, 0);
    setNoMedia(null); noMedia.hidden = true; // live shows the preview images instead
    setModeUI();
    preview.reset();
    clearInterval(camTimer);
    camTimer = setInterval(() => { chips.cam.textContent = preview.status(); }, 500);
    connect();
    clearTimeout(sessionTimer);
    const poll = async () => { await refreshSession(); if (!dead && mode === 'live') sessionTimer = setTimeout(poll, 4000); };
    sessionTimer = setTimeout(poll, 4000);
  }

  // ---------------------------------------------------------------- replay
  // rp: all = adapted frames by t, maps by t, tl = indices into all shown on the timeline, keys = timeline time per entry
  let rp = { all: [], allT: [], maps: [], tl: [], keys: [], tol: 0.1 };
  let cur = 0, curGap = false, speed = 1, video = null, mediaOk = false, vplay = false, vt = 0, vlast = 0, vraf = 0, vfc = 0, loadSeq = 0;
  const useVideo = () => video && mediaOk;

  function setNoMedia(v) {
    if (video && video !== v) { video.pause(); if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(vfc); video.remove(); }
    video = v; mediaOk = false;
    if (v) stage.prepend(v);
    noMedia.textContent = v ? 'loading video…' : 'no video is streamed: overlay on black';
    noMedia.hidden = false;
  }

  async function startReplay(recId) {
    stopLive(); pausePlayback();
    const seq = ++loadSeq;
    mode = 'replay'; recSel.value = recId || '';
    scene.reset(); overlay.set(null);
    rp = { all: [], allT: [], maps: [], tl: [], keys: [], tol: 0.1 };
    setNoMedia(null); setModeUI();
    setConnText('loading log…', 'warn'); setStatusChips(null, 0);
    let text;
    try { text = await getLiveLog(id, recId); } catch (e) {
      if (dead || seq !== loadSeq) return;
      setConnText('log unavailable', 'bad'); noMedia.textContent = 'Could not load log: ' + e.message; toast('Could not load log: ' + e.message);
      return;
    }
    if (dead || seq !== loadSeq) return;
    const maps = [], all = [];
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      let m; try { m = JSON.parse(line); } catch (e) { continue; }
      if (m.type === 'hello') onHello(m);
      else if (m.type === 'map') maps.push(m);
      else if (m.type === 'frame') all.push(adaptFrame(m));
    }
    all.sort((a, b) => a.t - b.t); maps.sort((a, b) => a.t - b.t);
    const rec = recId && session && (session.recordings || []).find((r) => r.recording_id === recId);
    let tl = all.map((_, i) => i), keys = all.map((f) => f.t), vurl = null;
    if (rec && rec.job_id) {
      const idx = all.map((f, i) => (f.rec && f.rec.recording_id === recId ? i : -1)).filter((i) => i >= 0)
        .sort((a, b) => all[a].rec.t - all[b].rec.t);
      if (idx.length) { tl = idx; keys = idx.map((i) => all[i].rec.t); vurl = jobMediaUrl(rec.job_id); }
    }
    const dts = [];
    for (let i = 1; i < keys.length; i++) dts.push(keys[i] - keys[i - 1]);
    dts.sort((a, b) => a - b);
    rp = { all, allT: all.map((f) => f.t), maps, tl, keys, tol: Math.max((dts.length ? dts[dts.length >> 1] : 0.1) * 1.5, 0.05) };
    cur = 0; curGap = false; vt = keys[0] || 0;
    slider.max = Math.max(0, tl.length - 1); slider.value = 0;
    drawStrip();
    setModeUI();
    if (!all.length) { setConnText('empty log', 'warn'); noMedia.textContent = 'This log has no frames.'; setStatusChips(null, 0); return; }
    setConnText(recId ? 'replay · recording' : 'replay', 'ok');
    if (vurl) attachVideo(vurl); else noMedia.textContent = 'no video is streamed: overlay on black';
    show(0, false);
    viewer.fit();
  }

  function attachVideo(url) {
    const v = el('video', { src: url, playsinline: true, muted: true, preload: 'auto', crossorigin: 'anonymous' });
    setNoMedia(v);
    v.playbackRate = speed;
    v.addEventListener('loadeddata', () => { mediaOk = true; noMedia.hidden = true; onVideoTime(v.currentTime); });
    v.addEventListener('error', () => { mediaOk = false; v.style.display = 'none'; noMedia.hidden = false; noMedia.textContent = 'video unavailable: overlay on black'; });
    v.addEventListener('play', () => { playBtn.textContent = 'Pause'; });
    v.addEventListener('pause', () => { playBtn.textContent = 'Play'; });
    v.addEventListener('ended', () => { playBtn.textContent = 'Play'; });
    for (const ev of ['seeked', 'timeupdate']) v.addEventListener(ev, () => { if (mediaOk) onVideoTime(v.currentTime); });
    if ('requestVideoFrameCallback' in v) {
      const cb = (now, meta) => { if (mediaOk) onVideoTime(meta.mediaTime); vfc = v.requestVideoFrameCallback(cb); };
      vfc = v.requestVideoFrameCallback(cb);
    }
  }

  function drawStrip() {
    tlStrip.width = Math.max(1, rp.tl.length);
    const g = tlStrip.getContext('2d');
    rp.tl.forEach((gi, i) => { g.fillStyle = STATUS_COLORS[rp.all[gi].status] || '#888'; g.fillRect(i, 0, 1, 1); });
  }

  function show(i, gap = false) {
    cur = i; curGap = gap;
    const gi = rp.tl[i], f = rp.all[gi];
    const mi = upperBound(rp.maps, f.t, (m) => m.t) - 1;
    const map = mi >= 0 ? rp.maps[mi] : null;
    const start = map ? upperBound(rp.allT, map.t) : Math.max(0, gi - TAIL);
    const tail = rp.all.slice(Math.max(start, gi - TAIL + 1), gi + 1);
    overlay.set(f, { dim: gap });
    scene.update(map, tail, f);
    setStatusChips(f, scene.landmarks.length);
    slider.value = i;
    counter.textContent = `${i + 1}/${rp.tl.length} · ${fmtTime(rp.keys[i])}`;
  }
  function nearest(t) {
    const k = rp.keys;
    let lo = 0, hi = k.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (k[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(k[lo - 1] - t) <= Math.abs(k[lo] - t)) lo--;
    return lo;
  }
  function onTime(t) {
    if (!rp.tl.length) return;
    const i = nearest(t), gap = Math.abs(rp.keys[i] - t) > rp.tol;
    if (i !== cur || gap !== curGap) show(i, gap);
  }
  function onVideoTime(t) { vt = t; onTime(t); }

  function setPlayLabel(p) { playBtn.textContent = p ? 'Pause' : 'Play'; }
  function pausePlayback() { vplay = false; cancelAnimationFrame(vraf); if (video) video.pause(); setPlayLabel(false); }
  function vloop(ts) {
    if (!vplay) return;
    vt += (ts - vlast) / 1000 * speed; vlast = ts;
    const end = rp.keys[rp.keys.length - 1];
    if (vt >= end) { vt = end; vplay = false; setPlayLabel(false); }
    onTime(vt);
    if (vplay) vraf = requestAnimationFrame(vloop);
  }
  function togglePlay() {
    if (!rp.tl.length) return;
    if (useVideo()) { if (video.paused) video.play().catch((e) => toast('Playback failed: ' + e.message)); else video.pause(); return; }
    vplay = !vplay; setPlayLabel(vplay);
    if (vplay) { if (vt >= rp.keys[rp.keys.length - 1] - 1e-6) vt = rp.keys[0]; vlast = performance.now(); vraf = requestAnimationFrame(vloop); }
  }
  slider.addEventListener('input', () => {
    pausePlayback();
    const i = Number(slider.value);
    vt = rp.keys[i];
    if (useVideo()) video.currentTime = vt + 0.001;
    show(i, false);
  });
  function stopReplay() { pausePlayback(); loadSeq++; setNoMedia(null); }

  // ---------------------------------------------------------------- start
  cleanups.push(() => { stopLive(); stopReplay(); });
  (async () => {
    try { session = await getLive(id); } catch (e) {
      if (dead) return;
      box.replaceChildren(el('div', { class: 'card err' }, 'Could not load live session: ' + e.message, el('div', {}, el('a', { href: '#/live' }, 'Back to live sessions'))));
      cleanups.forEach((c) => { try { c(); } catch (err) { console.error(err); } }); cleanups.length = 0;
      return;
    }
    if (dead) return;
    renderSide();
    if (session.active) startLive(); else startReplay(null);
  })();

  return { destroy() { dead = true; cleanups.forEach((c) => { try { c(); } catch (e) { console.error(e); } }); cleanups.length = 0; } };
}
