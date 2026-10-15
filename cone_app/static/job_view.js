import { getJob, getResult, mediaUrl, rerunJob, deleteJob, MOCK } from './api.js';
import { el, toast, fmtNum, fmtTime, fmtAgo, classColor, STATUS_COLORS, download, landmarksCsv, optionsFields, readOptions } from './util.js';
import { Overlay } from './overlay.js';
import { Viewer } from './viewer3d.js';

export function mountJobView(root, id) {
  let dead = false, timer = null, ui = null;
  const box = el('div', { class: 'page wide' });
  root.append(box);

  async function load() {
    clearTimeout(timer);
    let job;
    try { job = await getJob(id); } catch (e) {
      if (dead) return;
      box.replaceChildren(el('div', { class: 'card err' }, 'Could not load job: ' + e.message, el('div', {}, el('a', { href: '#/jobs' }, 'Back to jobs'))));
      return;
    }
    if (dead) return;
    if (job.status === 'done') {
      let result;
      try { result = await getResult(id); } catch (e) {
        box.replaceChildren(el('div', { class: 'card err' }, 'Could not load result: ' + e.message));
        return;
      }
      if (dead) return;
      teardown();
      ui = buildResultUI(box, job, result, { reload: load });
    } else {
      if (!ui || ui.kind !== 'pending') { teardown(); ui = { kind: 'pending', destroy() {} }; }
      renderPending(box, job, load);
      if (job.status === 'queued' || job.status === 'running') timer = setTimeout(load, 1500);
    }
  }
  function teardown() { if (ui) { ui.destroy(); ui = null; } box.replaceChildren(); }
  load();
  return { destroy() { dead = true; clearTimeout(timer); teardown(); } };
}

function backHeader(job, extra = []) {
  return el('div', { class: 'jv-head' },
    el('a', { class: 'btn small', href: '#/jobs' }, '← Jobs'),
    el('h1', {}, job.name || job.id),
    el('span', { class: 'tag' }, job.kind), el('span', { class: 'status ' + job.status }, job.status),
    el('span', { class: 'muted small hide-sm' }, job.id + ' · ' + fmtAgo(job.created_at)),
    el('span', { class: 'spacer' }), ...extra);
}

function rerunCard(job, options, onDone) {
  const fields = optionsFields(options || job.options || {}, 'rr');
  const btn = el('button', { class: 'btn primary', type: 'button', onclick: async () => {
    btn.disabled = true;
    try { await rerunJob(job.id, readOptions(fields)); toast('Job re-queued'); onDone(true); }
    catch (e) { toast('Rerun failed: ' + e.message); btn.disabled = false; }
  } }, 'Rerun with these options');
  return el('details', { class: 'card rerun' }, el('summary', {}, 'Rerun with modified options'), fields, btn);
}

function renderPending(box, job, reload) {
  const pct = Math.round((job.progress || 0) * 100);
  const failed = job.status === 'failed';
  box.replaceChildren(
    backHeader(job, [el('button', { class: 'btn small danger', onclick: async () => {
      if (!confirm('Delete this job?')) return;
      try { await deleteJob(job.id); location.hash = '#/jobs'; } catch (e) { toast(e.message); }
    } }, 'Delete')]),
    el('div', { class: 'card pending' },
      failed ? el('div', { class: 'err-text' }, 'Processing failed: ' + (job.error || 'unknown error'))
        : [el('div', {}, job.status === 'queued' ? 'Queued - waiting for the GPU worker…' : `Processing: ${job.stage || ''} (${pct}%)`),
          el('div', { class: 'bar big' }, el('div', { class: 'fill', style: { width: pct + '%' } })),
          el('div', { class: 'muted small' }, 'This page updates automatically and loads the result when it is ready.')]),
    ...(failed ? [rerunCard(job, job.options, reload)] : []));
}

// ===================================================================================================
function buildResultUI(box, job, result, { reload }) {
  const frames = result.frames || [];
  const [IW, IH] = result.image_size;
  const isVideo = result.kind === 'video' && frames.length > 1;
  const times = frames.map((f) => f.t);
  const dts = [];
  for (let i = 1; i < times.length; i++) dts.push(times[i] - times[i - 1]);
  dts.sort((a, b) => a - b);
  const dt = dts.length ? dts[Math.floor(dts.length / 2)] : 0.1;
  const tol = Math.max(dt * 0.75, 0.04);
  const cleanups = [];
  let cur = 0, curGap = false, sel = null, mediaOk = false;

  // ---------- media + overlay ----------
  const url = mediaUrl(job);
  const canvas = el('canvas', { class: 'overlay' });
  const stage = el('div', { class: 'stage', style: { '--ar': `${IW} / ${IH}` } });
  let media = null;
  if (url) {
    media = isVideo ? el('video', { src: url, controls: true, playsinline: true, muted: true, preload: 'auto', crossorigin: 'anonymous' })
      : el('img', { src: url, alt: job.name || 'photo' });
    media.addEventListener(isVideo ? 'loadeddata' : 'load', () => { mediaOk = true; });
    media.addEventListener('error', () => { mediaOk = false; media.style.display = 'none'; noMedia.hidden = false; });
    stage.append(media);
  }
  const noMedia = el('div', { class: 'nomedia', hidden: !!url }, MOCK ? 'mock mode: no media, overlay on black' : 'media unavailable');
  stage.append(noMedia, canvas);
  const overlay = new Overlay(stage, canvas, [IW, IH]);
  overlay.kpConf = (result.options && result.options.kp_conf) ?? 0.5;
  cleanups.push(() => overlay.destroy());

  const toggles = el('div', { class: 'toggles' },
    ...[['boxes', 'Boxes'], ['keypoints', 'Keypoints'], ['labels', 'Labels'], ['ids', 'Landmark ids']].map(([k, label]) => {
      const c = el('input', { type: 'checkbox', checked: true, onchange: () => { overlay.opts[k] = c.checked; overlay.draw(); } });
      return el('label', { class: 'chk' }, c, label);
    }));
  const mediaPane = el('section', { class: 'pane media-pane' }, el('div', { class: 'pane-title' }, 'Detections'), stage, toggles);

  // ---------- 3D ----------
  const vp = el('div', { class: 'viewport' });
  const viewer = new Viewer(vp);
  cleanups.push(() => viewer.destroy());
  const modeBtns = {};
  const mkBtn = (mode, label, title) => {
    const b = el('button', { class: 'btn small', type: 'button', title, onclick: () => viewer.setMode(mode) }, label);
    modeBtns[mode] = b; return b;
  };
  const vtoggles = ['ids:Ids', 'ellipses:Uncertainty', 'lines:Cam rays'].map((s) => {
    const [k, label] = s.split(':');
    const c = el('input', { type: 'checkbox', onchange: () => viewer.setOption(k, c.checked) });
    return el('label', { class: 'chk' }, c, label);
  });
  const vbar = el('div', { class: 'vbar' }, mkBtn('fit', 'Fit', 'Fit view to the scene'), mkBtn('orbit', 'Orbit', 'Free orbit'),
    mkBtn('follow', 'Follow', 'Chase camera'), mkBtn('camera', 'Through camera', 'View through the capture camera'),
    mkBtn('top', 'Top', 'Top-down'), el('span', { class: 'spacer' }), ...vtoggles);
  viewer.onMode = (m) => { for (const [k, b] of Object.entries(modeBtns)) b.classList.toggle('active', k === m); };
  const legend = el('div', { class: 'legend' }, ...['BLUE', 'YELLOW', 'ORANGE', 'ORANGE_BIG'].map((n) => el('span', {}, el('i', { style: { background: classColor(n) } }), n)));
  const viewPane = el('section', { class: 'pane view3d-pane' }, el('div', { class: 'pane-title' }, '3D scene (z up, metres)'), vp, vbar, legend);
  viewer.setResult(result);
  viewer.onMode('orbit');

  // ---------- timeline ----------
  const badge = el('span', { class: 'status' });
  const counter = el('span', { class: 'counter mono' });
  const slider = el('input', { type: 'range', min: 0, max: Math.max(0, frames.length - 1), step: 1, value: 0, 'aria-label': 'frame' });
  const playBtn = el('button', { class: 'btn small', type: 'button', onclick: () => togglePlay() }, 'Play');
  const strip = el('canvas', { class: 'strip', width: Math.max(1, frames.length), height: 1 });
  {
    const g = strip.getContext('2d');
    frames.forEach((f, i) => { g.fillStyle = STATUS_COLORS[f.status] || '#888'; g.fillRect(i, 0, 1, 1); });
  }
  const timeline = el('section', { class: 'timeline' },
    el('div', { class: 'tl-row' }, playBtn,
      el('button', { class: 'btn small', type: 'button', onclick: () => step(-1), title: 'previous frame (←)' }, '◀'),
      el('button', { class: 'btn small', type: 'button', onclick: () => step(1), title: 'next frame (→)' }, '▶'),
      el('div', { class: 'slider-wrap' }, slider, strip), counter, badge));
  if (!isVideo) timeline.hidden = true;

  // ---------- side panel ----------
  const detBox = el('div', { class: 'det-table-wrap' });
  const side = el('aside', { class: 'side' });
  const st = result.stats || {};
  const statRows = [
    ['Frames', st.n_frames_processed != null ? `${st.n_frames_processed} / ${st.n_frames_total}` : null],
    ['Tracked / lost', st.n_tracked != null ? `${st.n_tracked} / ${st.n_lost}` : null],
    ['Landmarks', st.n_landmarks], ['Runtime', st.runtime_s != null ? fmtNum(st.runtime_s, 1) + ' s' : null],
    ['Detector', st.detector_ms_per_frame != null ? fmtNum(st.detector_ms_per_frame, 1) + ' ms/frame' : null],
    ['Camera height', result.ground_plane ? fmtNum(result.ground_plane.camera_height_m, 2) + ' m' : 'no ground plane'],
    ['Image', `${IW}×${IH}`], ['Model imgsz', result.model && result.model.imgsz],
    ['KP correction', result.model && (result.model.kp_correction ? 'on' : 'off')],
  ].filter((r) => r[1] != null);
  const exportBtns = el('div', { class: 'row wrap' },
    el('button', { class: 'btn small', onclick: () => download(`${job.id}-result.json`, JSON.stringify(result)) }, 'Export JSON'),
    el('button', { class: 'btn small', onclick: () => download(`${job.id}-landmarks.csv`, landmarksCsv(result), 'text/csv') }, 'Export landmarks CSV'));
  side.append(
    el('div', { class: 'card' }, el('h3', {}, 'Stats'), el('dl', { class: 'stats' }, statRows.map(([k, v]) => [el('dt', {}, k), el('dd', {}, String(v))])), exportBtns),
    el('div', { class: 'card' }, el('h3', {}, 'Detections in frame'), detBox),
    rerunCard(job, result.options, (queued) => { if (queued) reloadSoon(); }));
  function reloadSoon() { reload(); }

  box.replaceChildren(
    backHeader(job, [el('button', { class: 'btn small danger', onclick: async () => {
      if (!confirm('Delete this job?')) return;
      try { await deleteJob(job.id); location.hash = '#/jobs'; } catch (e) { toast(e.message); }
    } }, 'Delete')]),
    el('div', { class: 'jv-grid' }, mediaPane, viewPane, timeline, side));

  // ---------- state update ----------
  function renderDetTable(f) {
    const dets = f.detections || [];
    if (!dets.length) { detBox.replaceChildren(el('div', { class: 'muted small' }, 'No detections.')); return; }
    const tbl = el('table', { class: 'det' }, el('thead', {}, el('tr', {}, ['Class', 'Conf', 'Range', 'LM', 'RMS'].map((h) => el('th', {}, h)))),
      el('tbody', {}, dets.map((d) => {
        const tr = el('tr', { class: sel === d.det_id ? 'sel' : '' },
          el('td', {}, el('i', { class: 'dot', style: { background: classColor(d.class_name) } }), d.class_name),
          el('td', {}, fmtNum(d.conf, 2)), el('td', {}, d.range_m != null ? fmtNum(d.range_m, 1) + ' m' : '-'),
          el('td', {}, d.landmark_id != null ? '#' + d.landmark_id : '-'), el('td', {}, fmtNum(d.reproj_rms_px, 1)));
        tr.addEventListener('mouseenter', () => { sel = d.det_id; overlay.set(frames[cur], { dim: curGap, sel }); tr.classList.add('sel'); });
        tr.addEventListener('mouseleave', () => { sel = null; overlay.set(frames[cur], { dim: curGap, sel }); tr.classList.remove('sel'); });
        return tr;
      })));
    detBox.replaceChildren(tbl);
  }

  function show(i, gap = false) {
    cur = i; curGap = gap;
    const f = frames[i];
    overlay.set(f, { dim: gap, sel });
    viewer.setFrame(i);
    renderDetTable(f);
    slider.value = i;
    counter.textContent = isVideo ? `${i + 1}/${frames.length} · #${f.index} · ${fmtTime(f.t)}` : `frame ${f.index}`;
    const status = gap ? 'gap' : f.status;
    badge.textContent = gap ? 'no data' : f.status;
    badge.className = 'status ' + status;
  }

  function nearest(t) {
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (times[m] < t) lo = m + 1; else hi = m; }
    if (lo > 0 && Math.abs(times[lo - 1] - t) <= Math.abs(times[lo] - t)) lo--;
    return lo;
  }
  function onTime(t) {
    const i = nearest(t), gap = Math.abs(times[i] - t) > tol;
    if (i !== cur || gap !== curGap) show(i, gap);
  }

  // ---------- playback: <video> or a virtual clock (mock / media missing) ----------
  let vplay = false, vt = times[0] || 0, vlast = 0, vraf = 0, vfc = 0;
  const useVideo = () => isVideo && media && mediaOk;
  function setPlayLabel(p) { playBtn.textContent = p ? 'Pause' : 'Play'; }
  function vloop(ts) {
    if (!vplay) return;
    vt += (ts - vlast) / 1000; vlast = ts;
    const end = times[times.length - 1];
    if (vt >= end) { vt = end; vplay = false; setPlayLabel(false); }
    onTime(vt);
    if (vplay) vraf = requestAnimationFrame(vloop);
  }
  function togglePlay() {
    if (!isVideo) return;
    if (useVideo()) { if (media.paused) media.play().catch((e) => toast('Playback failed: ' + e.message)); else media.pause(); return; }
    vplay = !vplay; setPlayLabel(vplay);
    if (vplay) { if (vt >= times[times.length - 1] - 1e-6) vt = times[0]; vlast = performance.now(); vraf = requestAnimationFrame(vloop); }
  }
  function seekIndex(i) {
    i = Math.max(0, Math.min(frames.length - 1, i));
    vt = times[i];
    if (useVideo()) media.currentTime = times[i] + 0.001;
    show(i, false);
  }
  function step(d) {
    if (useVideo()) media.pause();
    vplay = false; setPlayLabel(false);
    seekIndex(cur + d);
  }
  slider.addEventListener('input', () => { vplay = false; setPlayLabel(false); seekIndex(Number(slider.value)); });

  if (isVideo && media) {
    media.addEventListener('play', () => setPlayLabel(true));
    media.addEventListener('pause', () => setPlayLabel(false));
    media.addEventListener('ended', () => setPlayLabel(false));
    for (const ev of ['seeked', 'timeupdate', 'loadeddata']) media.addEventListener(ev, () => { if (mediaOk || ev === 'loadeddata') onTime(media.currentTime); });
    if ('requestVideoFrameCallback' in media) {
      const cb = (now, meta) => { if (mediaOk) onTime(meta.mediaTime); vfc = media.requestVideoFrameCallback(cb); };
      vfc = media.requestVideoFrameCallback(cb);
      cleanups.push(() => media.cancelVideoFrameCallback && media.cancelVideoFrameCallback(vfc));
    } else {
      let raf = 0;
      const loop = () => { if (mediaOk && !media.paused) onTime(media.currentTime); raf = requestAnimationFrame(loop); };
      raf = requestAnimationFrame(loop);
      cleanups.push(() => cancelAnimationFrame(raf));
    }
  }
  cleanups.push(() => { vplay = false; cancelAnimationFrame(vraf); if (media && media.pause) media.pause(); });

  // keyboard
  const onKey = (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
    if (e.target.tagName === 'VIDEO') return;
    if (!isVideo) return;
    if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    else if (e.key === ' ' && e.target.type !== 'checkbox') { togglePlay(); e.preventDefault(); }
  };
  window.addEventListener('keydown', onKey);
  cleanups.push(() => window.removeEventListener('keydown', onKey));

  // the through-camera view needs the same letterbox as the media; keep viewer sized properly on layout changes
  show(0, false);
  if (!frames.length) { box.append(el('div', { class: 'card' }, 'Result has no frames.')); }

  return { kind: 'result', destroy() { cleanups.forEach((c) => { try { c(); } catch (e) { console.error(e); } }); cleanups.length = 0; } };
}
