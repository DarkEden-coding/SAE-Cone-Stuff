// Small shared helpers (no framework).

export const CLASS_COLORS = {
  BLUE: '#2f6fff', YELLOW: '#ffd400', ORANGE: '#ff7a00', ORANGE_BIG: '#ff4d00', UNDEFINED: '#9aa0aa',
};
export const classColor = (name) => CLASS_COLORS[name] || CLASS_COLORS.UNDEFINED;

export const STATUS_COLORS = { tracked: '#4ade80', init: '#60a5fa', lost: '#f87171', single: '#fbbf24', gap: '#6b7280' };

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [p, val] of Object.entries(v)) { if (p.startsWith('--')) e.style.setProperty(p, val); else e.style[p] = val; }
    }
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function toast(msg, ms = 4000) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, ms);
}

export function fmtAgo(iso) {
  const t = Date.parse(iso);
  if (!t) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return Math.round(s) + ' s ago';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  return new Date(t).toLocaleDateString();
}

export const fmtNum = (x, d = 2) => (x == null || Number.isNaN(x) ? '-' : Number(x).toFixed(d));

export function fmtTime(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  return m + ':' + (s - m * 60).toFixed(2).padStart(5, '0');
}

export function download(filename, text, mime = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = el('a', { href: url, download: filename });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function landmarksCsv(result) {
  const rows = ['id,class,x,y,z,sx,sy,sz,n_obs'];
  for (const l of result.landmarks || []) {
    const p = l.p_world || [], s = l.sigma_m || [];
    rows.push([l.id, l.class_name, p[0], p[1], p[2], s[0], s[1], s[2], l.n_obs].map((v) => v ?? '').join(','));
  }
  return rows.join('\n') + '\n';
}

// ---- options form (shared by upload and rerun) ----
export const OPTION_FIELDS = [
  { key: 'target_fps', label: 'Target fps', type: 'number', step: 1, min: 1, hint: 'video frames processed per second' },
  { key: 'imgsz', label: 'Image size', type: 'number', step: 32, min: 320, hint: 'detector input size' },
  { key: 'conf', label: 'Box conf', type: 'number', step: 0.05, min: 0, max: 1 },
  { key: 'kp_conf', label: 'Keypoint conf', type: 'number', step: 0.05, min: 0, max: 1 },
  { key: 'max_range_m', label: 'Max range (m)', type: 'number', step: 1, min: 1 },
  { key: 'kp_correction', label: 'Keypoint correction', type: 'checkbox' },
  { key: 'flat_ground', label: 'Flat-ground assumption', type: 'checkbox' },
];

export function optionsFields(values = {}, prefix = 'opt') {
  const wrap = el('div', { class: 'opt-grid' });
  for (const f of OPTION_FIELDS) {
    const id = `${prefix}-${f.key}`;
    const input = el('input', { id, name: f.key, type: f.type, step: f.step, min: f.min, max: f.max });
    if (f.type === 'checkbox') input.checked = values[f.key] !== false && values[f.key] != null ? !!values[f.key] : f.key !== 'x';
    else if (values[f.key] != null) input.value = values[f.key];
    wrap.append(el('label', { class: f.type === 'checkbox' ? 'chk' : 'fld', for: id, title: f.hint || '' },
      f.type === 'checkbox' ? [input, f.label] : [el('span', {}, f.label), input]));
  }
  return wrap;
}

export function readOptions(root) {
  const o = {};
  for (const f of OPTION_FIELDS) {
    const i = root.querySelector(`[name="${f.key}"]`);
    if (!i) continue;
    if (f.type === 'checkbox') o[f.key] = i.checked;
    else if (i.value !== '') o[f.key] = Number(i.value);
  }
  return o;
}
