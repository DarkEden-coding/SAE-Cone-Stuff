// API client with optional bearer token and a ?mock=1 offline mode.
import { toast } from './util.js';

export const MOCK = new URLSearchParams(location.search).get('mock') === '1';
const BASE = new URL('./', import.meta.url).pathname;
const TOKEN_KEY = 'cone_app_token';
let health = null;

export const getToken = () => localStorage.getItem(TOKEN_KEY) || '';

function askToken(force = false) {
  if (!force && getToken()) return getToken();
  const t = window.prompt('This server requires an access token:', '');
  if (t) localStorage.setItem(TOKEN_KEY, t.trim());
  return getToken();
}

function authHeaders() {
  const t = getToken();
  return t ? { Authorization: 'Bearer ' + t } : {};
}

async function mockGet(path) {
  if (path.startsWith('/api/live/')) {
    const m = await import('./live_mock.js');
    const [p, query] = path.split('?');
    const lm = p.match(/^\/api\/live\/sessions(?:\/([^/]+)(\/log)?)?$/);
    if (!lm) throw new Error('404 ' + path);
    if (!lm[1]) return m.mockSessions();
    const s = m.mockSessions().find((x) => x.id === lm[1]);
    if (!s) throw new Error('404 session not found');
    return lm[2] ? m.mockLog(new URLSearchParams(query).get('recording_id')) : s;
  }
  const load = async (f) => (await fetch(BASE + 'dev/' + f)).json();
  if (path === '/api/health') return { ok: true, model: 'mock', imgsz: 1920, device: 'mock', version: 'mock', auth: false };
  if (path === '/api/jobs') return load('mock_jobs.json');
  if (path === '/api/intrinsics-profiles') {
    return [
      { id: 'iPhone15,2_3840x2160', device: 'iPhone15,2', image_size: [3840, 2160], K: [[3100, 0, 1920], [0, 3100, 1080], [0, 0, 1]], n_captures: 5, updated_at: new Date().toISOString() },
      { id: 'iPhone15,2_1920x1080', device: 'iPhone15,2', image_size: [1920, 1080], K: [[1550, 0, 960], [0, 1550, 540], [0, 0, 1]], n_captures: 2, updated_at: new Date().toISOString() },
    ];
  }
  let m = path.match(/^\/api\/jobs\/([^/]+)(\/result)?$/);
  if (m) {
    const job = (await load('mock_jobs.json')).find((j) => j.id === m[1]);
    if (!job) throw new Error('404 job not found');
    if (!m[2]) return job;
    if (job.status !== 'done') throw new Error('404 result not ready');
    return load(job.kind === 'photo' ? 'mock_result_photo.json' : 'mock_result_video.json');
  }
  throw new Error('404 ' + path);
}

export async function api(path, { method = 'GET', json, retry = true, text = false } = {}) {
  if (MOCK) {
    if (method !== 'GET') throw new Error('Not available in mock mode');
    return mockGet(path);
  }
  const headers = { ...authHeaders() };
  const init = { method, headers };
  if (json !== undefined) { headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(json); }
  const r = await fetch(path, init);
  if (r.status === 401 && retry) { if (askToken(true)) return api(path, { method, json, retry: false, text }); }
  if (!r.ok) {
    let detail = '';
    try { const j = await r.json(); detail = j.detail ? (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)) : ''; } catch (e) { /* ignore */ }
    const err = new Error(`${r.status} ${detail || r.statusText}`);
    err.status = r.status;
    throw err;
  }
  if (r.status === 204) return null;
  return text ? r.text() : r.json();
}

export async function init() {
  try {
    health = await api('/api/health');
    if (health.auth) askToken();
  } catch (e) {
    toast('Cannot reach server: ' + e.message);
  }
  return health;
}
export const getHealth = () => health;

export function annotatedUrl(job) {
  const t = getToken();
  return `/api/jobs/${encodeURIComponent(job.id)}/annotated` + (t ? '?token=' + encodeURIComponent(t) : '');
}

export function mediaUrl(job) {
  // mock mode has no media unless one is supplied for testing: ?mock=1&media=<url>
  if (MOCK) return new URLSearchParams(location.search).get('media');
  if (!job.media_url) return null;
  const t = getToken();
  return job.media_url + (t ? (job.media_url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(t) : '');
}

export const listJobs = () => api('/api/jobs');
export const getJob = (id) => api('/api/jobs/' + encodeURIComponent(id));
export const getResult = (id) => api('/api/jobs/' + encodeURIComponent(id) + '/result');
export const deleteJob = (id) => api('/api/jobs/' + encodeURIComponent(id), { method: 'DELETE' });
export const rerunJob = (id, options) => api('/api/jobs/' + encodeURIComponent(id) + '/rerun', { method: 'POST', json: options });
export const listProfiles = () => api('/api/intrinsics-profiles');

// ---- live sessions ----
export const listLive = () => api('/api/live/sessions');
export const getLive = (id) => api('/api/live/sessions/' + encodeURIComponent(id));
export const deleteLive = (id) => api('/api/live/sessions/' + encodeURIComponent(id), { method: 'DELETE' });
// NDJSON text of the stored messages; recordingId selects a recording's log
export const getLiveLog = (id, recordingId) => api('/api/live/sessions/' + encodeURIComponent(id) + '/log'
  + (recordingId ? '?recording_id=' + encodeURIComponent(recordingId) : ''), { text: true });
// browsers cannot set headers on WebSockets, hence ?token=
export function liveWatchSocket(id) {
  if (MOCK) return import('./live_mock.js').then((m) => new m.MockSocket());
  const t = getToken();
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return new WebSocket(`${proto}//${location.host}/api/live/sessions/${encodeURIComponent(id)}/watch` + (t ? '?token=' + encodeURIComponent(t) : ''));
}
export function jobMediaUrl(jobId) {
  return mediaUrl({ media_url: `/api/jobs/${encodeURIComponent(jobId)}/media` });
}

// multipart upload with progress (XHR); returns Job
export function uploadJob(form, onProgress) {
  return new Promise((resolve, reject) => {
    if (MOCK) return reject(new Error('Uploads are disabled in mock mode'));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    const t = getToken();
    if (t) xhr.setRequestHeader('Authorization', 'Bearer ' + t);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total, e.loaded, e.total); };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(e); }
      } else {
        let d = xhr.statusText;
        try { const j = JSON.parse(xhr.responseText); d = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail || j); } catch (e) { /* ignore */ }
        reject(new Error(`${xhr.status} ${d}`));
      }
    };
    xhr.send(form);
  });
}
