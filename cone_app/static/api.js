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

export async function api(path, { method = 'GET', json, retry = true } = {}) {
  if (MOCK) {
    if (method !== 'GET') throw new Error('Not available in mock mode');
    return mockGet(path);
  }
  const headers = { ...authHeaders() };
  const init = { method, headers };
  if (json !== undefined) { headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(json); }
  const r = await fetch(path, init);
  if (r.status === 401 && retry) { if (askToken(true)) return api(path, { method, json, retry: false }); }
  if (!r.ok) {
    let detail = '';
    try { const j = await r.json(); detail = j.detail ? (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)) : ''; } catch (e) { /* ignore */ }
    const err = new Error(`${r.status} ${detail || r.statusText}`);
    err.status = r.status;
    throw err;
  }
  if (r.status === 204) return null;
  return r.json();
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
