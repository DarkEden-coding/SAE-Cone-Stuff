import { listJobs, deleteJob, listProfiles, uploadJob, MOCK } from './api.js';
import { el, fmtAgo, toast, optionsFields, readOptions } from './util.js';

const ACTIVE = new Set(['queued', 'running']);

export function mountJobsView(root) {
  let timer = null, dead = false, jobs = [];
  const listBox = el('div', { class: 'job-list' });
  const upload = buildUploadCard();
  root.append(
    el('div', { class: 'page' },
      el('div', { class: 'page-head' }, el('h1', {}, 'Jobs'),
        el('p', { class: 'muted' }, 'Photos and videos captured with the ConeCapture iOS app are uploaded and processed automatically and appear here. ',
          'Use the form below only to upload a file by hand.')),
      listBox, upload.card));

  async function refresh() {
    try {
      jobs = await listJobs();
      if (dead) return;
      render();
    } catch (e) { listBox.replaceChildren(el('div', { class: 'card err' }, 'Failed to load jobs: ' + e.message)); }
    schedule();
  }
  function schedule() {
    clearTimeout(timer);
    if (!dead && jobs.some((j) => ACTIVE.has(j.status)) && !MOCK) timer = setTimeout(refresh, 2000);
  }

  function render() {
    if (!jobs.length) {
      listBox.replaceChildren(el('div', { class: 'card empty' }, 'No jobs yet. Capture something with the iOS app or upload a file below.'));
      return;
    }
    listBox.replaceChildren(...jobs.map(jobRow));
  }

  function jobRow(j) {
    const active = ACTIVE.has(j.status);
    const pct = Math.round((j.progress || 0) * 100);
    const del = el('button', { class: 'btn small danger', title: 'Delete job', onclick: async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete job "${j.name || j.id}"?`)) return;
      try { await deleteJob(j.id); refresh(); } catch (err) { toast('Delete failed: ' + err.message); }
    } }, 'Delete');
    return el('div', { class: 'card job-row', onclick: () => { location.hash = '#/job/' + encodeURIComponent(j.id); } },
      el('div', { class: 'jr-main' },
        el('div', { class: 'jr-title' }, j.name || j.id,
          el('span', { class: 'tag' }, j.kind), el('span', { class: 'tag ' + (j.source === 'ios' ? 'ios' : '') }, j.source || '?')),
        el('div', { class: 'muted small' }, j.id + ' · ' + fmtAgo(j.created_at) +
          (j.duration_s ? ` · ${j.duration_s.toFixed(1)} s` : '') + (j.image_size ? ` · ${j.image_size[0]}×${j.image_size[1]}` : ''))),
      el('div', { class: 'jr-status' },
        el('span', { class: 'status ' + j.status }, j.status),
        (active || j.status === 'failed') && el('span', { class: 'muted small' }, j.stage || ''),
        active && el('div', { class: 'bar' }, el('div', { class: 'fill', style: { width: pct + '%' } })),
        j.status === 'failed' && j.error && el('div', { class: 'small err-text', title: j.error }, j.error)),
      el('div', { class: 'jr-actions' },
        el('a', { class: 'btn small', href: '#/job/' + encodeURIComponent(j.id), onclick: (e) => e.stopPropagation() }, 'Open'), del));
  }

  refresh();
  upload.load();
  return { destroy() { dead = true; clearTimeout(timer); } };
}

function buildUploadCard() {
  const profileSel = el('select', { name: 'profile_id' });
  const profileHint = el('div', { class: 'muted small' });
  const file = el('input', { type: 'file', name: 'media', accept: 'image/*,video/*,.mp4,.mov,.jpg,.jpeg,.png', required: true });
  const nameIn = el('input', { type: 'text', name: 'name', placeholder: 'optional label' });
  const opts = optionsFields({ target_fps: 10, imgsz: 1920, conf: 0.25, kp_conf: 0.5, max_range_m: 40, kp_correction: true, flat_ground: true }, 'up');
  const bar = el('div', { class: 'bar', hidden: true }, el('div', { class: 'fill' }));
  const msg = el('div', { class: 'small' });
  const submit = el('button', { class: 'btn primary', type: 'submit' }, 'Upload and process');

  const form = el('form', { class: 'upload', onsubmit: onSubmit },
    el('div', { class: 'row' },
      el('label', { class: 'fld grow' }, el('span', {}, 'Photo or video (JPEG / H.264 MP4)'), file),
      el('label', { class: 'fld grow' }, el('span', {}, 'Name'), nameIn)),
    el('label', { class: 'fld' }, el('span', {}, 'Camera intrinsics profile'), profileSel, profileHint),
    el('details', {}, el('summary', {}, 'Processing options'), opts),
    el('div', { class: 'row' }, submit, msg), bar);

  const card = el('div', { class: 'card' },
    el('h2', {}, 'Manual upload'),
    el('p', { class: 'muted small' }, 'Intrinsics come only from iOS captures (ARKit). A profile is stored automatically from every capture ',
      'per device and resolution; pick one that matches the media you upload. The media resolution must equal the profile resolution.'),
    form);

  async function load() {
    try {
      const profiles = await listProfiles();
      profileSel.replaceChildren(...profiles.map((p) => el('option', { value: p.id },
        `${p.device} · ${p.image_size[0]}×${p.image_size[1]} · fx ${Number(p.K[0][0]).toFixed(0)} (${p.n_captures} capture${p.n_captures === 1 ? '' : 's'})`)));
      if (!profiles.length) {
        profileHint.textContent = 'No profiles yet: upload at least one capture from the iOS app first.';
        submit.disabled = true;
      }
    } catch (e) { profileHint.textContent = 'Could not load profiles: ' + e.message; }
  }

  function onSubmit(ev) {
    ev.preventDefault();
    if (MOCK) { toast('Uploads are disabled in mock mode'); return; }
    if (!file.files[0]) return;
    const fd = new FormData();
    fd.append('media', file.files[0]);
    fd.append('profile_id', profileSel.value);
    if (nameIn.value.trim()) fd.append('name', nameIn.value.trim());
    fd.append('options', JSON.stringify(readOptions(opts)));
    submit.disabled = true; bar.hidden = false; msg.className = 'small'; msg.textContent = 'Uploading…';
    const fill = bar.firstChild;
    uploadJob(fd, (f, l, t) => { fill.style.width = (f * 100).toFixed(1) + '%'; msg.textContent = `Uploading ${(l / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB`; })
      .then((job) => { location.hash = '#/job/' + encodeURIComponent(job.id); })
      .catch((e) => { msg.className = 'small err-text'; msg.textContent = 'Upload failed: ' + e.message; submit.disabled = false; bar.hidden = true; });
  }
  return { card, load };
}
