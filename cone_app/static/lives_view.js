import { listLive, deleteLive } from './api.js';
import { el, fmtAgo, toast } from './util.js';

const fmtDur = (s) => {
  s = Math.round(s || 0);
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
};

export function mountLivesView(root) {
  let timer = null, dead = false, sessions = null;
  const listBox = el('div', { class: 'job-list' });
  root.append(el('div', { class: 'page' },
    el('div', { class: 'page-head' }, el('h1', {}, 'Live'),
      el('p', { class: 'muted' }, 'Sessions streamed from the ConeCapture app in Live mode. Open one to watch it live or replay it.')),
    listBox));

  async function refresh() {
    clearTimeout(timer);
    try {
      sessions = await listLive();
      if (!dead) render();
    } catch (e) {
      if (!dead && !sessions) listBox.replaceChildren(el('div', { class: 'card err' }, 'Failed to load live sessions: ' + e.message));
    }
    if (!dead) timer = setTimeout(refresh, 3000);
  }

  function render() {
    if (!sessions.length) {
      listBox.replaceChildren(el('div', { class: 'card empty' }, 'No live sessions yet. Start Live mode in the iOS app.'));
      return;
    }
    listBox.replaceChildren(...sessions.map(row));
  }

  function row(s) {
    const open = () => { location.hash = '#/live/' + encodeURIComponent(s.id); };
    const del = el('button', { class: 'btn small danger', title: 'Delete session', onclick: async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete live session "${s.name || s.id}"?`)) return;
      try { await deleteLive(s.id); refresh(); } catch (err) { toast('Delete failed: ' + err.message); }
    } }, 'Delete');
    const recs = (s.recordings || []).map((r) => el('span', { class: 'rec-chip' },
      el('i', { class: 'rec-dot' }), (r.name || r.recording_id) + (r.t_end != null ? ` (${fmtDur(r.t_end - r.t_start)})` : ' (recording)'),
      r.job_id ? el('a', { href: '#/job/' + encodeURIComponent(r.job_id), onclick: (e) => e.stopPropagation() }, 'job') : null));
    return el('div', { class: 'card job-row live-row', onclick: open },
      el('div', { class: 'jr-main' },
        el('div', { class: 'jr-title' }, s.name || s.id, s.active ? el('span', { class: 'status tracked live-badge' }, 'active') : null),
        el('div', { class: 'muted small' }, `${s.id.slice(0, 8)} · ${fmtAgo(s.started_at)} · ${fmtDur(s.duration_s)} · ${s.n_frames ?? 0} frames · ${s.n_landmarks ?? 0} cones`),
        recs.length ? el('div', { class: 'rec-list' }, recs) : null),
      el('div', { class: 'jr-actions' },
        el('a', { class: 'btn small', href: '#/live/' + encodeURIComponent(s.id), onclick: (e) => e.stopPropagation() }, s.active ? 'Watch' : 'Replay'), del));
  }

  refresh();
  return { destroy() { dead = true; clearTimeout(timer); } };
}
