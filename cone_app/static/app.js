import { init, getHealth, MOCK } from './api.js';
import { mountJobsView } from './jobs_view.js';
import { mountJobView } from './job_view.js';
import { mountLivesView } from './lives_view.js';
import { mountLiveView } from './live_view.js';

const app = document.getElementById('app');
let current = null;

function route() {
  if (current) { try { current.destroy(); } catch (e) { console.error(e); } current = null; }
  app.replaceChildren();
  const h = location.hash.replace(/^#/, '') || '/jobs';
  const job = h.match(/^\/job\/(.+)$/);
  const live = h.match(/^\/live(?:\/(.+))?$/);
  document.body.dataset.view = job ? 'job' : live ? 'live' : 'jobs';
  for (const a of document.querySelectorAll('.topbar nav a')) a.classList.toggle('current', a.dataset.nav === document.body.dataset.view);
  if (job) current = mountJobView(app, decodeURIComponent(job[1]));
  else if (live) current = live[1] ? mountLiveView(app, decodeURIComponent(live[1])) : mountLivesView(app);
  else current = mountJobsView(app);
}

(async () => {
  await init();
  const hl = document.getElementById('health');
  const h = getHealth();
  hl.textContent = MOCK ? 'mock data' : h ? `${h.device} · imgsz ${h.imgsz}` : 'offline';
  hl.classList.toggle('bad', !h && !MOCK);
  window.addEventListener('hashchange', route);
  route();
})();
