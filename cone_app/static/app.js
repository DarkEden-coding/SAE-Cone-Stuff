import { init, getHealth, MOCK } from './api.js';
import { mountJobsView } from './jobs_view.js';
import { mountJobView } from './job_view.js';

const app = document.getElementById('app');
let current = null;

function route() {
  if (current) { try { current.destroy(); } catch (e) { console.error(e); } current = null; }
  app.replaceChildren();
  const h = location.hash.replace(/^#/, '') || '/jobs';
  const m = h.match(/^\/job\/(.+)$/);
  document.body.dataset.view = m ? 'job' : 'jobs';
  current = m ? mountJobView(app, decodeURIComponent(m[1])) : mountJobsView(app);
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
