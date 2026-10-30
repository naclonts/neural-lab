'use strict';
const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const finite = v => typeof v === 'number' && Number.isFinite(v);
const number = v => finite(v) ? v.toLocaleString(undefined, {maximumFractionDigits: 4}) : '—';
const percent = v => finite(v) ? `${(v * 100).toFixed(2)}%` : '—';
const date = v => finite(v) ? new Date(v * 1000).toLocaleString() : '—';
function age(v) {
  if (!finite(v)) return 'No timestamp';
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - v));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}
function el(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = String(content);
  return node;
}
function pill(status, label) {
  const known = ['running', 'succeeded', 'completed', 'failed', 'interrupted', 'queued'];
  return el('span', `pill ${known.includes(status) ? status : ''}`, label || status || 'Unknown');
}
let data = null;
let filter = 'all';
let busy = false;
let selectedNote = null;
let selectedVersion = null;
function experimentName(job) {
  const h = job.hypothesis || '';
  const patterns = [
    [/^Linear-control/i, 'Linear control'],
    [/^A three-layer MLP/i, 'Deep MLP'],
    [/^Residual connections improve/i, 'Residual MLP'],
    [/^An input-conditioned low-rank/i, 'Input-conditioned low-rank gate'],
    [/^Energy-normalized/i, 'Energy-normalized mixer'],
    [/^Centered low-rank product/i, 'Centered product mixer'],
    [/^Frozen training-initialized pre-up product/i, 'Product scale calibration'],
  ];
  return patterns.find(([pattern]) => pattern.test(h))?.[1] || h || 'Untitled experiment';
}
function renderJobs() {
  if (!data) return;
  const open = new Set([...$('jobs').querySelectorAll('details[open]')].map(n => n.dataset.id));
  const search = $('search').value.trim().toLowerCase();
  const jobs = data.queue.jobs.filter(job => {
    const allowed = filter === 'all' || (filter === 'active' && ['running', 'queued'].includes(job.status)) ||
      (filter === 'issues' && ['failed', 'interrupted'].includes(job.status)) || job.status === filter;
    return allowed && `${job.id} ${job.hypothesis} ${experimentName(job)}`.toLowerCase().includes(search);
  });
  const contents = jobs.map(job => {
    const result = job.result || {};
    const metrics = result.metrics && typeof result.metrics === 'object' ? result.metrics : {};
    const development = finite(metrics.scaled_product_dev_accuracy);
    const score = metrics.candidate_accuracy ?? metrics.validation_accuracy ?? metrics.scaled_product_dev_accuracy;
    const details = el('details', 'job');
    details.dataset.id = job.id;
    details.open = open.has(job.id);
    const summary = el('summary');
    const identity = el('div', 'job-identity');
    identity.append(el('span', 'job-title', experimentName(job)),
      el('span', 'job-meta', `${job.id.slice(0, 8)} · ${date(job.created_at)}`));
    const scoreNode = el('div', 'job-score');
    scoreNode.append(el('strong', '', percent(score)), el('small', '', development ? 'scaled dev accuracy' : finite(score) ? 'held-out accuracy' : 'metrics inside'));
    summary.append(identity, scoreNode, pill(job.status, job.status === 'succeeded' ? 'Completed' : job.status), el('span', 'chevron', '›'));
    const body = el('div', 'job-body');
    body.append(el('h3', '', 'HYPOTHESIS'), el('p', 'hypothesis', job.hypothesis));
    body.append(el('p', 'muted', `Recorded training device: ${job.device || 'not reported'} · GPU pulse above is a live machine-wide sample.`));
    if (finite(score) || finite(metrics.baseline_accuracy)) {
      const comparison = el('div', 'comparison');
      const arms = development ? Object.entries(metrics).filter(([key]) => key.endsWith('_dev_accuracy')).map(([key, value]) =>
        [key.replace('_dev_accuracy', '').replaceAll('_', ' ') + ' (dev)', value, metrics[key.replace('_accuracy', '_loss')]]) :
        [['Experiment', score, metrics.candidate_heldout_loss], ['Within-job baseline', metrics.baseline_accuracy, metrics.baseline_heldout_loss]];
      if (!development && finite(metrics.additive_accuracy)) arms.push(['Additive ablation', metrics.additive_accuracy, metrics.additive_heldout_loss]);
      for (const [label, accuracy, loss] of arms) {
        const arm = el('div');
        arm.append(el('span', '', label), el('strong', '', percent(accuracy)));
        if (finite(loss)) arm.append(el('small', '', `Held-out loss ${loss.toFixed(4)}`));
        comparison.append(arm);
      }
      body.append(comparison);
    }
    if (finite(metrics.accuracy_delta)) {
      const delta = metrics.accuracy_delta;
      body.append(el('p', `result-delta ${delta < 0 ? 'negative' : delta > 0 ? 'positive' : ''}`,
        `${delta > 0 ? '+' : ''}${(delta * 100).toFixed(2)} percentage points versus this job’s baseline. Exploratory result.`));
    }
    if (result.summary) body.append(el('h3', '', 'REPORTED RESULT'), el('p', 'result-summary', result.summary));
    if (job.error) body.append(el('p', 'alert', job.error));
    const entries = Object.entries(metrics);
    if (entries.length) {
      body.append(el('h3', '', 'ALL METRICS'));
      const grid = el('div', 'metric-grid');
      for (const [key, value] of entries) {
        const metric = el('div', 'metric');
        metric.append(el('span', '', key.replaceAll('_', ' ')), el('b', '', number(value)));
        grid.append(metric);
      }
      body.append(grid);
    }
    const duration = finite(job.started_at) ? Math.max(0, (job.finished_at || data.generated_at) - job.started_at).toFixed(1) : null;
    body.append(el('p', 'job-time', `Started: ${date(job.started_at)} · Finished: ${date(job.finished_at)}${duration ? ` · Wall time: ${duration}s` : ''} · Timeout: ${number(job.timeout_seconds)}s`));
    body.append(el('p', 'job-time', `Saved results: .neural-lab/queue-outputs/${job.id}/`));
    details.append(summary, body);
    return details;
  });
  $('jobs').replaceChildren(...(contents.length ? contents : [el('p', 'empty', search || filter !== 'all' ? 'No experiments match this view.' : 'No experiments yet. Queued jobs will appear here.')]));
  const total = Object.values(data.queue.counts).reduce((sum, value) => sum + value, 0);
  text('job-count', `${jobs.length} shown · ${total} total${total > 100 ? ' · latest 100 available' : ''}`);
}
function renderScientist(scientist) {
  const reason = typeof scientist.reason === 'string' ? scientist.reason : 'No report yet';
  const label = reason === 'running' ? 'Last reported: researching' : reason === 'completed' ? 'Session completed' : reason;
  const badge = pill(reason, label);
  badge.id = 'scientist-state';
  $('scientist-state').replaceWith(badge);
  text('model', scientist.model || 'Codex research orchestrator');
  text('scientist-updated', `Last checkpoint ${age(scientist.updated_at)}`);
  const seconds = Date.now() / 1000 - (scientist.updated_at || 0);
  text('scientist-explainer', reason === 'completed' ? 'Session finished. The controller normally waits five minutes before its next session.' :
    reason === 'running' ? (seconds > 180 ? 'Checkpoint is quiet. This is the last reported state, not a live process health check.' : 'Researching, designing, or implementing. GPU work is queued separately.') :
    'State comes from the scientist’s last saved checkpoint.');
  const tokens = scientist.last_tokens;
  text('context-label', finite(tokens) ? `${number(tokens)} / 250,000` : 'No telemetry yet');
  $('context-meter').value = finite(tokens) ? Math.min(250000, tokens) : 0;
  text('context-window', `Handoff above 250,000 tokens${finite(scientist.model_context_window) ? ` · effective window ${number(scientist.model_context_window)}` : ''}`);
  const messages = Array.isArray(scientist.recent_agent_messages) ? scientist.recent_agent_messages.filter(x => typeof x === 'string') : [];
  $('messages').replaceChildren(...(messages.length ? messages.map(message => el('p', 'message', message)) : [el('p', 'muted', 'No scientist messages saved yet.')]));
}
function renderGpu() {
  const gpu = data.gpu.devices[0];
  if (!gpu) {
    for (const id of ['gpu-util', 'gpu-temp', 'gpu-power', 'gpu-memory']) text(id, '—');
    text('gpu-name', 'GPU telemetry unavailable');
    text('gpu-state', 'Unavailable');
    $('memory-meter').value = 0;
    text('gpu-explainer', 'The experiment log remains available when NVIDIA telemetry is unavailable.');
    return;
  }
  text('gpu-name', gpu.name);
  text('gpu-util', finite(gpu.utilization) ? `${number(gpu.utilization)}%` : '—');
  text('gpu-temp', finite(gpu.temperature) ? `${number(gpu.temperature)}°C` : '—');
  text('gpu-power', finite(gpu.power) ? `${number(gpu.power)} W` : '—');
  text('gpu-memory', `${number(gpu.memory_used)} / ${number(gpu.memory_total)} MiB`);
  $('memory-meter').value = finite(gpu.memory_total) && gpu.memory_total > 0 ? gpu.memory_used / gpu.memory_total * 100 : 0;
  text('gpu-state', (gpu.utilization || 0) > 0 ? 'Active' : 'Idle');
  text('gpu-explainer', (data.queue.counts.running || 0) > 0 ? 'A job is running. Small training bursts may finish between five-second samples.' :
    (data.queue.counts.queued || 0) > 0 ? 'Jobs are queued. The worker checks for new work every 30 seconds.' :
    'No jobs are running. The GPU can rest while the scientist researches.');
}
async function fetchJson(url) {
  const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(8000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
async function selectNote(note, force = false) {
  if (!force && selectedNote === note.name && selectedVersion === note.updated_at) return;
  selectedNote = note.name;
  selectedVersion = note.updated_at;
  for (const button of $('note-list').querySelectorAll('button')) button.setAttribute('aria-current', String(button.dataset.name === selectedNote));
  text('note-title', note.name);
  text('note-date', date(note.updated_at));
  text('note-body', 'Loading note…');
  try {
    const result = await fetchJson(`/api/note?name=${encodeURIComponent(note.name)}`);
    if (selectedNote === note.name) text('note-body', result.text);
  } catch {
    if (selectedNote === note.name) { text('note-body', 'This note is unavailable. Select it again to retry.'); selectedVersion = null; }
  }
}
function renderNotes() {
  $('note-list').replaceChildren(...(data.notes.length ? data.notes.map(note => {
    const button = el('button', '', note.name);
    button.type = 'button';
    button.dataset.name = note.name;
    button.setAttribute('aria-current', String(note.name === selectedNote));
    button.append(el('small', '', age(note.updated_at)));
    button.addEventListener('click', () => selectNote(note));
    return button;
  }) : [el('p', 'muted', 'No Markdown notes yet.')]));
  const selected = data.notes.find(note => note.name === selectedNote);
  if (selected) selectNote(selected);
  else if (data.notes.length) selectNote(data.notes.find(note => note.name === 'handoff.md') || data.notes[0]);
}
async function refresh() {
  if (busy) return;
  busy = true;
  $('refresh').disabled = true;
  try {
    data = await fetchJson('/api/status');
    for (const [id, status] of [['completed', 'succeeded'], ['running', 'running'], ['queued', 'queued']]) text(id, data.queue.initialized ? number(data.queue.counts[status] || 0) : '—');
    text('failed', data.queue.initialized ? number((data.queue.counts.failed || 0) + (data.queue.counts.interrupted || 0)) : '—');
    text('worker-state', data.queue.counts.running ? 'experiment in progress' : 'no active experiment');
    text('disk', `${number(data.disk_free_gb)} GiB`);
    renderScientist(data.scientist);
    renderGpu();
    renderJobs();
    renderNotes();
    $('alert').hidden = !data.errors.length;
    text('alert', data.errors.join(' '));
    text('connection', 'Observing');
    $('connection-dot').className = 'dot live';
    text('updated', `Updated ${new Date(data.generated_at * 1000).toLocaleTimeString()}`);
  } catch {
    text('connection', 'Disconnected');
    $('connection-dot').className = 'dot';
    $('alert').hidden = false;
    text('alert', 'Cannot reach the dashboard server. Displayed observations may be out of date. Retrying automatically.');
  } finally { busy = false; $('refresh').disabled = false; }
}
for (const button of document.querySelectorAll('[data-filter]')) button.addEventListener('click', () => {
  filter = button.dataset.filter;
  for (const other of document.querySelectorAll('[data-filter]')) other.setAttribute('aria-pressed', String(other === button));
  renderJobs();
});
$('search').addEventListener('input', renderJobs);
$('refresh').addEventListener('click', refresh);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
setInterval(() => { if (!document.hidden) refresh(); }, 5000);
refresh();
