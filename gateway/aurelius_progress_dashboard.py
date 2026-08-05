"""Live local dashboard for Aurelius research progress.

Run:

    python -m gateway.aurelius_progress_dashboard --port 7871

The dashboard reads ``data/aurelius_progress/state.json`` and JSONL event/metric
logs. It intentionally avoids external JavaScript/CSS dependencies so it can run
offline on a laptop, inside a training VM, or on a rented GPU box.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import logging
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from src.monitoring.progress_store import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_STATE_PATH,
    append_event,
    append_metric,
    build_snapshot,
    import_metrics_from_jsonl,
    update_milestone,
)

logger = logging.getLogger(__name__)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Aurelius Progress Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080b14;
      --panel: #111827;
      --panel-2: #151f32;
      --panel-3: #1b263b;
      --border: #26364f;
      --text: #e5edf7;
      --muted: #91a4bd;
      --blue: #60a5fa;
      --cyan: #22d3ee;
      --green: #34d399;
      --yellow: #fbbf24;
      --orange: #fb923c;
      --red: #f87171;
      --purple: #a78bfa;
      --shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
      --radius: 16px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.16), transparent 32rem),
        radial-gradient(circle at 85% 10%, rgba(167, 139, 250, 0.13), transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(16px);
      background: rgba(8, 11, 20, 0.86);
      border-bottom: 1px solid var(--border);
      padding: 16px 22px;
    }
    .topline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 1.25rem;
      letter-spacing: 0.02em;
    }
    .subtitle {
      color: var(--muted);
      font-size: 0.82rem;
      margin-top: 4px;
    }
    .header-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, .button {
      border: 1px solid var(--border);
      background: var(--panel-3);
      color: var(--text);
      border-radius: 10px;
      padding: 8px 12px;
      font-weight: 650;
      cursor: pointer;
      font-family: inherit;
    }
    button:hover, .button:hover { border-color: var(--blue); }
    button.primary { background: linear-gradient(135deg, #1d4ed8, #0891b2); border-color: transparent; }
    .pill {
      border: 1px solid var(--border);
      background: rgba(17, 24, 39, 0.8);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 0.78rem;
      white-space: nowrap;
    }
    main {
      padding: 18px 22px 32px;
      max-width: 1500px;
      margin: 0 auto;
    }
    .progress-shell {
      height: 12px;
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 14px;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--blue), var(--purple));
      transition: width 0.35s ease;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }
    .card {
      background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(12, 18, 31, 0.96));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      min-width: 0;
    }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .card h2, .card h3 {
      margin: 0 0 12px;
      font-size: 0.92rem;
      color: #dbeafe;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .metric-value {
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.04em;
    }
    .metric-label {
      color: var(--muted);
      font-size: 0.82rem;
      margin-top: 4px;
    }
    .small { font-size: 0.82rem; color: var(--muted); }
    .muted { color: var(--muted); }
    .mono { font-family: var(--mono); }
    .row { display: flex; gap: 10px; align-items: center; justify-content: space-between; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      border: 1px solid transparent;
    }
    .status::before { content: ""; width: 7px; height: 7px; border-radius: 999px; background: currentColor; }
    .status-done { color: var(--green); background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.28); }
    .status-review { color: var(--blue); background: rgba(96, 165, 250, 0.12); border-color: rgba(96, 165, 250, 0.28); }
    .status-in_progress { color: var(--orange); background: rgba(251, 146, 60, 0.12); border-color: rgba(251, 146, 60, 0.28); }
    .status-planned { color: var(--yellow); background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.28); }
    .status-blocked { color: var(--red); background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.28); }
    .status-not_started { color: var(--muted); background: rgba(145, 164, 189, 0.10); border-color: rgba(145, 164, 189, 0.22); }
    .bar {
      height: 9px;
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--blue));
      border-radius: inherit;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 16px 0;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
    }
    .tab { background: transparent; color: var(--muted); }
    .tab.active { color: var(--text); border-color: var(--blue); background: rgba(96, 165, 250, 0.12); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .phase {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(17, 24, 39, 0.72);
      margin-bottom: 12px;
      overflow: hidden;
    }
    .phase-header {
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      cursor: pointer;
      background: rgba(21, 31, 50, 0.72);
    }
    .phase-body { padding: 0 14px 14px; }
    .milestone {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 12px;
      align-items: start;
      padding: 10px 0;
      border-top: 1px solid rgba(38, 54, 79, 0.7);
      font-size: 0.86rem;
    }
    .milestone:first-child { border-top: 0; }
    .milestone-title { font-weight: 700; color: #e5edf7; }
    .milestone-meta { color: var(--muted); margin-top: 3px; }
    .event {
      padding: 10px 0;
      border-bottom: 1px solid rgba(38, 54, 79, 0.55);
      font-size: 0.86rem;
    }
    .event:last-child { border-bottom: 0; }
    .event-time { color: var(--cyan); font-family: var(--mono); font-size: 0.76rem; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
    }
    th, td {
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid rgba(38, 54, 79, 0.6);
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .chart-box {
      width: 100%;
      min-height: 280px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(8, 11, 20, 0.55);
      padding: 10px;
      overflow: hidden;
    }
    svg { width: 100%; height: 260px; display: block; }
    .axis { stroke: #334155; stroke-width: 1; }
    .grid-line { stroke: rgba(51, 65, 85, 0.45); stroke-width: 1; }
    .line { fill: none; stroke: var(--cyan); stroke-width: 2.5; }
    .point { fill: var(--cyan); stroke: #08111f; stroke-width: 1; }
    .chart-label { fill: var(--muted); font-size: 11px; font-family: var(--mono); }
    .input, textarea, select {
      width: 100%;
      background: #0b1220;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      font-family: inherit;
      outline: none;
    }
    textarea { min-height: 92px; resize: vertical; }
    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .form-grid .full { grid-column: 1 / -1; }
    .danger { color: var(--red); }
    .ok { color: var(--green); }
    .warn { color: var(--yellow); }
    .tag {
      display: inline-flex;
      border-radius: 999px;
      padding: 3px 7px;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.72rem;
      margin-right: 4px;
      margin-bottom: 4px;
    }
    code {
      font-family: var(--mono);
      color: #bfdbfe;
    }
    @media (max-width: 980px) {
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 { grid-column: span 12; }
      .form-grid { grid-template-columns: 1fr; }
      header { padding: 14px 16px; }
      main { padding: 14px 16px 28px; }
    }
  </style>
</head>
<body>
<header>
  <div class="topline">
    <div>
      <h1>🜔 Aurelius Progress Dashboard</h1>
      <div class="subtitle">Live research command center for training, data, interpretability, safety, agentic deployment, and publication readiness.</div>
    </div>
    <div class="header-actions">
      <span class="pill" id="last-updated">Waiting for data…</span>
      <button id="refresh-btn">Refresh</button>
      <button id="seed-btn" class="primary">Seed Roadmap</button>
    </div>
  </div>
  <div class="progress-shell" title="Overall roadmap progress"><div class="progress-bar" id="overall-bar"></div></div>
</header>

<main>
  <nav class="tabs" aria-label="Dashboard sections">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="training">Training</button>
    <button class="tab" data-tab="data">Data</button>
    <button class="tab" data-tab="eval">Evaluation</button>
    <button class="tab" data-tab="interp">Interpretability</button>
    <button class="tab" data-tab="roadmap">Roadmap</button>
    <button class="tab" data-tab="events">Events</button>
    <button class="tab" data-tab="api">API / Updates</button>
  </nav>

  <section id="overview" class="tab-panel active">
    <div class="grid">
      <div class="card span-3">
        <h2>Overall Progress</h2>
        <div class="metric-value" id="overall-progress">0%</div>
        <div class="metric-label" id="milestone-count">0 milestones</div>
      </div>
      <div class="card span-3">
        <h2>Done Milestones</h2>
        <div class="metric-value" id="done-count">0</div>
        <div class="metric-label">Evidence-backed completions</div>
      </div>
      <div class="card span-3">
        <h2>High Risks</h2>
        <div class="metric-value" id="high-risk-count">0</div>
        <div class="metric-label">Active high-severity risks</div>
      </div>
      <div class="card span-3">
        <h2>Latest Event</h2>
        <div class="metric-value" id="latest-event-kind" style="font-size:1.25rem;">—</div>
        <div class="metric-label" id="latest-event-time">No events yet</div>
      </div>

      <div class="card span-8">
        <h2>Track Progress</h2>
        <div id="track-list"></div>
      </div>
      <div class="card span-4">
        <h2>Current Focus</h2>
        <div id="current-focus"></div>
      </div>

      <div class="card span-6">
        <h2>Risk Register</h2>
        <div id="risk-list"></div>
      </div>
      <div class="card span-6">
        <h2>Latest Metrics</h2>
        <div id="latest-metrics"></div>
      </div>
    </div>
  </section>

  <section id="training" class="tab-panel">
    <div class="grid">
      <div class="card span-12">
        <div class="row">
          <h2>Training Metric Series</h2>
          <select id="metric-select" style="max-width:260px;"></select>
        </div>
        <div class="chart-box"><svg id="metric-chart" role="img" aria-label="Selected metric chart"></svg></div>
      </div>
      <div class="card span-12">
        <h2>Metric Summary</h2>
        <table><thead><tr><th>Metric</th><th>Latest</th><th>Mean</th><th>Min</th><th>Max</th><th>Run</th><th>Step</th></tr></thead><tbody id="metric-summary"></tbody></table>
      </div>
    </div>
  </section>

  <section id="data" class="tab-panel">
    <div class="grid">
      <div class="card span-8">
        <h2>Data Pipeline Roadmap</h2>
        <div id="data-roadmap"></div>
      </div>
      <div class="card span-4">
        <h2>Data Quality Gates</h2>
        <ul class="small" id="data-gates"></ul>
      </div>
    </div>
  </section>

  <section id="eval" class="tab-panel">
    <div class="grid">
      <div class="card span-6">
        <h2>Evaluation Roadmap</h2>
        <div id="eval-roadmap"></div>
      </div>
      <div class="card span-6">
        <h2>Eval Metrics</h2>
        <table><thead><tr><th>Metric</th><th>Latest</th><th>Target / Direction</th></tr></thead><tbody id="eval-metrics"></tbody></table>
      </div>
    </div>
  </section>

  <section id="interp" class="tab-panel">
    <div class="grid">
      <div class="card span-8">
        <h2>Interpretability Roadmap</h2>
        <div id="interp-roadmap"></div>
      </div>
      <div class="card span-4">
        <h2>Mechanistic Evidence Checklist</h2>
        <ul class="small" id="interp-checklist"></ul>
      </div>
    </div>
  </section>

  <section id="roadmap" class="tab-panel">
    <div id="roadmap-list"></div>
  </section>

  <section id="events" class="tab-panel">
    <div class="card">
      <h2>Event Feed</h2>
      <div id="event-feed"></div>
    </div>
  </section>

  <section id="api" class="tab-panel">
    <div class="grid">
      <div class="card span-6">
        <h2>Append Event</h2>
        <div class="form-grid">
          <input class="input full" id="event-message" placeholder="Event message" />
          <input class="input" id="event-kind" placeholder="kind: training" value="general" />
          <select id="event-severity" class="input">
            <option value="info">info</option>
            <option value="warning">warning</option>
            <option value="error">error</option>
            <option value="critical">critical</option>
          </select>
          <button class="button full" id="event-submit">Append Event</button>
        </div>
      </div>
      <div class="card span-6">
        <h2>Append Metric</h2>
        <div class="form-grid">
          <input class="input" id="metric-name" placeholder="train_loss" />
          <input class="input" id="metric-value" placeholder="2.31" />
          <input class="input" id="metric-step" placeholder="step" />
          <input class="input" id="metric-run" placeholder="run id" value="default" />
          <input class="input full" id="metric-unit" placeholder="unit, optional" />
          <button class="button full" id="metric-submit">Append Metric</button>
        </div>
      </div>
      <div class="card span-12">
        <h2>Update Milestone</h2>
        <div class="form-grid">
          <input class="input full" id="milestone-id" placeholder="milestone id, e.g. phase1-baseline-1-4b" />
          <select class="input" id="milestone-status">
            <option value="done">done</option>
            <option value="review">review</option>
            <option value="in_progress">in_progress</option>
            <option value="planned">planned</option>
            <option value="blocked">blocked</option>
            <option value="not_started">not_started</option>
          </select>
          <input class="input" id="milestone-owner" placeholder="owner" />
          <input class="input full" id="milestone-evidence" placeholder="evidence / artifact path" />
          <textarea class="input full" id="milestone-notes" placeholder="notes"></textarea>
          <button class="button full" id="milestone-submit">Update Milestone</button>
        </div>
      </div>
      <div class="card span-12">
        <h2>API Shape</h2>
        <p class="small">
          <code>GET /api/snapshot</code>, <code>POST /api/event</code>, <code>POST /api/metric</code>,
          <code>POST /api/milestone</code>, <code>POST /api/import-metrics</code>.
          The dashboard polls every 5 seconds and renders all state from local JSON/JSONL files.
        </p>
      </div>
    </div>
  </section>
</main>

<script>
const state = { snapshot: null, selectedMetric: null };
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}
function fmtTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString();
}
function fmtIso(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}
function pct(value) { return `${Math.round(Number(value || 0))}%`; }
function statusClass(status) { return `status-${String(status || 'not_started').toLowerCase()}`; }

async function refresh() {
  try {
    const res = await fetch('/api/snapshot');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.snapshot = await res.json();
    render();
    $('last-updated').textContent = `Updated ${fmtIso(state.snapshot.generated_at)}`;
  } catch (err) {
    $('last-updated').textContent = `Disconnected: ${err.message}`;
  }
}
function render() {
  const s = state.snapshot;
  if (!s) return;
  $('overall-progress').textContent = pct(s.overall.progress);
  $('overall-bar').style.width = pct(s.overall.progress);
  $('milestone-count').textContent = `${s.overall.milestones} milestones across ${s.overall.roadmap_phases} phases`;
  const done = s.roadmap.reduce((sum, phase) => sum + phase.milestones.filter(m => m.status === 'done').length, 0);
  $('done-count').textContent = done;
  $('high-risk-count').textContent = s.risk_summary.high;
  const latest = s.recent_events[0];
  $('latest-event-kind').textContent = latest ? latest.kind : '—';
  $('latest-event-time').textContent = latest ? fmtTime(latest.timestamp) : 'No events yet';
  renderTracks(s.tracks);
  renderFocus(s.current_focus);
  renderRisks(s.risk_summary.items);
  renderLatestMetrics(s.latest_metrics);
  renderMetricSelect(s.metric_names);
  renderTraining();
  renderRoadmapSections(s);
  renderRoadmap();
  renderEvents(s.recent_events);
}
function renderTracks(tracks) {
  $('track-list').innerHTML = tracks.map(t => `
    <div style="margin-bottom:12px;">
      <div class="row"><strong>${escapeHtml(t.name)}</strong><span>${pct(t.progress)}</span></div>
      <div class="bar" style="margin-top:6px;"><span style="width:${Number(t.progress || 0).toFixed(2)}%"></span></div>
      <div class="small">${escapeHtml(t.objective || '')}</div>
    </div>
  `).join('');
}
function renderFocus(items) {
  $('current-focus').innerHTML = (items || []).map(item => `<div class="tag">${escapeHtml(item)}</div>`).join('');
}
function renderRisks(items) {
  $('risk-list').innerHTML = (items || []).map(r => `
    <div class="event">
      <div class="row"><strong>${escapeHtml(r.title)}</strong><span class="pill ${r.severity === 'high' ? 'danger' : r.severity === 'medium' ? 'warn' : 'ok'}">${escapeHtml(r.severity)}</span></div>
      <div class="small">${escapeHtml(r.mitigation || '')}</div>
    </div>
  `).join('');
}
function renderLatestMetrics(metrics) {
  const names = Object.keys(metrics || {}).slice(0, 8);
  $('latest-metrics').innerHTML = names.map(name => {
    const m = metrics[name];
    return `<div class="event"><div class="row"><strong>${escapeHtml(name)}</strong><span class="mono">${Number(m.latest).toPrecision(5)}</span></div><div class="small">mean ${Number(m.mean).toPrecision(5)} · min ${Number(m.min).toPrecision(5)} · max ${Number(m.max).toPrecision(5)}</div></div>`;
  }).join('') || '<div class="small">No metrics yet. Append a metric from the API tab or training hook.</div>';
}
function renderMetricSelect(names) {
  const select = $('metric-select');
  const current = state.selectedMetric;
  select.innerHTML = names.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
  if (current && names.includes(current)) select.value = current;
  state.selectedMetric = select.value;
}
function renderTraining() {
  const s = state.snapshot;
  const select = $('metric-select');
  const name = select.value;
  const series = s.metric_series?.[name] || [];
  drawChart(series, $('metric-chart'), name);
  const rows = Object.entries(s.latest_metrics || {}).map(([metric, m]) => `
    <tr>
      <td class="mono">${escapeHtml(metric)}</td>
      <td class="mono">${Number(m.latest).toPrecision(5)}</td>
      <td class="mono">${Number(m.mean).toPrecision(5)}</td>
      <td class="mono">${Number(m.min).toPrecision(5)}</td>
      <td class="mono">${Number(m.max).toPrecision(5)}</td>
      <td>${escapeHtml(m.latest_run_id || '')}</td>
      <td>${m.latest_step ?? '—'}</td>
    </tr>
  `).join('');
  $('metric-summary').innerHTML = rows || '<tr><td colspan="7" class="small">No metrics yet.</td></tr>';
}
function drawChart(series, svg, title) {
  svg.innerHTML = '';
  const w = 900, h = 260, pad = 34;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  if (!series.length) {
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pad);
    text.setAttribute('y', h / 2);
    text.setAttribute('fill', '#91a4bd');
    text.textContent = 'No metric samples yet.';
    svg.appendChild(text);
    return;
  }
  const values = series.map(d => Number(d.value)).filter(Number.isFinite);
  const min = Math.min(...values), max = Math.max(...values);
  const span = max === min ? 1 : max - min;
  const xFor = i => pad + (series.length === 1 ? 0 : i * (w - pad * 2) / (series.length - 1));
  const yFor = v => h - pad - ((Number(v) - min) / span) * (h - pad * 2);
  for (let i = 0; i <= 4; i++) {
    const y = pad + i * (h - pad * 2) / 4;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', pad); line.setAttribute('x2', w - pad); line.setAttribute('y1', y); line.setAttribute('y2', y);
    line.setAttribute('class', 'grid-line'); svg.appendChild(line);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', 4); label.setAttribute('y', y + 4); label.setAttribute('class', 'chart-label');
    label.textContent = (max - i * span / 4).toPrecision(4); svg.appendChild(label);
  }
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', series.map((d, i) => `${i ? 'L' : 'M'}${xFor(i).toFixed(2)} ${yFor(d.value).toFixed(2)}`).join(' '));
  path.setAttribute('class', 'line'); svg.appendChild(path);
  series.forEach((d, i) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', xFor(i)); circle.setAttribute('cy', yFor(d.value)); circle.setAttribute('r', 3.2); circle.setAttribute('class', 'point'); svg.appendChild(circle);
  });
  const titleNode = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  titleNode.setAttribute('x', pad); titleNode.setAttribute('y', 20); titleNode.setAttribute('fill', '#dbeafe'); titleNode.setAttribute('font-weight', '700');
  titleNode.textContent = title || 'Metric';
  svg.appendChild(titleNode);
}
function renderRoadmapSections(s) {
  const byTrack = id => s.tracks.find(t => t.id === id);
  $('data-roadmap').innerHTML = renderPhaseMilestones(s.roadmap.find(p => p.id === 'phase-1')?.milestones || [], s);
  $('eval-roadmap').innerHTML = renderPhaseMilestones(s.roadmap.find(p => p.id === 'phase-4')?.milestones || [], s);
  $('interp-roadmap').innerHTML = renderPhaseMilestones(s.roadmap.find(p => p.id === 'phase-2')?.milestones || [], s);
  $('data-gates').innerHTML = [
    'Deduplication rate and near-duplicate clusters',
    'Domain balance and curriculum ordering',
    'Contamination probes against target benchmarks',
    'Safety/toxicity/PII screening summary',
    'Manifest hashes and source provenance'
  ].map(x => `<li>${escapeHtml(x)}</li>`).join('');
  $('interp-checklist').innerHTML = [
    'Feature atlas with sparse autoencoders',
    'Layer probes for uncertainty and plan depth',
    'Causal tracing on answer-critical tokens',
    'Activation patching for memory gates',
    'Metacognition tasks where self-checking changes final answer'
  ].map(x => `<li>${escapeHtml(x)}</li>`).join('');
  const evalTargets = s.project.metric_targets || {};
  $('eval-metrics').innerHTML = Object.entries(evalTargets).map(([name, target]) => `
    <tr><td class="mono">${escapeHtml(name)}</td><td>${escapeHtml(target.target ?? '—')}</td><td>${escapeHtml(target.direction)}</td></tr>
  `).join('');
}
function renderPhaseMilestones(milestones, s) {
  return (milestones || []).map(m => `
    <div class="milestone">
      <div><div class="milestone-title">${escapeHtml(m.title)}</div><div class="milestone-meta">${escapeHtml(m.next_action || '')}</div><div class="small">${escapeHtml(m.evidence || '')}</div></div>
      <span class="${statusClass(m.status)}">${escapeHtml(m.status)}</span>
    </div>
  `).join('') || '<div class="small">No milestones yet.</div>';
}
function renderRoadmap() {
  const s = state.snapshot;
  $('roadmap-list').innerHTML = s.roadmap.map(phase => `
    <div class="phase">
      <div class="phase-header">
        <div><strong>${escapeHtml(phase.phase)}: ${escapeHtml(phase.title)}</strong><div class="small">${escapeHtml(phase.objective || '')}</div></div>
        <span class="${statusClass(phase.status)}">${escapeHtml(phase.status)} · ${pct(phaseProgress(phase))}</span>
      </div>
      <div class="phase-body">${renderPhaseMilestones(phase.milestones || [], s)}</div>
    </div>
  `).join('');
}
function phaseProgress(phase) {
  const total = phase.milestones.reduce((sum, m) => sum + Number(m.weight || 1), 0);
  const done = phase.milestones.reduce((sum, m) => sum + Number(m.weight || 1) * ({done:1, review:.9, in_progress:.6, planned:.2, blocked:.05, not_started:0}[m.status] || 0), 0);
  return total ? 100 * done / total : 0;
}
function renderEvents(events) {
  $('event-feed').innerHTML = (events || []).map(e => `
    <div class="event">
      <div class="event-time">${escapeHtml(fmtTime(e.timestamp))}</div>
      <div class="row"><strong>${escapeHtml(e.kind)}</strong><span class="pill">${escapeHtml(e.severity)}</span></div>
      <div>${escapeHtml(e.message)}</div>
      <div class="small mono">${escapeHtml(JSON.stringify(e.metadata || {}))}</div>
    </div>
  `).join('') || '<div class="small">No events yet.</div>';
}

document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  $(btn.dataset.tab).classList.add('active');
}));
$('metric-select').addEventListener('change', renderTraining);
$('refresh-btn').addEventListener('click', refresh);
$('seed-btn').addEventListener('click', async () => {
  const res = await fetch('/api/reset-demo', { method: 'POST' });
  const data = await res.json();
  alert(data.ok ? 'Roadmap seeded.' : data.error);
  refresh();
});
$('event-submit').addEventListener('click', async () => {
  await post('/api/event', { message: $('event-message').value, kind: $('event-kind').value, severity: $('event-severity').value });
  refresh();
});
$('metric-submit').addEventListener('click', async () => {
  await post('/api/metric', { name: $('metric-name').value, value: Number($('metric-value').value), step: $('metric-step').value ? Number($('metric-step').value) : null, run_id: $('metric-run').value, unit: $('metric-unit').value });
  refresh();
});
$('milestone-submit').addEventListener('click', async () => {
  await post('/api/milestone', { milestone_id: $('milestone-id').value, status: $('milestone-status').value, evidence: $('milestone-evidence').value, notes: $('milestone-notes').value, owner: $('milestone-owner').value });
  refresh();
});
async function post(path, payload) {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  if (!res.ok) alert(data.error || 'Request failed');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class ProgressDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Aurelius progress dashboard."""

    server_version = "AureliusProgressDashboard/1.0"

    def _send_json(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = _HTML_TEMPLATE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/api/snapshot":
            self._send_json(
                200,
                build_snapshot(
                    state_path=DEFAULT_STATE_PATH,
                    events_path=DEFAULT_EVENTS_PATH,
                    metrics_path=DEFAULT_METRICS_PATH,
                ),
            )
            return
        if path == "/api/events":
            from src.monitoring.progress_store import read_jsonl

            self._send_json(200, {"events": read_jsonl(DEFAULT_EVENTS_PATH, limit=200)})
            return
        if path == "/api/metrics":
            from src.monitoring.progress_store import read_jsonl

            self._send_json(200, {"metrics": read_jsonl(DEFAULT_METRICS_PATH, limit=1000)})
            return
        self._send_json(404, {"error": f"Unknown path: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/event":
                message = str(payload.get("message", "")).strip()
                if not message:
                    raise ValueError("message is required")
                event = append_event(
                    message,
                    kind=str(payload.get("kind", "general")),
                    severity=str(payload.get("severity", "info")),
                    metadata=payload.get("metadata") or {},
                )
                self._send_json(200, {"ok": True, "event": event})
                return

            if path == "/api/metric":
                name = str(payload.get("name", "")).strip()
                if not name:
                    raise ValueError("name is required")
                value = float(payload["value"])
                metric = append_metric(
                    name,
                    value,
                    step=payload.get("step"),
                    run_id=str(payload.get("run_id", "default")),
                    unit=payload.get("unit"),
                    tags=payload.get("tags") or {},
                )
                self._send_json(200, {"ok": True, "metric": metric})
                return

            if path == "/api/milestone":
                milestone_id = str(payload.get("milestone_id", "")).strip()
                if not milestone_id:
                    raise ValueError("milestone_id is required")
                milestone = update_milestone(
                    milestone_id,
                    status=payload.get("status"),
                    evidence=payload.get("evidence"),
                    notes=payload.get("notes"),
                    owner=payload.get("owner"),
                    blockers=payload.get("blockers"),
                    next_action=payload.get("next_action"),
                )
                self._send_json(200, {"ok": True, "milestone": milestone})
                return

            if path == "/api/import-metrics":
                source_path = str(payload.get("path", "")).strip()
                if not source_path:
                    raise ValueError("path is required")
                result = import_metrics_from_jsonl(
                    source_path,
                    run_id=payload.get("run_id"),
                )
                self._send_json(200, {"ok": True, **result})
                return

            if path == "/api/reset-demo":
                from src.monitoring.progress_store import ensure_state

                ensure_state(DEFAULT_STATE_PATH, overwrite=True)
                append_event(
                    "Seeded the Aurelius progress roadmap.",
                    kind="roadmap",
                    severity="info",
                )
                self._send_json(200, {"ok": True})
                return

            self._send_json(404, {"error": f"Unknown path: {path}"})
        except Exception as exc:  # noqa: BLE001 - HTTP API should return JSON errors.
            logger.exception("Dashboard request failed")
            self._send_json(400, {"error": str(exc)})


def create_dashboard_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ProgressDashboardHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurelius Progress Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=7871, help="Port to listen on.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server = create_dashboard_server(args.host, args.port)
    url = f"http://{args.host}:{args.port}"
    logger.info("Serving Aurelius Progress Dashboard at %s", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


__all__ = [
    "ProgressDashboardHandler",
    "create_dashboard_server",
]
