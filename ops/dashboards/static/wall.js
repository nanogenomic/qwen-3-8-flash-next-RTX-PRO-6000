/* Copyright © 2025 Ligandal, Inc.
   SPDX-License-Identifier: Apache-2.0
   Wall shell: SSE transport, pane lifecycle, ribbon, animation loop.
   Draws only what /api/state and /events send. A missing field renders as an
   em dash; nothing here invents or carries forward a value. */
'use strict';

(() => {
  const wall = document.getElementById('wall');
  const panes = [];
  let lastState = null;
  let connected = false;
  let lastMsg = 0;

  function buildPane(idx, cfg) {
    const pane = document.createElement('div');
    pane.className = 'pane';

    const ribbon = document.createElement('div');
    ribbon.className = 'ribbon';
    ribbon.innerHTML = `
      <div class="brand"><b>DASHWALL</b><span data-host>—</span>
        <span class="mon">${cfg ? cfg.label : ''}</span></div>
      <div class="metric"><span class="k">GPU SM</span><span class="v" data-r-util>—</span></div>
      <canvas class="spark" data-r-sputil></canvas>
      <div class="metric"><span class="k">VRAM</span><span class="v" data-r-mem>—</span></div>
      <canvas class="spark" data-r-spmem></canvas>
      <div class="metric"><span class="k">Temp</span><span class="v" data-r-temp>—</span></div>
      <div class="metric"><span class="k">Power</span><span class="v" data-r-pow>—</span></div>
      <div class="metric"><span class="k">CPU</span><span class="v" data-r-cpu>—</span></div>
      <canvas class="spark" data-r-spcpu></canvas>
      <div class="metric"><span class="k">RAM</span><span class="v" data-r-ram>—</span></div>
      <div class="metric"><span class="k">Net</span><span class="v" data-r-net>—</span></div>
      <div class="spacer"></div>
      <div class="metric"><span class="k">Link</span><span class="v" data-r-link>…</span></div>
      <div class="clock"><span data-r-clock>--:--:--</span><small data-r-date></small></div>`;
    pane.appendChild(ribbon);

    const header = document.createElement('div');
    header.className = 'panel-header';
    header.innerHTML = `<h1 data-h-title>—</h1><span class="sub" data-h-sub></span>
      <span class="spacer"></span><span class="pill idle" data-h-pill><i class="dot"></i>—</span>`;
    pane.appendChild(header);

    const body = document.createElement('div');
    body.className = 'panel-body';
    pane.appendChild(body);

    const footer = document.createElement('div');
    footer.className = 'panel-footer';
    footer.innerHTML = `<span data-f-chips style="display:flex;gap:7px"></span>
      <span class="spacer"></span><span data-f-rot></span>
      <span class="rotbar"><i data-f-bar style="width:0%"></i></span>`;
    pane.appendChild(footer);

    const toast = document.createElement('div');
    toast.className = 'toast';
    pane.appendChild(toast);

    /* Screen picker: every view the rotating pane can show. */
    const picker = document.createElement('div');
    picker.className = 'picker';
    pane.appendChild(picker);

    wall.appendChild(pane);
    return { idx, cfg, pane, ribbon, header, body, footer, toast, picker,
             viewKey: null, view: null, state: null, root: null };
  }

  function ensureView(p, panel) {
    const key = panel.view;
    if (p.viewKey === key) return;
    p.body.innerHTML = '';
    const root = document.createElement('div');
    root.style.height = '100%';
    p.body.appendChild(root);
    const view = Views[key] || Views['__fallback'];
    p.root = root;
    p.view = view;
    p.state = view.mount ? view.mount(root) : {};
    p.viewKey = key;
  }

  function paintRibbon(p, r) {
    const q = (s) => p.ribbon.querySelector(s);
    const num = (v, d, suf) => (typeof v === 'number' && isFinite(v)
      ? v.toFixed(d) + (suf || '') : '—');
    q('[data-host]').textContent = r.host || '';
    q('[data-r-util]').innerHTML = num(r.gpu_util, 0) + '<small>%</small>';
    q('[data-r-mem]').innerHTML = r.gpu_mem_total
      ? (r.gpu_mem_used / 1024).toFixed(1) + '<small>/' + (r.gpu_mem_total / 1024).toFixed(0) + ' GiB</small>'
      : '—';
    q('[data-r-temp]').innerHTML = num(r.gpu_temp, 0) + '<small>°C</small>';
    q('[data-r-pow]').innerHTML = num(r.gpu_power, 0) + '<small>/' + num(r.gpu_power_limit, 0) + ' W</small>';
    q('[data-r-cpu]').innerHTML = num(r.cpu_pct, 0) + '<small>%</small>';
    q('[data-r-ram]').innerHTML = r.ram_total
      ? DW.bytes(r.ram_used) + '<small>/' + DW.bytes(r.ram_total) + '</small>' : '—';
    q('[data-r-net]').innerHTML = '↓' + DW.bytes(r.net_rx || 0) + '<small> ↑' + DW.bytes(r.net_tx || 0) + '</small>';
    const h = r.history || {};
    DW.spark(q('[data-r-sputil]'), h.gpu_util, { color: DW.P.s1, min: 0, max: 100 });
    DW.spark(q('[data-r-spmem]'), h.gpu_mem, { color: DW.P.s3, min: 0, max: 100 });
    DW.spark(q('[data-r-spcpu]'), h.cpu, { color: DW.P.s2, min: 0, max: 100 });
  }

  function paintClock(p) {
    const now = new Date();
    p.ribbon.querySelector('[data-r-clock]').textContent =
      now.toLocaleTimeString('en-GB', { hour12: false });
    p.ribbon.querySelector('[data-r-date]').textContent =
      now.toLocaleDateString('en-GB', { weekday: 'short', day: '2-digit', month: 'short' });
    const link = p.ribbon.querySelector('[data-r-link]');
    const age = (performance.now() - lastMsg) / 1000;
    if (connected && age < 12) { link.textContent = 'live'; link.style.color = 'var(--good)'; }
    else if (connected) { link.textContent = age.toFixed(0) + 's'; link.style.color = 'var(--warning)'; }
    else { link.textContent = 'down'; link.style.color = 'var(--critical)'; }
  }

  let lastActionN = -1;
  let titledHost = null;
  function apply(state) {
    lastState = state;
    // index.html ships a static <title>, and several instances can run this
    // same bundle, so name the document after the host that is actually
    // answering, taken from the measured ribbon.
    const rh = (state.ribbon || {}).host;
    const wn = ((state.panes || [])[0] || {}).pane;
    if (rh && rh !== titledHost) {
      titledHost = rh;
      document.title = 'dashwall — ' + rh + (wn && wn.label ? ' · ' + wn.label : '');
    }
    const la = state.last_action;
    if (la && la.n !== lastActionN) {
      const first = lastActionN === -1;
      lastActionN = la.n;
      if (!first && la.label) {
        panes.forEach((p) => {
          p.toast.textContent = la.label;
          p.toast.classList.remove('show');
          void p.toast.offsetWidth;   /* restart the animation */
          p.toast.classList.add('show');
        });
      }
    }
    const assigned = state.panes || [];
    while (panes.length < assigned.length) {
      panes.push(buildPane(panes.length, (assigned[panes.length] || {}).pane));
    }
    panes.forEach((p, i) => {
      const a = assigned[i];
      if (!a) return;
      const panel = a.panel;
      ensureView(p, panel);
      p.header.querySelector('[data-h-title]').textContent = panel.title;
      p.header.querySelector('[data-h-sub]').textContent = panel.subtitle || '';
      const pill = p.header.querySelector('[data-h-pill]');
      pill.className = 'pill ' + (panel.state || 'idle');
      pill.innerHTML = '<i class="dot"></i>' + (panel.state || '');
      paintRibbon(p, state.ribbon || {});

      const chips = (state.groups || []).map((g) =>
        `<span class="chip ${g.group === state.primary_group ? 'on' : ''}">${g.group}
         <span style="opacity:.6">${g.panels.length}</span></span>`).join('');
      p.footer.querySelector('[data-f-chips]').innerHTML = chips;
      const secs = Math.ceil(a.rotate_in || 0);
      p.footer.querySelector('[data-f-rot]').textContent = state.frozen && a.rotating
        ? 'HELD — ← → step · ↑ ↓ all screens · Esc resumes rotation'
        : state.frozen
        ? 'rotation held · Esc resumes'
        : (a.rotating ? `next view in ${secs}s`
                      : (i === 0 ? 'pinned to the highest-priority panel' : 'only view'));
      const frac = a.rotate_seconds ? (1 - a.rotate_in / a.rotate_seconds) : 0;
      p.footer.querySelector('[data-f-bar]').style.width =
        (a.rotating && !state.frozen ? Math.max(0, Math.min(1, frac)) * 100 : 0) + '%';

      const pk = state.picker || {};
      const showPk = pk.open && a.rotating;
      p.picker.classList.toggle('open', !!showPk);
      if (showPk) {
        let lastG = null;
        p.picker.innerHTML = '<div class="pk-h">ALL SCREENS <span>'
          + (pk.items || []).length + '</span></div>'
          + (pk.left ? `<div class="pk-left">pinned pane: ${escape_(pk.left)}</div>` : '')
          + (pk.items || []).map((it, j) => {
            const g = it.group !== lastG ? `<div class="pk-g">${escape_(it.group)}</div>` : '';
            lastG = it.group;
            return g + `<div class="pk-i${j === pk.index ? ' on' : ''}">`
              + `<i class="dot ${escape_(it.state || '')}"></i>${escape_(it.title)}</div>`;
          }).join('');
        const on = p.picker.querySelector('.pk-i.on');
        if (on) on.scrollIntoView({ block: 'nearest' });
      }

      try {
        if (p.view.update) p.view.update(p.root, panel, p.state, state);
      } catch (err) {
        console.error('view update failed', panel.view, err);
      }
      p.lastPanel = panel;
    });
  }

  /* ---- animation loop: views that animate get a tick() ---------------- */
  let prev = performance.now();
  function frame(now) {
    const dt = Math.min(60, now - prev); prev = now;
    panes.forEach((p) => {
      paintClock(p);
      if (p.view && p.view.tick && p.lastPanel) {
        try { p.view.tick(p.root, p.state, dt); } catch (e) { /* keep the wall up */ }
      }
    });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  /* ---- transport ------------------------------------------------------ */
  function connect() {
    const es = new EventSource('/events');
    es.onopen = () => { connected = true; };
    es.onmessage = (ev) => {
      lastMsg = performance.now();
      connected = true;
      try { apply(JSON.parse(ev.data)); }
      catch (e) { console.error('bad frame', e); }
    };
    es.onerror = () => {
      connected = false;
      es.close();
      setTimeout(connect, 2500);
    };
  }
  fetch('/api/state').then((r) => r.json()).then((s) => { lastMsg = performance.now(); connected = true; apply(s); })
    .catch(() => {}).finally(connect);
})();
