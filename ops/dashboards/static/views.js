/* Copyright © 2025 Ligandal, Inc.
   SPDX-License-Identifier: Apache-2.0

   Pane renderers. Each view: mount() builds the DOM once, update() fills it
   with real data, tick() drives animation.

   ⛔ A VIEW NEVER FABRICATES A VALUE. Absent data renders as an explicit gap --
   an em dash, a shaded no-data band, or the word STALE -- never as a zero and
   never as the last reading carried forward. Every "n/r", every shaded stretch
   and every UNREACHABLE row below is there because the alternative was a chart
   that looked fine and was wrong. */
'use strict';

const Views = {};
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const card = (title, note) => {
  const c = el('div', 'card');
  const h = el('h2', null, title);
  if (note) h.appendChild(el('span', 'note', note));
  c.appendChild(h);
  return c;
};
const canvasIn = (parent, cls) => {
  const w = el('div', cls || 'fill');
  const cv = el('canvas');
  w.appendChild(cv); parent.appendChild(w);
  return cv;
};
const fmtClock = (t) => new Date(t * 1000).toLocaleTimeString('en-GB', { hour12: false });


/* HTML-escape anything that came from a filesystem, a model name, a hostname or
   a config file. Every interpolation of external text below goes through it. */
function escape_(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ===================== gpu.host =======================================
   Local card telemetry. The per-GPU slot DOM is rebuilt when, and ONLY when,
   the CARD COUNT changes -- a card installed, a driver reset, a card dropping
   out mid-run -- so the dashboard follows the hardware without an edit and
   without a restart. With no NVIDIA GPU present it says "no GPU telemetry",
   which is the truth, rather than drawing an idle 0% card. */
Views['gpu.host'] = {
  mount(root) {
    root.className = 'grid';
    root.style.gridTemplateColumns = '1fr 1fr 1fr';
    root.style.gridTemplateRows = '260px 1fr';
    const c1 = card('GPU');
    c1.style.gridColumn = '1 / 4';
    const gpuRow = el('div');
    gpuRow.style.cssText = 'display:flex;flex-direction:row;gap:16px;flex:1 1 auto;min-height:0';
    c1.appendChild(gpuRow);
    root.appendChild(c1);

    const c3 = card('GPU processes', 'resident on the card right now');
    const t = el('div'); t.style.overflow = 'hidden'; t.style.flex = '1 1 auto';
    c3.appendChild(t);
    c3.style.gridColumn = '1 / 3';
    root.appendChild(c3);

    const c4 = card('Host');
    c4.style.gridColumn = '3';
    const cv4 = canvasIn(c4);
    const kv = el('dl', 'kv');
    c4.appendChild(kv);
    root.appendChild(c4);
    return { gpuRow, cv4, table: t, kv, gpuCount: -1, gpuSlots: [] };
  },
  update(root, p, st) {
    const d = p.data, gpus = d.gpus || [];
    const gh = (d.history && d.history.gpu) || {};

    /* Rebuild the per-GPU slot DOM only when the GPU count actually changes
       (installing/removing a card) -- never on every poll. */
    if (gpus.length !== st.gpuCount) {
      st.gpuRow.innerHTML = '';
      st.gpuSlots = (gpus.length ? gpus : [null]).map((_, i) => {
        const slot = el('div');
        slot.style.cssText = 'display:flex;flex-direction:column;gap:6px;flex:1 1 0;min-width:0;'
          + (i > 0 ? 'border-left:1px solid var(--hairline);padding-left:16px' : '');
        const label = el('div', 'd');
        label.style.cssText = 'font-family:var(--font-mono);font-size:12px;color:var(--text-secondary)';
        const hero = el('div', 'hero');
        hero.innerHTML = '<div><span class="n" data-gpu-util style="font-size:38px">—</span>'
          + '<span class="u">% SM</span></div><div class="d" data-gpu-mem></div>';
        const cv = el('canvas');
        const cvWrap = el('div'); cvWrap.style.cssText = 'flex:1 1 auto;min-height:0;position:relative';
        cvWrap.appendChild(cv);
        slot.appendChild(label); slot.appendChild(hero); slot.appendChild(cvWrap);
        st.gpuRow.appendChild(slot);
        return { slot, label, hero, cv };
      });
      st.gpuCount = gpus.length;
    }

    const colors = DW.SERIES();
    if (!gpus.length) {
      const s = st.gpuSlots[0];
      s.label.textContent = 'no GPU telemetry';
      s.hero.querySelector('[data-gpu-util]').textContent = '—';
      s.hero.querySelector('[data-gpu-mem]').textContent = '';
    } else {
      gpus.forEach((g, i) => {
        const s = st.gpuSlots[i];
        const idx = g.index === undefined ? i : g.index;
        s.label.textContent = `GPU ${idx} · ${g.name || '?'}`
          + (g.temperature_gpu !== undefined ? ` · ${Math.round(g.temperature_gpu)}°C` : '')
          + (g.power_draw !== undefined ? ` · ${Math.round(g.power_draw)}/${Math.round(g.power_limit || 0)}W` : '');
        s.hero.querySelector('[data-gpu-util]').textContent =
          g.utilization_gpu === undefined ? '—' : Math.round(g.utilization_gpu);
        s.hero.querySelector('[data-gpu-mem]').textContent = g.memory_total
          ? `${(g.memory_used / 1024).toFixed(1)} / ${(g.memory_total / 1024).toFixed(0)} GiB`
          : '';
        const h = gh[String(idx)] || {};
        DW.spark(s.cv, h.util, { color: colors[i % colors.length], min: 0, max: 100 });
      });
    }

    const multiGpu = gpus.length > 1;
    const rows = (d.gpu_procs || []).map((pr) => `<tr>
      ${multiGpu ? `<td class="mono">${pr.gpu === null || pr.gpu === undefined ? '—' : pr.gpu}</td>` : ''}
      <td class="mono">${pr.pid}</td>
      <td>${escape_(pr.label || '')}</td>
      <td class="num">${(pr.mem_mib / 1024).toFixed(1)} GiB</td>
      <td class="num">${pr.cpu === undefined ? '—' : pr.cpu.toFixed(0) + '%'}</td>
      <td class="mono" style="color:var(--text-muted)">${escape_((pr.cwd || '').split('/').slice(-2).join('/'))}</td>
    </tr>`).join('');
    st.table.innerHTML = `<table class="dw"><thead><tr>
      ${multiGpu ? '<th class="mono">GPU</th>' : ''}
      <th class="mono">PID</th><th>workload</th><th class="num">VRAM</th>
      <th class="num">CPU</th><th>cwd</th></tr></thead><tbody>${rows}</tbody></table>`;

    DW.spark(st.cv4, d.history.cpu, { color: DW.P.s2, min: 0, max: 100 });
    st.kv.innerHTML = `
      <dt>CPU</dt><dd>${d.cpu_pct.toFixed(0)}% of ${d.cpu_count} thr</dd>
      <dt>load</dt><dd>${d.load.map((x) => x.toFixed(1)).join(' · ')}</dd>
      <dt>RAM</dt><dd>${DW.bytes(d.ram_used)} / ${DW.bytes(d.ram_total)}</dd>
      <dt>net</dt><dd>↓${DW.bytes(d.net_rx)}/s ↑${DW.bytes(d.net_tx)}/s</dd>
      ${(d.disks || []).map((k) => `<dt>${k.mount}</dt><dd>${k.pct.toFixed(0)}% · ${DW.bytes(k.total - k.used)} free</dd>`).join('')}
      <dt>uptime</dt><dd>${DW.dur(d.uptime)}</dd>`;
  },
};

Views['__fallback'] = {
  mount(root) {
    root.className = '';
    const e = el('div', 'empty');
    e.innerHTML = '<div class="big" data-t></div><div data-s></div>';
    root.appendChild(e);
    return {};
  },
  update(root, p) {
    root.querySelector('[data-t]').textContent = p.title;
    root.querySelector('[data-s]').innerHTML =
      escape_(p.subtitle) + `<br><code>no renderer registered for view "${escape_(p.view)}"</code>`;
  },
};

/* ===================== tokens.usage ===================================
   Agent token consumption per model family, contribution-graph style. Source:
   the snapshot written by tools/tokens_export.py from the agent framework's
   session_model_usage ledger.

   INPUT is the UNCACHED prompt (the exporter subtracts cached_tokens); CACHE
   READ is the prefix-cache share; OUTPUT is the completion. An engine that
   never reports cached_tokens shows "n/r", NEVER 0% -- those are different
   claims, and conflating them invents a cache-miss rate nobody measured.
   Prompt and output are drawn on SEPARATE charts: output runs ~2% of prompt
   and vanishes on a shared axis. */
const HT_F = { input: 0, cached: 1, output: 2, calls: 3, cacheRep: 4 };
const htInt = (n) => (n >= 1e4 ? DW.nice(n) : String(Math.round(n || 0)));
const htTot = (v) => (v ? v[0] + v[1] + v[2] : 0);
const htHit = (inp, cached, rep) => (!rep ? 'n/r'
  : (inp + cached ? (100 * cached / (inp + cached)).toFixed(0) + '%' : '—'));

/* ---- live tok/s strip ------------------------------------------------
   Source: the VRAM/KV sampler's dated JSONL ring (see ../../observability.md),
   which records the engine's cumulative realtime_tokens_total{mode=decode}.
   The provider TAILS those files; nothing here or there queries the engine.
     rate   = Δ(decode counter) / Δt: exact mean generated tok/s
     smooth = trailing 5-min mean of that counter
     gauge  = the engine's gen_throughput at the sample instant: speed WHILE decoding
              (aggregate over streams), 0 when idle -- faint dots only
   warm = first decoding interval after an observed backend restart (the first
   request after boot measured ~4.5x slower); kept visible, excluded from the
   y-scale and from peak/p95. */
const HT_WINS = ['15m', '1h', '6h', '24h', '7d'];
const HT_WIN_KEY = 'dw.tokens.tput.win';
const htWinLoad = () => { try { const v = localStorage.getItem(HT_WIN_KEY); return HT_WINS.includes(v) ? v : '1h'; } catch (e) { return '1h'; } };
const htWinSave = (v) => { try { localStorage.setItem(HT_WIN_KEY, v); } catch (e) { /* storage blocked */ } };
const htRate = (v) => (v === null || v === undefined ? '—' : (v >= 100 ? v.toFixed(0) : v.toFixed(1)));
const htPct = (arr, q) => { if (!arr.length) return null; const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))]; };
const htNiceCeil = (v) => {
  if (!(v > 0)) return 10;
  const e = 10 ** Math.floor(Math.log10(v)); const m = v / e;
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((k) => m <= k) * e;
};
const htTimeFmt = (span) => (t) => {
  const dt = new Date(t * 1000);
  if (span > 2 * 86400) return dt.toLocaleString('en-US', { weekday: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
  return dt.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
};

/* One tok/s chart for window w of tput t. x = [now - span, now] always, so an
   empty stretch reads as "no data", never as a compressed axis. */
function htTputChart(cv, t, key, o = {}) {
  const { ctx, w, h } = DW.fit(cv);
  const win = (t.windows || {})[key];
  const S = DW.SERIES();
  const font = getComputedStyle(document.body).fontFamily;
  const pad = { l: 44, r: 8, t: 8, b: o.compact ? 16 : 18 };
  const x1 = t.now, x0 = x1 - (win ? win.span : 3600);
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  if (!win) return;
  const C = Object.fromEntries(win.cols.map((c, i) => [c, i]));
  const pts = win.pts || [];
  const raw = !('peak' in C);
  // y-scale: p98 of every non-warm tok/s value on the chart (gauge included,
  // same unit), so one post-boot or spike sample cannot flatten the rest.
  const vals = [];
  pts.forEach((p) => {
    if (p[C.warm]) return;
    ['rate', raw ? 'smooth' : 'peak', 'gauge'].forEach((c) => { const v = p[C[c]]; if (v !== null && v !== undefined && v > 0) vals.push(v); });
  });
  const yhi = htNiceCeil(Math.max(10, (htPct(vals, 0.98) || 10) * 1.08));
  const Y = (v) => h - pad.b - Math.min(1, Math.max(0, v / yhi)) * (h - pad.t - pad.b);

  // no-data zones: before the sampler history, and before counters existed
  ctx.fillStyle = 'rgba(255,255,255,0.035)';
  const cov0 = Math.max(x0, t.first_tput || x1);
  if (cov0 > x0) ctx.fillRect(pad.l, pad.t, X(cov0) - pad.l, h - pad.t - pad.b);

  // grid + labels
  ctx.font = `${o.compact ? 10 : 11}px ${font}`; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  [0, 0.5, 1].forEach((f) => {
    const y = Y(yhi * f);
    ctx.fillStyle = DW.P.hair; ctx.fillRect(pad.l, Math.round(y), w - pad.l - pad.r, 1);
    ctx.fillStyle = DW.P.muted; ctx.fillText(htRate(yhi * f), pad.l - 6, Math.max(7, y));
  });
  ctx.save(); ctx.translate(9, (h - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
  ctx.fillText('tok/s', 0, 0); ctx.restore();
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  const tf = htTimeFmt(win.span);
  const nT = o.compact ? 4 : 6;
  for (let i = 0; i <= nT; i++) {
    const tv = x0 + (x1 - x0) * i / nT;
    ctx.textAlign = i === 0 ? 'left' : i === nT ? 'right' : 'center';
    ctx.fillText(i === nT ? 'now' : tf(tv), X(tv), h - 3);
  }
  ctx.textAlign = 'left';
  if (cov0 > x0 + (x1 - x0) * 0.08) {
    ctx.fillStyle = DW.P.muted; ctx.textBaseline = 'top';
    ctx.fillText(t.first_tput ? `no counter data before ${tf(t.first_tput)}` : 'no counter data yet', pad.l + 6, pad.t + 4);
  }

  // backend restarts
  (t.resets || []).filter((r) => r >= x0).forEach((r) => {
    const x = Math.round(X(r)) + 0.5;
    ctx.setLineDash([3, 3]); ctx.strokeStyle = S[7]; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke(); ctx.setLineDash([]);
  });
  if (!o.compact) {
    const rs = (t.resets || []).filter((r) => r >= x0);
    if (rs.length) { ctx.fillStyle = S[7]; ctx.textBaseline = 'top'; ctx.fillText(rs.length > 1 ? `${rs.length} restarts` : 'restart', X(rs[rs.length - 1]) + 4, pad.t + 18); }
  }

  const path = (col, color, width, alpha) => {
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
    let on = false; let prevT = null;
    const maxGap = (win.bucket || 20) * 3.5;
    ctx.beginPath();
    pts.forEach((p) => {
      const v = p[C[col]];
      if (v === null || v === undefined || (prevT !== null && p[0] - prevT > maxGap)) { on = false; }
      if (v === null || v === undefined) return;
      const x = X(p[0] + (raw ? 0 : (win.bucket || 0) / 2)), y = Y(v);
      if (on) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      on = true; prevT = p[0];
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  // gauge dots (faint)
  ctx.fillStyle = S[3]; ctx.globalAlpha = 0.45;
  pts.forEach((p) => { const v = p[C.gauge]; if (v > 0) { ctx.beginPath(); ctx.arc(X(p[0] + (raw ? 0 : win.bucket / 2)), Y(v), o.compact ? 1.4 : 2, 0, 6.2832); ctx.fill(); } });
  ctx.globalAlpha = 1;
  if (raw) {
    path('rate', S[0], 1.2, 0.55);
    path('smooth', S[0], 2.6, 1);
  } else {
    path('peak', S[0], 1, 0.4);
    path('rate', S[0], 2.4, 1);
  }
  // warm-up markers: hollow ring on the raw value
  pts.forEach((p) => {
    if (!p[C.warm]) return;
    const v = p[C.rate]; if (v === null || v === undefined) return;
    const x = X(p[0] + (raw ? 0 : win.bucket / 2)), y = Y(v);
    ctx.strokeStyle = S[3]; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 5, 0, 6.2832); ctx.stroke();
    if (!o.compact) { ctx.fillStyle = S[3]; ctx.textBaseline = 'bottom'; ctx.fillText('post-boot warm-up', x + 7, y - 3); }
  });
  // values above the scale
  if (vals.some((v) => v > yhi)) {
    ctx.fillStyle = DW.P.muted; ctx.textBaseline = 'top'; ctx.textAlign = 'right';
    ctx.fillText(`▲ clipped > ${htRate(yhi)}`, w - pad.r, pad.t + 2); ctx.textAlign = 'left';
  }
}

/* Small context strip: running vs queued (steps), or pool %. */
function htStrip(cv, t, key, kind) {
  const { ctx, w, h } = DW.fit(cv);
  const win = (t.windows || {})[key]; if (!win) return;
  const C = Object.fromEntries(win.cols.map((c, i) => [c, i]));
  const S = DW.SERIES();
  const font = getComputedStyle(document.body).fontFamily;
  const pad = { l: 44, r: 8, t: 6, b: 4 };
  const x1 = t.now, x0 = x1 - win.span;
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const pts = win.pts || [];
  const series = kind === 'pool' ? [['usage', S[6]]] : [['run', S[2]], ['queue', S[7]]];
  const hi = kind === 'pool' ? 100 : Math.max(4, ...pts.map((p) => Math.max(p[C.run] || 0, p[C.queue] || 0)));
  const Y = (v) => h - pad.b - Math.min(1, v / hi) * (h - pad.t - pad.b);
  ctx.fillStyle = DW.P.hair; ctx.fillRect(pad.l, h - pad.b, w - pad.l - pad.r, 1); ctx.fillRect(pad.l, pad.t, w - pad.l - pad.r, 1);
  ctx.font = `10px ${font}`; ctx.fillStyle = DW.P.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(kind === 'pool' ? '100%' : String(hi), pad.l - 6, pad.t + 4); ctx.fillText('0', pad.l - 6, h - pad.b - 3);
  ctx.textAlign = 'left';
  const bw = (win.bucket || 20);
  series.forEach(([col, color]) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    let on = false, prevT = null;
    pts.forEach((p) => {
      const v = p[C[col]];
      if (v === null || v === undefined || (prevT !== null && p[0] - prevT > bw * 3.5)) on = false;
      if (v === null || v === undefined) return;
      const xa = X(p[0]), xb = X(p[0] + bw), y = Y(v);
      if (on) ctx.lineTo(xa, y); else ctx.moveTo(xa, y);
      ctx.lineTo(xb, y); on = true; prevT = p[0];
    });
    ctx.stroke();
  });
}

function htRenderTput(st, t) {
  HT_WINS.forEach((k) => {
    const on = k === st.win;
    st.btns[k].style.background = on ? 'var(--series-1)' : 'transparent';
    st.btns[k].style.color = on ? '#fff' : 'var(--text-secondary)';
    st.btns[k].style.borderColor = on ? 'var(--series-1)' : 'var(--hairline)';
  });
  if (!t || !t.ok) {
    st.tpNow.innerHTML = `<span style="color:var(--critical)">NO THROUGHPUT DATA</span> — ${escape_((t && t.reason) || 'provider sent none')}`;
    [st.tpCv, st.stRun.cv, st.stPool.cv, st.fx24.cv, st.fx7.cv].forEach((cv) => DW.fit(cv));
    return;
  }
  const c = t.current || {};
  const w = (t.windows || {})[st.win] || {};
  const ws = w.stats || {};
  const stale = t.age_s > (t.stale_after_s || 90);
  const bucketTxt = (w.bucket || 20) >= 60 ? `${(w.bucket / 60)}-min means` : '20 s samples';
  st.tpNow.innerHTML =
    `<span style="font-size:20px;font-weight:700;color:${stale ? 'var(--critical)' : 'var(--text-primary)'}">${htRate(c.smooth)}</span>`
    + `<span style="color:var(--text-muted)"> tok/s 5-min</span> · `
    + `<b>${htRate(c.rate)}</b> last 20 s · engine ${htRate(c.gauge)} · `
    + `run ${c.running ?? '—'} / queue <b style="color:${c.queued ? 'var(--series-8)' : 'inherit'}">${c.queued ?? '—'}</b> · pool ${c.usage_pct ?? '—'}%<br>`
    + `<span style="color:var(--text-muted)">${st.win}: mean ${htRate(ws.mean)} · busy ${htRate(ws.busy_mean)} (${ws.busy_frac !== null && ws.busy_frac !== undefined ? Math.round(ws.busy_frac * 100) + '%' : '—'}) · `
    + `p95 ${htRate(ws.p95)} · ${DW.nice(ws.tokens)} tok · ${bucketTxt} · sample ${DW.dur(t.age_s)} old${stale ? ' — STALE' : ''}</span>`;
  htTputChart(st.tpCv, t, st.win);
  st.stRun.hd.innerHTML = `RUNNING <span style="color:var(--series-3)">■</span> vs QUEUED <span style="color:var(--series-8)">■</span> <span style="color:var(--text-muted)">· ${st.win}${(w.bucket || 20) >= 60 ? ' · mean / max per bucket' : ''} · a queue = contention</span>`;
  st.stPool.hd.innerHTML = `KV POOL USED % <span style="color:var(--text-muted)">· ${st.win}</span>`;
  htStrip(st.stRun.cv, t, st.win, 'run');
  htStrip(st.stPool.cv, t, st.win, 'pool');
  const s24 = ((t.windows || {})['24h'] || {}).stats || {}, s7 = ((t.windows || {})['7d'] || {}).stats || {};
  st.fx24.hd.innerHTML = `LAST 24 h · 5-min means <span style="color:var(--text-muted)">· mean ${htRate(s24.mean)} · p95 ${htRate(s24.p95)} · ${DW.nice(s24.tokens)} tok</span>`;
  st.fx7.hd.innerHTML = `LAST 7 d · 15-min means <span style="color:var(--text-muted)">· mean ${htRate(s7.mean)} · ${DW.nice(s7.tokens)} tok</span>`;
  htTputChart(st.fx24.cv, t, '24h', { compact: true });
  htTputChart(st.fx7.cv, t, '7d', { compact: true });
  const S = DW.SERIES();
  const sw = (col, a, dash) => `<span style="display:inline-block;width:14px;height:${dash ? 1 : 3}px;background:${col};opacity:${a};vertical-align:middle;margin-right:4px"></span>`;
  st.tpLeg.innerHTML =
    `${sw(S[0], 1)}${(w.bucket || 20) >= 60 ? 'bucket mean' : '5-min mean'} (Δ decode counter / Δt) &nbsp; ${sw(S[0], 0.55, 1)}${(w.bucket || 20) >= 60 ? 'peak 20 s interval' : '20 s interval'} &nbsp; `
    + `<span style="color:${S[3]}">●</span> engine gen_throughput gauge (speed while decoding, 0 when idle) &nbsp; `
    + `<span style="color:${S[3]}">○</span> post-boot warm-up (off-scale) &nbsp; <span style="color:${S[7]}">┆</span> backend restart<br>`
    + `counters recorded since ${t.first_tput ? new Date(t.first_tput * 1000).toLocaleString('en-GB', { hour12: false }) : '—'} · shaded = no data`;
}

Views['tokens.usage'] = {
mount(root) {
root.className = 'grid';
root.style.gridTemplateColumns = '1.15fr 1fr 1fr';
root.style.gridTemplateRows = '142px 372px 1fr';
const hero = card('TODAY · THIS MONTH', 'uncached input · cache read · output');
hero.style.gridColumn = '1 / span 3'; hero.style.gridRow = '1';
const heroBody = el('div'); heroBody.style.cssText = 'display:flex;gap:26px;flex:1 1 auto';
hero.appendChild(heroBody); root.appendChild(hero);

/* live throughput row */
const tp = card('GENERATION THROUGHPUT', 'tok/s from the engine decode-token counter · sampler ring, 20 s cadence');
tp.style.gridColumn = '1 / span 3'; tp.style.gridRow = '2';
const tpBody = el('div'); tpBody.style.cssText = 'display:flex;gap:18px;flex:1 1 auto;min-height:0';
const tpL = el('div'); tpL.style.cssText = 'flex:1.65 1 0;min-width:0;display:flex;flex-direction:column';
const tpBar = el('div'); tpBar.style.cssText = 'display:flex;align-items:center;gap:14px;height:44px;white-space:nowrap;overflow:hidden';
const tpBtns = el('div'); tpBtns.style.cssText = 'display:flex;gap:4px;flex:0 0 auto';
const btns = {};
HT_WINS.forEach((k) => {
  const b = el('button', null, k);
  b.style.cssText = 'font:600 12px var(--font-mono);padding:3px 9px;border-radius:4px;cursor:pointer;'
    + 'border:1px solid var(--hairline);background:transparent;color:var(--text-secondary)';
  tpBtns.appendChild(b); btns[k] = b;
});
const tpNow = el('div'); tpNow.style.cssText = 'font-size:12px;color:var(--text-secondary);font-family:var(--font-mono);overflow:hidden;line-height:1.35;min-width:0';
tpBar.appendChild(tpBtns); tpBar.appendChild(tpNow); tpL.appendChild(tpBar);
const tpCv = el('canvas'); tpCv.style.cssText = 'width:100%;height:194px;display:block';
tpL.appendChild(tpCv);
const tpStrips = el('div'); tpStrips.style.cssText = 'display:flex;gap:14px;margin-top:4px';
const mkStrip = (label) => {
  const box = el('div'); box.style.cssText = 'flex:1 1 0;min-width:0';
  const hd = el('div', null, label); hd.style.cssText = 'font-size:11px;color:var(--text-secondary);letter-spacing:.05em;height:15px;white-space:nowrap;overflow:hidden';
  const cv = el('canvas'); cv.style.cssText = 'width:100%;height:56px;display:block';
  box.appendChild(hd); box.appendChild(cv); tpStrips.appendChild(box); return { hd, cv };
};
const stRun = mkStrip(''), stPool = mkStrip('');
tpL.appendChild(tpStrips);
const tpR = el('div'); tpR.style.cssText = 'flex:1 1 0;min-width:0;display:flex;flex-direction:column;gap:6px';
const mkFixed = (label) => {
  const hd = el('div', null, label); hd.style.cssText = 'font-size:11px;color:var(--text-secondary);letter-spacing:.05em;white-space:nowrap;overflow:hidden;height:15px';
  const cv = el('canvas'); cv.style.cssText = 'width:100%;height:124px;display:block';
  tpR.appendChild(hd); tpR.appendChild(cv); return { hd, cv };
};
const fx24 = mkFixed(''), fx7 = mkFixed('');
const tpLeg = el('div'); tpLeg.style.cssText = 'font-size:10.5px;color:var(--text-muted);line-height:1.35;overflow:hidden';
tpR.appendChild(tpLeg);
tpBody.appendChild(tpL); tpBody.appendChild(tpR); tp.appendChild(tpBody); root.appendChild(tp);

const hm = card('DAILY TOKENS', 'total = uncached input + cache read + output · log colour');
hm.style.gridColumn = '1'; hm.style.gridRow = '3';
const hmCv = canvasIn(hm); root.appendChild(hm);

const pr = card('PROMPT TOKENS / DAY — last 45 d', 'solid = uncached · pale = cache read');
pr.style.gridColumn = '2'; pr.style.gridRow = '3';
const prCv = el('canvas'); prCv.style.cssText = 'width:100%;height:150px;display:block';
pr.appendChild(prCv);
const leg = el('div'); leg.style.cssText = 'font-size:11px;color:var(--text-muted);margin-top:4px;white-space:nowrap;overflow:hidden';
pr.appendChild(leg);
const outCv = el('canvas'); outCv.style.cssText = 'width:100%;height:66px;display:block;margin-top:4px';
const outH = el('div', null, 'OUTPUT TOKENS / DAY'); outH.style.cssText = 'font-size:11px;color:var(--text-secondary);margin-top:6px;letter-spacing:.06em';
pr.appendChild(outH); pr.appendChild(outCv);
root.appendChild(pr);

const tb = card('MONTHLY · BY PROFILE', '');
tb.style.gridColumn = '3'; tb.style.gridRow = '3';
const tbl = el('div'); tbl.style.cssText = 'overflow:hidden;flex:1 1 auto';
tb.appendChild(tbl); root.appendChild(tb);
const st = { heroBody, hmCv, prCv, outCv, leg, tbl, tpNow, tpCv, stRun, stPool, fx24, fx7, tpLeg, btns, win: htWinLoad(), lastTput: null };
HT_WINS.forEach((k) => btns[k].addEventListener('click', () => {
  st.win = k; htWinSave(k);
  if (st.lastTput) htRenderTput(st, st.lastTput);
}));
return st;
},

update(root, p, st) {
const d = p.data || {};
st.lastTput = d.tput || null;
htRenderTput(st, st.lastTput);
if (d.error) {
  st.heroBody.innerHTML = '<div class="empty"><div class="big">NO SNAPSHOT</div>'
    + `<div style="color:var(--text-muted);font-size:12px">${escape_(d.error)}</div></div>`;
  return;
}
const S = DW.SERIES();
/* The example generator stamps synthetic:true. Say so on screen -- a chart of
   fabricated numbers that does not announce itself is the worst artifact in
   this directory. */
st.heroBody.style.outline = d.synthetic ? '2px solid var(--warning)' : '';
st.heroBody.title = d.synthetic ? 'SYNTHETIC EXAMPLE DATA — not a measurement' : '';
const fams = d.families || [];
const col = {}; fams.forEach((f, i) => { col[f.key] = S[i]; });
const cal = d.calendar || [];
const month = (d.today || '').slice(0, 7);
const mrow = (d.months || {})[month] || {};

/* ---- hero tiles: one per family ---- */
st.heroBody.innerHTML = fams.map((f) => {
  const t = (cal[cal.length - 1] || {})[f.key];
  const m = mrow[f.key] || [0, 0, 0, 0];
  const rep = cal.some((c) => c[f.key] && c[f.key][HT_F.cacheRep]);
  const split = (v) => `<span style="color:${col[f.key]}">${DW.nice(v[0])}</span> in · `
    + `${DW.nice(v[1])} cached · ${DW.nice(v[2])} out · ${htInt(v[3])} calls · hit ${htHit(v[0], v[1], rep)}`;
  return '<div style="flex:1 1 0;min-width:0">'
    + `<div style="font-size:13px;color:var(--text-secondary);letter-spacing:.05em">`
    + `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${col[f.key]};margin-right:7px"></span>`
    + `${escape_(f.label.toUpperCase())} <span style="color:var(--text-muted);font-size:11px">${escape_((f.models || []).join(', '))}</span></div>`
    + '<div style="display:flex;gap:34px;margin-top:8px">'
    + `<div class="hero"><div><span class="n">${DW.nice(htTot(t))}</span><span class="u">today</span></div>`
    + `<div class="d" style="font-family:var(--font-mono);font-size:11px;white-space:nowrap">${t ? split(t) : 'no usage today'}</div></div>`
    + `<div class="hero"><div><span class="n" style="color:var(--text-secondary)">${DW.nice(m[0] + m[1] + m[2])}</span><span class="u">${escape_(month)}</span></div>`
    + `<div class="d" style="font-family:var(--font-mono);font-size:11px;white-space:nowrap">${split(m)}</div></div>`
    + '</div></div>';
}).join('')
+ `<div style="flex:0 0 190px;font-size:11px;color:var(--text-muted);line-height:1.5">`
+ (d.synthetic ? '<b style="color:var(--warning)">SYNTHETIC EXAMPLE DATA</b><br>' : '')
+ `snapshot <b style="color:${d.stale ? 'var(--critical)' : 'var(--text-secondary)'}">${DW.dur(d.age_s)}</b> old${d.stale ? ' — STALE' : ''}<br>`
+ `last agent call ${d.last_usage_at ? DW.dur(Date.now() / 1000 - d.last_usage_at) + ' ago' : '—'}<br>`
+ `days in ${escape_(d.tz || 'client local')}; multi-day sessions split by turn<br>`
+ `src: ${(d.sources || []).length} ledger(s)</div>`;

/* ---- GitHub-style heatmaps, one per family, first active week -> today ---- */
{
  const { ctx, w, h } = DW.fit(st.hmCv);
  const weeks = Math.ceil(cal.length / 7);
  const labW = 150, top = 22, gap = 3, dowW = 30, nF = fams.length, botW = 18;
  // Side by side when history is short (big squares), stacked when long.
  const sideStep = Math.min((h - top - botW - 4) / 7, (w / nF - labW - dowW - 16) / weeks);
  const stackStep = Math.min((h - top - 4) / (nF * 7 + 1.2), (w - labW - dowW - 4) / weeks);
  const side = sideStep >= stackStep;
  const step = Math.floor(side ? sideStep : stackStep), cell = step - gap;
  // Short history: widen columns (up to 2.6x) so a day's value fits on one line.
  const stepX = side ? Math.floor(Math.min((w / nF - labW - dowW - 16) / weeks, step * 2.6)) : step;
  const cellW = stepX - gap;
  const font = getComputedStyle(document.body).fontFamily;
  fams.forEach((f, fi) => {
    const x0 = (side ? fi * (w / nF) : 0) + labW + dowW;
    const y0 = top + (side ? 0 : fi * (7 * step + Math.round(step * 1.2)));
    const vals = cal.map((c) => htTot(c[f.key])).filter((v) => v > 0);
    const lo = vals.length ? Math.log10(Math.min(...vals)) : 0;
    const hi = vals.length ? Math.log10(Math.max(...vals)) : 1;
    ctx.font = '12px ' + font; ctx.textBaseline = 'middle';
    ctx.fillStyle = col[f.key]; ctx.fillRect(x0 - labW - dowW, y0 + 3.5 * step - 26, 10, 10);
    ctx.fillStyle = DW.P.sec; ctx.fillText(f.label, x0 - labW - dowW + 16, y0 + 3.5 * step - 21);
    ctx.font = '11px ' + font; ctx.fillStyle = DW.P.muted;
    ctx.fillText(`${vals.length} active days`, x0 - labW - dowW, y0 + 3.5 * step - 2);
    ctx.fillText(vals.length ? `${DW.nice(10 ** lo)}–${DW.nice(10 ** hi)}/day` : 'no usage', x0 - labW - dowW, y0 + 3.5 * step + 14);
    ctx.fillText('log colour', x0 - labW - dowW, y0 + 3.5 * step + 30);
    ['', 'Mon', '', 'Wed', '', 'Fri', ''].forEach((t, r) => t && ctx.fillText(t, x0 - dowW, y0 + r * step + cell / 2));
    let lastM = '';
    if (fi === 0 || side) {
      ctx.textBaseline = 'alphabetic';
      cal.forEach((c, i) => {
        const m = c.d.slice(5, 7);
        if (i % 7 === 0 && m !== lastM) {
          lastM = m;
          ctx.fillText(new Date(c.d + 'T12:00').toLocaleString('en-US', { month: 'short' }), x0 + Math.floor(i / 7) * stepX, top - 8);
        }
      });
    }
    cal.forEach((c, i) => {
      const x = x0 + Math.floor(i / 7) * stepX, y = y0 + (i % 7) * step;
      const v = htTot(c[f.key]);
      const future = c.d > d.today;
      if (future) return;
      ctx.fillStyle = v > 0 ? DW.rampDark(hi > lo ? (Math.log10(v) - lo) / (hi - lo) : 1) : DW.P.hair;
      ctx.fillRect(x, y, cellW, cell);
      if (cellW >= 50 && cell >= 18) {
        ctx.font = '11px ' + font; ctx.textBaseline = 'middle';
        ctx.fillStyle = v > 0 ? 'rgba(0,0,0,.8)' : DW.P.muted;
        ctx.fillText(c.d.slice(8), x + 4, y + cell / 2);
        if (v > 0) { ctx.textAlign = 'right'; ctx.fillText(DW.nice(v), x + cellW - 4, y + cell / 2); ctx.textAlign = 'left'; }
      }
      if (c.d === d.today) { ctx.strokeStyle = DW.P.pri; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, y - 1, cellW + 2, cell + 2); }
    });
    if (side && stepX >= 44) {                   // weekly totals under each column, when they fit
      ctx.font = '10px ' + font; ctx.textBaseline = 'top'; ctx.fillStyle = DW.P.muted;
      for (let wk = 0; wk < weeks; wk++) {
        const t = cal.slice(wk * 7, wk * 7 + 7).reduce((acc, c) => acc + htTot(c[f.key]), 0);
        if (t) ctx.fillText('wk ' + DW.nice(t), x0 + wk * stepX, y0 + 7 * step + 3);
      }
    }
  });
}

/* ---- daily bars: prompt (uncached + cached), then output ---- */
const last = cal.slice(-45);
const bars = (cv, parts) => {
  const { ctx, w, h } = DW.fit(cv);
  const padL = 52, padB = 16, nF = fams.length;
  const max = Math.max(1, ...last.map((c) => Math.max(...fams.map((f) => parts(c[f.key]).reduce((a, b) => a + b, 0)))));
  const bw = (w - padL) / last.length;
  ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
  ctx.fillStyle = DW.P.muted; ctx.textBaseline = 'middle';
  [0, 0.5, 1].forEach((t) => {
    const y = (h - padB) * (1 - t);
    ctx.fillText(DW.nice(max * t), 0, Math.max(6, y));
    ctx.fillStyle = DW.P.hair; ctx.fillRect(padL, y, w - padL, 1); ctx.fillStyle = DW.P.muted;
  });
  last.forEach((c, i) => {
    fams.forEach((f, fi) => {
      const x = padL + i * bw + 1 + fi * (bw - 2) / nF, ww = Math.max(1, (bw - 2) / nF - 1);
      let y = h - padB;
      parts(c[f.key]).forEach((v, pi) => {
        const hh = (h - padB) * v / max;
        ctx.globalAlpha = pi === 0 ? 1 : 0.42;
        ctx.fillStyle = col[f.key]; ctx.fillRect(x, y - hh, ww, hh); y -= hh;
      });
      ctx.globalAlpha = 1;
    });
    if (i % 7 === 0) { ctx.fillStyle = DW.P.muted; ctx.fillText(c.d.slice(5), padL + i * bw, h - 6); }
  });
};
bars(st.prCv, (v) => (v ? [v[0], v[1]] : [0, 0]));
bars(st.outCv, (v) => (v ? [v[2]] : [0]));
st.leg.innerHTML = fams.map((f) => `<span style="margin-right:16px"><span style="display:inline-block;width:9px;height:9px;background:${col[f.key]};margin-right:5px"></span>${escape_(f.label)}</span>`).join('')
  + 'cache read n/r = the engine never reported cached_tokens for that family, not 0%';

/* ---- tables ---- */
const row = (name, v, rep) => `<tr><td>${name}</td><td class="num">${DW.nice(v[0])}</td><td class="num">${DW.nice(v[1])}</td>`
  + `<td class="num">${DW.nice(v[2])}</td><td class="num">${htInt(v[3])}</td><td class="num">${htHit(v[0], v[1], rep)}</td></tr>`;
const head = (c0) => `<thead><tr><th>${c0}</th><th class="num">INPUT (uncached)</th><th class="num">CACHE READ</th>`
  + '<th class="num">OUTPUT</th><th class="num">CALLS</th><th class="num">HIT</th></tr></thead>';
const months = Object.entries(d.months || {}).slice(-3).reverse();
let html = `<table class="dw tight">${head('MONTH · MODEL')}<tbody>`;
months.forEach(([m, fv]) => fams.forEach((f) => {
  const v = fv[f.key]; if (!v || !(v[0] + v[1] + v[2])) return;
  html += row(`${m} <span style="color:${col[f.key]}">${escape_(f.label.replace(/^\S+\s(?=\S)/, ''))}</span>`, v, v[1] > 0);
}));
html += '</tbody></table>';
const profs = Object.entries(d.profiles || {}).map(([pn, fv]) => [pn, fv])
  .sort((a, b) => fams.reduce((s, f) => s + htTot(b[1][f.key]), 0) - fams.reduce((s, f) => s + htTot(a[1][f.key]), 0));
html += `<table class="dw tight" style="margin-top:12px">${head('PROFILE · MODEL (all-time)')}<tbody>`;
profs.forEach(([pn, fv]) => fams.forEach((f) => {
  const v = fv[f.key]; if (!v || !(v[0] + v[1] + v[2])) return;
  html += row(`${escape_(pn)} <span style="color:${col[f.key]}">${escape_(f.label.replace(/^\S+\s(?=\S)/, ''))}</span>`, v, v[1] > 0);
}));
st.tbl.innerHTML = html + '</tbody></table>';
},
};

/* ===================== gpu.fleet =====================================
   The fleet inventory from tools/fleet_probe.py: one row per accelerator the
   probe actually SAW, seconds ago.

   ⛔ THREE STATES, NOT TWO. `live` is a card that answered. `UNREACHABLE` is a
   host that did not answer, and carries the error -- it is NOT a host with no
   GPUs. `no-gpu` is a host whose nvidia-smi answered and listed nothing, which
   is a genuine observation. Collapsing the last two is how an inventory comes
   to describe a one-GPU fleet while most of the hardware is missing from it. */
Views['gpu.fleet'] = {
  mount(root) {
    root.className = 'grid';
    root.style.gridTemplateColumns = '1fr';
    root.style.gridTemplateRows = '96px 1fr';
    const hero = card('FLEET', 'nvidia-smi, every host, this poll');
    const heroBody = el('div');
    heroBody.style.cssText = 'display:flex;gap:30px;flex:1 1 auto;align-items:center';
    hero.appendChild(heroBody); root.appendChild(hero);
    const c = card('ACCELERATORS', 'one row per card the probe saw');
    const tbl = el('div'); tbl.style.cssText = 'overflow:hidden;flex:1 1 auto';
    c.appendChild(tbl); root.appendChild(c);
    return { heroBody, tbl };
  },
  update(root, p, st) {
    const d = p.data || {};
    if (d.error) {
      st.heroBody.innerHTML = '<div class="empty"><div class="big">PROBE FAILED</div>'
        + `<div style="color:var(--text-muted);font-size:12px">${escape_(d.error)}</div></div>`;
      st.tbl.innerHTML = '';
      return;
    }
    const age = d.probed_at ? (Date.now() / 1000 - d.probed_at) : null;
    const tile = (n, u, color) => '<div class="hero"><div>'
      + `<span class="n" ${color ? `style="color:${color}"` : ''}>${n}</span>`
      + `<span class="u">${u}</span></div></div>`;
    st.heroBody.innerHTML =
      tile(d.n_live ?? '—', 'live cards')
      + tile(d.n_unreachable ?? 0, 'unreachable', d.n_unreachable ? 'var(--critical)' : null)
      + tile(d.n_nogpu ?? 0, 'hosts with no CUDA device')
      + tile(d.total_vram_mb ? (d.total_vram_mb / 1024).toFixed(0) : '—', 'GiB VRAM total')
      + `<div style="margin-left:auto;font-size:11px;color:var(--text-muted);line-height:1.6;text-align:right">`
      + `probed ${age === null ? '—' : DW.dur(age) + ' ago'}<br>`
      + `probe took ${d.probe_s ? d.probe_s.toFixed(2) + ' s' : '—'}<br>`
      + `${(d.hosts || []).length} host(s) in GPU_FLEET</div>`;

    const cell = (v, suf) => (v === null || v === undefined ? '—' : v + (suf || ''));
    const rows = (d.accelerators || []).map((a) => {
      const bad = a.status === 'UNREACHABLE';
      const used = a.observed_used_vram_mb, tot = a.total_vram_mb;
      return `<tr${bad ? ' style="color:var(--critical)"' : ''}>
        <td class="mono">${escape_(a.host)}</td>
        <td class="mono">${cell(a.index)}</td>
        <td>${escape_(a.name || '')}</td>
        <td class="mono">${escape_((a.uuid || '').slice(0, 12) || '—')}</td>
        <td class="num">${tot ? (used / 1024).toFixed(1) + ' / ' + (tot / 1024).toFixed(0) + ' GiB' : '—'}</td>
        <td class="num">${cell(a.utilization, '%')}</td>
        <td class="num">${cell(a.temperature_c, '°C')}</td>
        <td class="mono">${escape_(a.status)}</td>
        <td class="mono" style="color:var(--text-muted)">${escape_(a.error || '')}</td>
      </tr>`;
    }).join('');
    st.tbl.innerHTML = `<table class="dw"><thead><tr>
      <th class="mono">HOST</th><th class="mono">IDX</th><th>NAME</th>
      <th class="mono">UUID</th><th class="num">VRAM</th><th class="num">UTIL</th>
      <th class="num">TEMP</th><th class="mono">STATUS</th><th>NOTE</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  },
};
