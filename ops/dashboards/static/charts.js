/* Copyright © 2025 Ligandal, Inc.
   SPDX-License-Identifier: Apache-2.0
   Canvas chart primitives for dashwall. Canvas (not SVG) because the kiosk
   renders in software and these redraw every couple of seconds. */
'use strict';

const DW = (() => {
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const P = {
    get s1() { return css('--series-1'); }, get s2() { return css('--series-2'); },
    get s3() { return css('--series-3'); }, get s4() { return css('--series-4'); },
    get s5() { return css('--series-5'); }, get s6() { return css('--series-6'); },
    get s7() { return css('--series-7'); }, get s8() { return css('--series-8'); },
    get surface() { return css('--surface-1'); },
    get hair() { return css('--hairline'); },
    get muted() { return css('--text-muted'); },
    get sec() { return css('--text-secondary'); },
    get pri() { return css('--text-primary'); },
  };
  const SERIES = () => [P.s1, P.s2, P.s3, P.s4, P.s5, P.s6, P.s7, P.s8];

  /* Size a canvas to its CSS box exactly once per resize. */
  function fit(cv) {
    const w = cv.clientWidth | 0, h = cv.clientHeight | 0;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  const nice = (v) => {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + 'G';
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(0) + 'k';
    if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k';
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    if (a >= 0.001) return v.toFixed(4);
    return v.toExponential(1);
  };
  const bytes = (b) => {
    if (!isFinite(b)) return '—';
    const u = ['B', 'K', 'M', 'G', 'T']; let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return b.toFixed(i ? 1 : 0) + u[i];
  };
  const dur = (s) => {
    if (s === null || s === undefined || !isFinite(s)) return '—';
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + 's';
    if (s < 3600) return Math.floor(s / 60) + 'm' + String(s % 60).padStart(2, '0');
    if (s < 86400) return Math.floor(s / 3600) + 'h' + String(Math.floor(s % 3600 / 60)).padStart(2, '0');
    return Math.floor(s / 86400) + 'd' + String(Math.floor(s % 86400 / 3600)).padStart(2, '0') + 'h';
  };

  /* ---- sparkline: a bare trend, no axes ------------------------------- */
  function spark(cv, values, opts = {}) {
    const { ctx, w, h } = fit(cv);
    const vals = (values || []).filter((v) => typeof v === 'number' && isFinite(v));
    if (vals.length < 2) return;
    const lo = opts.min !== undefined ? opts.min : Math.min(...vals);
    const hi = opts.max !== undefined ? opts.max : Math.max(...vals);
    const span = (hi - lo) || 1;
    const col = opts.color || P.s1;
    const X = (i) => (i / (vals.length - 1)) * (w - 2) + 1;
    const Y = (v) => h - 2 - ((v - lo) / span) * (h - 4);
    if (opts.fill !== false) {
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, col + '4d'); g.addColorStop(1, col + '00');
      ctx.beginPath(); ctx.moveTo(X(0), h);
      vals.forEach((v, i) => ctx.lineTo(X(i), Y(v)));
      ctx.lineTo(X(vals.length - 1), h); ctx.closePath();
      ctx.fillStyle = g; ctx.fill();
    }
    ctx.beginPath();
    vals.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.stroke();
    if (opts.dot !== false) {
      ctx.beginPath();
      ctx.arc(X(vals.length - 1), Y(vals[vals.length - 1]), 2.6, 0, 6.2832);
      ctx.fillStyle = col; ctx.fill();
    }
  }

  /* ---- axes frame shared by line/bar/histogram ------------------------ */
  function frame(ctx, w, h, pad, yTicks, xLabels, opts = {}) {
    ctx.strokeStyle = P.hair; ctx.lineWidth = 1;
    ctx.font = '10px ' + (opts.mono ? 'Fira Code, monospace' : 'Inter, DejaVu Sans, sans-serif');
    ctx.fillStyle = P.muted; ctx.textBaseline = 'middle';
    yTicks.forEach((t) => {
      const y = Math.round(t.y) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(t.label, pad.l - 6, y);
    });
    ctx.textBaseline = 'top'; ctx.textAlign = 'center';
    (xLabels || []).forEach((t) => ctx.fillText(t.label, t.x, h - pad.b + 5));
  }

  /* Axis labels get their own formatter: enough precision to distinguish
     adjacent ticks, never more. `nice` is for values, not axes. */
  function axisFmt(step) {
    const dec = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
    return (v) => {
      if (!isFinite(v)) return '';
      if (Math.abs(v) >= 1e4) return nice(v);
      return v.toFixed(dec);
    };
  }

  function ticks(lo, hi, n = 4) {
    const out = []; const span = hi - lo || 1;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
    out.step = step;
    return out;
  }

  /* ---- multi-series line chart ---------------------------------------- */
  function line(cv, series, opts = {}) {
    const { ctx, w, h } = fit(cv);
    const pad = Object.assign({ l: 46, r: 10, t: 10, b: 20 }, opts.pad);
    const all = series.flatMap((s) => s.points);
    if (all.length < 2) return;
    let xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).filter(isFinite);
    if (!ys.length) return;
    let ylo = opts.ymin !== undefined ? opts.ymin : Math.min(...ys);
    let yhi = opts.ymax !== undefined ? opts.ymax : Math.max(...ys);
    if (opts.log) { ylo = Math.max(ylo, 1e-12); }
    if (ylo === yhi) { yhi = ylo + 1; }
    const xlo = Math.min(...xs), xhi = Math.max(...xs);
    const tf = opts.log ? ((v) => Math.log10(Math.max(v, 1e-12))) : ((v) => v);
    const tlo = tf(ylo), thi = tf(yhi);
    const X = (v) => pad.l + ((v - xlo) / ((xhi - xlo) || 1)) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - ((tf(v) - tlo) / ((thi - tlo) || 1)) * (h - pad.t - pad.b);

    const _yt = ticks(ylo, yhi, 4); const _yf = axisFmt(_yt.step);
    const yt = _yt.map((v) => ({ y: Y(v), label: opts.yfmt ? opts.yfmt(v, _yf) : _yf(v) }));
    const xt = [xlo, (xlo + xhi) / 2, xhi].map((v) => ({ x: X(v), label: opts.xfmt ? opts.xfmt(v) : nice(v) }));
    frame(ctx, w, h, pad, yt, xt, { mono: true });

    series.forEach((s, i) => {
      const pts = s.points.filter((p) => isFinite(p[1]));
      if (pts.length < 2) return;
      ctx.beginPath();
      pts.forEach((p, j) => (j ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
      ctx.strokeStyle = s.color || SERIES()[i % 8];
      ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.stroke();
      const last = pts[pts.length - 1];
      ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 3, 0, 6.2832);
      ctx.fillStyle = s.color || SERIES()[i % 8];
      ctx.strokeStyle = P.surface; ctx.lineWidth = 2;
      ctx.fill(); ctx.stroke();
    });
  }

  /* ---- histogram (counts over a fixed 0..1 domain by default) --------- */
  function histogram(cv, counts, opts = {}) {
    const { ctx, w, h } = fit(cv);
    const pad = Object.assign({ l: 34, r: 8, t: 8, b: 20 }, opts.pad);
    const n = counts.length;
    const hi = Math.max(1, ...counts);
    const _yt = ticks(0, hi, 3); const _yf = axisFmt(_yt.step);
    const yt = _yt.map((v) => ({ y: h - pad.b - (v / hi) * (h - pad.t - pad.b), label: _yf(v) }));
    const bw = (w - pad.l - pad.r) / n;
    const xlabels = (opts.xlabels || []).map((l) => ({ x: pad.l + l.at * (w - pad.l - pad.r), label: l.label }));
    frame(ctx, w, h, pad, yt, xlabels, { mono: true });
    counts.forEach((c, i) => {
      if (!c) return;
      const x = pad.l + i * bw, bh = (c / hi) * (h - pad.t - pad.b);
      const y = h - pad.b - bh;
      /* 2px surface gap between adjacent bars, 4px rounded data-end */
      ctx.fillStyle = opts.colorAt ? opts.colorAt(i, n) : P.s1;
      roundRectTop(ctx, x + 1, y, Math.max(1, bw - 2), bh, Math.min(4, bw / 2));
      ctx.fill();
    });
    if (opts.markers) {
      opts.markers.forEach((m) => {
        const x = Math.round(pad.l + m.at * (w - pad.l - pad.r)) + 0.5;
        ctx.beginPath(); ctx.setLineDash([3, 3]);
        ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b);
        ctx.strokeStyle = m.color || P.sec; ctx.lineWidth = 1; ctx.stroke();
        ctx.setLineDash([]);
        if (m.label) {
          ctx.font = '10px Fira Code, monospace'; ctx.fillStyle = m.color || P.sec;
          ctx.textAlign = 'left'; ctx.textBaseline = 'top';
          ctx.fillText(m.label, x + 3, pad.t + 1);
        }
      });
    }
  }

  /* ---- horizontal bars (categories) ----------------------------------- */
  function hbars(cv, items, opts = {}) {
    const { ctx, w, h } = fit(cv);
    if (!items.length) return;
    const labelW = opts.labelW || 74;
    const valW = opts.valW || 52;
    const rowH = Math.min(opts.maxRow || 30, h / items.length);
    const hi = opts.max !== undefined ? opts.max : Math.max(...items.map((i) => i.value || 0), 1e-9);
    ctx.textBaseline = 'middle';
    items.forEach((it, i) => {
      const y = i * rowH + rowH / 2;
      ctx.font = '12px Inter, DejaVu Sans, sans-serif';
      ctx.fillStyle = P.sec; ctx.textAlign = 'left';
      ctx.fillText(clip(ctx, it.label, labelW - 6), 0, y);
      const x0 = labelW, bwMax = w - labelW - valW;
      const bw = Math.max(2, (Math.max(0, it.value) / hi) * bwMax);
      const bh = Math.max(6, rowH * 0.44);
      ctx.fillStyle = it.color || SERIES()[i % 8];
      roundRectRight(ctx, x0, y - bh / 2, bw, bh, Math.min(4, bh / 2));
      ctx.fill();
      ctx.font = '12px Fira Code, monospace';
      ctx.fillStyle = P.pri; ctx.textAlign = 'right';
      ctx.fillText(it.text !== undefined ? it.text : nice(it.value), w, y);
    });
  }

  function clip(ctx, s, maxw) {
    s = String(s);
    if (ctx.measureText(s).width <= maxw) return s;
    while (s.length > 1 && ctx.measureText(s + '…').width > maxw) s = s.slice(0, -1);
    return s + '…';
  }

  function roundRectTop(ctx, x, y, w, h, r) {
    r = Math.min(r, h);
    ctx.beginPath();
    ctx.moveTo(x, y + h); ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h); ctx.closePath();
  }
  function roundRectRight(ctx, x, y, w, h, r) {
    r = Math.min(r, w);
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x, y + h); ctx.closePath();
  }

  /* ---- stacked horizontal bar (one row per entity) --------------------- */
  function stackRows(cv, rows, opts = {}) {
    const { ctx, w, h } = fit(cv);
    if (!rows.length) return;
    const labelW = opts.labelW || 118;
    const totalW = opts.totalW || 70;
    const rowH = Math.min(opts.maxRow || 62, h / rows.length);
    const hi = Math.max(...rows.map((r) => r.segments.reduce((a, s) => a + s.value, 0)));
    ctx.textBaseline = 'middle';
    rows.forEach((r, ri) => {
      const y = ri * rowH + rowH / 2;
      const bh = Math.max(14, rowH * 0.40);
      ctx.font = '13px Inter, DejaVu Sans, sans-serif';
      ctx.fillStyle = P.sec; ctx.textAlign = 'left';
      ctx.fillText(clip(ctx, r.label, labelW - 8), 0, y - 1);
      if (r.sub) {
        ctx.font = '10px Inter, DejaVu Sans, sans-serif';
        ctx.fillStyle = P.muted;
        ctx.fillText(clip(ctx, r.sub, labelW - 8), 0, y + 13);
      }
      let x = labelW;
      const span = w - labelW - totalW;
      r.segments.forEach((sg) => {
        const bw = (sg.value / hi) * span;
        if (bw < 0.4) return;
        /* 2px surface gap between segments, per the mark spec. */
        const drawW = Math.max(1, bw - 2);
        if (sg.skipped) {
          ctx.save();
          ctx.strokeStyle = sg.color; ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.strokeRect(x + 0.5, y - bh / 2 + 0.5, drawW - 1, bh - 1);
          ctx.restore();
          ctx.globalAlpha = 0.16; ctx.fillStyle = sg.color;
          ctx.fillRect(x, y - bh / 2, drawW, bh);
          ctx.globalAlpha = 1;
        } else {
          ctx.fillStyle = sg.color;
          ctx.fillRect(x, y - bh / 2, drawW, bh);
        }
        if (bw > 46) {
          ctx.font = '10px Fira Code, monospace';
          ctx.fillStyle = sg.skipped ? P.muted : '#ffffff';
          ctx.textAlign = 'center';
          ctx.fillText(sg.text || nice(sg.value), x + drawW / 2, y);
        }
        x += bw;
      });
      ctx.font = '13px Fira Code, monospace';
      ctx.fillStyle = P.pri; ctx.textAlign = 'right';
      ctx.fillText(r.total, w, y);
    });
  }

  /* ---- dot + confidence interval, against a reference line ------------- */
  function dotCI(cv, items, opts = {}) {
    const { ctx, w, h } = fit(cv);
    if (!items.length) return;
    const labelW = opts.labelW || 66;
    const pad = { l: labelW, r: 58, t: 26, b: 22 };
    const los = items.map((i) => (i.ci ? i.ci[0] : i.value));
    const his = items.map((i) => (i.ci ? i.ci[1] : i.value));
    let lo = Math.min(...los), hi = Math.max(...his);
    if (opts.ref !== undefined) { lo = Math.min(lo, opts.ref); hi = Math.max(hi, opts.ref); }
    const m = (hi - lo) * 0.12 || 0.05; lo -= m; hi += m;
    const X = (v) => pad.l + ((v - lo) / (hi - lo)) * (w - pad.l - pad.r);
    const rowH = (h - pad.t - pad.b) / items.length;

    if (opts.ref !== undefined) {
      const x = Math.round(X(opts.ref)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t - 8); ctx.lineTo(x, h - pad.b);
      ctx.strokeStyle = P.sec; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.font = '10px Fira Code, monospace'; ctx.fillStyle = P.sec;
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(opts.refLabel || String(opts.ref), x, pad.t - 9);
    }
    ctx.textBaseline = 'middle';
    items.forEach((it, i) => {
      const y = pad.t + rowH * (i + 0.5);
      ctx.font = '12px Inter, DejaVu Sans, sans-serif';
      ctx.fillStyle = P.sec; ctx.textAlign = 'left';
      ctx.fillText(it.label, 0, y);
      if (it.ci) {
        ctx.beginPath(); ctx.moveTo(X(it.ci[0]), y); ctx.lineTo(X(it.ci[1]), y);
        ctx.strokeStyle = it.color || P.s1; ctx.lineWidth = 2; ctx.lineCap = 'round';
        ctx.globalAlpha = 0.45; ctx.stroke(); ctx.globalAlpha = 1;
        [it.ci[0], it.ci[1]].forEach((v) => {
          ctx.beginPath(); ctx.moveTo(X(v), y - 4); ctx.lineTo(X(v), y + 4);
          ctx.strokeStyle = it.color || P.s1; ctx.lineWidth = 2; ctx.stroke();
        });
      }
      ctx.beginPath(); ctx.arc(X(it.value), y, 5, 0, 6.2832);
      ctx.fillStyle = it.color || P.s1;
      ctx.strokeStyle = P.surface; ctx.lineWidth = 2;
      ctx.fill(); ctx.stroke();
      ctx.font = '12px Fira Code, monospace';
      ctx.fillStyle = P.pri; ctx.textAlign = 'right';
      ctx.fillText(it.value.toFixed(3), w, y);
    });
  }

  /* ---- box / quantile strip -------------------------------------------
     A NUMERIC SPREAD, not a count. Each row is one population drawn on a
     SHARED domain so rows are comparable: whisker p10..p90, box q1..q3,
     median rule, and the true min/max as open ticks so a long tail is never
     hidden by the whisker clip. The numbers are printed beside the geometry
     because a box read from across the room is a shape, not a measurement. */
  function box(cv, items, opts = {}) {
    const { ctx, w, h } = fit(cv);
    if (!items || !items.length) return;
    const labelW = opts.labelW || 92;
    const pad = Object.assign({ l: labelW, r: opts.textW || 118, t: 8, b: 22 }, opts.pad);
    const all = items.filter((it) => it.q);
    if (!all.length) return;
    const lo = opts.min !== undefined ? opts.min
      : Math.min(...all.map((it) => it.q.min));
    const hi = opts.max !== undefined ? opts.max
      : Math.max(...all.map((it) => it.q.max));
    const span = (hi - lo) || 1;
    const X = (v) => pad.l + ((v - lo) / span) * (w - pad.l - pad.r);
    const rowH = (h - pad.t - pad.b) / items.length;

    /* domain grid + axis, drawn under the boxes */
    const xt = ticks(lo, hi, 4);
    /* One more decimal than axisFmt's default: on a 0..1 domain its 0.25 step
       rounds the 0.25 tick to "0.3", which mislabels where the box actually is. */
    const xdec = xt.step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(xt.step)) + 1);
    const xf = (v) => (isFinite(v) ? v.toFixed(xdec) : '');
    ctx.strokeStyle = P.hair; ctx.lineWidth = 1;
    ctx.font = '10px Fira Code, monospace'; ctx.fillStyle = P.muted;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    xt.forEach((v) => {
      const x = Math.round(X(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.fillText(xf(v), x, h - pad.b + 5);
    });
    (opts.markers || []).forEach((m) => {
      const x = Math.round(X(m.at)) + 0.5;
      ctx.beginPath(); ctx.setLineDash([3, 3]);
      ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b);
      ctx.strokeStyle = m.color || P.sec; ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([]);
      if (m.label) {
        ctx.fillStyle = m.color || P.sec; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(m.label, x + 3, pad.t);
      }
    });

    items.forEach((it, i) => {
      const y = pad.t + rowH * (i + 0.5);
      const col = it.color || SERIES()[i % 8];
      ctx.textBaseline = 'middle';
      ctx.font = (it.strong ? '600 ' : '') + '12px Inter, DejaVu Sans, sans-serif';
      ctx.fillStyle = it.strong ? P.pri : P.sec; ctx.textAlign = 'left';
      ctx.fillText(clip(ctx, it.label, labelW - 8), 0, y);
      const q = it.q;
      if (!q) {
        ctx.font = '11px Fira Code, monospace'; ctx.fillStyle = P.muted;
        ctx.fillText('no values', pad.l, y);
        return;
      }
      const bh = Math.max(9, Math.min(opts.boxH || 20, rowH * 0.46));
      /* whisker p10..p90 */
      ctx.beginPath(); ctx.moveTo(X(q.p10), y); ctx.lineTo(X(q.q1), y);
      ctx.moveTo(X(q.q3), y); ctx.lineTo(X(q.p90), y);
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.6;
      ctx.stroke(); ctx.globalAlpha = 1;
      [q.p10, q.p90].forEach((v) => {
        ctx.beginPath(); ctx.moveTo(Math.round(X(v)) + 0.5, y - bh * 0.34);
        ctx.lineTo(Math.round(X(v)) + 0.5, y + bh * 0.34);
        ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.6;
        ctx.stroke(); ctx.globalAlpha = 1;
      });
      /* min/max as open ticks -- the tail the whisker clipped */
      [q.min, q.max].forEach((v) => {
        ctx.beginPath(); ctx.arc(X(v), y, 2.2, 0, 6.2832);
        ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.globalAlpha = 0.75;
        ctx.stroke(); ctx.globalAlpha = 1;
      });
      /* box q1..q3 */
      const x1 = X(q.q1), x3 = X(q.q3);
      ctx.fillStyle = col; ctx.globalAlpha = 0.30;
      ctx.fillRect(x1, y - bh / 2, Math.max(1.5, x3 - x1), bh);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(x1) + 0.5, Math.round(y - bh / 2) + 0.5,
                     Math.max(1, Math.round(x3 - x1)), Math.round(bh));
      /* median */
      const xm = Math.round(X(q.median)) + 0.5;
      ctx.beginPath(); ctx.moveTo(xm, y - bh / 2 - 1); ctx.lineTo(xm, y + bh / 2 + 1);
      ctx.strokeStyle = P.pri; ctx.lineWidth = 2; ctx.stroke();
      /* the measurement, in text */
      ctx.font = '11px Fira Code, monospace';
      ctx.fillStyle = P.sec; ctx.textAlign = 'right';
      ctx.fillText(it.text !== undefined ? it.text
        : `${q.median.toFixed(3)}  n=${q.n}`, w, y);
    });
  }

  /* ---- heatmap (rows x cols) ------------------------------------------- */
  function heatmap(cv, mat, opts = {}) {
    const { ctx, w, h } = fit(cv);
    if (!mat || !mat.length) return;
    const rows = mat.length, cols = mat[0].length;
    const pad = Object.assign({ l: 34, r: 8, t: 8, b: 18 }, opts.pad);
    const pw = (w - pad.l - pad.r) / cols;
    const ph = (h - pad.t - pad.b) / rows;
    let lo = opts.min, hi = opts.max;
    if (lo === undefined || hi === undefined) {
      /* Full min..max washes out when the distribution is narrow with a few
         outliers -- clip to percentiles so the structure is actually visible.
         The real range is reported in the legend so nothing is hidden. */
      const flat = [];
      for (const r of mat) for (const v of r) if (isFinite(v)) flat.push(v);
      flat.sort((a, b) => a - b);
      const q = opts.clip === undefined ? 0.02 : opts.clip;
      lo = flat[Math.floor(flat.length * q)];
      hi = flat[Math.min(flat.length - 1, Math.floor(flat.length * (1 - q)))];
      if (!(hi > lo)) { lo = flat[0]; hi = flat[flat.length - 1]; }
      opts._trueLo = flat[0]; opts._trueHi = flat[flat.length - 1];
    }
    const span = (hi - lo) || 1;
    const col = opts.color || ((t) => rampDark(t));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        ctx.fillStyle = col((mat[i][j] - lo) / span);
        ctx.fillRect(pad.l + j * pw, pad.t + i * ph, Math.max(1, pw + 0.5), Math.max(1, ph + 0.5));
      }
    }
    ctx.font = '10px Fira Code, monospace';
    ctx.fillStyle = P.muted;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(opts.ylabel || String(rows), pad.l - 5, pad.t + ph / 2);
    ctx.fillText('1', pad.l - 5, pad.t + (rows - 0.5) * ph);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('1', pad.l, h - pad.b + 4);
    ctx.fillText(String(cols), w - pad.r, h - pad.b + 4);
    if (opts.xlabel) {
      ctx.fillStyle = P.sec;
      ctx.fillText(opts.xlabel, (pad.l + w - pad.r) / 2, h - pad.b + 4);
    }
    return { lo, hi, trueLo: opts._trueLo, trueHi: opts._trueHi };
  }

  /* ---- scatter --------------------------------------------------------- */
  function scatter(cv, pts, opts = {}) {
    const { ctx, w, h } = fit(cv);
    const pad = Object.assign({ l: 40, r: 10, t: 10, b: 20 }, opts.pad);
    if (!pts.length) return;
    const xlo = opts.xmin !== undefined ? opts.xmin : Math.min(...pts.map((p) => p.x));
    const xhi = opts.xmax !== undefined ? opts.xmax : Math.max(...pts.map((p) => p.x));
    const ylo = opts.ymin !== undefined ? opts.ymin : Math.min(...pts.map((p) => p.y));
    const yhi = opts.ymax !== undefined ? opts.ymax : Math.max(...pts.map((p) => p.y));
    const X = (v) => pad.l + ((v - xlo) / ((xhi - xlo) || 1)) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - ((v - ylo) / ((yhi - ylo) || 1)) * (h - pad.t - pad.b);
    const _yt = ticks(ylo, yhi, 3); const _yf = axisFmt(_yt.step);
    const yt = _yt.map((v) => ({ y: Y(v), label: _yf(v) }));
    const xt = ticks(xlo, xhi, 3).map((v) => ({ x: X(v), label: opts.xfmt ? opts.xfmt(v) : nice(v) }));
    frame(ctx, w, h, pad, yt, xt, { mono: true });
    pts.forEach((p) => {
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 4, 0, 6.2832);
      ctx.fillStyle = p.color || P.s1;
      ctx.globalAlpha = p.alpha === undefined ? 0.85 : p.alpha;
      ctx.fill(); ctx.globalAlpha = 1;
      if (p.ring) { ctx.strokeStyle = P.surface; ctx.lineWidth = 2; ctx.stroke(); }
    });
  }

  /* ---- ordered colour ramps ------------------------------------------- */
  const BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  function ramp(t) {  /* 0..1 -> blue sequential, light->dark (light surfaces) */
    t = Math.max(0, Math.min(1, t));
    const i = Math.min(BLUE.length - 1, Math.floor(t * BLUE.length));
    return BLUE[i];
  }
  /* The wall is dark, so the step nearest the surface is the DARKEST one and
     must mean "near zero" -- the light->dark direction is inverted here or the
     strongest values render as near-black and vanish. Floor is step 550 so even
     t=0 clears the surface. */
  const BLUE_ON_DARK = ['#1c5cab', '#256abf', '#2a78d6', '#3987e5',
                        '#5598e7', '#86b6ef', '#b7d3f6'];
  function rampDark(t) {
    t = Math.max(0, Math.min(1, t));
    const i = Math.min(BLUE_ON_DARK.length - 1, Math.floor(t * BLUE_ON_DARK.length));
    return BLUE_ON_DARK[i];
  }
  /* AlphaFold pLDDT confidence bands -- the domain standard, on purpose. */
  function plddtColor(v) {
    if (v >= 90) return css('--plddt-vhigh');
    if (v >= 70) return css('--plddt-high');
    if (v >= 50) return css('--plddt-low');
    return css('--plddt-vlow');
  }
  /* Diverging blue<->red about a neutral midpoint, for error-type layers. */
  function diverge(t) {
    t = Math.max(0, Math.min(1, t));
    const cool = ['#0d366b', '#256abf', '#3987e5', '#86b6ef'];
    const warm = ['#f0b0af', '#e66767', '#cf3b3a', '#8f1f1e'];
    if (t < 0.5) return cool[Math.min(3, Math.floor(t * 2 * 4))];
    return warm[Math.min(3, Math.floor((t - 0.5) * 2 * 4))];
  }

  return { P, SERIES, fit, spark, line, histogram, hbars, scatter, nice, bytes,
           dur, ramp, rampDark, plddtColor, diverge, clip, ticks, axisFmt, stackRows, dotCI,
           box,
           heatmap };
})();
