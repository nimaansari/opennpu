// Small, dependency-free SVG charts for the dashboard (works offline). Each chart: hairline
// grid, one y axis, thin marks, a hover tooltip, and a "Table" toggle with the same data.
"use strict";
const NS = "http://www.w3.org/2000/svg";
const tip = (() => {
  let el;
  return {
    show(html, x, y) {
      if (!el) { el = document.createElement("div"); el.className = "chart-tip"; el.setAttribute("role", "status"); document.body.append(el); }
      el.innerHTML = html; el.hidden = false;
      const r = el.getBoundingClientRect();
      el.style.left = Math.min(innerWidth - r.width - 8, Math.max(8, x - r.width / 2)) + "px";
      el.style.top = Math.max(8, y - r.height - 12) + "px";
    },
    hide() { if (el) el.hidden = true; },
  };
})();
function svgEl(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.append(e);
  return e;
}
function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v)), n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
}
const compact = (v) => (v >= 1e9 ? (v / 1e9).toFixed(1) + "B" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e4 ? Math.round(v / 1e3) + "k"
  : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : Number.isInteger(v) ? String(v) : v.toFixed(v < 10 ? 1 : 0));

// A card: title, optional note, body, and a Table toggle that swaps the chart for its data.
function chartCard(root, title, opts = {}) {
  const card = document.createElement("section");
  card.className = "chart-card" + (opts.wide ? " wide" : "");
  card.innerHTML = `<header><h3>${esc(title)}</h3>${opts.note ? `<span class="fine">${esc(opts.note)}</span>` : ""}
    <button class="link-btn" type="button" aria-pressed="false">Table</button></header><div class="chart-body"></div><div class="chart-table" tabindex="0" role="region" aria-label="Chart data as a table" hidden></div>`;
  root.append(card);
  const btn = card.querySelector(".link-btn"), body = card.querySelector(".chart-body"), table = card.querySelector(".chart-table");
  btn.onclick = () => {
    const on = btn.getAttribute("aria-pressed") !== "true";
    btn.setAttribute("aria-pressed", String(on)); btn.textContent = on ? "Chart" : "Table";
    body.hidden = on; table.hidden = !on;
  };
  return { card, body, table };
}
function fillTable(el, head, rows) {
  el.innerHTML = `<table class="data"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr>${r.map((c) => `<td>${esc(String(c ?? "-"))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function empty(el, text) { el.innerHTML = `<p class="chart-empty muted small">${esc(text)}</p>`; }

// Columns over a category or time axis. data: [{label, value, tip}]
function columns(el, data, { fmt = compact, xEvery = 1, height = 180, label = "" } = {}) {
  el.innerHTML = "";
  if (!data.length || data.every((d) => !d.value)) return empty(el, "Nothing in this range yet.");
  const W = Math.max(el.clientWidth || 600, 280), H = height, L = 40, B = 22, T = 8;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label }, el);
  const max = niceMax(Math.max(...data.map((d) => d.value)));
  const plotW = W - L - 6, band = plotW / data.length, bw = Math.max(2, Math.min(24, band - 2));
  for (const f of [0, 0.5, 1]) {
    const y = T + (H - T - B) * (1 - f);
    svgEl("line", { x1: L, x2: W - 6, y1: y, y2: y, class: "grid" }, svg);
    svgEl("text", { x: L - 6, y: y + 4, class: "axis", "text-anchor": "end" }, svg).textContent = fmt(max * f);
  }
  data.forEach((d, i) => {
    const h = (H - T - B) * (d.value / max), x = L + i * band + (band - bw) / 2, y = H - B - h;
    if (d.value > 0) {
      const r = Math.min(4, bw / 2, h);
      svgEl("path", { class: "bar", d: `M${x},${H - B}V${y + r}a${r},${r} 0 0 1 ${r},${-r}H${x + bw - r}a${r},${r} 0 0 1 ${r},${r}V${H - B}Z` }, svg);
    }
    if (i % xEvery === 0) svgEl("text", { x: x + bw / 2, y: H - 6, class: "axis", "text-anchor": "middle" }, svg).textContent = d.label;
    const hit = svgEl("rect", { x: L + i * band, y: T, width: band, height: H - T - B, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => tip.show(d.tip || `<b>${esc(d.label)}</b><br>${fmt(d.value)}`, e.clientX, e.clientY));
    hit.addEventListener("mouseleave", tip.hide);
  });
}

// One line over time with a 10% area wash, crosshair and tooltip. points: [{t, v}]
function line(el, points, { fmt = (v) => v.toFixed(0), unit = "", min = 0, max = null, xfmt, height = 150, label = "" } = {}) {
  el.innerHTML = "";
  const pts = points.filter((p) => p.v != null);
  if (pts.length < 2) return empty(el, "Not enough history yet: the sampler stores one point per minute.");
  const W = Math.max(el.clientWidth || 400, 240), H = height, L = 40, B = 20, T = 8;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label }, el);
  const t0 = pts[0].t, t1 = pts[pts.length - 1].t, hi = max ?? niceMax(Math.max(...pts.map((p) => p.v)) * 1.05);
  const X = (t) => L + (W - L - 8) * ((t - t0) / Math.max(1, t1 - t0)), Y = (v) => T + (H - T - B) * (1 - (v - min) / (hi - min || 1));
  for (const f of [0, 0.5, 1]) {
    const v = min + (hi - min) * f, y = Y(v);
    svgEl("line", { x1: L, x2: W - 8, y1: y, y2: y, class: "grid" }, svg);
    svgEl("text", { x: L - 6, y: y + 4, class: "axis", "text-anchor": "end" }, svg).textContent = fmt(v) + unit;
  }
  for (const f of [0, 0.5, 1]) {
    const t = t0 + (t1 - t0) * f;
    svgEl("text", { x: X(t), y: H - 4, class: "axis", "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" }, svg).textContent = xfmt(t);
  }
  // gaps in the history (server off) break the line rather than bridging it
  const step = Math.min(...pts.slice(1).map((p, i) => p.t - pts[i].t).filter((d) => d > 0));
  let d = "", area = "", seg = [];
  const flush = () => {
    if (!seg.length) return;
    d += seg.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p.v).toFixed(1)}`).join("");
    area += `M${X(seg[0].t).toFixed(1)},${Y(min)}` + seg.map((p) => `L${X(p.t).toFixed(1)},${Y(p.v).toFixed(1)}`).join("") + `L${X(seg[seg.length - 1].t).toFixed(1)},${Y(min)}Z`;
    seg = [];
  };
  pts.forEach((p, i) => { if (i && p.t - pts[i - 1].t > step * 3) flush(); seg.push(p); });
  flush();
  svgEl("path", { d: area, class: "area" }, svg);
  svgEl("path", { d, class: "line" }, svg);
  const cross = svgEl("line", { y1: T, y2: H - B, class: "cross", visibility: "hidden" }, svg);
  const dot = svgEl("circle", { r: 4, class: "dot", visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: L, y: T, width: W - L - 8, height: H - T - B, class: "hit" }, svg);
  hit.addEventListener("mousemove", (e) => {
    const bx = svg.getBoundingClientRect(), t = t0 + ((e.clientX - bx.left - L) / (W - L - 8)) * (t1 - t0);
    let best = pts[0];
    for (const p of pts) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    cross.setAttribute("x1", X(best.t)); cross.setAttribute("x2", X(best.t)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(best.t)); dot.setAttribute("cy", Y(best.v)); dot.setAttribute("visibility", "visible");
    tip.show(`<b>${fmt(best.v)}${unit}</b>${best.max != null ? ` (peak ${fmt(best.max)}${unit})` : ""}<br>${xfmt(best.t, true)}`, e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); });
}

// Horizontal bars with the value written at the end: identity is the label, not a color.
function hbars(el, rows, { fmt = compact, label = "" } = {}) {
  if (!rows.length) return empty(el, "Nothing in this range yet.");
  const max = Math.max(...rows.map((r) => r.value)) || 1;
  el.innerHTML = `<ul class="hbars" aria-label="${esc(label)}">${rows.map((r) => `<li title="${esc(r.title || r.label)}">
    <span class="hb-label">${esc(r.label)}${r.sub ? `<span class="hb-sub">${esc(r.sub)}</span>` : ""}</span>
    <span class="hb-bar"><span style="width:${Math.max(1.5, (100 * r.value) / max)}%"></span></span>
    <span class="hb-val num">${fmt(r.value)}</span></li>`).join("")}</ul>`;
}
