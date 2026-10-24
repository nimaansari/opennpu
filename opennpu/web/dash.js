// Dashboard: local usage analytics + hardware history (GET /api/usage, /api/hardware).
// Everything shown here lives in a SQLite file on this computer; nothing is sent anywhere.
"use strict";
const D = { tab: store.get("dash.tab", "usage"), range: { usage: store.get("dash.ur", "7d"), hardware: store.get("dash.hr", "24h") },
  timer: 0, metric: store.get("dash.metric", "tokens") };
const RANGES = { usage: [["24h", "24 h"], ["7d", "7 days"], ["30d", "30 days"], ["all", "All"]],
  hardware: [["1h", "1 h"], ["24h", "24 h"], ["7d", "7 days"], ["30d", "30 days"]] };

const fmtTime = (ts, long) => {
  const d = new Date(ts * 1000);
  return long ? d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};
const fmtDay = (ts) => new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
const fmtDur = (s) => (s == null ? "-" : s < 60 ? s.toFixed(1) + " s" : s < 3600 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h");
const gib = (b) => (b == null ? "-" : (b / 2 ** 30).toFixed(1) + " GB");
const STATUS = { ok: "Done", error: "Failed", cancelled: "Stopped" };

async function renderDashboard() {
  $$("#dashTabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === D.tab)));
  $("#dashUsage").hidden = D.tab !== "usage";
  $("#dashHardware").hidden = D.tab !== "hardware";
  $("#dashCsv").hidden = $("#dashClear").hidden = D.tab !== "usage";
  $("#dashRange").innerHTML = RANGES[D.tab].map(([k, l]) => `<button type="button" data-r="${k}" aria-pressed="${k === D.range[D.tab]}">${l}</button>`).join("");
  $$("#dashRange button").forEach((b) => (b.onclick = () => {
    D.range[D.tab] = b.dataset.r; store.set(D.tab === "usage" ? "dash.ur" : "dash.hr", b.dataset.r); renderDashboard();
  }));
  clearTimeout(D.timer);
  try {
    if (D.tab === "usage") await renderUsage(); else await renderHardware();
  } catch (e) {
    $(D.tab === "usage" ? "#dashUsage" : "#dashHardware").innerHTML =
      `<div class="panel"><h2>Could not load the dashboard</h2><p class="muted">${esc(e.message)}</p></div>`;
  }
  // live-ish: refresh every 15 s while visible
  D.timer = setTimeout(() => { if (!$("#view-dashboard").hidden && !document.hidden) renderDashboard(); }, 15000);
}

function tiles(items) {
  return `<div class="tiles">${items.map(([label, value, sub]) => `<div class="tile"><div class="tile-label">${esc(label)}</div>
    <div class="tile-value">${esc(value)}</div>${sub ? `<div class="tile-sub">${esc(sub)}</div>` : ""}</div>`).join("")}</div>`;
}

async function renderUsage() {
  const u = await (await api(`/api/usage?range=${D.range.usage}`)).json();
  $("#dashRecord").checked = u.enabled;
  $("#dashWhere").textContent = `Stored only on this computer: ${u.stored_at}`;
  const t = u.totals, root = $("#dashUsage");
  if (!t.requests) {
    root.innerHTML = `<div class="panel dash-empty"><h2>No usage in this range yet</h2><p class="muted">${u.enabled
      ? "Chat, embeddings, images, video and voice from any app (this page, the desktop app, the npu CLI, or an OpenAI / Ollama client) show up here."
      : "Recording is off. Turn it on above to start keeping usage on this computer."}</p></div>`;
    return;
  }
  root.innerHTML = tiles([
    ["Requests", t.requests.toLocaleString(), t.errors || t.cancelled ? `${t.errors} failed, ${t.cancelled} stopped` : "all completed"],
    ["Tokens generated", t.completion_tokens.toLocaleString(), `${t.prompt_tokens.toLocaleString()} tokens read`],
    ["Average speed", t.avg_tok_s ? `${t.avg_tok_s.toFixed(1)} tok/s` : "-", "decode, weighted by tokens"],
    ["First token", t.avg_ttft_s != null ? `${t.avg_ttft_s} s` : "-", "average for chat"],
    ["NPU working time", fmtDur(t.busy_s), "time spent generating"],
    ["Energy", t.energy_wh ? `${t.energy_wh < 1 ? t.energy_wh.toFixed(2) : t.energy_wh.toFixed(1)} Wh` : "-", "measured while on battery"],
  ]) + `<div class="chart-grid" id="usageCharts"></div>`;
  const grid = $("#usageCharts");

  // over time: one measure at a time (no second axis)
  const daily = u.bucket_s >= 86400;
  const ot = chartCard(grid, "Over time", { wide: true });
  ot.card.querySelector(".link-btn").insertAdjacentHTML("beforebegin", `<div class="seg seg-sm" role="group" aria-label="Measure">
    <button type="button" data-m="tokens" aria-pressed="${D.metric === "tokens"}">Tokens</button>
    <button type="button" data-m="requests" aria-pressed="${D.metric === "requests"}">Requests</button></div>`);
  $$("[data-m]", ot.card).forEach((b) => (b.onclick = () => { D.metric = b.dataset.m; store.set("dash.metric", D.metric); renderDashboard(); }));
  const tl = u.timeline.map((p) => ({ label: daily ? fmtDay(p.t) : fmtTime(p.t), value: p[D.metric],
    tip: `<b>${(daily ? fmtDay(p.t) : fmtTime(p.t, true))}</b><br>${p.tokens.toLocaleString()} tokens, ${p.requests} requests` }));
  columns(ot.body, tl, { xEvery: Math.ceil(tl.length / 8), label: `${D.metric} over time` });
  fillTable(ot.table, ["Time", "Requests", "Tokens"], u.timeline.map((p) => [daily ? fmtDay(p.t) : fmtTime(p.t, true), p.requests, p.tokens]));

  const bm = chartCard(grid, "By model", { note: "tokens generated" });
  hbars(bm.body, u.by_model.map((m) => ({ label: m.model.split(":").slice(1).join(":") || m.model, sub: `${m.requests} req${m.avg_tok_s ? `, ${m.avg_tok_s} tok/s` : ""}`,
    value: m.completion_tokens || m.requests, title: m.model })), { label: "Tokens by model" });
  fillTable(bm.table, ["Model", "Engine", "Requests", "Tokens out", "Tokens in", "tok/s", "Busy", "Last used"],
    u.by_model.map((m) => [m.model, m.engine, m.requests, m.completion_tokens, m.prompt_tokens, m.avg_tok_s, fmtDur(m.busy_s), fmtTime(m.last_ts, true)]));

  const hd = chartCard(grid, "When you use it", { note: "requests by hour of day" });
  columns(hd.body, u.hour_of_day.map((v, h) => ({ label: String(h).padStart(2, "0"), value: v, tip: `<b>${String(h).padStart(2, "0")}:00</b><br>${v} requests` })),
    { xEvery: 3, height: 150, label: "Requests by hour of day" });
  fillTable(hd.table, ["Hour", "Requests"], u.hour_of_day.map((v, h) => [`${String(h).padStart(2, "0")}:00`, v]));

  const KIND = { chat: "Chat", embed: "Embeddings", image: "Images", video: "Video", speech: "Voice" };
  const bk = chartCard(grid, "By kind");
  hbars(bk.body, Object.entries(u.by_kind).map(([k, v]) => ({ label: KIND[k] || k, value: v })), { label: "Requests by kind" });
  fillTable(bk.table, ["Kind", "Requests"], Object.entries(u.by_kind).map(([k, v]) => [KIND[k] || k, v]));
  const APP = { web: "Web UI", desktop: "Desktop app", cli: "npu CLI", "openai sdk": "OpenAI client", "ollama client": "Ollama client" };
  const ba = chartCard(grid, "By app");
  hbars(ba.body, Object.entries(u.by_client).map(([k, v]) => ({ label: APP[k] || k, value: v })), { label: "Requests by app" });
  fillTable(ba.table, ["App", "Requests"], Object.entries(u.by_client).map(([k, v]) => [APP[k] || k, v]));

  const rc = document.createElement("section");
  rc.className = "chart-card wide";
  rc.innerHTML = `<header><h3>Recent requests</h3><span class="fine">latest 50 in this range</span></header><div class="table-wrap" tabindex="0" role="region" aria-label="Recent requests"></div>`;
  grid.append(rc);
  fillTable($(".table-wrap", rc), ["Time", "Kind", "Model", "App", "Result", "Tokens in", "Tokens out", "tok/s", "First token", "Duration"],
    u.recent.map((r) => [fmtTime(r.ts, true), KIND[r.kind] || r.kind, r.model, APP[r.client] || r.client, STATUS[r.status] || r.status,
      r.prompt_tokens, r.completion_tokens, r.decode_tok_s, r.ttft_s != null ? r.ttft_s.toFixed(2) + " s" : null, fmtDur(r.total_s)]));
}

async function renderHardware() {
  const h = await (await api(`/api/hardware?range=${D.range.hardware}`)).json();
  $("#dashRecord").checked = h.enabled;
  const i = h.info, n = h.now || {}, root = $("#dashHardware");
  const npu = i.npu || {};
  const disk = i.disks && (i.disks.models || i.disks.home);
  root.innerHTML = tiles([
    ["NPU busy", n.npu_busy != null ? `${Math.round(n.npu_busy)}%` : "-", "last 5 s"],
    ["CPU", n.cpu != null ? `${Math.round(n.cpu)}%` : "-", `${i.cores} threads`],
    ["Memory", n.ram_used != null ? `${gib(n.ram_used)}` : "-", `of ${gib(i.ram_total)}`],
    ["CPU temperature", n.cpu_temp != null ? `${Math.round(n.cpu_temp)} °C` : "-", n.gpu_temp != null ? `GPU ${Math.round(n.gpu_temp)} °C` : ""],
    ["Power", n.power_w != null ? `${n.power_w.toFixed(1)} W` : "plugged in", "whole laptop, on battery"],
    ["Model disk", disk ? `${gib(disk.free)} free` : "-", disk ? `of ${gib(disk.total)}` : ""],
  ]) + `<div class="chart-grid" id="hwCharts"></div>
  <section class="panel hw-info"><h2>This machine</h2><dl class="kv">${[
    ["Processor", i.cpu], ["NPU", npu.device ? `${npu.device} (${npu.arch || "?"})` : "not detected"], ["NPU firmware", npu.firmware],
    ["XRT", npu.xrt], ["Graphics", i.gpu], ["Memory", gib(i.ram_total)], ["Swap", gib(i.swap_total)],
    ["System", i.os], ["Kernel", i.kernel],
    ["Battery", i.battery ? `${i.battery.capacity ?? "?"}%, ${(i.battery.status || "").toLowerCase()}${i.battery.health ? `, health ${i.battery.health}%` : ""}` : null],
    ...Object.entries(i.disks || {}).map(([k, d]) => [`Disk (${k})`, `${gib(d.free)} free of ${gib(d.total)}  ${d.path}`]),
  ].filter(([, v]) => v != null).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("")}</dl></section>`;
  const grid = $("#hwCharts"), s = h.series;
  const long = D.range.hardware === "7d" || D.range.hardware === "30d";
  const xfmt = (t, full) => (full ? fmtTime(t, true) : long ? fmtDay(t) : fmtTime(t));
  const add = (title, key, opts) => {
    const c = chartCard(grid, title, { note: opts.note });
    const pts = s.map((p) => ({ t: p.t, v: p[key] == null ? null : p[key] * (opts.scale || 1), max: opts.peak && p[opts.peak] != null ? p[opts.peak] * (opts.scale || 1) : null }));
    line(c.body, pts, { xfmt, label: title, ...opts });
    fillTable(c.table, ["Time", title + (opts.unit ? ` (${opts.unit.trim()})` : "")], pts.filter((p) => p.v != null).map((p) => [fmtTime(p.t, true), opts.fmt(p.v)]));
  };
  const pct = (v) => v.toFixed(0);
  add("NPU busy", "npu_busy", { unit: "%", max: 100, fmt: pct, peak: "npu_busy_max", note: "share of time generating" });
  add("CPU", "cpu", { unit: "%", max: 100, fmt: pct, peak: "cpu_max" });
  add("Memory used", "ram_used", { unit: " GB", scale: 1 / 2 ** 30, max: i.ram_total ? Math.ceil(i.ram_total / 2 ** 30) : null, fmt: (v) => v.toFixed(0) });
  add("CPU temperature", "cpu_temp", { unit: " °C", min: 20, fmt: pct, peak: "cpu_temp_max" });
  add("Power", "power_w", { unit: " W", fmt: (v) => v.toFixed(0), note: "on battery only" });
  add("Server memory", "server_rss", { unit: " GB", scale: 1 / 2 ** 30, fmt: (v) => v.toFixed(1), note: "the model resident in RAM" });
  $("#dashWhere").textContent = "Hardware history: one point per minute, kept 30 days, stored only on this computer.";
}

function initDashboard() {
  $$("#dashTabs button").forEach((b) => (b.onclick = () => { D.tab = b.dataset.tab; store.set("dash.tab", D.tab); renderDashboard(); }));
  $("#dashRecord").onchange = async (e) => {
    try { await api("/api/usage/settings", { enabled: e.target.checked }); toast(e.target.checked ? "Recording usage on this computer" : "Recording paused"); }
    catch (err) { e.target.checked = !e.target.checked; toast("Could not change recording: " + err.message); }
  };
  $("#dashClear").onclick = async () => {
    if (!confirm("Delete all recorded usage from this computer? Hardware history is kept.")) return;
    try { await api("/api/usage?what=usage", null, { method: "DELETE", headers: HDR }); toast("Usage history deleted"); renderDashboard(); }
    catch (err) { toast("Could not delete: " + err.message); }
  };
  addEventListener("resize", () => { if (!$("#view-dashboard").hidden) { clearTimeout(D.rz); D.rz = setTimeout(renderDashboard, 200); } });
}
initDashboard();
