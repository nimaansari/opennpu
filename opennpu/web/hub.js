// OpenNPU web UI: the Discover view. Searches Hugging Face through the local server (/api/hub/*),
// shows how each model would run on the NPU, downloads with progress and cancel, then opens it in
// Chat or Create. Uses app.js helpers ($, $$, esc, store, toast, announce, show, loadModels, HDR).
"use strict";
(() => {                                         // own scope: gen.js / dash.js share the global one
const H = {
  task: store.get("hub.task", "chat"), sort: store.get("hub.sort", "downloads"), q: "",
  cards: [], pulls: new Map(), freeDisk: null, seq: 0, ctl: null, searched: false, error: "",
  open: null,            // id whose details panel is open: {id, info|null, error, gguf, busy}
  confirm: null,         // id whose Remove is waiting for confirmation
  timer: 0, debounce: 0, wasDownloading: new Set(),
};
const HUB_TASKS = ["chat", "vision", "embed", "speech", "image", "video", "all"];
const HUB_SORTS = ["downloads", "likes", "trending_score", "last_modified"];
const TASK_NAME = {
  "text-generation": "Chat", "image-text-to-text": "Vision", "feature-extraction": "Embeddings",
  "sentence-similarity": "Embeddings", "automatic-speech-recognition": "Speech to text",
  "text-to-speech": "Text to speech", "text-to-audio": "Text to audio", "text-to-image": "Image",
  "image-to-image": "Image to image", "text-to-video": "Video", "image-to-video": "Image to video",
};
const CREATE_TASK = { "text-to-image": "image", "image-to-image": "image", "text-to-video": "video",
  "image-to-video": "video", "text-to-speech": "speech", "text-to-audio": "speech" };

// ---------------------------------------------------------------- formatting
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
function fmtBytes(b) {
  if (b == null) return "?";
  if (b < 1e6) return Math.max(1, Math.round(b / 1e3)) + " KB";
  if (b < 1e9) return Math.round(b / 1e6) + " MB";
  return (b / 1e9).toFixed(b >= 1e11 ? 0 : 1) + " GB";
}
function fmtParams(p) {
  if (!p) return "";
  if (p >= 1e9) return (p / 1e9).toFixed(p >= 1e11 ? 0 : 1) + "B";
  return Math.round(p / 1e6) + "M";
}
function fmtEta(s) {
  if (s == null) return "estimating";
  if (s < 60) return Math.max(1, Math.round(s)) + " s left";
  if (s < 3600) return Math.round(s / 60) + " min left";
  return Math.floor(s / 3600) + " h " + Math.round((s % 3600) / 60) + " min left";
}
const taskName = (t) => TASK_NAME[t] || t || "Unknown task";
const rowId = (id) => "hub-" + id.replace(/[^\w-]/g, "_");

async function hubFetch(path, opts = {}) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = j.error;
    const err = new Error((e && (e.message || (typeof e === "string" ? e : ""))) || r.statusText || "request failed");
    err.status = r.status;
    throw err;
  }
  return j;
}
// errors in words a user can act on
function hubErrText(e) {
  const m = e.message || String(e);
  if (e.status === 502 || /offline|reach huggingface/i.test(m))
    return m + ". Discover needs internet access for search and downloads; installed models keep working offline.";
  if (/gated/i.test(m) && !/hf auth login/.test(m))
    return m + ". Accept the license on huggingface.co, run `hf auth login` in a terminal, then try again.";
  if (e.name === "TypeError") return "Can't reach the OpenNPU server. Start it with `npu start`; this page reconnects by itself.";
  return m;
}

// ---------------------------------------------------------------- the model id to chat with
function chatId(card, job) {
  if (job && job.model) return job.model;
  const repo = card.id, name = repo.split("/").pop();
  const hit = S.models.find((m) => m.id === "npu4:" + repo || m.id === "universal:" + repo) ||
    S.models.find((m) => (m.engine === "native" || m.engine === "moe") && m.name === name);
  if (hit) return hit.id;
  const eng = (card.plan || {}).engine;
  return eng === "native" || eng === "moe" ? `${eng}:${name}` : (eng === "npu4" ? "npu4:" : "universal:") + repo;
}
// A finished download reaches /v1/models within a few seconds (the catalog is rescanned every 5 s).
async function chatWith(mid) {
  for (let i = 0; i < 10 && !S.models.some((m) => m.id === mid); i++) {
    await loadModels();
    if (S.models.some((m) => m.id === mid)) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (!S.models.some((m) => m.id === mid)) { toast("Downloaded. It appears in the model list in a few seconds"); return; }
  $("#modelSel").value = mid; store.set("model", mid); refreshAttach();
  newConvo(); show("chat");
  announce("Chat with " + mid.split(":").slice(1).join(":"));
}
function openInCreate(card) {
  const t = CREATE_TASK[card.task] || "image";
  store.set("gen.task", t); store.set("gen.model." + t, card.id);
  if (typeof G === "object") { G.task = t; G.models = null; }          // re-read: it may be new
  show("create");
}

// ---------------------------------------------------------------- render
function renderDiscover() {
  $("#hubQ").value = H.q;
  $("#hubSort").value = H.sort;
  $$("#hubTasks button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.task === H.task)));
  if (!H.searched) search();
  pollPulls();
}
function cardFor(id) { return H.cards.find((c) => c.id === id); }

function badges(c) {
  const p = c.plan || {}, out = [];
  out.push(`<span class="badge${p.fast ? " fast" : ""}" title="${esc(p.fast ? "Fastest path on this NPU" : "How OpenNPU runs it")}">` +
    `${p.fast ? '<span class="ico i-lightning" aria-hidden="true"></span>' : ""}${esc(p.label || p.engine || "Unknown engine")}</span>`);
  if (c.gated) out.push('<span class="badge gated"><span class="ico i-lock-simple" aria-hidden="true"></span>Gated</span>');
  if (c.gguf) out.push('<span class="badge">GGUF</span>');
  if (c.installed) out.push('<span class="badge ok"><span class="ico i-check-circle" aria-hidden="true"></span>Installed</span>');
  return out.join("");
}
function factsHtml(c) {
  const bits = [`<span>${esc(taskName(c.task))}</span>`];
  if (c.params) bits.push(`<span title="Parameters">${fmtParams(c.params)}</span>`);
  bits.push(`<span title="Downloads"><span class="ico i-download-simple" aria-hidden="true"></span>${compact.format(c.downloads || 0)}<span class="sr"> downloads</span></span>`);
  bits.push(`<span title="Likes"><span class="ico i-heart" aria-hidden="true"></span>${compact.format(c.likes || 0)}<span class="sr"> likes</span></span>`);
  if (c.updated) bits.push(`<span title="Last updated">${esc(c.updated)}</span>`);
  return bits.join("");
}
function rowHtml(c) {
  const [org, ...rest] = c.id.split("/");
  const warn = c.plan && c.plan.fits === false && c.plan.note
    ? `<p class="hub-warn"><span class="ico i-warning-circle" aria-hidden="true"></span>${esc(c.plan.note)}</p>` : "";
  return `<article class="hub-row" id="${rowId(c.id)}" data-id="${esc(c.id)}">
    <div class="hub-main">
      <h3 class="hub-name"><span class="hub-org">${esc(org)}/</span>${esc(rest.join("/"))}</h3>
      <div class="hub-facts">${factsHtml(c)}</div>
      <div class="hub-badges">${badges(c)}</div>${warn}
    </div>
    <div class="hub-act"></div>
    <div class="hub-detail" id="${rowId(c.id)}-detail" hidden></div>
  </article>`;
}
function pullStatusHtml(job, card, label) {
  const name = esc(label || job.id);
  if (job.state === "downloading") {
    const pct = job.total ? Math.min(100, Math.floor((job.bytes / job.total) * 100)) : 0;
    return `<div class="hub-progress">
      <div class="gen-progress" role="progressbar" aria-label="Download progress for ${name}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${pct}%"></span></div>
      <div class="hub-prog-row"><span class="num">${fmtBytes(job.bytes)} of ${fmtBytes(job.total)}</span>
        <span class="num">${job.rate ? fmtBytes(job.rate) + "/s" : ""}</span><span>${fmtEta(job.eta_s)}</span>
        <button class="btn" data-hub-cancel="${esc(job.id)}" aria-label="Cancel download of ${name}"><span class="ico i-x"></span>Cancel</button></div></div>`;
  }
  if (job.state === "done") return doneActions(card || { id: job.id, plan: { engine: job.engine } }, job, name);
  if (job.state === "failed") {
    return `<div class="hub-state err"><span class="ico i-x-circle" aria-hidden="true"></span><span>Download failed: ${esc(job.error || "unknown error")}</span>
      <button class="btn" data-hub-retry="${esc(job.id)}" aria-label="Retry download of ${name}">Retry</button>
      <button class="btn" data-hub-dismiss="${esc(job.id)}" aria-label="Dismiss ${name}">Dismiss</button></div>`;
  }
  return "";
}
function doneActions(c, job, name) {
  const eng = (c.plan || {}).engine || (job && job.engine);
  name = name || esc(c.id);
  let main;
  if (eng === "gen" && CREATE_TASK[c.task])
    main = `<button class="btn primary" data-hub-create="${esc(c.id)}" aria-label="Open ${name} in Create"><span class="ico i-sparkle"></span>Open in Create</button>`;
  else if (eng === "gen") main = '<span class="muted small">Ready for voice input in Chat</span>';
  else if (eng === "embed") main = '<span class="muted small">Ready for /v1/embeddings</span>';
  else main = `<button class="btn primary" data-hub-chat="${esc(c.id)}" aria-label="${job ? "Chat now with" : "Chat with"} ${name}"><span class="ico i-chat-circle-text"></span>${job ? "Chat now" : "Chat"}</button>`;
  if (H.confirm === c.id) {
    return `<div class="hub-state confirm" role="group" aria-label="Confirm removal"><span>Remove ${name}? Its files are deleted from this computer.</span>
      <button class="btn danger" data-hub-remove-yes="${esc(c.id)}">Remove</button><button class="btn" data-hub-remove-no="${esc(c.id)}">Keep</button></div>`;
  }
  return `<div class="hub-state">${job ? '<span class="ok-text"><span class="ico i-check-circle" aria-hidden="true"></span>Downloaded</span>' : ""}${main}
    <button class="btn" data-hub-remove="${esc(c.id)}" aria-label="Remove ${name}"><span class="ico i-trash"></span>Remove</button></div>`;
}
function actHtml(c) {
  const job = H.pulls.get(c.id);
  if (job && job.state !== "cancelled") {
    const s = pullStatusHtml(job, c);
    if (s) return s;
  }
  if (c.installed) return doneActions(c, null);
  const open = !!(H.open && H.open.id === c.id);
  const note = job && job.state === "cancelled" ? '<span class="muted small">Cancelled</span>' : "";
  return `<div class="hub-state">${note}<button class="btn" data-hub-details="${esc(c.id)}" aria-expanded="${open}" aria-controls="${rowId(c.id)}-detail"
    aria-label="Download ${esc(c.id)}"><span class="ico i-download-simple"></span>Download</button></div>`;
}
function detailHtml(c) {
  const o = H.open;
  if (o.error) return `<div class="gen-error" role="alert">${esc(o.error)}</div><div class="hub-detail-actions"><button class="btn" data-hub-close>Close</button></div>`;
  if (!o.info) return '<div class="skeleton w1"></div><div class="skeleton w2"></div>';
  const d = o.info, p = d.plan || {};
  const ggufSize = (n) => ((d.gguf || []).find((g) => g.name === n) || {}).size || 0;
  const defGguf = (d.download.files || []).find((f) => f.endsWith(".gguf"));
  const bytes = o.gguf && defGguf ? d.download.bytes - ggufSize(defGguf) + ggufSize(o.gguf) : d.download.bytes;
  const free = d.free_disk, short = free != null && bytes + 2 * 2 ** 30 > free;
  const rows = [
    ["Runs on", esc(p.label || p.engine) + (p.note ? `<br><span class="hub-note">${esc(p.note)}</span>` : "")],
    ["Download", `<span class="num">${fmtBytes(bytes)}</span> <span class="muted">(${(d.download.files || []).length} files)</span>`],
    ["Free disk", `<span class="num${short ? " bad-text" : ""}">${fmtBytes(free)}</span>${short ? ' <span class="bad-text">not enough space</span>' : ""}`],
    ["Parameters", esc(fmtParams(d.params || c.params) || "unknown")],
    ["RAM needed", p.ram_gb ? `about ${p.ram_gb} GB${p.fits === false ? ' <span class="warn-text">more than this machine has</span>' : ""}` : "unknown"],
    ["License", esc(d.license || "not stated")],
  ];
  const picker = (d.gguf || []).length ? `<label class="gen-label" for="${rowId(c.id)}-gguf">GGUF variant</label>
    <div class="select-wrap"><select id="${rowId(c.id)}-gguf" data-hub-gguf>${d.gguf.filter((g) => !/mmproj/i.test(g.name)).map((g) =>
      `<option value="${esc(g.name)}"${g.name === (o.gguf || defGguf) ? " selected" : ""}>${esc(g.name)} (${fmtBytes(g.size)})</option>`).join("")}</select><span class="ico i-caret-down"></span></div>` : "";
  const gated = d.gated ? '<p class="hub-note"><span class="ico i-lock-simple" aria-hidden="true"></span>Gated: accept the license on huggingface.co and run <code>hf auth login</code> before downloading.</p>' : "";
  return `<dl class="kv hub-kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>${picker}${gated}
    ${o.pullError ? `<div class="gen-error" role="alert">${esc(o.pullError)}</div>` : ""}
    <div class="hub-detail-actions"><button class="btn" data-hub-close>Close</button>
      <button class="btn primary" data-hub-pull="${esc(c.id)}"${o.busy ? " disabled" : ""}><span class="ico i-download-simple"></span>Download ${fmtBytes(bytes)}</button></div>`;
}
function paintRow(c) {
  const el = document.getElementById(rowId(c.id));
  if (!el) return;
  const act = $(".hub-act", el), html = actHtml(c);
  if (act._html !== html) { act.innerHTML = html; act._html = html; }
  const det = $(".hub-detail", el), open = H.open && H.open.id === c.id;
  det.id = rowId(c.id) + "-detail";
  det.hidden = !open;
  if (open) {
    const dh = detailHtml(c);
    if (det._html !== dh) { det.innerHTML = dh; det._html = dh; }
  } else if (det._html) { det.innerHTML = ""; det._html = ""; }
  const bd = $(".hub-badges", el), bh = badges(c);
  if (bd._html !== bh) { bd.innerHTML = bh; bd._html = bh; }
}
function renderResults() {
  const root = $("#hubResults");
  $("#hubError").hidden = !H.error;
  $("#hubError").textContent = H.error;
  if (H.error && !H.cards.length) { root.innerHTML = ""; return; }
  if (!H.cards.length) {
    root.innerHTML = H.searched ? `<p class="muted hub-empty">No ${H.task === "all" ? "" : esc(H.task === "embed" ? "embedding" : H.task) + " "}models match${H.q ? ` "${esc(H.q)}"` : ""}.</p>` : "";
    return;
  }
  // keep keyboard focus on the same control when fresh results replace the rows
  const a = document.activeElement, keep = a && root.contains(a) && a.closest(".hub-row") &&
    [a.closest(".hub-row").dataset.id, [...a.attributes].find((x) => x.name.startsWith("data-hub-"))];
  root.innerHTML = H.cards.map(rowHtml).join("");
  H.cards.forEach(paintRow);
  if (keep && keep[1] && cardFor(keep[0]))
    document.getElementById(rowId(keep[0]))?.querySelector(`[${keep[1].name}]`)?.focus();
}
function renderPulls() {
  // downloads that are not in the current results (started elsewhere, or before the search changed)
  const box = $("#hubPulls");
  const extra = [...H.pulls.values()].filter((j) => !cardFor(j.id) && j.state !== "cancelled");
  box.hidden = !extra.length;
  if (!extra.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<h2>Downloads</h2>` + extra.map((j) => `<div class="hub-row hub-pull" data-id="${esc(j.id)}">
    <div class="hub-main"><h3 class="hub-name">${esc(j.id)}</h3></div><div class="hub-act">${pullStatusHtml(j, null)}</div></div>`).join("");
}

// ---------------------------------------------------------------- search
async function search() {
  const seq = ++H.seq;
  if (H.ctl) H.ctl.abort();
  H.ctl = new AbortController();
  $("#hubResults").setAttribute("aria-busy", "true");
  if (!H.cards.length) $("#hubResults").innerHTML = '<div class="skeleton w1"></div><div class="skeleton w2"></div><div class="skeleton w3"></div>';
  const qs = new URLSearchParams({ q: H.q, task: H.task, sort: H.sort, limit: "30" });
  try {
    const d = await hubFetch("/api/hub/search?" + qs, { signal: H.ctl.signal });
    if (seq !== H.seq) return;
    H.cards = d.data || []; H.error = "";
  } catch (e) {
    if (e.name === "AbortError" || seq !== H.seq) return;
    H.cards = []; H.error = "Search failed: " + hubErrText(e);
  }
  H.searched = true;
  if (H.open && !cardFor(H.open.id)) H.open = null;
  $("#hubResults").setAttribute("aria-busy", "false");
  $("#hubCount").textContent = H.error ? "" : `${H.cards.length} model${H.cards.length === 1 ? "" : "s"} found`;
  renderResults(); renderPulls();
}

// ---------------------------------------------------------------- downloads
async function pollPulls() {
  clearTimeout(H.timer);
  let d;
  try { d = await hubFetch("/api/hub/pulls"); } catch { H.timer = setTimeout(pollPulls, 3000); return; }
  H.pulls = new Map((d.data || []).map((j) => [j.id, j]));
  H.freeDisk = d.free_disk;
  let finished = false;
  for (const j of H.pulls.values()) {
    if (j.state === "downloading") H.wasDownloading.add(j.id);
    else if (H.wasDownloading.has(j.id)) {
      H.wasDownloading.delete(j.id);
      const c = cardFor(j.id);
      if (j.state === "done") { finished = true; if (c) c.installed = true; announce(`${j.id} downloaded`); }
      else if (j.state === "failed") announce(`Download of ${j.id} failed`);
    }
  }
  if (finished) loadModels();
  H.cards.forEach(paintRow); renderPulls();
  if ([...H.pulls.values()].some((j) => j.state === "downloading")) H.timer = setTimeout(pollPulls, 1000);
}
async function openDetails(id) {
  if (H.open && H.open.id === id) { H.open = null; paintRow(cardFor(id)); return; }
  const prev = H.open && cardFor(H.open.id);
  H.open = { id, info: null, error: "", gguf: null, busy: false };
  if (prev) paintRow(prev);
  const c = cardFor(id);
  paintRow(c);
  try {
    const info = await hubFetch("/api/hub/model?id=" + encodeURIComponent(id));
    if (!H.open || H.open.id !== id) return;
    H.open.info = info;
    if (info.installed && !c.installed) { c.installed = true; H.open = null; }
  } catch (e) {
    if (!H.open || H.open.id !== id) return;
    H.open.error = "Could not read the model details: " + hubErrText(e);
  }
  paintRow(c);
  $(`#${rowId(id)} [data-hub-pull], #${rowId(id)} [data-hub-close]`)?.focus();
}
async function startPull(id) {
  const o = H.open, c = cardFor(id);
  const body = { id };
  if (o && o.id === id && o.info && (o.info.gguf || []).length) {
    const sel = $(`#${rowId(id)} [data-hub-gguf]`);
    if (sel) body.gguf = sel.value;
  }
  if (o) { o.busy = true; o.pullError = ""; paintRow(c); }
  try {
    const job = await hubFetch("/api/hub/pull", { method: "POST", headers: HDR, body: JSON.stringify(body) });
    H.pulls.set(id, job); H.wasDownloading.add(id);
    H.open = null; H.confirm = null;
    announce(`Downloading ${id}`);
    if (c) paintRow(c); else renderPulls();
    $(`#${rowId(id)} [data-hub-cancel]`)?.focus();
    pollPulls();
  } catch (e) {
    if (o && H.open === o) { o.busy = false; o.pullError = hubErrText(e); paintRow(c); }
    else toast("Could not start the download: " + hubErrText(e));
  }
}
async function cancelPull(id) {
  try { await hubFetch("/api/hub/cancel", { method: "POST", headers: HDR, body: JSON.stringify({ id }) }); }
  catch (e) { toast("Could not cancel: " + hubErrText(e)); }
  H.wasDownloading.delete(id);
  announce(`Download of ${id} cancelled`);
  await pollPulls();
  $(`#${rowId(id)} [data-hub-details]`)?.focus();
}
async function removeModel(id) {
  const c = cardFor(id);
  try {
    await hubFetch("/api/hub/model?id=" + encodeURIComponent(id), { method: "DELETE", headers: HDR });
  } catch (e) {
    H.confirm = null;
    if (c) paintRow(c);
    toast("Could not remove: " + hubErrText(e));
    return;
  }
  // the finished job would still offer "Chat now": clear it (cancel on a finished job drops it)
  if (H.pulls.has(id) && H.pulls.get(id).state !== "downloading")
    await fetch("/api/hub/cancel", { method: "POST", headers: HDR, body: JSON.stringify({ id }) }).catch(() => {});
  H.pulls.delete(id); H.confirm = null;
  if (c) { c.installed = false; paintRow(c); }
  renderPulls(); loadModels();
  toast(`Removed ${id}`); announce(`Removed ${id}`);
  $(`#${rowId(id)} [data-hub-details]`)?.focus();
}

// ---------------------------------------------------------------- events (delegated: rows are re-rendered)
function hubClick(e) {
  const b = e.target.closest("button");
  if (!b) return;
  const d = b.dataset, c = (id) => cardFor(id) || { id, plan: { engine: (H.pulls.get(id) || {}).engine } };
  if (d.hubDetails) openDetails(d.hubDetails);
  else if (d.hubClose !== undefined) { const o = H.open; H.open = null; if (o) { paintRow(cardFor(o.id)); $(`#${rowId(o.id)} [data-hub-details]`)?.focus(); } }
  else if (d.hubPull) startPull(d.hubPull);
  else if (d.hubRetry) startPull(d.hubRetry);
  else if (d.hubCancel) cancelPull(d.hubCancel);
  else if (d.hubDismiss) cancelPull(d.hubDismiss);
  else if (d.hubChat) chatWith(chatId(c(d.hubChat), H.pulls.get(d.hubChat)));
  else if (d.hubCreate) openInCreate(c(d.hubCreate));
  else if (d.hubRemove) { H.confirm = d.hubRemove; paintRow(c(d.hubRemove)); $(`#${rowId(d.hubRemove)} [data-hub-remove-no]`)?.focus(); renderPulls(); }
  else if (d.hubRemoveNo) { H.confirm = null; paintRow(c(d.hubRemoveNo)); renderPulls(); $(`#${rowId(d.hubRemoveNo)} [data-hub-remove]`)?.focus(); }
  else if (d.hubRemoveYes) removeModel(d.hubRemoveYes);
}
function hubBoot() {
  if (!HUB_TASKS.includes(H.task)) H.task = "chat";
  if (!HUB_SORTS.includes(H.sort)) H.sort = "downloads";
  $("#hubQ").addEventListener("input", () => {
    clearTimeout(H.debounce);
    H.debounce = setTimeout(() => { const q = $("#hubQ").value.trim(); if (q !== H.q) { H.q = q; search(); } }, 350);
  });
  $("#hubQ").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { clearTimeout(H.debounce); H.q = $("#hubQ").value.trim(); search(); }
  });
  $("#hubSort").onchange = (e) => { H.sort = e.target.value; store.set("hub.sort", H.sort); search(); };
  $$("#hubTasks button").forEach((b) => (b.onclick = () => {
    H.task = b.dataset.task; store.set("hub.task", H.task);
    $$("#hubTasks button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    search();
  }));
  $("#view-discover").addEventListener("click", hubClick);
  $("#view-discover").addEventListener("change", (e) => {
    if (e.target.matches("[data-hub-gguf]") && H.open) { H.open.gguf = e.target.value; paintRow(cardFor(H.open.id)); $(`#${rowId(H.open.id)} [data-hub-gguf]`)?.focus(); }
  });
  pollPulls();                                   // a download started before this page loaded keeps showing
}
hubBoot();                                       // app.js boot() (DOMContentLoaded) shows the saved view after this
window.renderDiscover = renderDiscover;
window.HUB = H;                                  // for tests and debugging
})();
