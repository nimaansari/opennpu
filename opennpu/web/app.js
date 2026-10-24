// OpenNPU web UI: plain JS, no build step, works offline. Talks to the same server it is served from.
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ---------------------------------------------------------------- storage (never throws)
let storeWarned = false;
const store = {
  get(k, d) { try { const v = localStorage.getItem("opennpu." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) {
    try { localStorage.setItem("opennpu." + k, JSON.stringify(v)); return true; } catch {
      if (!storeWarned) { storeWarned = true; toast("Could not save: browser storage is full or disabled"); }
      return false;
    }
  },
};
function loadConvos() {
  const c = store.get("convos", []);
  if (!Array.isArray(c)) return [];
  return c.filter((x) => x && typeof x.id === "string" && Array.isArray(x.messages))
    .map((x) => ({ ...x, title: String(x.title || "Conversation"),
      // a reply cut off by a crash or a closed tab was saved empty: drop it
      messages: x.messages.filter((m) => m && (m.role === "user" || m.content || m.error || m.stopped)) }));
}

// S.stream is the one reply being generated (the server runs one at a time): {convo, reply, abort}
// remote: conversations live on the server (shared by every app, saved on this computer);
// before the server answers (or on an old server) the browser's own store is used.
const S = { models: [], stats: null, up: null, convos: loadConvos(), cur: store.get("cur", null), stream: null, spark: [],
  remote: false, attach: [], pendingDelete: new Map() };

// ---------------------------------------------------------------- utils
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const gb = (b) => (b / 1e9).toFixed(b >= 1e10 ? 0 : 1) + " GB";
const fmtS = (s) => (s == null ? "-" : s < 10 ? s.toFixed(1) + " s" : Math.round(s) + " s");
const errText = (e) => (e && typeof e === "object" ? e.message : e) || "";
function toast(t, action) {
  $$(".toast").forEach((x) => x.remove());
  const d = document.createElement("div");
  d.className = "toast"; d.setAttribute("role", "status");
  d.append(t);
  if (action) {
    const b = document.createElement("button");
    b.className = "toast-act"; b.textContent = action.label;
    b.onclick = () => { action.run(); d.remove(); };
    d.append(b);
  }
  document.body.append(d);
  setTimeout(() => d.remove(), action ? 6000 : 1800);
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); } catch { toast("Copy failed"); }
}
const HDR = { "Content-Type": "application/json", "X-OpenNPU-Client": "web" };   // lets the dashboard say which app asked
async function api(path, body, opts = {}) {
  const r = await fetch(path, { ...(body ? { method: "POST", headers: HDR, body: JSON.stringify(body) } : {}), ...opts });
  if (!r.ok) throw new Error(errText((await r.json().catch(() => ({}))).error) || r.statusText);
  return r;
}
function announce(t) { const a = $("#announce"); a.textContent = ""; setTimeout(() => (a.textContent = t), 50); }

// ---------------------------------------------------------------- markdown (safe: everything is escaped first)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0001${codes.length - 1}\u0001`; });
  s = s.replace(/\*\*(?=\S)([^*\n]*?\S)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[+i]}</code>`);
}
const ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
function block(lines) {
  const out = [];
  let i = 0;
  const startsBlock = (l) => /^\s*```/.test(l) || /^#{1,6}\s/.test(l) || ITEM.test(l) || /^&gt;\s?/.test(l) || /^\s*(-{3,}|\*{3,})\s*$/.test(l);
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    if (/^\s*```/.test(l)) {                                   // fenced code (may be unclosed while streaming)
      const body = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) body.push(lines[i]);
      i++;
      out.push(`<pre><code>${body.join("\n")}</code></pre>`);
      continue;
    }
    const h = l.match(/^(#{1,6})\s+(.*)$/);
    if (h) { const n = Math.min(h[1].length + 2, 6); out.push(`<h${n}>${inline(h[2])}</h${n}>`); i++; continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(l)) { out.push("<hr>"); i++; continue; }
    if (/^&gt;\s?/.test(l)) {
      const q = [];
      for (; i < lines.length && /^&gt;\s?/.test(lines[i]); i++) q.push(lines[i].replace(/^&gt;\s?/, ""));
      out.push(`<blockquote>${block(q)}</blockquote>`);
      continue;
    }
    const m = l.match(ITEM);
    if (m) {
      const base = m[1].length, ordered = /\d/.test(m[2]);
      const items = [];
      while (i < lines.length) {
        const x = lines[i], mm = x.match(ITEM);
        if (mm && mm[1].length <= base + 1 && /\d/.test(mm[2]) === ordered) { items.push({ text: mm[3], sub: [] }); i++; continue; }
        if (mm && mm[1].length > base + 1 && items.length) { items[items.length - 1].sub.push(x.slice(base + 2)); i++; continue; }
        if (!x.trim()) {                                       // a blank line ends the list unless it continues
          let j = i + 1;
          while (j < lines.length && !lines[j].trim()) j++;
          const nx = j < lines.length && lines[j].match(ITEM);
          if (nx && nx[1].length <= base + 1 && /\d/.test(nx[2]) === ordered) { i = j; continue; }
          break;
        }
        if (/^\s+\S/.test(x) && items.length && !mm) { items[items.length - 1].text += " " + x.trim(); i++; continue; }
        break;
      }
      const start = ordered ? parseInt(m[2], 10) : 1;
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}${ordered && start !== 1 ? ` start="${start}"` : ""}>` +
        items.map((it) => `<li>${inline(it.text)}${it.sub.length ? block(it.sub) : ""}</li>`).join("") + `</${tag}>`);
      continue;
    }
    const para = [];
    for (; i < lines.length && lines[i].trim() && !(para.length && startsBlock(lines[i])); i++) para.push(lines[i]);
    out.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }
  return out.join("");
}
const md = (src) => block(esc(src).split("\n"));

// Reasoning models: "<think>...</think>answer", or (template pre-filled <think>) "...</think>answer".
function splitThink(text) {
  let m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (m) return { reasoning: m[1].trim(), thinking: !m[2], answer: m[3] };
  m = text.match(/^([\s\S]*?)<\/think>([\s\S]*)$/);
  if (m) return { reasoning: m[1].trim(), thinking: false, answer: m[2] };
  return { reasoning: null, thinking: false, answer: text };
}
// Update a reply bubble in place: the <details> element survives so the user's open/closed
// choice and text selection are not reset on every streamed token.
function fillReply(bubble, m, live) {
  const p = splitThink(m.content || "");
  let det = $("details.reasoning", bubble), ans = $(".answer", bubble), note = $(".reply-note", bubble);
  if (!ans) {
    bubble.innerHTML = "";
    ans = document.createElement("div"); ans.className = "answer";
    bubble.append(ans);
  }
  if (p.reasoning !== null) {
    if (!det) {
      det = document.createElement("details"); det.className = "reasoning";
      det.innerHTML = "<summary></summary><div></div>";
      det.open = live && p.thinking;
      bubble.prepend(det);
    }
    $("summary", det).textContent = p.thinking && live ? "Thinking" : "Reasoning";
    $("div", det).textContent = p.reasoning;
  } else if (det) det.remove();
  ans.innerHTML = md(p.answer);
  ans.classList.toggle("cursor", live);
  const noteText = m.error ? m.error : m.stopped ? "Stopped." : !live && !m.content ? "The model returned an empty reply." : "";
  if (noteText) {
    if (!note) { note = document.createElement("p"); bubble.append(note); }
    note.className = "reply-note" + (m.error ? " err" : "");
    note.textContent = noteText;
  } else if (note) note.remove();
}

// ---------------------------------------------------------------- views + mobile rail
const mobile = matchMedia("(max-width: 760px)");
function setRail(open) {
  $("#rail").classList.toggle("open", open);
  $("#backdrop").hidden = !open;
  $("#rail").inert = mobile.matches && !open;
  if (open) $(".nav-item[aria-current]")?.focus();
}
function show(view) {
  if (!$("#view-" + view)) view = "chat";
  $$(".view").forEach((v) => (v.hidden = v.id !== "view-" + view));
  $$(".nav-item").forEach((b) => (b.dataset.view === view ? b.setAttribute("aria-current", "page") : b.removeAttribute("aria-current")));
  setRail(false);
  store.set("view", view);
  if (view === "models") renderModels();
  if (view === "system") renderSystem();
  if (view === "create" && typeof renderCreate === "function") renderCreate();
  if (view === "dashboard" && typeof renderDashboard === "function") renderDashboard();
  if (view === "discover" && typeof renderDiscover === "function") renderDiscover();
  if (view === "chat" && !mobile.matches) $("#input").focus();
}

// ---------------------------------------------------------------- models
const ENGINE_ORDER = ["moe", "native", "npu4", "llama", "universal"];
const ENGINE_NOTE = {
  moe: "Mixture-of-experts on OpenNPU's own open kernels",
  native: "Dense models on OpenNPU's own open fused kernels",
  npu4: "Any other LLM, dense or mixture-of-experts: 4-bit weights, every large matmul on the NPU",
  llama: "GGUF models (incl. your LM Studio folder) run by llama.cpp, its matrix products on the NPU",
  universal: "Vision and other Hugging Face models: linear layers on the NPU",
};
async function loadModels() {
  try {
    S.models = (await (await api("/v1/models", null, { signal: AbortSignal.timeout(15000) })).json()).data || [];
  } catch { return; }                           // keep what we had; poll() retries when the server is back
  const sel = $("#modelSel");
  const ids = new Set(S.models.map((m) => m.id));
  let want = store.get("model", null);
  if (!ids.has(want)) want = (S.stats && ids.has(S.stats.loaded) && S.stats.loaded) || (S.models[0] && S.models[0].id);
  sel.innerHTML = "";
  for (const eng of ENGINE_ORDER) {
    const ms = S.models.filter((m) => m.engine === eng);
    if (!ms.length) continue;
    const g = document.createElement("optgroup"); g.label = ms[0].engine_label;
    for (const m of ms) g.append(new Option(m.name, m.id, false, m.id === want));
    sel.append(g);
  }
  if (!S.models.length) sel.append(new Option("No models installed", ""));
  if (!$("#view-models").hidden) renderModels();
  renderThread();
}
function renderModels() {
  const root = $("#modelGroups");
  if (S.up === false) { root.innerHTML = downPanel(); return; }
  if (!S.models.length) {
    root.innerHTML = `<div class="panel"><h2>No models yet</h2><p class="muted">Find one on Hugging Face in Discover, or download one from a terminal:</p>
      <pre class="snippet">npu pull Qwen/Qwen3-8B          # dense model, OpenNPU engine\nnpu chat Qwen/Qwen3-0.6B         # any Hugging Face model, by name</pre>
      <p><button class="btn primary" data-go-discover><span class="ico i-compass"></span>Open Discover</button></p></div>`;
    $("[data-go-discover]", root).onclick = () => show("discover");
    return;
  }
  const loaded = S.stats && S.stats.loaded;
  root.innerHTML = ENGINE_ORDER.map((eng) => {
    const ms = S.models.filter((m) => m.engine === eng);
    if (!ms.length) return "";
    return `<section class="group"><h2>${esc(ms[0].engine_label)} <span class="muted">${esc(ENGINE_NOTE[eng] || "")}</span></h2><div class="cards">` +
      ms.map((m) => `<article class="card${m.id === loaded ? " loaded" : ""}">
        <h3 class="card-name">${esc(m.name)}</h3>
        <div class="card-facts"><span>${m.family && m.family !== m.name ? esc(m.family) : esc(m.engine_label)}</span><span>${m.size_bytes ? gb(m.size_bytes) : ""}</span></div>
        <div class="card-actions"><button class="btn primary" data-chat="${esc(m.id)}" aria-label="Chat with ${esc(m.name)}"><span class="ico i-chat-circle-text"></span>Chat</button>
        ${m.id === loaded ? '<span class="tag-loaded">Loaded on the NPU</span>' : ""}</div></article>`).join("") +
      `</div></section>`;
  }).join("");
  $$("[data-chat]", root).forEach((b) => (b.onclick = () => { $("#modelSel").value = b.dataset.chat; store.set("model", b.dataset.chat); newConvo(); show("chat"); }));
}
const downPanel = () => `<div class="panel"><h2>Can't reach the OpenNPU server</h2>
  <p class="muted">Start it from a terminal; this page reconnects by itself.</p><pre class="snippet">npu start\nnpu logs      # if it does not come up</pre></div>`;

// ---------------------------------------------------------------- system
function renderSystem() {
  const n = (S.stats && S.stats.npu) || {};
  const rows = [["Device", n.device], ["Architecture", n.arch], ["NPU firmware", n.firmware], ["XRT", n.xrt],
    ["Server version", $("#version").textContent || null], ["Idle unload", S.stats ? S.stats.idle_unload_s + " s" : null]];
  $("#devKv").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v == null ? (S.up === false ? "server not reachable" : "...") : v)}</dd>`).join("");
  const base = location.origin;
  $("#oaiUrl").textContent = base + "/v1";
  $("#olUrl").textContent = base;
  const m = $("#modelSel").value || "MODEL";
  $("#curlSnip").textContent = `curl ${base}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model": "${m}", "messages": [{"role": "user", "content": "Hello"}]}'`;
}

// ---------------------------------------------------------------- live telemetry
function setState(s, text) {
  const el = $("#state");
  if (el.dataset.s !== s) el.dataset.s = s;
  if ($("#stateText").textContent !== text) $("#stateText").textContent = text;
}
async function poll() {
  try {
    const st = await (await api("/api/stats", null, { signal: AbortSignal.timeout(5000) })).json();
    const wasDown = S.up !== true, first = !S.stats;
    S.stats = st; S.up = true;
    if (wasDown) {                                // first contact, or the server came back
      loadModels().then(refreshAttach);
      if (!S.remote) initSessions();
      if (typeof genServerUp === "function") genServerUp();
    }
    if ((first || wasDown) && !$("#view-system").hidden) renderSystem();
    const want = $("#modelSel").value;
    if (st.generating) setState("busy", "Generating on the NPU");
    else if (S.stream && !S.stream.started && st.loaded !== want) setState("loading", "Loading model onto the NPU");
    else if (st.loaded) setState("idle", "Ready on the NPU");
    else setState("idle", "NPU idle");
    $("#mBusy").textContent = Math.round(st.npu_busy_pct) + "%";
    $("#mCpu").textContent = st.cpu_pct.toFixed(0) + "%";
    $("#mPow").textContent = st.system_power_w == null ? "plugged in" : st.system_power_w.toFixed(1) + " W";
    $("#mMem").textContent = gb(st.rss_bytes);
    $("#mIdle").textContent = st.unload_in_s == null ? "-" : Math.floor(st.unload_in_s / 60) + ":" + String(st.unload_in_s % 60).padStart(2, "0");
    S.spark.push(st.npu_busy_pct); if (S.spark.length > 60) S.spark.shift();
    drawSpark();
    const l = st.last || {};
    $("#lTok").textContent = l.decode_tok_s ? l.decode_tok_s + " tok/s" : "-";
    $("#lTtft").textContent = fmtS(l.ttft_s);
    $("#lN").textContent = l.completion_tokens ?? "-";
    $("#lJ").textContent = l.joules_per_token ? l.joules_per_token + " J/token" : "-";
  } catch {
    const wasUp = S.up !== false;
    S.up = false;
    setState("down", "Server not reachable");
    if (wasUp) { renderThread(); if (!$("#view-models").hidden) renderModels(); if (!$("#view-system").hidden) renderSystem(); }
  }
}
let pollTimer = 0;
function schedulePoll(ms = 1500) {
  clearTimeout(pollTimer);
  if (document.hidden) return;                     // resumes on visibilitychange
  pollTimer = setTimeout(async () => { await poll(); schedulePoll(); }, ms);
}
function drawSpark() {
  const c = $("#spark"); if (!c || !c.offsetWidth) return;
  const dpr = devicePixelRatio || 1, w = c.offsetWidth, h = 56;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr);
  const col = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const pts = S.spark, n = 60, step = w / (n - 1), off = n - pts.length;
  g.beginPath();
  pts.forEach((v, i) => { const x = (i + off) * step, y = h - 2 - (v / 100) * (h - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
  if (pts.length) { g.lineTo((n - 1) * step, h); g.lineTo(off * step, h); g.closePath(); g.globalAlpha = .12; g.fillStyle = col; g.fill(); }
}

// ---------------------------------------------------------------- conversations
const curConvo = () => S.convos.find((c) => c.id === S.cur);
const newId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random()));
const modelInfo = (id) => S.models.find((m) => m.id === id);

// Switch to server sessions once the server has them; move the browser's old conversations over once.
async function initSessions() {
  let list;
  try { list = (await (await api("/v1/sessions")).json()).data; } catch { return; }
  const local = loadConvos().filter((c) => c.messages.length);
  if (!store.get("sessions.imported", false) && local.length) {
    try {
      await api("/v1/sessions/import", { sessions: local.map((c) => ({ id: c.id, title: c.title, model: c.model,
        messages: c.messages.map(({ role, content, images, stats, error, stopped }) => ({ role, content, images, stats, error, stopped })) })) });
      store.set("convos.backup", local);
      list = (await (await api("/v1/sessions")).json()).data;
    } catch { return; }                                     // stay local; retried next time
  }
  store.set("sessions.imported", true);
  try { localStorage.removeItem("opennpu.convos"); } catch { /* ignore */ }
  S.remote = true;
  const live = S.stream && S.stream.convo;
  S.convos = list.map((x) => ({ id: x.id, title: x.title, model: x.model, messages: null, n: x.messages }));
  if (live && !S.convos.find((c) => c.id === live.id)) S.convos.unshift(live);
  if (!curConvo()) S.cur = S.convos[0] ? S.convos[0].id : null;
  renderConvos();
  await ensureMessages(curConvo());
  renderThread(); refreshComposer();
}
async function ensureMessages(c) {
  if (!c || c.messages) return;
  try {
    const d = await (await api(`/v1/sessions/${encodeURIComponent(c.id)}`)).json();
    c.messages = d.messages || []; c.model = d.model;
  } catch { c.messages = []; }
}
function saveConvos() {
  if (S.remote) { store.set("cur", S.cur && !String(S.cur).startsWith("new-") ? S.cur : null); return; }   // the server saves turns
  let list = S.convos.slice(0, 100);
  while (!store.set("convos", list) && list.length > 1) list = list.slice(0, Math.ceil(list.length / 2));  // quota: keep the newest
  store.set("cur", S.cur);
}
function newConvo() {
  const c = curConvo();
  if (c && !c.messages.length) { renderThread(); return; }
  S.convos.unshift({ id: S.remote ? "new-" + newId() : newId(), title: "New conversation", messages: [] });
  S.cur = S.convos[0].id; saveConvos(); renderConvos(); renderThread(); refreshComposer();
}
async function openConvo(id) {
  S.cur = id; saveConvos(); renderConvos(); show("chat");
  const c = curConvo();
  if (c && !c.messages) { $("#thread").innerHTML = '<div class="skeleton w1"></div><div class="skeleton w2"></div>'; await ensureMessages(c); }
  if (S.cur !== id) return;
  if (c && c.model && modelInfo(c.model)) { $("#modelSel").value = c.model; refreshAttach(); }   // a session remembers its model
  renderThread(); refreshComposer();
}
function deleteConvo(id) {
  const idx = S.convos.findIndex((c) => c.id === id);
  if (idx < 0) return;
  const [gone] = S.convos.splice(idx, 1);
  if (S.stream && S.stream.convo === gone) S.stream.abort.abort();
  if (S.cur === id) S.cur = S.convos[0] ? S.convos[0].id : null;
  saveConvos(); renderConvos(); renderThread(); refreshComposer();
  if (S.remote && !String(id).startsWith("new-")) {        // deleted on the server after the Undo window
    S.pendingDelete.set(id, setTimeout(() => { S.pendingDelete.delete(id); fetch(`/v1/sessions/${encodeURIComponent(id)}`, { method: "DELETE", headers: HDR }).catch(() => {}); }, 6500));
  }
  toast("Conversation deleted", { label: "Undo", run: () => {
    clearTimeout(S.pendingDelete.get(id)); S.pendingDelete.delete(id);
    S.convos.splice(idx, 0, gone); S.cur = gone.id; saveConvos(); renderConvos(); renderThread(); refreshComposer();
  } });
}
function renderConvos() {
  const ul = $("#convos");
  ul.innerHTML = S.convos.map((c) => `<li class="${c.id === S.cur ? "active" : ""}">
    <button class="c-open" data-id="${esc(c.id)}"${c.id === S.cur ? ' aria-current="true"' : ""}>${esc(c.title)}</button>
    <button class="icon-btn c-del" data-del="${esc(c.id)}" aria-label="Delete conversation: ${esc(c.title)}" title="Delete"><span class="ico i-trash"></span></button></li>`).join("");
  $$(".c-open", ul).forEach((b) => (b.onclick = () => openConvo(b.dataset.id)));
  $$(".c-del", ul).forEach((b) => (b.onclick = () => deleteConvo(b.dataset.del)));
}
function metaHtml(m, i) {
  const st = m.stats, bits = [];
  if (st) {
    if (st.decode_tok_s) bits.push(`<span><span class="num">${+st.decode_tok_s}</span> tok/s</span>`);
    if (st.completion_tokens != null) bits.push(`<span><span class="num">${+st.completion_tokens}</span> tokens</span>`);
    if (st.ttft_s != null) bits.push(`<span>first token <span class="num">${fmtS(+st.ttft_s)}</span></span>`);
    if (st.cached_tokens) bits.push(`<span title="Prompt tokens reused from this conversation's cache">reused <span class="num">${+st.cached_tokens}</span> tokens</span>`);
    if (st.joules_per_token) bits.push(`<span><span class="num">${+st.joules_per_token}</span> J/token</span>`);
    if (st.engine) bits.push(`<span>${esc(st.engine)}</span>`);
  }
  if (!m.content) return bits.length ? `<div class="meta">${bits.join("")}</div>` : "";
  return `<div class="meta">${bits.join("")}<button class="icon-btn speak-msg" data-speak="${i}" aria-label="Read aloud" title="Read aloud"><span class="ico i-speaker-high"></span></button><button class="icon-btn copy-msg" data-copy-msg="${i}" aria-label="Copy reply" title="Copy"><span class="ico i-copy"></span></button></div>`;
}
function renderThread() {
  const th = $("#thread"), c = curConvo();
  if (c && !c.messages) { th.innerHTML = '<div class="skeleton w1"></div><div class="skeleton w2"></div>'; return; }
  if (!c || !c.messages.length) {
    let head, body, starters = "";
    if (S.up === false) { head = "Can't reach the OpenNPU server"; body = "Start it with <code>npu start</code>. This page reconnects by itself."; }
    else if (S.up && !S.models.length) { head = "No models installed yet"; body = "Find one in Discover, or add one with <code>npu pull &lt;huggingface repo&gt;</code>."; }
    else {
      head = "What should the NPU work on?";
      body = "Everything runs locally on your Ryzen AI NPU, so the CPU and GPU stay free and nothing leaves this machine.";
      starters = `<div class="starters">${["Explain how a CPU cache works, briefly.", "Write a Python function that dedupes a list, keeping order.",
        "Summarize the trade-offs of running LLMs on an NPU.", "Draft a polite email declining a meeting."].map((q) => `<button type="button">${q}</button>`).join("")}</div>`;
    }
    th.innerHTML = `<div class="empty"><h1>${head}</h1><p class="muted">${body}</p>${starters}</div>`;
    $$(".starters button", th).forEach((b) => (b.onclick = () => { $("#input").value = b.textContent; send(); }));
    return;
  }
  th.innerHTML = c.messages.map((m, i) => `<div class="msg ${m.role === "user" ? "user" : "assistant"}${m.error ? " error" : ""}" data-i="${i}"><div class="bubble"></div>${m.role === "assistant" ? metaHtml(m, i) : ""}</div>`).join("");
  $$(".msg", th).forEach((el) => {
    const m = c.messages[+el.dataset.i], b = $(".bubble", el);
    if (m.role === "user") {
      b.textContent = m.content;
      if (m.images && m.images.length) {
        const row = document.createElement("div");
        row.className = "msg-images";
        for (const u of m.images) {
          if (!/^(data:image\/|\/v1\/sessions\/media\/)/.test(u)) continue;     // only our own or inline images
          const img = document.createElement("img"); img.src = u; img.alt = "Attached image"; img.loading = "lazy";
          row.append(img);
        }
        el.insertBefore(row, b);
      }
    } else fillReply(b, m, S.stream && S.stream.reply === m);
  });
  $$("[data-copy-msg]", th).forEach((b) => (b.onclick = () => copy(splitThink(c.messages[+b.dataset.copyMsg].content || "").answer.trim())));
  $$("[data-speak]", th).forEach((b) => (b.onclick = () => speak(b, splitThink(c.messages[+b.dataset.speak].content || "").answer)));
  th.scrollTop = th.scrollHeight;
}

// ---------------------------------------------------------------- sending
// History the model sees: no failed turns (and the question that failed), no empty replies,
// strictly alternating roles (merging adjacent user turns) so strict chat templates accept it.
function history(c, upto) {
  const msgs = [];
  for (let i = 0; i < upto; i++) {
    const m = c.messages[i];
    if (m.role === "user" && c.messages[i + 1] && c.messages[i + 1].error && i + 1 < upto) { i++; continue; }
    if (m.error || (m.role !== "user" && !m.content)) continue;
    const content = m.role === "user" ? m.content : splitThink(m.content).answer.trim() || m.content;
    const last = msgs[msgs.length - 1];
    if (last && last.role === m.role && !m.images && !last.images) last.content += "\n\n" + content;
    else msgs.push(m.images && m.images.length ? { role: m.role, content, images: m.images } : { role: m.role, content });
  }
  // OpenAI shape: images become image_url parts of their message
  return msgs.map((m) => (m.images ? { role: m.role, content: [{ type: "text", text: m.content },
    ...m.images.map((u) => ({ type: "image_url", image_url: { url: u } }))] } : m));
}
function refreshComposer() {
  const here = S.stream && S.stream.convo === curConvo();
  const elsewhere = S.stream && !here;
  const s = $("#send");
  s.innerHTML = `<span class="ico ${here ? "i-stop" : "i-arrow-up"}"></span>`;
  s.setAttribute("aria-label", here ? "Stop generating" : "Send");
  s.title = here ? "Stop" : elsewhere ? "Another conversation is still generating" : "Send";
  s.disabled = !!elsewhere;
  s.type = here ? "button" : "submit";
  $("#hint").textContent = elsewhere ? "Another conversation is still generating" : "Enter to send, Shift+Enter for a new line";
  $("#thread").setAttribute("aria-busy", here ? "true" : "false");
}
let modelsLoading = null;
async function send() {
  const input = $("#input"), text = input.value.trim();
  if (text && !S.stream && !S.models.length && S.up !== false) {       // sent before the model list arrived
    await (modelsLoading = modelsLoading || loadModels().finally(() => (modelsLoading = null)));
  }
  const model = $("#modelSel").value;
  if (S.stream) { toast(S.stream.convo === curConvo() ? "Wait for this reply, or press Stop" : "Another conversation is still generating"); return; }
  if (!text && !S.attach.length) return;
  if (S.attach.length && !(modelInfo(model) || {}).vision) {
    const v = S.models.find((m) => m.vision);
    toast(v ? `${(modelInfo(model) || {}).name || model} can't see images: pick a vision model, e.g. ${v.name}` : "No vision model installed (e.g. npu chat Qwen/Qwen3-VL-2B-Instruct)");
    return;
  }
  if (!model) { toast(S.up === false ? "The server is not reachable" : S.models.length ? "Pick a model first" : "No models installed yet: see Models"); return; }
  if (!curConvo()) newConvo();
  const c = curConvo();
  if (!c.messages.length) c.title = (text || "Image").slice(0, 60);
  const um = { role: "user", content: text };
  if (S.attach.length) { um.images = S.attach.map((a) => a.url); S.attach = []; renderAttach(); }
  c.messages.push(um);
  c.model = model;
  const reply = { role: "assistant", content: "", stats: null, model };
  c.messages.push(reply);
  const msgs = history(c, c.messages.length - 1);
  S.convos = [c, ...S.convos.filter((x) => x !== c)];            // most recent first
  const stream = S.stream = { convo: c, reply, abort: new AbortController(), started: false };
  input.value = ""; autosize(); saveConvos(); renderConvos(); renderThread(); refreshComposer();
  const bubbleNow = () => (S.cur === c.id ? $(`#thread .msg[data-i="${c.messages.indexOf(reply)}"] .bubble`) : null);
  const b0 = bubbleNow();
  if (b0) b0.innerHTML = '<div class="skeleton w1"></div><div class="skeleton w2"></div><div class="skeleton w3"></div>';
  const think = $("#think").checked;
  let raf = 0, lastSave = Date.now();
  const paint = () => {
    raf = 0;
    const b = bubbleNow(); if (!b) return;
    const th = $("#thread"), stick = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
    fillReply(b, reply, true);
    if (stick) th.scrollTop = th.scrollHeight;
    if (Date.now() - lastSave > 1500) { lastSave = Date.now(); saveConvos(); }
  };
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: HDR, signal: stream.abort.signal,
      body: JSON.stringify({ model, stream: true, max_tokens: 2048, chat_template_kwargs: { enable_thinking: think },
        reasoning_effort: think ? "medium" : "low", messages: msgs,
        ...(S.remote ? { session: String(c.id).startsWith("new-") ? "new" : c.id } : {}) }),
    });
    if (!r.ok) throw new Error(errText((await r.json().catch(() => ({}))).error) || r.statusText);
    const sid = r.headers.get("X-OpenNPU-Session");
    if (sid && c.id !== sid) {                                // the server created the session: adopt its id
      if (S.cur === c.id) S.cur = sid;
      c.id = sid; saveConvos(); renderConvos();
    }
    const rd = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 2);
        if (!line.startsWith("data:") || line.endsWith("[DONE]")) continue;
        let d;
        try { d = JSON.parse(line.slice(5)); } catch { continue; }
        if (d.error) throw new Error(errText(d.error));
        stream.started = true;
        const delta = d.choices && d.choices[0] && d.choices[0].delta && d.choices[0].delta.content;
        if (delta) { reply.content += delta; if (!raf) raf = requestAnimationFrame(paint); }
        if (d.npu) reply.stats = d.npu;
      }
    }
  } catch (e) {
    if (e.name === "AbortError") reply.stopped = true;
    else reply.error = (reply.content ? "The reply was cut off: " : "Could not get a reply: ") + e.message;
  } finally {
    cancelAnimationFrame(raf);
    S.stream = null;
    saveConvos(); refreshComposer();
    if (S.cur === c.id) renderThread();
    announce(reply.error ? "Reply failed" : reply.stopped ? "Stopped" : "Reply finished");
    poll();
  }
}
// ---------------------------------------------------------------- images
const IMG_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
function addFiles(files) {
  for (const f of files) {
    if (!IMG_TYPES.includes(f.type)) { toast("Images must be PNG, JPEG, WebP or GIF"); continue; }
    if (f.size > 10 * 2 ** 20) { toast(`${f.name || "Image"} is over 10 MB`); continue; }
    if (S.attach.length >= 4) { toast("At most 4 images per message"); break; }
    const rd = new FileReader();
    rd.onload = () => { S.attach.push({ url: rd.result, name: f.name || "image" }); renderAttach(); };
    rd.readAsDataURL(f);
  }
}
function renderAttach() {
  const box = $("#attachments");
  box.hidden = !S.attach.length;
  box.innerHTML = S.attach.map((a, i) => `<div class="att"><img src="${esc(a.url)}" alt="${esc(a.name)}"><button type="button" data-rm="${i}" aria-label="Remove ${esc(a.name)}"><span class="ico i-x"></span></button></div>`).join("");
  $$("[data-rm]", box).forEach((b) => (b.onclick = () => { S.attach.splice(+b.dataset.rm, 1); renderAttach(); }));
}
function refreshAttach() {
  const v = (modelInfo($("#modelSel").value) || {}).vision;
  $("#attachBtn").title = v ? "Attach an image" : "Attach an image (needs a vision model)";
}

// ---------------------------------------------------------------- voice input: record, transcribe on the NPU
const MIC = { rec: null, chunks: [] };
async function toggleMic() {
  const btn = $("#micBtn");
  if (MIC.rec) { MIC.rec.stop(); return; }
  if (!navigator.mediaDevices || !window.MediaRecorder) { toast("This browser can't record audio"); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { toast("Microphone permission was denied"); return; }
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  MIC.rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
  MIC.chunks = [];
  MIC.rec.ondataavailable = (e) => e.data.size && MIC.chunks.push(e.data);
  MIC.rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    MIC.rec = null;
    btn.classList.remove("rec"); btn.setAttribute("aria-label", "Start voice input");
    const blob = new Blob(MIC.chunks, { type: mime || "audio/webm" });
    if (blob.size < 2000) { toast("Recording too short, try again"); return; }
    btn.disabled = true; btn.title = "Transcribing on the NPU";
    try {
      const fd = new FormData();
      fd.append("file", blob, "voice.webm");
      fd.append("response_format", "json");
      const r = await fetch("/v1/audio/transcriptions", { method: "POST", headers: { "X-OpenNPU-Client": "web" }, body: fd });
      if (!r.ok) throw new Error(errText((await r.json().catch(() => ({}))).error) || r.statusText);
      const text = ((await r.json()).text || "").trim();
      if (text) { const i = $("#input"); i.value = (i.value ? i.value + " " : "") + text; autosize(); i.focus(); }
      else toast("No speech heard");
    } catch (e) { toast("Could not transcribe: " + e.message); }
    finally { btn.disabled = false; btn.title = "Speak (speech to text on the NPU)"; }
  };
  MIC.rec.start();
  btn.classList.add("rec"); btn.setAttribute("aria-label", "Stop recording");
  announce("Recording");
}

// ---------------------------------------------------------------- read aloud (text to speech on the NPU)
let SPEAK = null;
async function speak(btn, text) {
  if (SPEAK) { SPEAK.audio.pause(); URL.revokeObjectURL(SPEAK.url); SPEAK.btn.classList.remove("on"); const same = SPEAK.btn === btn; SPEAK = null; if (same) return; }
  const plain = text.replace(/```[\s\S]*?```/g, " (code) ").replace(/[*_`#>]/g, "").replace(/\s+/g, " ").trim().slice(0, 4000);
  if (!plain) return;
  btn.classList.add("on");
  try {
    const r = await fetch("/v1/audio/speech", { method: "POST", headers: HDR, body: JSON.stringify({ input: plain, response_format: "wav" }) });
    if (!r.ok) throw new Error(errText((await r.json().catch(() => ({}))).error) || r.statusText);
    const url = URL.createObjectURL(await r.blob()), audio = new Audio(url);
    SPEAK = { audio, url, btn };
    audio.onended = () => { if (SPEAK && SPEAK.audio === audio) { btn.classList.remove("on"); URL.revokeObjectURL(url); SPEAK = null; } };
    await audio.play();
  } catch (e) { btn.classList.remove("on"); SPEAK = null; toast("Could not read aloud: " + e.message); }
}

function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, innerHeight * 0.4) + "px"; }

// ---------------------------------------------------------------- theme: system / light / dark
const THEMES = ["system", "light", "dark"];
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = t === "dark" || (t !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const label = { system: "Theme: follow system", light: "Theme: light", dark: "Theme: dark" }[t] || "Theme: follow system";
  $("#themeBtn").innerHTML = `<span class="ico ${t === "system" || !t ? "i-circle-half" : dark ? "i-moon" : "i-sun"}"></span>`;
  $("#themeBtn").title = label + " (click to change)";
  $("#themeBtn").setAttribute("aria-label", label);
  drawSpark();
}

// ---------------------------------------------------------------- boot
function boot() {
  $$(".nav-item").forEach((b) => (b.onclick = () => show(b.dataset.view)));
  $("#newChat").onclick = () => { newConvo(); show("chat"); };
  $$(".rail-toggle").forEach((b) => (b.onclick = () => setRail(!$("#rail").classList.contains("open"))));
  $("#backdrop").onclick = () => setRail(false);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#rail").classList.contains("open")) setRail(false); });
  mobile.addEventListener("change", () => setRail(false));
  setRail(false);
  $("#panelBtn").onclick = () => { const cb = $(".chat-body"); cb.classList.toggle("no-live"); store.set("nolive", cb.classList.contains("no-live")); drawSpark(); };
  if (store.get("nolive", false)) $(".chat-body").classList.add("no-live");
  let theme = store.get("theme", "system");
  if (!THEMES.includes(theme)) theme = "system";
  $("#themeBtn").onclick = () => { theme = THEMES[(THEMES.indexOf(theme) + 1) % 3]; store.set("theme", theme); applyTheme(theme); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(theme));
  applyTheme(theme);
  $("#modelSel").onchange = (e) => { store.set("model", e.target.value); refreshAttach(); };
  $("#attachBtn").onclick = () => $("#fileIn").click();
  $("#fileIn").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
  $("#input").addEventListener("paste", (e) => {
    const files = [...(e.clipboardData && e.clipboardData.files) || []].filter((f) => f.type.startsWith("image/"));
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  const comp = $("#composer");
  comp.addEventListener("dragover", (e) => { if ([...e.dataTransfer.types].includes("Files")) { e.preventDefault(); comp.classList.add("drop"); } });
  comp.addEventListener("dragleave", () => comp.classList.remove("drop"));
  comp.addEventListener("drop", (e) => { e.preventDefault(); comp.classList.remove("drop"); addFiles([...e.dataTransfer.files]); });
  $("#micBtn").onclick = toggleMic;
  $("#composer").onsubmit = (e) => { e.preventDefault(); send(); };
  $("#send").addEventListener("click", (e) => { if (S.stream && S.stream.convo === curConvo()) { e.preventDefault(); S.stream.abort.abort(); } });
  $("#input").addEventListener("input", autosize);
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  $$("[data-copy]").forEach((b) => (b.onclick = () => copy($("#" + b.dataset.copy).textContent)));
  addEventListener("storage", (e) => {                     // another tab changed the conversations
    if (e.key === "opennpu.convos" && !S.stream && !S.remote) { S.convos = loadConvos(); if (!curConvo()) S.cur = S.convos[0] ? S.convos[0].id : null; renderConvos(); renderThread(); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { poll(); schedulePoll(); } });
  fetch("/api/version").then((r) => r.json()).then((v) => { $("#version").textContent = "v" + v.version; if (!$("#view-system").hidden) renderSystem(); }).catch(() => {});
  if (S.cur && !curConvo()) S.cur = null;
  renderConvos(); renderThread(); refreshComposer();
  show(store.get("view", "chat"));
  poll().then(() => schedulePoll());
}
document.addEventListener("DOMContentLoaded", boot);
