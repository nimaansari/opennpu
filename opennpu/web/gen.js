// OpenNPU web UI: the Create view (images, video, voice). Uses app.js helpers ($, $$, esc, store, toast).
"use strict";
const G = { models: null, task: store.get("gen.task", "image"), busy: false, results: [] };
const TASK_LABEL = { image: "Image", video: "Video", speech: "Voice" };
const TASK_PATH = { image: "/v1/images/generations", video: "/v1/videos/generations", speech: "/v1/audio/speech" };
const TASK_HINT = {
  image: "Describe the picture. Turbo models need 1 to 4 steps; others look best at 20 to 30.",
  video: "Describe the scene and the motion. A short clip takes several minutes on a laptop.",
  speech: "Type what the voice should say.",
};

async function genFetch(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: HDR, body: JSON.stringify(body) } : {});
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    const e = j.error;
    const err = new Error((e && (e.message || e)) || r.statusText || "request failed");
    err.cancelled = r.status === 499 || (e && e.type === "cancelled");
    throw err;
  }
  return r;
}
const LIMITS = { image: 2000, video: 2000, speech: 4096 };       // the server's limits, enforced here first

async function renderCreate() {
  $$("#genTabs button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.task === G.task)));
  $("#genHint").textContent = TASK_HINT[G.task];
  $("#genImageOpts").hidden = G.task !== "image";
  $("#genVideoOpts").hidden = G.task !== "video";
  $("#genVoiceOpts").hidden = G.task !== "speech";
  $(".gen-common").hidden = G.task === "speech";
  $("#genPrompt").placeholder = G.task === "speech" ? "Hello from the NPU." : "A red fox sitting in fresh snow, golden hour";
  $("#genPrompt").maxLength = LIMITS[G.task];
  if (!G.models) {
    $("#genModel").innerHTML = '<option value="">Loading models</option>';
    $("#genGo").disabled = true;
    try {
      G.models = (await (await genFetch("/v1/gen/models")).json()).data || [];
      showGenError("");
    } catch (e) {
      showGenError("Could not load the model list: " + e.message);   // not cached: retried when the server is back
    }
  }
  const sel = $("#genModel");
  const ms = (G.models || []).filter((m) => m.task === G.task);
  const want = store.get("gen.model." + G.task, ms[0] && ms[0].id);
  sel.innerHTML = ms.map((m) => `<option value="${esc(m.id)}"${m.id === want ? " selected" : ""}>${esc(m.id)}  (${m.ram_gb} GB RAM${m.downloaded ? "" : ", downloads on first use"})</option>`).join("")
    || `<option value="">${G.models ? "No " + TASK_LABEL[G.task].toLowerCase() + " models" : "Model list unavailable"}</option>`;
  modelNote();
  $("#genGo").disabled = !G.busy && !ms.length;       // Cancel stays usable while a job runs
  renderResults();
}

function modelNote() {
  const m = (G.models || []).find((x) => x.id === $("#genModel").value);
  $("#genModelNote").textContent = m ? m.note : "";
}

function showGenError(msg) {
  const box = $("#genError");
  box.hidden = !msg;
  box.textContent = msg || "";
}

const MAX_RESULTS = 12;
function cardHtml(r) {
    if (r.pending) {
      return `<article class="gen-card"><div class="skeleton gen-skel gen-skel-${r.task}"></div>
        <div class="gen-progress" role="progressbar" aria-label="Generation progress" aria-valuemin="0" aria-valuemax="100"><span></span></div>
        <div class="gen-meta"><span>${esc(r.model)}</span><span class="gen-phase">Starting</span></div></article>`;
    }
    const media = r.task === "image" ? `<img src="${r.url}" alt="${esc(r.prompt)}">`
      : r.task === "video" ? `<video src="${r.url}" controls loop playsinline></video>`
      : `<audio src="${r.url}" controls></audio>`;
    return `<article class="gen-card">${media}
      <p class="gen-prompt">${esc(r.prompt)}</p>
      <div class="gen-meta"><span>${esc(r.model)}</span><span class="num">${r.seconds != null ? r.seconds.toFixed(1) + " s" : ""}</span>
        <a class="btn" href="${r.url}" download="${esc(r.file)}"><span class="ico i-download-simple"></span>Download</a></div></article>`;
}
// Cards are added / replaced one at a time so a playing video or voice clip is not restarted.
function renderResults() {
  const root = $("#genResults");
  const rs = G.results.filter((r) => r.task === G.task);
  if (!rs.length) {
    root.innerHTML = `<p class="muted small gen-empty">Results appear here. They stay until you reload the page.</p>`;
    return;
  }
  $(".gen-empty", root)?.remove();
  const want = new Set(rs);
  [...root.children].forEach((el) => { if (!want.has(el._r)) el.remove(); });
  rs.forEach((r, i) => {
    let el = [...root.children].find((x) => x._r === r);
    if (!el || el._pending !== !!r.pending) {
      const t = document.createElement("template");
      t.innerHTML = cardHtml(r).trim();
      const fresh = t.content.firstElementChild;
      fresh._r = r; fresh._pending = !!r.pending;
      if (el) el.replaceWith(fresh); else root.insertBefore(fresh, root.children[i] || null);
      el = fresh;
    }
    if (root.children[i] !== el) root.insertBefore(el, root.children[i] || null);
  });
}
function dropOld() {
  const done = G.results.filter((r) => !r.pending);
  for (const r of done.slice(MAX_RESULTS)) { URL.revokeObjectURL(r.url); G.results.splice(G.results.indexOf(r), 1); }
}

async function b64Url(b64, mime) {
  // the browser decodes a data: URL natively, without a JS string and byte-array copy of a big video
  return URL.createObjectURL(await (await fetch(`data:${mime};base64,${b64}`)).blob());
}

function num(id) {
  const v = $(id).value.trim();
  return v === "" ? undefined : Number(v);
}

async function generate(ev) {
  ev.preventDefault();
  if (G.busy) return;
  const form = $("#genForm");
  if (!form.checkValidity()) { form.reportValidity(); return; }
  const prompt = $("#genPrompt").value.trim();
  const model = $("#genModel").value;
  if (!model) { showGenError(G.models ? "Pick a model first." : "The model list is still loading."); return; }
  if (!prompt) { showGenError(G.task === "speech" ? "Type what the voice should say." : "Describe what to create."); $("#genPrompt").focus(); return; }
  showGenError("");
  const task = G.task;
  const entry = { task, model, prompt, pending: true };
  G.results.unshift(entry);
  G.busy = true;
  setGo(true);
  renderResults();
  const t0 = performance.now();
  const stopWatch = watchProgress(entry, t0);
  try {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    if (task === "speech") {
      const fmt = $("#genFormat").value;
      const r = await genFetch(TASK_PATH.speech, { model, input: prompt, response_format: fmt });
      entry.url = URL.createObjectURL(await r.blob());
      entry.file = `speech-${stamp}.${fmt}`;
    } else {
      const body = { model, prompt, steps: num("#genSteps"), seed: num("#genSeed") };
      if (task === "image") body.size = $("#genSize").value;
      else { body.frames = num("#genFrames"); }
      const j = await (await genFetch(TASK_PATH[task], body)).json();
      const d = j.data[0];
      entry.url = await b64Url(d.b64_json, task === "image" ? "image/png" : "video/mp4");
      entry.file = `${task}-${stamp}.${task === "image" ? "png" : "mp4"}`;
      const t = Array.isArray(j.timings) ? j.timings[0] : j.timings;
      entry.seconds = t && t.seconds;
    }
    if (entry.seconds == null) entry.seconds = (performance.now() - t0) / 1000;
    entry.pending = false;
    const m = (G.models || []).find((x) => x.id === model);
    if (m) m.downloaded = true;
    dropOld();
  } catch (e) {
    G.results = G.results.filter((r) => r !== entry);
    if (e.cancelled) toast("Cancelled");
    else showGenError(e.message);
  } finally {
    stopWatch();
    G.busy = false;
    setGo(false);
    renderResults();
  }
}

// Generate <-> Cancel on the same button while a job runs
function setGo(busy) {
  const b = $("#genGo");
  b.disabled = false;
  b.classList.toggle("primary", !busy);
  b.innerHTML = busy ? '<span class="ico i-stop"></span>Cancel'
    : '<span class="ico i-sparkle"></span>Generate';
  b.type = busy ? "button" : "submit";
}
async function cancelGen() {
  const b = $("#genGo");
  b.disabled = true;
  b.innerHTML = '<span class="ico i-stop"></span>Cancelling';
  try { await fetch("/v1/gen/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); }
  catch { /* the request itself will report */ }
}
// Poll /v1/gen/progress (no lock on the server) and show phase, step k of N, elapsed time.
function watchProgress(entry, t0) {
  let alive = true, timer = 0;
  const tick = async () => {
    if (!alive) return;
    try {
      const p = await (await fetch("/v1/gen/progress", { signal: AbortSignal.timeout(3000) })).json();
      const card = [...$("#genResults").children].find((el) => el._r === entry);
      if (card && alive) {
        const secs = Math.round((performance.now() - t0) / 1000);
        const el = Math.floor(secs / 60) + ":" + String(secs % 60).padStart(2, "0");
        const pct = p.steps ? Math.min(100, Math.round((100 * p.step) / p.steps)) : 0;
        const phase = p.phase === "loading" ? "Loading the model" : p.phase === "generating" && p.steps > 1
          ? `Step ${Math.min(p.step, p.steps)} of ${p.steps}` : "Generating on the NPU";
        $(".gen-phase", card).textContent = `${phase}, ${el}`;
        const bar = $(".gen-progress", card);
        bar.setAttribute("aria-valuenow", String(pct));
        $("span", bar).style.width = pct + "%";
      }
    } catch { /* keep polling */ }
    if (alive) timer = setTimeout(tick, 1000);
  };
  tick();
  return () => { alive = false; clearTimeout(timer); };
}

function initCreate() {
  $$("#genTabs button").forEach((b) => (b.onclick = () => { G.task = b.dataset.task; store.set("gen.task", G.task); showGenError(""); renderCreate(); }));
  $("#genModel").onchange = () => { store.set("gen.model." + G.task, $("#genModel").value); modelNote(); };
  $("#genForm").onsubmit = generate;
  $("#genGo").addEventListener("click", (e) => { if (G.busy) { e.preventDefault(); cancelGen(); } });
  $("#genPrompt").onkeydown = (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#genForm").requestSubmit(); } };
  // integers only, in range: the browser enforces these on submit
  $("#genSteps").step = "1"; $("#genSeed").step = "1"; $("#genSeed").max = "4294967295";
}
// called by app.js when the server becomes reachable (again)
function genServerUp() {
  if (G.models && G.models.length) return;
  G.models = null;
  if (!$("#view-create").hidden) renderCreate();
}
initCreate();
