// OpenNPU web UI, end to end. Fake engines script replies by keyword (see tests/qa/fakeserver.py).
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const errors = [];
test.beforeEach(async ({ page }) => {
  errors.length = 0;
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });
  // every test starts with no saved sessions on the (fake) server and an empty browser store
  const list = await (await page.request.get("/v1/sessions")).json();
  for (const x of list.data || []) await page.request.delete(`/v1/sessions/${x.id}`);
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.locator("#state")).toHaveAttribute("data-s", /idle|busy/);
});
test.afterEach(async () => { expect(errors, "no JS errors").toEqual([]); });

async function ask(page, text) {
  const n = await page.locator("#thread .msg.assistant").count();
  await page.fill("#input", text);
  await page.keyboard.press("Enter");
  // send() may first wait for the model list; only then does the reply (and the Stop state) exist
  await expect(page.locator("#thread .msg.assistant")).toHaveCount(n + 1);
}
const lastReply = (page) => page.locator("#thread .msg.assistant").last();
const done = (page) => expect(page.locator("#send")).toHaveAttribute("aria-label", "Send", { timeout: 20000 });

// ---------------------------------------------------------------- shell
test("loads every view, keeps the selected view on reload", async ({ page }) => {
  for (const v of ["create", "models", "discover", "system", "chat"]) {
    await page.click(`.nav-item[data-view="${v}"]`);
    await expect(page.locator(`#view-${v}`)).toBeVisible();
    await expect(page.locator(`.nav-item[data-view="${v}"]`)).toHaveAttribute("aria-current", "page");
  }
  await page.click('.nav-item[data-view="create"]');
  await page.reload();
  await expect(page.locator("#view-create")).toBeVisible();
  await expect(page.locator("#genModel option").first()).not.toHaveText("");     // Create renders after reload
  await expect(page.locator("#genHint")).not.toBeEmpty();
});

test("theme cycles system, light, dark and persists", async ({ page }) => {
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  await page.emulateMedia({ colorScheme: "dark" });
  expect(await bg()).toBe("rgb(15, 17, 19)");                                   // system = dark
  await page.click("#themeBtn");                                                 // light
  expect(await bg()).toBe("rgb(246, 247, 248)");
  await expect(page.locator("#themeBtn")).toHaveAttribute("aria-label", "Theme: light");
  await page.click("#themeBtn");                                                 // dark
  expect(await bg()).toBe("rgb(15, 17, 19)");
  await page.emulateMedia({ colorScheme: "light" });
  await page.reload();
  expect(await bg()).toBe("rgb(15, 17, 19)");                                   // forced dark survives
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme)).toBe("dark");
  await page.click("#themeBtn");                                                 // back to system (light)
  expect(await bg()).toBe("rgb(246, 247, 248)");
});

// ---------------------------------------------------------------- chat
test("streams a reply with stats, titles the conversation", async ({ page }) => {
  await ask(page, "hello there npu");
  await done(page);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo: hello there npu");
  await expect(lastReply(page).locator(".meta")).toContainText("tok/s");
  await expect(page.locator("#convos li.active .c-open")).toHaveText("hello there npu");
  await expect(page.locator("#lTok")).not.toHaveText("-");
});

test("markdown renders headings, lists, code, and keeps code literal", async ({ page }) => {
  await ask(page, "__md__");
  await done(page);
  const a = lastReply(page).locator(".answer");
  await expect(a.locator("h3")).toHaveText("Title");
  await expect(a.locator("ul li")).toHaveCount(2);
  await expect(a.locator("ol li")).toHaveCount(2);
  await expect(a.locator("pre code")).toHaveText("print('hi')");
  await expect(a.locator("strong")).toHaveText("bold");
});

test("model output cannot inject script or markup", async ({ page }) => {
  await ask(page, "__xss__");
  await done(page);
  expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
  await expect(page.locator("#thread script, #thread img, #thread a[href^='javascript']")).toHaveCount(0);
  await expect(lastReply(page)).toContainText("<script>window.__pwned=1</script>");
});

test("reasoning goes into a collapsible block, copy excludes it", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await ask(page, "__think__");
  await done(page);
  await expect(lastReply(page).locator("details.reasoning summary")).toHaveText("Reasoning");
  await expect(lastReply(page).locator(".answer")).toHaveText("The answer is 42.");
  await lastReply(page).locator(".copy-msg").click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("The answer is 42.");
});

test("stop ends the reply, keeps the partial text, next send works", async ({ page }) => {
  await ask(page, "__slow__");
  await expect(lastReply(page).locator(".answer")).toContainText("s3", { timeout: 10000 });
  await expect(page.locator("#send")).toHaveAttribute("aria-label", "Stop generating");
  await page.keyboard.press("Enter");                                   // Enter must NOT stop it
  await expect(page.locator("#send")).toHaveAttribute("aria-label", "Stop generating");
  await page.click("#send");
  await done(page);
  await expect(lastReply(page).locator(".reply-note")).toHaveText("Stopped.");
  await expect(lastReply(page).locator(".answer")).toContainText("s0 s1");
  await ask(page, "after stop");
  await done(page);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo: after stop");
});

test("an engine error is shown, then the next turn works and skips the failed turn", async ({ page }) => {
  await ask(page, "__error0__");
  await done(page);
  await expect(lastReply(page).locator(".reply-note.err")).toContainText("Could not get a reply");
  const sent = page.waitForRequest((r) => r.url().endsWith("/v1/chat/completions"));
  await ask(page, "second try");
  const body = JSON.parse((await sent).postData());
  expect(body.messages).toEqual([{ role: "user", content: "second try" }]);
  await done(page);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo: second try");
});

test("a mid-stream error keeps the partial reply", async ({ page }) => {
  await ask(page, "__error__ a b c");
  await done(page);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo:");
  await expect(lastReply(page).locator(".reply-note.err")).toContainText("cut off");
});

test("switching conversations mid-stream: reply lands in the right one, still live on return", async ({ page }) => {
  await ask(page, "__slow__");
  await expect(lastReply(page).locator(".answer")).toContainText("s2", { timeout: 10000 });
  await page.click("#newChat");
  await expect(page.locator("#send")).toBeDisabled();                    // the server runs one reply at a time
  await expect(page.locator("#hint")).toHaveText("Another conversation is still generating");
  await page.locator("#convos li .c-open").nth(1).click();               // back to the streaming one
  const before = await lastReply(page).locator(".answer").innerText();
  await page.waitForTimeout(600);
  expect((await lastReply(page).locator(".answer").innerText()).length).toBeGreaterThan(before.length);
  await page.click("#send");                                             // stop
  await done(page);
});

test("deleting the streaming conversation stops it, undo restores it", async ({ page }) => {
  await ask(page, "__slow__");
  await expect(lastReply(page).locator(".answer")).toContainText("s1", { timeout: 10000 });
  await page.locator("#convos li.active .c-del").click();
  await expect(page.locator("#convos li")).toHaveCount(0);
  await page.click(".toast-act");
  await expect(page.locator("#convos li")).toHaveCount(1);
  await expect(lastReply(page).locator(".reply-note")).toHaveText("Stopped.");
  await expect(page.locator("#send")).toHaveAttribute("aria-label", "Send");
});

test("conversations persist across reload and survive corrupt storage", async ({ page }) => {
  await ask(page, "remember me");
  await done(page);
  await page.reload();
  await expect(page.locator("#convos .c-open")).toHaveText(["remember me"]);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo: remember me");
  for (const bad of ["{not json", "null", '{"a":1}', '[1, {"id": 5}, {"id": "x"}]']) {
    await page.evaluate((v) => localStorage.setItem("opennpu.convos", v), bad);
    await page.reload();
    await expect(page.locator("#thread .empty h1")).toBeVisible();
  }
});

test("server down: clear state, then recovers by itself", async ({ page }) => {
  await page.route("**/api/stats", (r) => r.abort());
  await page.route("**/v1/models", (r) => r.abort());
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.locator("#stateText")).toHaveText("Server not reachable", { timeout: 10000 });
  await expect(page.locator("#thread .empty h1")).toHaveText("Can't reach the OpenNPU server");
  await page.unroute("**/api/stats");
  await page.unroute("**/v1/models");
  await expect(page.locator("#stateText")).not.toHaveText("Server not reachable", { timeout: 10000 });
  await expect(page.locator("#modelSel option")).not.toHaveCount(0);
  await expect(page.locator("#thread .empty h1")).toHaveText("What should the NPU work on?");
});

test("long reply: the user can scroll up while it streams", async ({ page }) => {
  await ask(page, "__slow__");
  await expect(lastReply(page).locator(".answer")).toContainText("s40", { timeout: 15000 });
  const th = page.locator("#thread");
  await th.evaluate((el) => (el.scrollTop = 0));
  await page.waitForTimeout(500);
  expect(await th.evaluate((el) => el.scrollTop)).toBe(0);
  await page.click("#send");
  await done(page);
});

// ---------------------------------------------------------------- models, system
test("models page groups engines and starts a chat with the chosen model", async ({ page }) => {
  await page.click('.nav-item[data-view="models"]');
  await expect(page.locator("#modelGroups .group")).toHaveCount(3);
  await page.locator('[data-chat="native:Fake-Mistral-Nemo-12B"]').click();
  await expect(page.locator("#view-chat")).toBeVisible();
  await expect(page.locator("#modelSel")).toHaveValue("native:Fake-Mistral-Nemo-12B");
  await ask(page, "which model");
  await done(page);
  await expect(lastReply(page).locator(".meta")).toContainText("OpenNPU fused engine");
});

test("system page shows device and connection details", async ({ page }) => {
  await page.click('.nav-item[data-view="system"]');
  await expect(page.locator("#devKv")).toContainText("RyzenAI-npu6");
  await expect(page.locator("#devKv")).toContainText("v0.");
  await expect(page.locator("#oaiUrl")).toHaveText(/\/v1$/);
});

// ---------------------------------------------------------------- create
test("create: image result with download, voice result, errors inline", async ({ page }) => {
  await page.click('.nav-item[data-view="create"]');
  await expect(page.locator("#genGo")).toBeEnabled();                    // enabled once the model list is in
  await page.click("#genGo");                                            // empty prompt: told what to do
  await expect(page.locator("#genError")).toHaveText("Describe what to create.");
  await page.fill("#genPrompt", "a fox");
  await page.selectOption("#genSize", "512x512");
  await page.click("#genGo");
  await expect(page.locator("#genResults img")).toHaveCount(1, { timeout: 15000 });
  await expect(page.locator("#genResults a[download]")).toHaveAttribute("download", /^image-.*\.png$/);
  await page.fill("#genSteps", "500");                                   // out of range: browser blocks it
  await page.locator("#genPrompt").press("Control+Enter");
  await expect(page.locator("#genResults img")).toHaveCount(1);
  await page.fill("#genSteps", "");
  await page.route("**/v1/images/generations", (r) => r.fulfill({ status: 501, contentType: "application/json",
    body: JSON.stringify({ error: { message: "generation support is not installed (diffusers missing): run ./install.sh --gen" } }) }));
  await page.click("#genGo");
  await expect(page.locator("#genError")).toContainText("./install.sh --gen");
  await page.click('#genTabs button[data-task="speech"]');
  await page.fill("#genPrompt", "hello");
  await page.selectOption("#genFormat", "wav");
  await page.click("#genGo");
  await expect(page.locator("#genResults audio")).toHaveCount(1, { timeout: 15000 });
});

test("create: progress shows steps, Cancel stops the job, then the next job runs", async ({ page }) => {
  await page.click('.nav-item[data-view="create"]');
  await expect(page.locator("#genGo")).toBeEnabled();
  await page.fill("#genPrompt", "__slow__ fox");
  await page.click("#genGo");
  await expect(page.locator("#genGo")).toHaveText("Cancel");
  await expect(page.locator(".gen-phase")).toContainText(/Step \d+ of 20/, { timeout: 8000 });
  await page.click('#genTabs button[data-task="speech"]');               // switching tabs keeps Cancel usable
  await page.click('#genTabs button[data-task="image"]');
  await expect(page.locator("#genGo")).toBeEnabled();
  await page.click("#genGo");                                             // Cancel
  await expect(page.locator("#genGo")).toHaveText("Generate", { timeout: 8000 });
  await expect(page.locator(".toast")).toHaveText("Cancelled");
  await expect(page.locator("#genError")).toBeHidden();
  await expect(page.locator("#genResults img")).toHaveCount(0);
  await page.fill("#genPrompt", "a quick fox");
  await page.click("#genGo");
  await expect(page.locator("#genResults img")).toHaveCount(1, { timeout: 10000 });
});

test("create: prompt length is capped at the server's limit", async ({ page }) => {
  await page.click('.nav-item[data-view="create"]');
  expect(await page.locator("#genPrompt").getAttribute("maxlength")).toBe("2000");
  await page.click('#genTabs button[data-task="speech"]');
  expect(await page.locator("#genPrompt").getAttribute("maxlength")).toBe("4096");
});

// ---------------------------------------------------------------- mobile
test("mobile: menu opens on every page, closes with Escape and backdrop, no sideways scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const v of ["chat", "create", "models", "system"]) {
    await page.locator(`#view-${v === "chat" ? "chat" : v} .rail-toggle`).first().click().catch(async () => {
      await page.locator(".rail-toggle:visible").first().click();
    });
    await expect(page.locator("#rail")).toHaveClass(/open/);
    await page.click(`.nav-item[data-view="${v === "chat" ? "create" : v === "create" ? "models" : v === "models" ? "system" : "chat"}"]`);
    await expect(page.locator("#rail")).not.toHaveClass(/open/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  }
  await page.locator(".rail-toggle:visible").first().click();
  await page.keyboard.press("Escape");
  await expect(page.locator("#rail")).not.toHaveClass(/open/);
  await page.locator(".rail-toggle:visible").first().click();
  await page.mouse.click(370, 400);                                     // backdrop
  await expect(page.locator("#rail")).not.toHaveClass(/open/);
});

test("tablet widths: create page does not overflow", async ({ page }) => {
  for (const w of [768, 820, 900, 1024]) {
    await page.setViewportSize({ width: w, height: 1000 });
    await page.click('.nav-item[data-view="create"]').catch(() => {});
    await page.goto("/"); await page.evaluate(() => localStorage.setItem("opennpu.view", '"create"')); await page.reload();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `width ${w}`).toBe(true);
  }
});

// ---------------------------------------------------------------- accessibility + copy rules
for (const scheme of ["light", "dark"]) {
  test(`axe: no serious accessibility violations (${scheme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await ask(page, "__md__");
    await done(page);
    for (const v of ["chat", "create", "models", "system", "dashboard", "discover"]) {
      await page.click(`.nav-item[data-view="${v}"]`);
      await page.waitForTimeout(600);
      const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
      const bad = r.violations.filter((x) => ["serious", "critical"].includes(x.impact));
      expect(bad.map((x) => `${v}: ${x.id} ${x.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(", ")}`)).toEqual([]);
    }
  });
}

test("keyboard: skip link and visible focus", async ({ page }) => {
  await page.evaluate(() => localStorage.setItem("opennpu.view", '"models"'));   // a page that does not autofocus
  await page.reload();
  await page.keyboard.press("Tab");
  await expect(page.locator(".skip")).toBeFocused();
  await expect(page.locator(".skip")).toBeInViewport();                   // visible when focused
  await page.click('.nav-item[data-view="chat"]');
  await expect(page.locator("#input")).toBeFocused();
});

test("no em or en dashes in any visible text", async ({ page }) => {
  for (const v of ["chat", "create", "models", "system", "dashboard", "discover"]) {
    await page.click(`.nav-item[data-view="${v}"]`);
    await page.waitForTimeout(300);
    const t = await page.evaluate(() => document.body.innerText + [...document.querySelectorAll("[title],[aria-label],[placeholder]")]
      .map((e) => (e.title || "") + (e.getAttribute("aria-label") || "") + (e.placeholder || "")).join(" "));
    expect(t).not.toMatch(/[–—]/);
  }
});


// ---------------------------------------------------------------- dashboard (local analytics)
test("dashboard: usage reflects what was just done, tables, ranges, CSV", async ({ page }) => {
  await page.request.delete("/api/usage?what=usage");
  await ask(page, "count me please");
  await done(page);
  await page.click('.nav-item[data-view="dashboard"]');
  await expect(page.locator(".tile").first()).toContainText("Requests");
  await expect(page.locator(".tile-value").first()).toHaveText("1");
  await expect(page.locator(".hbars").first()).toContainText("Fake-Qwen3");
  await expect(page.locator("#dashUsage")).toContainText("Web UI");                  // the app that asked
  await page.locator(".chart-card", { hasText: "By model" }).locator(".link-btn").click();
  await expect(page.locator(".chart-card", { hasText: "By model" }).locator("table.data")).toContainText("native:Fake-Qwen3-8B");
  await page.click('#dashRange [data-r="30d"]');
  await expect(page.locator('#dashRange [data-r="30d"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".chart-card", { hasText: "Over time" }).locator("svg .bar")).not.toHaveCount(0);
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#dashCsv")]);
  expect(dl.suggestedFilename()).toBe("opennpu-usage.csv");
  expect(await page.textContent("#dashUsage")).not.toContain("count me please");   // no content, ever
});

test("dashboard: recording toggle and clear history", async ({ page }) => {
  await page.click('.nav-item[data-view="dashboard"]');
  await expect(page.locator("#dashRecord")).toBeChecked();
  await page.uncheck("#dashRecord");
  await expect(page.locator(".toast")).toHaveText("Recording paused");
  expect((await (await page.request.get("/api/usage")).json()).enabled).toBe(false);
  await page.check("#dashRecord");
  page.once("dialog", (d) => d.accept());
  await page.click("#dashClear");
  await expect(page.locator(".toast")).toHaveText("Usage history deleted");
  await expect(page.locator("#dashUsage h2")).toHaveText("No usage in this range yet");
});

test("dashboard: hardware tab shows machine info and history charts with tooltips", async ({ page }) => {
  await page.click('.nav-item[data-view="dashboard"]');
  await page.click('#dashTabs [data-tab="hardware"]');
  await expect(page.locator(".hw-info")).toContainText("RyzenAI-npu6");
  await expect(page.locator("#dashCsv")).toBeHidden();
  await expect(page.locator("#hwCharts .chart-card")).toHaveCount(6);
  const first = page.locator("#hwCharts .chart-card").first();
  await expect(first.locator("svg path.line")).toHaveCount(1);
  await first.locator("svg .hit").hover({ position: { x: 120, y: 30 } });
  await expect(page.locator(".chart-tip")).toBeVisible();
  await expect(page.locator(".chart-tip")).toContainText("%");
});


// ---------------------------------------------------------------- sessions, vision, voice
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "base64");

test("sessions are saved on the server, survive a cleared browser, reuse the cache", async ({ page }) => {
  await ask(page, "first message for the session");
  await done(page);
  await ask(page, "and a follow up");
  await done(page);
  await expect(lastReply(page).locator(".meta")).toContainText("reused");
  const list = await (await page.request.get("/v1/sessions")).json();
  expect(list.data.length).toBe(1);
  expect(list.data[0].title).toBe("first message for the session");
  await page.evaluate(() => localStorage.clear());                // nothing in the browser
  await page.reload();
  await expect(page.locator("#convos .c-open")).toHaveText(["first message for the session"]);
  await page.locator("#convos .c-open").first().click();
  await expect(page.locator("#thread .msg")).toHaveCount(4);
});

test("a session remembers its model", async ({ page }) => {
  await page.selectOption("#modelSel", "native:Fake-Mistral-Nemo-12B");
  await ask(page, "on the flm model");
  await done(page);
  await page.click("#newChat");
  await page.selectOption("#modelSel", "native:Fake-Qwen3-8B");
  await ask(page, "on the native model");
  await done(page);
  await page.locator("#convos .c-open", { hasText: "on the flm model" }).click();
  await expect(page.locator("#modelSel")).toHaveValue("native:Fake-Mistral-Nemo-12B");
});

test("old browser conversations are imported once", async ({ page }) => {
  await page.evaluate(() => {
    localStorage.removeItem("opennpu.sessions.imported");
    localStorage.setItem("opennpu.convos", JSON.stringify([{ id: "legacy-1", title: "From before", messages: [
      { role: "user", content: "hi" }, { role: "assistant", content: "hello" }] }]));
  });
  await page.reload();
  await expect(page.locator("#convos .c-open")).toContainText(["From before"]);
  // the list shows the browser's copy at once; the import to the server runs in the background
  await expect.poll(async () => (await (await page.request.get("/v1/sessions/legacy-1")).json()).title, { timeout: 10000 }).toBe("From before");
  await expect.poll(() => page.evaluate(() => localStorage.getItem("opennpu.convos"))).toBeNull();   // moved, backup kept
  expect(await page.evaluate(() => localStorage.getItem("opennpu.convos.backup"))).toContain("From before");
});

test("vision: attach an image, it is shown and answered; text models refuse with a hint", async ({ page }) => {
  await page.setInputFiles("#fileIn", { name: "dot.png", mimeType: "image/png", buffer: PNG });
  await expect(page.locator("#attachments img")).toHaveCount(1);
  await page.fill("#input", "what is this?");
  await page.keyboard.press("Enter");
  await expect(page.locator(".toast")).toContainText("can't see images");          // default model is text-only
  await page.selectOption("#modelSel", "universal:org/fake-vlm");
  await ask(page, "what is this?");
  await done(page);
  await expect(page.locator("#thread .msg.user .msg-images img")).toHaveCount(1);
  await expect(lastReply(page).locator(".answer")).toContainText("I see 1 image");
  await expect(page.locator("#attachments")).toBeHidden();
  const s = (await (await page.request.get("/v1/sessions")).json()).data[0];
  const full = await (await page.request.get(`/v1/sessions/${s.id}`)).json();
  expect(full.messages[0].images[0]).toMatch(/^\/v1\/sessions\/media\//);       // stored as a file, not inline
});

test("voice: the mic records, transcribes on the server, and fills the message box", async ({ page }) => {
  const log = [];
  page.on("console", (m) => log.push(`console ${m.type()}: ${m.text()}`));
  page.on("request", (r) => r.url().includes("/audio/") && log.push(`request ${r.method()} ${r.url()}`));
  await page.click("#micBtn");
  await expect(page.locator("#micBtn")).toHaveClass(/rec/);
  await page.waitForTimeout(2500);                 // slow CI machines start the fake mic late
  await page.click("#micBtn");
  // on failure, say what the page showed (toast, console, requests) instead of only "empty input"
  await expect(async () => {
    const v = await page.locator("#input").inputValue();
    const toasts = await page.locator(".toast").allTextContents();
    expect(v, `toasts: ${JSON.stringify(toasts)}; ${log.join(" | ")}`).toBe("hello from the microphone");
  }).toPass({ timeout: 15000 });
});

test("read aloud plays the reply", async ({ page }) => {
  await ask(page, "say something");
  await done(page);
  const req = page.waitForRequest((r) => r.url().endsWith("/v1/audio/speech"));
  await lastReply(page).locator(".speak-msg").click();
  expect(JSON.parse((await req).postData()).input).toContain("Echo: say something");
});


// ---------------------------------------------------------------- discover (Hugging Face through the server; fake Hub)
const row = (page, id) => page.locator(`.hub-row[data-id="${id}"]`);
async function resetHub(page, id) {                     // not downloaded, no job: every test starts clean
  await page.request.delete(`/api/hub/model?id=${id}`);
  for (let i = 0; i < 2; i++) await page.request.post("/api/hub/cancel", { data: { id } });
}
async function discover(page, q, task) {
  await page.click('.nav-item[data-view="discover"]');
  if (task) await page.click(`#hubTasks button[data-task="${task}"]`);
  await page.fill("#hubQ", q);
  await page.press("#hubQ", "Enter");                   // search now (skips the debounce): no late re-render
  await expect(page.locator("#hubResults")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#hubCount")).toHaveText(/found/);
}
async function pullDone(page, id) {
  await page.request.post("/api/hub/pull", { data: { id } });
  await expect.poll(async () => ((await (await page.request.get("/api/hub/pulls")).json()).data
    .find((j) => j.id === id) || {}).state, { timeout: 20000 }).toBe("done");
}

test("discover: search with task chips and sort shows engine, size, gated and RAM warnings", async ({ page }) => {
  await page.click('.nav-item[data-view="discover"]');
  await expect(page.locator("#hubTasks button[data-task=chat]")).toHaveAttribute("aria-pressed", "true");
  const q = row(page, "Qwen/Qwen3-0.6B");
  await expect(q).toBeVisible();                                              // opens with the most downloaded chat models
  await expect(q.locator(".badge.fast")).toHaveText("Fused NPU engine (fastest)");
  await expect(q.locator(".hub-facts")).toContainText("752M");
  await expect(q.locator(".hub-facts")).toContainText("9.1M");
  await expect(row(page, "Qwen/Qwen3-30B-A3B").locator(".badge.fast")).toHaveText("NPU MoE engine");
  await expect(row(page, "Qwen/Qwen3-VL-2B-Instruct")).toHaveCount(0);       // a vision model is not a chat chip hit
  await expect(row(page, "meta-llama/Llama-3.2-1B-Instruct").locator(".badge.gated")).toHaveText("Gated");
  await expect(row(page, "unsloth/gpt-oss-20b-GGUF").locator(".hub-facts")).toContainText("20.9B");
  await expect(row(page, "deepseek-ai/DeepSeek-V3").locator(".hub-warn")).toContainText(/needs ~\d+ GB RAM, this machine has \d+ GB/);
  await expect(row(page, "Qwen/Qwen3-0.6B").locator(".hub-warn")).toHaveCount(0);
  await page.click("#hubTasks button[data-task=image]");
  await expect(row(page, "stabilityai/sd-turbo")).toBeVisible();
  await expect(row(page, "Qwen/Qwen3-0.6B")).toHaveCount(0);
  await page.click("#hubTasks button[data-task=all]");
  await page.selectOption("#hubSort", "likes");
  await expect(page.locator(".hub-row").first()).toHaveAttribute("data-id", "deepseek-ai/DeepSeek-V3");   // 3900 likes
  await discover(page, "no-such-model-anywhere");
  await expect(page.locator(".hub-empty")).toContainText("No models match");
  await page.reload();                                                        // task and sort are remembered
  await page.click('.nav-item[data-view="discover"]');
  await expect(page.locator("#hubTasks button[data-task=all]")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#hubSort")).toHaveValue("likes");
});

test("discover: details, download with progress, then Chat now selects the model and it answers", async ({ page }) => {
  test.setTimeout(60000);
  await resetHub(page, "Qwen/Qwen3-0.6B");
  await discover(page, "qwen3-0.6b", "chat");
  const r = row(page, "Qwen/Qwen3-0.6B");
  await r.getByRole("button", { name: "Download Qwen/Qwen3-0.6B" }).click();
  const det = r.locator(".hub-detail");
  await expect(det).toBeVisible();
  await expect(det.locator(".hub-kv")).toContainText("Fused NPU engine (fastest)");
  await expect(det.locator(".hub-kv")).toContainText("License");
  await expect(det.locator(".hub-kv")).toContainText("apache-2.0");
  await expect(det.locator(".hub-kv")).toContainText("Free disk");
  await expect(det.locator("[data-hub-pull]")).toHaveText("Download 27 MB");
  await det.locator("[data-hub-pull]").click();
  const bar = r.getByRole("progressbar", { name: "Download progress for Qwen/Qwen3-0.6B" });
  await expect(bar).toBeVisible();
  await expect(r.locator(".hub-prog-row")).toContainText(/of 27 MB/);
  await expect(r.getByRole("button", { name: "Cancel download of Qwen/Qwen3-0.6B" })).toBeVisible();
  const chatNow = r.getByRole("button", { name: "Chat now with Qwen/Qwen3-0.6B" });
  await expect(chatNow).toBeVisible({ timeout: 20000 });
  await expect(r.locator(".badge.ok")).toHaveText("Installed");
  await chatNow.click();
  await expect(page.locator("#view-chat")).toBeVisible({ timeout: 15000 });
  await expect(page.locator("#modelSel")).toHaveValue("native:Qwen3-0.6B");
  await ask(page, "hi from discover");
  await done(page);
  await expect(lastReply(page).locator(".answer")).toHaveText("Echo: hi from discover");
  await resetHub(page, "Qwen/Qwen3-0.6B");
});

test("discover: GGUF variant picker sets the download size and the file fetched", async ({ page }) => {
  await resetHub(page, "unsloth/gpt-oss-20b-GGUF");
  await discover(page, "gpt-oss", "chat");
  const r = row(page, "unsloth/gpt-oss-20b-GGUF");
  await r.locator("[data-hub-details]").click();
  const sel = r.getByLabel("GGUF variant");
  await expect(sel).toHaveValue("gpt-oss-20b-Q4_K_M.gguf");                // Q4_K_M by default
  await expect(sel.locator("option")).toHaveCount(3);
  await expect(r.locator("[data-hub-pull]")).toHaveText("Download 13 MB");
  await sel.selectOption("gpt-oss-20b-Q8_0.gguf");
  await expect(r.locator("[data-hub-pull]")).toHaveText("Download 17 MB");
  await r.locator("[data-hub-pull]").click();
  await expect(r.getByRole("button", { name: /Chat now with/ })).toBeVisible({ timeout: 20000 });
  const job = (await (await page.request.get("/api/hub/pulls")).json()).data.find((j) => j.id === "unsloth/gpt-oss-20b-GGUF");
  expect(job.total).toBe(16 * 2 ** 20);                                      // the Q8_0 file, not the default
  expect(job.model).toBe("universal:unsloth/gpt-oss-20b-GGUF");
  await r.locator("[data-hub-close]").count().then((n) => expect(n).toBe(0)); // the details panel closed
  await resetHub(page, "unsloth/gpt-oss-20b-GGUF");
});

test("discover: cancel stops a download; a failed download says why and offers Retry", async ({ page }) => {
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
  await resetHub(page, "opennpu-test/Broken-Qwen3-4B");
  await discover(page, "opennpu-test", "chat");
  const r = row(page, "opennpu-test/Slow-Qwen3-1.7B");
  await r.locator("[data-hub-details]").click();
  await r.locator("[data-hub-pull]").click();
  await expect(r.getByRole("progressbar")).toBeVisible();
  await expect.poll(async () => +(await r.getByRole("progressbar").getAttribute("aria-valuenow")), { timeout: 10000 }).toBeGreaterThan(0);
  await r.getByRole("button", { name: "Cancel download of opennpu-test/Slow-Qwen3-1.7B" }).click();
  await expect(r.getByRole("progressbar")).toHaveCount(0);
  await expect(r.locator(".hub-act")).toContainText("Cancelled");
  await expect(r.getByRole("button", { name: "Download opennpu-test/Slow-Qwen3-1.7B" })).toBeVisible();
  const job = (await (await page.request.get("/api/hub/pulls")).json()).data.find((j) => j.id === "opennpu-test/Slow-Qwen3-1.7B");
  expect(job.state).toBe("cancelled");
  const b = row(page, "opennpu-test/Broken-Qwen3-4B");
  await b.locator("[data-hub-details]").click();
  await b.locator("[data-hub-pull]").click();
  await expect(b.locator(".hub-state.err")).toContainText("Download failed: huggingface_hub.errors.HfHubHTTPError: 500 Server Error", { timeout: 15000 });
  await expect(b.getByRole("button", { name: "Retry download of opennpu-test/Broken-Qwen3-4B" })).toBeVisible();
  await b.getByRole("button", { name: "Dismiss opennpu-test/Broken-Qwen3-4B" }).click();
  await expect(b.getByRole("button", { name: "Download opennpu-test/Broken-Qwen3-4B" })).toBeVisible();
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
});

test("discover: installed models offer Chat and Remove, Remove confirms inline (no browser dialog)", async ({ page }) => {
  page.on("dialog", (d) => { errors.push("unexpected dialog: " + d.message()); d.dismiss(); });
  await resetHub(page, "Qwen/Qwen3-30B-A3B");
  await pullDone(page, "Qwen/Qwen3-30B-A3B");
  await page.request.post("/api/hub/cancel", { data: { id: "Qwen/Qwen3-30B-A3B" } });   // forget the job: installed only
  await discover(page, "qwen3-30b", "chat");
  const r = row(page, "Qwen/Qwen3-30B-A3B");
  await expect(r.locator(".badge.ok")).toHaveText("Installed");
  await expect(r.getByRole("button", { name: "Chat with Qwen/Qwen3-30B-A3B" })).toBeVisible();
  await r.getByRole("button", { name: "Remove Qwen/Qwen3-30B-A3B" }).click();
  const confirmBox = r.getByRole("group", { name: "Confirm removal" });
  await expect(confirmBox).toContainText("Remove Qwen/Qwen3-30B-A3B?");
  await expect(confirmBox.getByRole("button", { name: "Keep" })).toBeFocused();
  await confirmBox.getByRole("button", { name: "Keep" }).click();
  await expect(r.locator(".badge.ok")).toHaveText("Installed");
  await r.getByRole("button", { name: "Remove Qwen/Qwen3-30B-A3B" }).click();
  await r.getByRole("group", { name: "Confirm removal" }).getByRole("button", { name: "Remove", exact: true }).click();
  await expect(page.locator(".toast")).toContainText("Removed Qwen/Qwen3-30B-A3B");
  await expect(r.locator(".badge.ok")).toHaveCount(0);
  await expect(r.getByRole("button", { name: "Download Qwen/Qwen3-30B-A3B" })).toBeVisible();
  expect((await (await page.request.get("/api/hub/model?id=Qwen/Qwen3-30B-A3B")).json()).installed).toBe(false);
});

test("discover: offline, gated and unknown-model errors are shown inline with what to do", async ({ page }) => {
  await page.click('.nav-item[data-view="discover"]');
  await page.fill("#hubQ", "offline");
  await expect(page.locator("#hubError")).toContainText("can't reach huggingface.co");
  await expect(page.locator("#hubError")).toContainText("installed models keep working offline");
  await discover(page, "llama", "chat");
  await expect(page.locator("#hubError")).toBeHidden();
  const r = row(page, "meta-llama/Llama-3.2-1B-Instruct");
  await r.locator("[data-hub-details]").click();
  await expect(r.locator(".hub-detail")).toContainText("Gated: accept the license on huggingface.co");
  await expect(r.locator(".hub-kv")).toContainText("llama3.2");
  await r.locator("[data-hub-pull]").click();
  await expect(r.locator(".hub-detail .gen-error")).toContainText("hf auth login");
  await r.locator("[data-hub-close]").click();
  await expect(r.locator(".hub-detail")).toBeHidden();
});

test("discover: a download keeps going and stays visible when the search changes", async ({ page }) => {
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
  await page.request.post("/api/hub/pull", { data: { id: "opennpu-test/Slow-Qwen3-1.7B" } });   // e.g. from another app
  await discover(page, "sd-turbo", "all");
  const p = page.locator("#hubPulls");
  await expect(p).toBeVisible();
  await expect(p.getByRole("progressbar", { name: "Download progress for opennpu-test/Slow-Qwen3-1.7B" })).toBeVisible();
  await p.getByRole("button", { name: "Cancel download of opennpu-test/Slow-Qwen3-1.7B" }).click();
  await expect(p).toBeHidden();
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
});

for (const scheme of ["light", "dark"]) {
  test(`axe: discover with a details panel and a download in progress (${scheme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
    await discover(page, "", "all");
    await row(page, "opennpu-test/Slow-Qwen3-1.7B").locator("[data-hub-details]").click();
    await row(page, "opennpu-test/Slow-Qwen3-1.7B").locator("[data-hub-pull]").click();
    await row(page, "unsloth/gpt-oss-20b-GGUF").locator("[data-hub-details]").click();
    await expect(row(page, "unsloth/gpt-oss-20b-GGUF").getByLabel("GGUF variant")).toBeVisible();
    await expect(row(page, "opennpu-test/Slow-Qwen3-1.7B").getByRole("progressbar")).toBeVisible();
    await row(page, "deepseek-ai/DeepSeek-V3").getByRole("button", { name: "Download deepseek-ai/DeepSeek-V3" }).focus();
    const r = await new AxeBuilder({ page }).include("#view-discover").withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    const bad = r.violations.filter((x) => ["serious", "critical"].includes(x.impact));
    expect(bad.map((x) => `${x.id} ${x.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(", ")}`)).toEqual([]);
    await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
  });
}

test("mobile: discover fits a phone, details and progress included", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
  await page.evaluate(() => localStorage.setItem("opennpu.view", '"discover"'));
  await page.reload();
  await expect(row(page, "Qwen/Qwen3-0.6B")).toBeVisible();
  await page.locator("#view-discover .rail-toggle").click();                 // the menu works here too
  await expect(page.locator("#rail")).toHaveClass(/open/);
  await page.keyboard.press("Escape");
  await row(page, "unsloth/gpt-oss-20b-GGUF").locator("[data-hub-details]").click();
  await expect(row(page, "unsloth/gpt-oss-20b-GGUF").getByLabel("GGUF variant")).toBeVisible();
  await row(page, "opennpu-test/Slow-Qwen3-1.7B").locator("[data-hub-details]").click();
  await row(page, "opennpu-test/Slow-Qwen3-1.7B").locator("[data-hub-pull]").click();
  await expect(row(page, "opennpu-test/Slow-Qwen3-1.7B").getByRole("progressbar")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(await page.evaluate(() => [...document.querySelectorAll(".hub-row")].every((r) => r.scrollWidth <= r.clientWidth + 1))).toBe(true);
  await resetHub(page, "opennpu-test/Slow-Qwen3-1.7B");
});
