// Web UI end-to-end tests against the real server with fake engines (tests/qa/fakeserver.py).
//   cd tests/e2e && npm install && npx playwright test
import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";
const CHROME = process.env.CHROME || (existsSync("/usr/bin/google-chrome") ? "/usr/bin/google-chrome" : "");
const LOCAL_PY = `${process.env.HOME}/.local/share/opennpu/env/bin/python`;
const PY = process.env.OPENNPU_PY || (existsSync(LOCAL_PY) ? LOCAL_PY : "python3");
const PORT = 18611;
export default defineConfig({
  testDir: ".",
  timeout: 30000,
  retries: 0,
  workers: 1,                         // one server, one model slot: tests share it like users would
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // a system Chrome if there is one (no download), else Playwright's own Chromium (CI)
    // fake microphone (a tone) so voice input can be tested headless
    launchOptions: { ...(CHROME ? { executablePath: CHROME } : {}),
      args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
    permissions: ["microphone"],
    viewport: { width: 1280, height: 800 },
    trace: "retain-on-failure",
  },
  webServer: { command: `${PY} ../qa/fakeserver.py ${PORT}`, url: `http://127.0.0.1:${PORT}/health`, reuseExistingServer: false, timeout: 30000 },
});
