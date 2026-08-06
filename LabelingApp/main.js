const { app, BrowserWindow, dialog, ipcMain, screen } = require("electron");
const fs = require("fs/promises");
const fsSync = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

const BACKEND_CWD = path.resolve(__dirname, "..");
let backendProcess = null;

// Everything about *this* application rather than the labeling shell: its
// window title, the folder layout a new project gets scaffolded with, and
// whether there is a Python backend behind it at all. Kept in app.config.json
// so main.js itself is the same file in every build.
const APP = (() => {
  const defaults = {
    title: "Labeler",
    projectDirs: ["labels", path.join("models", "checkpoints")],
    manifestVersion: 1,
    backend: null,        // no backend: labeling only
  };
  try {
    return { ...defaults, ...JSON.parse(fsSync.readFileSync(path.join(__dirname, "app.config.json"), "utf8")) };
  } catch {
    return defaults;      // running the shell standalone, with no host config
  }
})();

// Two applications with a backend must not share a port, or whichever starts
// second silently drives the first one's server — its UI would look fine while
// running the wrong pipeline.
const BACKEND_PORT = (APP.backend && APP.backend.port) || 8741;

// A backend has to be both configured and actually on disk. The standalone
// labeler has neither; a host checked out without its Python side has the
// first but not the second, and spawning uvicorn there just fails 30 times.
function backendAvailable() {
  if (!APP.backend) return false;
  const pkg = (APP.backend.module || "backend.server:app").split(".")[0];
  return fsSync.existsSync(path.join(BACKEND_CWD, pkg));
}

// ── Backend lifecycle ─────────────────────────────────────────────────────

// Choose the Python interpreter that runs the backend (and therefore training).
// Resolution order:
//   1. RTASOID_PYTHON / PYTHON env override (advanced escape hatch).
//   2. The project-local pixi env at <root>/.pixi/envs/default — created by
//      `pixi install`. This is the supported setup: no manual env or env var.
//   3. Bare `python` / `python3` on PATH (legacy / not recommended).
function pixiEnvPython() {
  return process.platform === "win32"
    ? path.join(BACKEND_CWD, ".pixi", "envs", "default", "python.exe")
    : path.join(BACKEND_CWD, ".pixi", "envs", "default", "bin", "python");
}

function resolveBackendPython() {
  const override = process.env.RTASOID_PYTHON || process.env.PYTHON;
  if (override && override.trim()) return override.trim();

  const envPy = pixiEnvPython();
  if (fsSync.existsSync(envPy)) return envPy;

  return process.platform === "win32" ? "python" : "python3";
}

function startBackend() {
  const pythonCmd = resolveBackendPython();
  console.log(`[backend] launching with interpreter: ${pythonCmd}`);
  backendProcess = spawn(
    pythonCmd,
    ["-m", "uvicorn", (APP.backend && APP.backend.module) || "backend.server:app",
     "--port", String(BACKEND_PORT), "--host", "127.0.0.1"],
    {
      cwd: BACKEND_CWD,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    }
  );

  backendProcess.stdout.on("data", (data) => {
    console.log(`[backend] ${data.toString().trim()}`);
  });
  backendProcess.stderr.on("data", (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });
  backendProcess.on("error", (err) => {
    console.error("[backend] Failed to start:", err.message);
    backendProcess = null;
  });
  backendProcess.on("exit", (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProcess = null;
  });
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

function waitForBackend(maxAttempts = 30, intervalMs = 1000) {
  return new Promise((resolve) => {
    let attempts = 0;
    const check = () => {
      attempts++;
      const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/api/health`, (res) => {
        if (res.statusCode === 200) {
          console.log("[backend] ready");
          resolve(true);
        } else {
          retry();
        }
      });
      req.on("error", retry);
      req.setTimeout(2000, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (attempts >= maxAttempts) {
        console.error("[backend] failed to start after", maxAttempts, "attempts");
        resolve(false);
      } else {
        setTimeout(check, intervalMs);
      }
    };
    check();
  });
}

// ── Window ────────────────────────────────────────────────────────────────

const createWindow = () => {
  const preloadPath = path.resolve(__dirname, "preload.js");
  if (!fsSync.existsSync(preloadPath)) {
    dialog.showErrorBox(
      "Preload missing",
      `Preload script not found at ${preloadPath}`
    );
  }
  const iconPath = path.join(
    __dirname,
    process.platform === "win32" ? "labelerlogo.ico" : "labelerlogo.png"
  );
  // Open relative to the actual display rather than a fixed 1400x900, which is
  // a small window on a large monitor — and the video shell now flex-fills, so
  // window size translates directly into how big the video is.
  const { workAreaSize } = screen.getPrimaryDisplay();
  const win = new BrowserWindow({
    width: Math.max(1100, Math.min(1920, Math.round(workAreaSize.width * 0.92))),
    height: Math.max(720, Math.min(1200, Math.round(workAreaSize.height * 0.92))),
    minWidth: 1000,
    minHeight: 680,
    center: true,
    icon: fsSync.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  win.on("page-title-updated", (event) => {
    event.preventDefault();
  });
  win.loadFile("index.html");

  win.webContents.on("did-finish-load", () => {
    win.setTitle(APP.title);
    win.webContents
      .executeJavaScript("Boolean(window.api)")
      .then((hasApi) => {
        if (!hasApi) {
          dialog.showMessageBox(win, {
            type: "warning",
            title: "Preload missing",
            message:
              "Preload did not attach to this window. WebCodecs will be disabled."
          });
        }
      })
      .catch(() => {});
  });
};

// ── App lifecycle ─────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  // Show the window first. Labeling needs no backend at all — the player,
  // bouts, CSV and project files are pure Electron — so making someone stare at
  // nothing while uvicorn boots (or while it fails to, thirty seconds of
  // retries) was backwards. The renderer polls health and flips its own status
  // indicator when the backend turns up.
  createWindow();

  if (backendAvailable()) {
    startBackend();
    waitForBackend();
  } else if (APP.backend) {
    console.warn(`[backend] configured but not found under ${BACKEND_CWD} — labeling only`);
  } else {
    console.log("[backend] none configured — labeling only");
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

// ── IPC handlers (existing) ───────────────────────────────────────────────

ipcMain.handle("open-csv", async () => {
  const result = await dialog.showOpenDialog({
    title: "Load Labels",
    filters: [{ name: "Label Files", extensions: ["csv", "xlsx", "xls"] }],
    properties: ["openFile"]
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  const filePath = result.filePaths[0];
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".xlsx" || ext === ".xls") {
    return { canceled: false, filePath, isXlsx: true };
  }
  const content = await fs.readFile(filePath, "utf8");
  return { canceled: false, filePath, content };
});

ipcMain.handle("save-video-dialog", async (_event, payload) => {
  const { defaultPath } = payload || {};
  const result = await dialog.showSaveDialog({
    title: "Save labeled video",
    defaultPath: defaultPath || "labeled.mp4",
    filters: [{ name: "MP4 Video", extensions: ["mp4"] }]
  });
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }
  return { canceled: false, filePath: result.filePath };
});

ipcMain.handle("reencode-video", async (_event, payload) => {
  const { inputPath } = payload || {};
  if (!inputPath) return { success: false, error: "No input path" };

  const ext = path.extname(inputPath);
  const base = path.basename(inputPath, ext);
  const dir = path.dirname(inputPath);
  const outputPath = path.join(dir, base + "_reencoded.mp4");

  const { execFile } = require("child_process");
  return new Promise((resolve) => {
    const args = [
      "-y", "-i", inputPath,
      "-c:v", "libx264", "-preset", "fast", "-crf", "18",
      "-c:a", "aac",
      outputPath
    ];
    const proc = execFile("ffmpeg", args, { maxBuffer: 50 * 1024 * 1024 }, (error, _stdout, stderr) => {
      if (error) {
        resolve({ success: false, error: error.message, stderr: stderr || "", outputPath });
      } else {
        resolve({ success: true, outputPath });
      }
    });
    proc.on("error", (err) => {
      resolve({ success: false, error: err.message || "ffmpeg not found", outputPath });
    });
  });
});

// Remux a non-MP4 container (e.g. MKV) into MP4 for the frame-accurate WebCodecs
// path, which demuxes with mp4box (ISO/MP4 only). Tries a fast stream-copy first
// (no re-encode, no quality loss — works when the video codec is MP4-compatible:
// H.264/H.265/AV1) and falls back to an H.264 re-encode otherwise. Audio is
// dropped: the labeling app never uses it, and this makes the copy succeed
// regardless of the source audio codec. faststart puts moov up front so the
// renderer's chunked reader reaches track metadata early.
ipcMain.handle("remux-video", async (_event, payload) => {
  const { inputPath } = payload || {};
  if (!inputPath) return { success: false, error: "No input path" };

  const fs = require("fs");
  const ext = path.extname(inputPath);
  const base = path.basename(inputPath, ext);
  const dir = path.dirname(inputPath);
  const outputPath = path.join(dir, base + "_remuxed.mp4");

  // Reuse a previous successful conversion (fast re-open across switches/sessions).
  try {
    if (fs.statSync(outputPath).size > 0) {
      return { success: true, outputPath, method: "cached" };
    }
  } catch { /* not present yet */ }

  const { execFile } = require("child_process");
  const run = (args) => new Promise((resolve) => {
    const proc = execFile("ffmpeg", args, { maxBuffer: 50 * 1024 * 1024 }, (error, _o, stderr) => {
      resolve({ error, stderr: stderr || "" });
    });
    proc.on("error", (err) => resolve({ error: err, stderr: err.message || "ffmpeg not found", spawnFail: true }));
  });
  const produced = () => { try { return fs.statSync(outputPath).size > 0; } catch { return false; } };

  // 1) Fast path: stream-copy the video, drop audio.
  let r = await run(["-y", "-i", inputPath, "-map", "0:v:0", "-c:v", "copy",
                     "-an", "-movflags", "+faststart", outputPath]);
  if (r.spawnFail) return { success: false, error: "ffmpeg not found", stderr: r.stderr };
  if (!r.error && produced()) return { success: true, outputPath, method: "copy" };

  // 2) Fallback: re-encode (codec not MP4-compatible, or copy produced nothing).
  try { fs.unlinkSync(outputPath); } catch { /* ignore */ }
  r = await run(["-y", "-i", inputPath, "-map", "0:v:0",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-an", "-movflags", "+faststart", outputPath]);
  if (!r.error && produced()) return { success: true, outputPath, method: "reencode" };

  return { success: false, error: (r.error && r.error.message) || "Remux failed",
           stderr: r.stderr, outputPath };
});

ipcMain.handle("run-ffmpeg", async (_event, payload) => {
  const { inputPath, outputPath, filterStr } = payload || {};
  if (!inputPath || !outputPath || !filterStr) {
    return { success: false, error: "Missing input/output/filter" };
  }

  const { execFile } = require("child_process");
  return new Promise((resolve) => {
    const args = [
      "-y", "-i", inputPath,
      "-vf", filterStr,
      "-c:v", "libx264", "-preset", "fast", "-crf", "18",
      "-c:a", "aac",
      outputPath
    ];
    const proc = execFile("ffmpeg", args, { maxBuffer: 50 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        resolve({ success: false, error: error.message, stderr: stderr || "" });
      } else {
        resolve({ success: true, stdout, stderr });
      }
    });
    proc.on("error", (err) => {
      resolve({ success: false, error: err.message || "ffmpeg not found" });
    });
  });
});

ipcMain.handle("save-csv", async (_event, payload) => {
  const { defaultPath, content } = payload || {};
  const result = await dialog.showSaveDialog({
    title: "Save bouts CSV",
    defaultPath: defaultPath || "bouts.csv",
    filters: [{ name: "CSV", extensions: ["csv"] }]
  });

  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }

  await fs.writeFile(result.filePath, content, "utf8");
  return { canceled: false, filePath: result.filePath };
});

// ── IPC handlers (new — file/directory dialogs) ──────────────────────────

ipcMain.handle("select-directory", async () => {
  const result = await dialog.showOpenDialog({
    title: "Select Directory",
    properties: ["openDirectory", "createDirectory"]
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  return { canceled: false, path: result.filePaths[0] };
});

ipcMain.handle("select-file", async (_event, payload) => {
  const { title, filters } = payload || {};
  const result = await dialog.showOpenDialog({
    title: title || "Select File",
    filters: filters || [{ name: "All Files", extensions: ["*"] }],
    properties: ["openFile"]
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  return { canceled: false, path: result.filePaths[0] };
});

ipcMain.handle("select-files", async (_event, payload) => {
  const { title, filters } = payload || {};
  const result = await dialog.showOpenDialog({
    title: title || "Select Files",
    filters: filters || [{ name: "All Files", extensions: ["*"] }],
    properties: ["openFile", "multiSelections"]
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  return { canceled: false, paths: result.filePaths };
});

ipcMain.handle("import-project", async () => {
  // Step 1: Select video file
  const videoResult = await dialog.showOpenDialog({
    title: "Select Video File",
    filters: [
      { name: "Video Files", extensions: ["mp4", "webm", "avi", "mkv", "mov"] },
      { name: "All Files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (videoResult.canceled || !videoResult.filePaths || videoResult.filePaths.length === 0) {
    return { canceled: true };
  }
  const videoPath = videoResult.filePaths[0];

  // Step 2: Select label file (CSV or XLSX)
  const labelResult = await dialog.showOpenDialog({
    title: "Select Label File",
    filters: [{ name: "Label Files", extensions: ["csv", "xlsx", "xls"] }],
    properties: ["openFile"],
  });
  if (labelResult.canceled || !labelResult.filePaths || labelResult.filePaths.length === 0) {
    return { canceled: true };
  }
  const labelPath = labelResult.filePaths[0];

  // Step 3: Read label content (text for CSV, flag for XLSX)
  const ext = path.extname(labelPath).toLowerCase();
  if (ext === ".xlsx" || ext === ".xls") {
    return { canceled: false, videoPath, labelPath, isXlsx: true };
  }
  const labelContent = await fs.readFile(labelPath, "utf8");
  return { canceled: false, videoPath, labelPath, labelContent };
});

ipcMain.handle("get-backend-port", () => {
  return BACKEND_PORT;
});

// ── IPC handlers (project management) ─────────────────────────────────────

function isPathWithin(parent, child) {
  const rel = path.relative(parent, child);
  return rel && !rel.startsWith("..") && !path.isAbsolute(rel);
}

ipcMain.handle("project:select-dir", async () => {
  const result = await dialog.showOpenDialog({
    title: "Choose Project Location",
    properties: ["openDirectory", "createDirectory"]
  });
  if (result.canceled || !result.filePaths.length) return { canceled: true };
  return { canceled: false, path: result.filePaths[0] };
});

ipcMain.handle("project:prompt-name", async () => {
  // Use a message box with an input field workaround — Electron has no native input dialog,
  // so we use showSaveDialog to let the user type a folder name, creating the dir in one step.
  const result = await dialog.showSaveDialog({
    title: "Create New Project — enter project folder name",
    buttonLabel: "Create Project",
    properties: ["createDirectory", "showOverwriteConfirmation"]
  });
  if (result.canceled || !result.filePath) return { canceled: true };
  // Use the chosen path as both the project dir and derive the name from it
  const projectDir = result.filePath;
  const name = path.basename(projectDir);
  await fs.mkdir(projectDir, { recursive: true });
  return { canceled: false, path: projectDir, name };
});

ipcMain.handle("project:open-dialog", async () => {
  const result = await dialog.showOpenDialog({
    title: "Open Project Directory",
    properties: ["openDirectory"]
  });
  if (result.canceled || !result.filePaths.length) return { canceled: true };
  const rootDir = result.filePaths[0];
  const manifestPath = path.join(rootDir, "project.json");
  try {
    await fs.access(manifestPath);
  } catch {
    return { canceled: false, error: "No project.json found in selected directory." };
  }
  return { canceled: false, path: rootDir };
});

ipcMain.handle("project:create", async (_event, payload) => {
  const { rootDir, name } = payload;
  // What a fresh project contains is the pipeline's business — pose features
  // for one, extracted frames and backgrounds for another — so it comes from
  // app.config.json rather than being written into the shared shell.
  for (const sub of APP.projectDirs) {
    await fs.mkdir(path.join(rootDir, sub), { recursive: true });
  }
  const manifest = {
    name: name || path.basename(rootDir),
    created: new Date().toISOString().slice(0, 10),
    version: APP.manifestVersion,
    behaviors: [],
    videos: [],
    preprocessing: {},
    training: {},
    models: []
  };
  await fs.writeFile(path.join(rootDir, "project.json"), JSON.stringify(manifest, null, 2), "utf8");
  return manifest;
});

ipcMain.handle("project:open", async (_event, payload) => {
  const { rootDir } = payload;
  const raw = await fs.readFile(path.join(rootDir, "project.json"), "utf8");
  return JSON.parse(raw);
});

ipcMain.handle("project:save", async (_event, payload) => {
  const { rootDir, manifest } = payload;
  await fs.writeFile(path.join(rootDir, "project.json"), JSON.stringify(manifest, null, 2), "utf8");
  return { ok: true };
});

ipcMain.handle("project:save-labels", async (_event, payload) => {
  const { rootDir, relativePath, content } = payload;
  const resolved = path.resolve(rootDir, relativePath);
  if (!isPathWithin(rootDir, resolved)) {
    return { error: "Path traversal rejected" };
  }
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  await fs.writeFile(resolved, content, "utf8");
  return { ok: true };
});

ipcMain.handle("project:read-labels", async (_event, payload) => {
  const { rootDir, relativePath } = payload;
  const resolved = path.resolve(rootDir, relativePath);
  if (!isPathWithin(rootDir, resolved)) {
    return { error: "Path traversal rejected" };
  }
  try {
    const content = await fs.readFile(resolved, "utf8");
    return { content };
  } catch {
    return { content: null };
  }
});

ipcMain.handle("project:ensure-dir", async (_event, payload) => {
  const { rootDir, relativePath } = payload;
  const resolved = path.resolve(rootDir, relativePath);
  if (!isPathWithin(rootDir, resolved)) {
    return { error: "Path traversal rejected" };
  }
  await fs.mkdir(resolved, { recursive: true });
  return { ok: true };
});

// ── IPC handlers (.rlt file dialogs) ─────────────────────────────────────

ipcMain.handle("rlt:save-dialog", async (_event, payload) => {
  const { defaultPath } = payload || {};
  const result = await dialog.showSaveDialog({
    title: "Export Project as .rlt",
    defaultPath: defaultPath || "project.rlt",
    filters: [{ name: "RealtimeASOID Project", extensions: ["rlt"] }],
  });
  if (result.canceled || !result.filePath) return { canceled: true };
  return { canceled: false, path: result.filePath };
});

ipcMain.handle("rlt:open-dialog", async () => {
  const result = await dialog.showOpenDialog({
    title: "Import .rlt Project File",
    filters: [{ name: "RealtimeASOID Project", extensions: ["rlt"] }],
    properties: ["openFile"],
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  return { canceled: false, path: result.filePaths[0] };
});

ipcMain.handle("rlt:select-output-dir", async () => {
  const result = await dialog.showOpenDialog({
    title: "Choose Folder to Unpack Project Into",
    properties: ["openDirectory", "createDirectory"],
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  return { canceled: false, path: result.filePaths[0] };
});
