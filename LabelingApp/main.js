const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const fs = require("fs/promises");
const fsSync = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

const BACKEND_PORT = 8741;
const BACKEND_CWD = path.resolve(__dirname, "..");
let backendProcess = null;

// ── Backend lifecycle ─────────────────────────────────────────────────────

function startBackend() {
  const pythonCmd = process.platform === "win32" ? "python" : "python3";
  backendProcess = spawn(
    pythonCmd,
    ["-m", "uvicorn", "backend.server:app", "--port", String(BACKEND_PORT), "--host", "127.0.0.1"],
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
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
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
    win.setTitle("RealtimeASOID");
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
  startBackend();
  await waitForBackend();
  createWindow();

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
    title: "Load bouts CSV",
    filters: [{ name: "CSV", extensions: ["csv"] }],
    properties: ["openFile"]
  });
  if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
    return { canceled: true };
  }
  const filePath = result.filePaths[0];
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

  // Step 2: Select label CSV
  const labelResult = await dialog.showOpenDialog({
    title: "Select Label CSV",
    filters: [{ name: "CSV Files", extensions: ["csv"] }],
    properties: ["openFile"],
  });
  if (labelResult.canceled || !labelResult.filePaths || labelResult.filePaths.length === 0) {
    return { canceled: true };
  }
  const labelPath = labelResult.filePaths[0];

  // Step 3: Read CSV content
  const labelContent = await fs.readFile(labelPath, "utf8");

  return { canceled: false, videoPath, labelPath, labelContent };
});

ipcMain.handle("get-backend-port", () => {
  return BACKEND_PORT;
});
