const { contextBridge, ipcRenderer } = require("electron");
const fs = require("fs/promises");
const path = require("path");
const { createRequire } = require("module");
let MP4Box = null;
let mp4boxError = "";
let preloadError = "";

const requireMp4Box = () => {
  const localPath = path.join(
    __dirname,
    "node_modules",
    "mp4box",
    "dist",
    "mp4box.all.cjs"
  );
  try {
    return require(localPath);
  } catch (error) {
    return null;
  }
};

const loadMp4Box = () => {
  if (MP4Box) {
    return MP4Box;
  }
  mp4boxError = "";
  try {
    let moduleRef = null;
    try {
      const requireFromApp = createRequire(path.join(__dirname, "preload.js"));
      moduleRef = requireFromApp("mp4box/dist/mp4box.all.cjs");
    } catch (error) {
      moduleRef = requireMp4Box();
      if (!moduleRef) {
        const requireFromApp = createRequire(path.join(__dirname, "preload.js"));
        moduleRef = requireFromApp("mp4box");
      }
    }
    if (moduleRef && typeof moduleRef.createFile === "function") {
      MP4Box = moduleRef;
    } else if (moduleRef && moduleRef.default && typeof moduleRef.default.createFile === "function") {
      MP4Box = moduleRef.default;
    } else {
      mp4boxError = "createFile export not found.";
    }
  } catch (error) {
    mp4boxError = error && error.message ? error.message : "Unknown error.";
    MP4Box = null;
  }
  return MP4Box;
};

process.on("uncaughtException", (error) => {
  preloadError = error && error.message ? error.message : "Unknown preload error.";
});
process.on("unhandledRejection", (error) => {
  preloadError = error && error.message ? error.message : "Unknown preload error.";
});

try {
  loadMp4Box();
} catch (error) {
  preloadError = error && error.message ? error.message : "Unknown preload error.";
}

contextBridge.exposeInMainWorld("api", {
  preloadReady: true,
  preloadError,
  saveCsv: (payload) => ipcRenderer.invoke("save-csv", payload),
  openCsv: () => ipcRenderer.invoke("open-csv"),
  saveVideoDialog: (payload) => ipcRenderer.invoke("save-video-dialog", payload),
  reencodeVideo: (payload) => ipcRenderer.invoke("reencode-video", payload),
  runFfmpeg: (payload) => ipcRenderer.invoke("run-ffmpeg", payload),
  createMp4File: () => {
    const moduleRef = loadMp4Box();
    if (!moduleRef || typeof moduleRef.createFile !== "function") {
      return null;
    }
    const created = moduleRef.createFile();
    if (typeof created === "function") {
      return created();
    }
    return created;
  },
  readFileAsArrayBuffer: async (filePath) => {
    const data = await fs.readFile(filePath);
    return new Uint8Array(data).buffer;
  },
  getFileSize: async (filePath) => {
    const stats = await fs.stat(filePath);
    return stats.size;
  },
  readFileChunk: async (filePath, offset, length) => {
    const handle = await fs.open(filePath, "r");
    try {
      const buffer = Buffer.alloc(length);
      const { bytesRead } = await handle.read(buffer, 0, length, offset);
      const slice = buffer.subarray(0, bytesRead);
      return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
    } finally {
      await handle.close();
    }
  },
  getMp4BoxStatus: () => ({
    available: Boolean(loadMp4Box()),
    error: mp4boxError
  }),
  getApiInfo: () => ({
    apiVersion: "1",
    hasCreateMp4File: true,
    mp4boxAvailable: Boolean(loadMp4Box()),
    mp4boxError,
    preloadError
  }),
  // ── New: backend integration ──
  backendPort: 8741,
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  selectFile: (payload) => ipcRenderer.invoke("select-file", payload),
  selectFiles: (payload) => ipcRenderer.invoke("select-files", payload),
  getBackendPort: () => ipcRenderer.invoke("get-backend-port"),
  importProject: () => ipcRenderer.invoke("import-project"),
});
