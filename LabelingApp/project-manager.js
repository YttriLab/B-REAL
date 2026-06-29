// project-manager.js — Project directory management for RealtimeASOID
// Loaded as a <script> before renderer.js. Exposes window.projectManager.

(function () {
  "use strict";

  let activeProject = null; // null = localStorage mode; { rootDir, manifest }

  let saveManifestTimeout = null;

  function sanitizeId(filename) {
    return filename
      .replace(/\.[^/.]+$/, "") // strip extension
      .replace(/[^a-zA-Z0-9_-]/g, "_") // sanitize
      .replace(/_+/g, "_")
      .replace(/^_|_$/g, "");
  }

  // ── Core API ──────────────────────────────────────────────────────────────

  async function createProject(name) {
    // If no name provided, use the save dialog to get both name and location
    let rootDir;
    if (!name) {
      const result = await window.api.projectPromptName();
      if (result.canceled) return null;
      rootDir = result.path;
      name = result.name;
    } else {
      const dirResult = await window.api.projectSelectDir();
      if (dirResult.canceled) return null;
      rootDir = dirResult.path;
    }
    const manifest = await window.api.projectCreate({ rootDir, name });
    activeProject = { rootDir, manifest };
    localStorage.setItem("lastProjectDir", rootDir);
    window.dispatchEvent(new CustomEvent("project-opened", { detail: { rootDir, manifest } }));
    return activeProject;
  }

  async function openProject() {
    const dirResult = await window.api.projectOpenDialog();
    if (dirResult.canceled) return null;
    if (dirResult.error) {
      window.dispatchEvent(new CustomEvent("project-error", { detail: dirResult.error }));
      return null;
    }

    const rootDir = dirResult.path;
    const manifest = await window.api.projectOpen({ rootDir });
    activeProject = { rootDir, manifest };
    localStorage.setItem("lastProjectDir", rootDir);
    window.dispatchEvent(new CustomEvent("project-opened", { detail: { rootDir, manifest } }));
    return activeProject;
  }

  async function openProjectFromDir(rootDir) {
    try {
      const manifest = await window.api.projectOpen({ rootDir });
      activeProject = { rootDir, manifest };
      localStorage.setItem("lastProjectDir", rootDir);
      window.dispatchEvent(new CustomEvent("project-opened", { detail: { rootDir, manifest } }));
      return activeProject;
    } catch (err) {
      console.warn("[project] Failed to open from dir:", rootDir, err);
      return null;
    }
  }

  async function closeProject() {
    if (!activeProject) return;
    await flushManifest();
    activeProject = null;
    localStorage.removeItem("lastProjectDir");
    window.dispatchEvent(new CustomEvent("project-closed"));
  }

  // ── Manifest persistence ──────────────────────────────────────────────────

  function saveManifest() {
    if (!activeProject) return;
    if (saveManifestTimeout) clearTimeout(saveManifestTimeout);
    saveManifestTimeout = setTimeout(flushManifest, 500);
  }

  async function flushManifest() {
    if (!activeProject) return;
    saveManifestTimeout = null;
    try {
      await window.api.projectSave({
        rootDir: activeProject.rootDir,
        manifest: activeProject.manifest
      });
    } catch (err) {
      console.warn("[project] Failed to save manifest:", err);
    }
  }

  // ── Video management ──────────────────────────────────────────────────────

  function addVideoToManifest(videoPath) {
    if (!activeProject) return null;
    const filename = videoPath.split(/[\\/]/).pop();
    const id = sanitizeId(filename);

    // If this video is already tracked, refresh its path (re-link) rather than
    // adding a duplicate.
    const existing = activeProject.manifest.videos.find(v => v.id === id);
    if (existing) {
      if (existing.path !== videoPath) {
        existing.path = videoPath;
        saveManifest();
      }
      return existing;
    }

    const entry = {
      id,
      path: videoPath,
      labels: "labels/" + id + "_bouts.csv",
      frames: "frames/" + id + "/",
      background: "backgrounds/bg_" + id + ".png"
    };
    activeProject.manifest.videos.push(entry);
    saveManifest();
    return entry;
  }

  // Re-link a tracked video (matched by display name) to a new file path.
  function setVideoPath(videoName, newPath) {
    if (!activeProject) return;
    const entry = getVideoEntry(videoName);
    if (entry && newPath && entry.path !== newPath) {
      entry.path = newPath;
      saveManifest();
    }
  }

  function removeVideoFromManifest(videoId) {
    if (!activeProject) return;
    activeProject.manifest.videos = activeProject.manifest.videos.filter(v => v.id !== videoId);
    saveManifest();
  }

  function getVideoEntry(videoName) {
    if (!activeProject) return null;
    const id = sanitizeId(videoName);
    return activeProject.manifest.videos.find(v => v.id === id) || null;
  }

  function getVideoIdFromName(videoName) {
    return sanitizeId(videoName);
  }

  // ── Label I/O ─────────────────────────────────────────────────────────────

  async function saveVideoLabels(videoName, csvContent) {
    if (!activeProject) return;
    const entry = getVideoEntry(videoName);
    if (!entry) return;
    await window.api.projectSaveLabels({
      rootDir: activeProject.rootDir,
      relativePath: entry.labels,
      content: csvContent
    });
  }

  async function loadVideoLabels(videoName) {
    if (!activeProject) return null;
    const entry = getVideoEntry(videoName);
    if (!entry || !entry.labels) return null;
    const result = await window.api.projectReadLabels({
      rootDir: activeProject.rootDir,
      relativePath: entry.labels
    });
    return result.content || null;
  }

  // ── Active-learning predictions I/O ──────────────────────────────────────
  // Persist a video's inference results as predictions/<id>.json so the
  // uncertainty/misclassified overlays and relabel data survive video switches
  // and restarts without re-running inference. Reuses the generic project file
  // IPC (same as labels), just with a different relative path.

  async function savePredictions(videoName, jsonContent) {
    if (!activeProject) return;
    const id = sanitizeId(videoName);
    await window.api.projectSaveLabels({
      rootDir: activeProject.rootDir,
      relativePath: "predictions/" + id + ".json",
      content: jsonContent,
    });
  }

  async function loadPredictions(videoName) {
    if (!activeProject) return null;
    const id = sanitizeId(videoName);
    const result = await window.api.projectReadLabels({
      rootDir: activeProject.rootDir,
      relativePath: "predictions/" + id + ".json",
    });
    return (result && result.content) || null;
  }

  // ── Behaviors ─────────────────────────────────────────────────────────────

  function updateBehaviors(behaviors) {
    if (!activeProject) return;
    activeProject.manifest.behaviors = behaviors.map(b => ({ name: b.name, key: b.key }));
    saveManifest();
  }

  // ── Preprocessing config ──────────────────────────────────────────────────

  function updatePreprocessingConfig(params) {
    if (!activeProject) return;
    activeProject.manifest.preprocessing = { ...params };
    saveManifest();
  }

  // ── Training config ───────────────────────────────────────────────────────

  function updateTrainingConfig(params) {
    if (!activeProject) return;
    activeProject.manifest.training = { ...params };
    saveManifest();
  }

  function addModel(modelEntry) {
    if (!activeProject) return;
    if (!Array.isArray(activeProject.manifest.models)) {
      activeProject.manifest.models = [];
    }
    // Dedupe by path: replace an existing entry for the same checkpoint.
    const existingIdx = activeProject.manifest.models.findIndex(
      (m) => m.path && modelEntry.path && m.path === modelEntry.path
    );
    if (existingIdx >= 0) {
      activeProject.manifest.models[existingIdx] = modelEntry;
    } else {
      activeProject.manifest.models.push(modelEntry);
    }
    saveManifest();
  }

  function getModels() {
    if (!activeProject || !Array.isArray(activeProject.manifest.models)) return [];
    return activeProject.manifest.models;
  }

  function updateModelPath(oldPath, newPath) {
    if (!activeProject) return;
    const entry = (activeProject.manifest.models || []).find((m) => m.path === oldPath);
    if (entry) {
      entry.path = newPath;
      saveManifest();
    }
  }

  // ── Path helpers ──────────────────────────────────────────────────────────

  const isAbsolutePath = (p) => /^([a-zA-Z]:[\\/]|\/)/.test(p);

  const toAbsolute = (relOrAbs) => {
    if (!relOrAbs) return null;
    return isAbsolutePath(relOrAbs) ? relOrAbs : activeProject.rootDir + "/" + relOrAbs;
  };

  // Store a background path on a video entry (relative to the project root when
  // possible, so the project stays portable).
  function setVideoBackground(videoName, absPath) {
    if (!activeProject || !absPath) return;
    const entry = getVideoEntry(videoName);
    if (!entry) return;
    const root = activeProject.rootDir.replace(/\\/g, "/").replace(/\/$/, "");
    const norm = String(absPath).replace(/\\/g, "/");
    entry.background = norm.startsWith(root + "/") ? norm.slice(root.length + 1) : absPath;
    saveManifest();
  }

  function getVideoBackgroundAbs(videoName) {
    if (!activeProject) return null;
    const entry = getVideoEntry(videoName);
    return entry && entry.background ? toAbsolute(entry.background) : null;
  }

  function getVideoLabelsAbs(videoName) {
    if (!activeProject) return null;
    const entry = getVideoEntry(videoName);
    return entry && entry.labels ? toAbsolute(entry.labels) : null;
  }

  function getFramesDir(videoName) {
    if (!activeProject) return null;
    const entry = getVideoEntry(videoName);
    if (!entry) return null;
    return activeProject.rootDir + "/" + entry.frames;
  }

  function getBackgroundsDir() {
    if (!activeProject) return null;
    return activeProject.rootDir + "/backgrounds";
  }

  function getModelsDir() {
    if (!activeProject) return null;
    return activeProject.rootDir + "/models";
  }

  function getCheckpointsDir() {
    if (!activeProject) return null;
    return activeProject.rootDir + "/models/checkpoints";
  }

  function getExportsDir() {
    if (!activeProject) return null;
    return activeProject.rootDir + "/models/exports";
  }

  function getProjectRoot() {
    return activeProject ? activeProject.rootDir : null;
  }

  // ── .rlt export / import ────────────────────────────────────────────────

  async function exportRlt(options) {
    if (!activeProject) {
      throw new Error("No active project to export.");
    }
    await flushManifest();

    const pkg = !!(options && options.package);
    const suffix = pkg ? "-package.rlt" : ".rlt";
    const defaultName = (activeProject.manifest.name || "project") + suffix;
    const saveResult = await window.api.rltSaveDialog({ defaultPath: defaultName });
    if (saveResult.canceled) return null;

    const backendPort = window.api.backendPort || 8741;
    const res = await fetch(`http://127.0.0.1:${backendPort}/api/project/export-rlt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_dir: activeProject.rootDir,
        output_path: saveResult.path,
        package: pkg,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Export failed");
    }
    return await res.json();
  }

  async function importRlt() {
    const openResult = await window.api.rltOpenDialog();
    if (openResult.canceled) return null;

    const dirResult = await window.api.rltSelectOutputDir();
    if (dirResult.canceled) return null;

    const backendPort = window.api.backendPort || 8741;
    const res = await fetch(`http://127.0.0.1:${backendPort}/api/project/import-rlt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rlt_path: openResult.path,
        output_dir: dirResult.path,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Import failed");
    }
    const data = await res.json();

    // Open the unpacked project
    activeProject = { rootDir: dirResult.path, manifest: data.manifest };
    localStorage.setItem("lastProjectDir", dirResult.path);
    window.dispatchEvent(
      new CustomEvent("project-opened", {
        detail: { rootDir: dirResult.path, manifest: data.manifest },
      })
    );

    return { rootDir: dirResult.path, manifest: data.manifest };
  }

  async function inspectRlt(rltPath) {
    const backendPort = window.api.backendPort || 8741;
    const res = await fetch(`http://127.0.0.1:${backendPort}/api/project/inspect-rlt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rlt_path: rltPath }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Inspect failed");
    }
    return await res.json();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  window.projectManager = {
    // State
    isActive: () => activeProject !== null,
    getProject: () => activeProject,
    getProjectRoot,
    getManifest: () => activeProject ? activeProject.manifest : null,

    // Lifecycle
    createProject,
    openProject,
    openProjectFromDir,
    closeProject,

    // Manifest
    saveManifest,
    flushManifest,

    // Videos
    addVideoToManifest,
    removeVideoFromManifest,
    setVideoPath,
    getVideoEntry,
    getVideoIdFromName,

    // Labels
    saveVideoLabels,
    loadVideoLabels,

    // Active-learning predictions
    savePredictions,
    loadPredictions,

    // Config
    updateBehaviors,
    updatePreprocessingConfig,
    updateTrainingConfig,
    addModel,
    getModels,
    updateModelPath,

    // Paths
    getFramesDir,
    getBackgroundsDir,
    getModelsDir,
    getCheckpointsDir,
    getExportsDir,
    setVideoBackground,
    getVideoBackgroundAbs,
    getVideoLabelsAbs,

    // .rlt file format
    exportRlt,
    importRlt,
    inspectRlt,

    // Utility
    sanitizeId
  };
})();
