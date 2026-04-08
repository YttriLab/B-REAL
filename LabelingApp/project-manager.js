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

    // Skip duplicates
    if (activeProject.manifest.videos.some(v => v.id === id)) {
      return activeProject.manifest.videos.find(v => v.id === id);
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
    activeProject.manifest.models.push(modelEntry);
    saveManifest();
  }

  // ── Path helpers ──────────────────────────────────────────────────────────

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
    getVideoEntry,
    getVideoIdFromName,

    // Labels
    saveVideoLabels,
    loadVideoLabels,

    // Config
    updateBehaviors,
    updatePreprocessingConfig,
    updateTrainingConfig,
    addModel,

    // Paths
    getFramesDir,
    getBackgroundsDir,
    getModelsDir,
    getCheckpointsDir,
    getExportsDir,

    // Utility
    sanitizeId
  };
})();
