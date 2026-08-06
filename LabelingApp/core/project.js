// core/project.js — Project lifecycle, the project menu, and .rlt export/import.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Project management ────────────────────────────────────────────────────

function updateProjectUI() {
  const nameEl = document.getElementById("projectName");
  const saveBtn = document.getElementById("saveProject");
  const closeBtn = document.getElementById("closeProject");
  const exportRltBtn = document.getElementById("exportRlt");
  const exportRltPkgBtn = document.getElementById("exportRltPackage");
  if (window.projectManager && window.projectManager.isActive()) {
    const manifest = window.projectManager.getManifest();
    nameEl.textContent = manifest.name || "Untitled";
    saveBtn.classList.remove("hidden");
    closeBtn.classList.remove("hidden");
    exportRltBtn.classList.toggle("hidden", !BACKEND_ENABLED);
    exportRltPkgBtn.classList.toggle("hidden", !BACKEND_ENABLED);
  } else {
    nameEl.textContent = "";
    saveBtn.classList.add("hidden");
    closeBtn.classList.add("hidden");
    exportRltBtn.classList.add("hidden");
    exportRltPkgBtn.classList.add("hidden");
  }
  // Anything the pipeline restores from a project — its iteration, its video
  // checklists, its newest checkpoint — happens inside this hook. emit() already
  // swallows a throwing plugin, so a half-built panel can't block the gate.
  emitPlugin("onProjectOpened");
  updateProjectGate();
}

// Show the project gate (blocking overlay) whenever no project is active.
function updateProjectGate() {
  const gate = document.getElementById("projectGate");
  if (!gate) return;
  const active = window.projectManager && window.projectManager.isActive();
  gate.classList.toggle("hidden", Boolean(active));
}

// Gate buttons reuse the existing Project-menu handlers.
document.getElementById("gateNewProject").addEventListener("click", () => {
  document.getElementById("newProject").click();
});
document.getElementById("gateOpenProject").addEventListener("click", () => {
  document.getElementById("openProjectBtn").click();
});
document.getElementById("gateImportRlt").addEventListener("click", () => {
  document.getElementById("importRlt").click();
});

// ── .rlt is a backend format ───────────────────────────────────────────────
// An .rlt is an HDF5 bundle written by the Python side: the lightweight flavor
// carries the manifest and labels, the package flavor also carries whatever the
// pipeline produced (pose files and feature sets, or extracted frames and
// backgrounds) plus model blobs. A build with no backend can neither write nor
// read one, so it should not offer to.
//
// The code stays here rather than moving into a plugin because both pipelines
// drive it through the identical endpoints and the identical project-manager
// API — pushing it out would mean maintaining the same 60 lines twice, which is
// the duplication this whole split exists to remove. Hiding the entry points is
// enough: nothing below is reachable without them.
// Hidden rather than removed: the handlers below are attached to these same
// elements further down this file, and updateProjectUI reaches for two of them
// every time a project opens. Hiding is what the user sees either way.
if (!BACKEND_ENABLED) {
  for (const id of ["exportRlt", "exportRltPackage", "importRlt", "gateImportRlt"]) {
    const el = document.getElementById(id);
    if (el) el.classList.add("hidden");
  }
}

async function restoreProjectVideos(manifest) {
  // Restore behaviors from manifest
  if (Array.isArray(manifest.behaviors) && manifest.behaviors.length > 0) {
    behaviors = manifest.behaviors.map((b, i) => ({
      id: Date.now() + "-" + i,
      name: normalizeBehaviorName(b.name),
      key: b.key || KEY_ORDER[i] || ""
    }));
    saveBehaviors();
    renderBehaviors();
  }

  // Add videos from manifest to videoList
  for (const vEntry of manifest.videos) {
    if (!vEntry.path) continue;
    const name = vEntry.path.split(/[\\/]/).pop();
    addVideoToList({ name, path: vEntry.path });

    // Load labels from project if they exist
    const labelContent = await window.projectManager.loadVideoLabels(name);
    if (labelContent && labelContent.trim()) {
      // Store as progress so restoreProgress picks them up
      // We need to temporarily switch context to parse for this video
      const prevBouts = [...bouts];
      bouts = [];
      parseCsvAndImport(labelContent, "project:" + vEntry.id);
      const importedBouts = [...bouts];
      bouts = prevBouts;

      // Save imported bouts to localStorage for this video
      if (importedBouts.length > 0) {
        const progress = { frameIndex: 0, bouts: importedBouts, activeStarts: [] };
        localStorage.setItem(getProgressKey(name), JSON.stringify(progress));
      }
    }
  }

  // Collapse any duplicate behaviors introduced across the manifest + labels.
  mergeDuplicateBehaviors();
  renderBehaviors();

  // Load first video
  if (videoList.length > 0) {
    switchToVideo(videoList[0].name);
  }

  // Re-link any trained models whose checkpoint path no longer resolves
  await relinkProjectModels(manifest);
}

async function fileExists(path) {
  if (!path || !window.api || typeof window.api.getFileSize !== "function") return false;
  try {
    await window.api.getFileSize(path);
    return true;
  } catch {
    return false;
  }
}

async function relinkProjectModels(manifest) {
  const models = Array.isArray(manifest.models) ? manifest.models : [];
  if (models.length === 0) return;

  const missing = [];
  for (const m of models) {
    if (!(await fileExists(m.path))) missing.push(m);
  }
  if (missing.length === 0) return;

  setStatus(
    `${missing.length} model checkpoint(s) need re-linking (copied separately, or re-train to regenerate).`
  );

  for (const m of missing) {
    const ok = window.confirm(
      `Model "${m.name}" was not found at:\n${m.path}\n\nLocate the checkpoint file now? ` +
      `(Cancel to skip — you can re-train to regenerate it.)`
    );
    if (!ok) continue;
    const res = await window.api.selectFile({
      title: `Locate model for "${m.name}"`,
      filters: modelFileFilters(),
    });
    if (res && !res.canceled) {
      window.projectManager.updateModelPath(m.path, res.path);
      // If this is the model the Active Learning panel points at, update it.
      if (alState && alState.checkpointPath === m.path) {
        alState.checkpointPath = res.path;
        setAlCheckpointDisplay(res.path, m.iteration);
      }
      setStatus(`Re-linked model "${m.name}".`);
    }
  }
}

document.getElementById("newProject").addEventListener("click", async () => {
  if (!window.projectManager) {
    setStatus("Project manager not available.");
    return;
  }
  const result = await window.projectManager.createProject();
  if (!result) return;

  // A new project starts empty — clear the current session so the previously
  // open project's videos/bouts don't carry over into it.
  videoList = [];
  saveVideoList();
  currentVideoName = "";
  currentFile = null;
  if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute("src"); }
  resetSession();
  renderVideoSidebar();
  // Seed the manifest with the (global) behavior set.
  window.projectManager.updateBehaviors(behaviors);

  updateProjectUI();
  setStatus(`Project "${result.manifest.name}" created (empty).`);
});

document.getElementById("openProjectBtn").addEventListener("click", async () => {
  if (!window.projectManager) return;

  // Close current project if open
  if (window.projectManager.isActive()) {
    await window.projectManager.closeProject();
  }

  const result = await window.projectManager.openProject();
  if (!result) return;

  updateProjectUI();
  await restoreProjectVideos(result.manifest);
  setStatus(`Project "${result.manifest.name}" opened.`);
});

document.getElementById("saveProject").addEventListener("click", async () => {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  saveProgress();
  // Write labels for all videos in the project
  for (const v of videoList) {
    if (v.name === currentVideoName) continue; // already saved by saveProgress
    const stored = loadProgress(v.name);
    if (stored && Array.isArray(stored.bouts) && stored.bouts.length > 0) {
      const csv = generateTrainingLabelCsv(stored.bouts);
      await window.projectManager.saveVideoLabels(v.name, csv);
    }
  }
  await window.projectManager.flushManifest();
  setStatus("Project saved.");
});

document.getElementById("closeProject").addEventListener("click", async () => {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  // Save current state before closing
  saveProgress();
  await window.projectManager.closeProject();
  updateProjectUI();
  setStatus("Project closed.");
});

// ── Project dropdown menu ──
(() => {
  const menu = document.getElementById("projectMenu");
  const trigger = document.getElementById("projectMenuBtn");
  if (!menu || !trigger) return;

  const setOpen = (open) => {
    menu.classList.toggle("open", open);
    trigger.setAttribute("aria-expanded", String(open));
  };

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!menu.classList.contains("open"));
  });
  // Close when clicking outside, on Escape, or after picking an action.
  document.addEventListener("click", (e) => {
    if (!menu.contains(e.target)) setOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });
  menu.querySelectorAll(".menu-panel button").forEach((b) =>
    b.addEventListener("click", () => setOpen(false))
  );
})();

// ── .rlt export / import ──

async function doExportRlt(pkg) {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  // Save all current state first
  saveProgress();
  for (const v of videoList) {
    if (v.name === currentVideoName) continue;
    const stored = loadProgress(v.name);
    if (stored && Array.isArray(stored.bouts) && stored.bouts.length > 0) {
      const csv = generateTrainingLabelCsv(stored.bouts);
      await window.projectManager.saveVideoLabels(v.name, csv);
    }
  }
  await window.projectManager.flushManifest();

  setStatus(pkg ? "Exporting full package (.rlt) — this can take a while..." : "Exporting project to .rlt...");
  try {
    const result = await window.projectManager.exportRlt({ package: pkg });
    if (!result) {
      setStatus("Export canceled.");
      return;
    }
    const s = result.summary;
    if (s.packaged) {
      // What a package contains is up to the backend — pose files and feature
      // sets for one pipeline, extracted frames and backgrounds for another —
      // so report whatever it counted rather than naming the parts here.
      const counted = Object.entries(s)
        .filter(([k, v]) => k !== "packaged" && typeof v === "number" && v > 0)
        .map(([k, v]) => `${v} ${k}`)
        .join(", ");
      setStatus(`Exported package .rlt: ${counted || "nothing to package"}.`);
    } else {
      setStatus(`Exported .rlt: ${s.videos} video reference(s), ${s.labels} label set(s).`);
    }
  } catch (err) {
    setStatus("Export failed: " + (err.message || err));
  }
}

document.getElementById("exportRlt").addEventListener("click", () => doExportRlt(false));
document.getElementById("exportRltPackage").addEventListener("click", () => doExportRlt(true));

document.getElementById("importRlt").addEventListener("click", async () => {
  if (!window.projectManager) {
    setStatus("Project manager not available.");
    return;
  }
  // Close any open project first
  if (window.projectManager.isActive()) {
    saveProgress();
    await window.projectManager.closeProject();
  }

  setStatus("Importing .rlt file...");
  try {
    const result = await window.projectManager.importRlt();
    if (!result) {
      setStatus("Import canceled.");
      return;
    }
    updateProjectUI();
    await restoreProjectVideos(result.manifest);
    setStatus(
      `Imported project "${result.manifest.name}". ` +
      "Re-link any video that doesn't load; frames regenerate from the video + config, " +
      "and models can be re-linked (if copied over) or re-trained."
    );
  } catch (err) {
    setStatus("Import failed: " + (err.message || err));
  }
});

// Try to restore last project on startup
(async () => {
  const lastDir = localStorage.getItem("lastProjectDir");
  if (lastDir && window.projectManager) {
    try {
      const result = await window.projectManager.openProjectFromDir(lastDir);
      if (result) {
        updateProjectUI();
        await restoreProjectVideos(result.manifest);
      }
    } catch (err) {
      console.warn("[project] Failed to restore last project:", err);
      localStorage.removeItem("lastProjectDir");
    }
  }
  // Show the gate if nothing restored (no project active).
  updateProjectGate();
})();

// ── Save settings on page unload ──────────────────────────────────────────
window.addEventListener("beforeunload", () => {
  saveSettings();
  // Flush project manifest on unload
  if (window.projectManager && window.projectManager.isActive()) {
    saveProgress();
    window.projectManager.flushManifest();
  }
});
