// core/hud.js — The on-video heads-up display: prediction, label, confidence.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── On-video HUD ──────────────────────────────────────────────────────────
// Draws the current frame's prediction, confidence and label straight onto the
// video, so reviewing a model doesn't mean reading a number in a side panel
// while watching the animal out of the corner of your eye.
function renderVideoHud(info) {
  const hud = document.getElementById("videoHud");
  const toggle = document.getElementById("showHud");
  if (!hud || !toggle) return;

  if (!toggle.checked) {
    hud.classList.remove("visible");
    return;
  }

  const i = info || currentFrameInfo();
  // Nothing to report on this frame — stay entirely out of the way rather than
  // covering the video with em dashes.
  if (!i.hasPrediction && !i.bout) {
    hud.classList.remove("visible");
    return;
  }

  const predRow = hud.querySelector(".hud-pred");
  const labelRow = hud.querySelector(".hud-label");
  const bar = document.getElementById("hudBar");
  const barFill = document.getElementById("hudBarFill");
  const confEl = document.getElementById("hudConf");

  predRow.style.display = i.hasPrediction ? "" : "none";
  document.getElementById("hudPred").textContent = i.predName || "—";

  const confidence = confidenceFrom(i.uncertainty);
  if (i.hasPrediction && confidence !== null) {
    confEl.textContent = `${Math.round(confidence * 100)}%`;
    bar.style.display = "";
    barFill.style.width = `${confidence * 100}%`;
    // Match the timeline heatmap's reading of the same number: green = the
    // model is sure, red = it isn't and this frame is worth a look.
    barFill.style.background = `hsl(${Math.round(confidence * 120)}, 70%, 48%)`;
  } else {
    confEl.textContent = "";
    bar.style.display = "none";
  }

  labelRow.style.display = i.bout ? "" : "none";
  if (i.bout) {
    document.getElementById("hudLabel").textContent =
      i.truth && i.truth !== i.bout.behaviorName
        ? `${i.bout.behaviorName} → ${i.truth}`
        : i.bout.behaviorName;
  }

  hud.classList.toggle("mismatch", i.mismatch);
  hud.classList.add("visible");
}

document.getElementById("showHud").addEventListener("change", () => {
  renderVideoHud();
  debouncedSave();
});

function buildFrameIndex() {
  alState.frameByIndex = (alState.results && alState.results.frames)
    ? new Map(alState.results.frames.map((f) => [f.index, f]))
    : null;
}

// Apply a results object to alState and rebuild all derived views/overlays.
function applyAlResults(results) {
  alState.results = results;
  alState.lowConfCursor = -1; // new predictions → restart the worst-first walk
  buildUncertaintyArray();
  buildMisclassifiedArray();
  buildFrameIndex();
  renderUncertaintyOverlay();
  updateAlStats();
  updateAlFrameInfo();
}

// Drop in-memory predictions + derived overlays. Called on video switch so one
// video's predictions never leak onto another (they're indexed by native frame,
// so a stale array would be silently misaligned).
function clearAlPredictions() {
  if (alState.sseSource) { try { alState.sseSource.close(); } catch { /* already closed */ } }
  alState.taskId = null;
  alState.sseSource = null;
  alState.results = null;
  alState.uncertaintyArray = null;
  alState.misclassifiedArray = null;
  alState.frameByIndex = null;
  alState.lowConfCursor = -1;
  renderUncertaintyOverlay();
  updateAlStats();
  updateAlFrameInfo();
  const statusEl = document.getElementById("alStatus");
  if (statusEl) statusEl.textContent = "Idle";
}

// Persist the current video's predictions to the project (no-op without one).
function persistPredictions() {
  try {
    const pm = window.projectManager;
    if (pm && pm.isActive() && currentVideoName && alState.results) {
      pm.savePredictions(currentVideoName, JSON.stringify(alState.results)).catch(() => {});
    }
  } catch { /* project manager not ready */ }
}

// Load a video's saved predictions (if any) and rebuild overlays. Guarded so a
// late async result can't apply to a video the user has since switched away from.
async function restorePredictions(videoName) {
  try {
    const pm = window.projectManager;
    if (!pm || !pm.isActive()) return;
    const content = await pm.loadPredictions(videoName);
    if (!content) return;
    if (currentVideoName !== videoName) return; // user switched videos meanwhile
    const results = JSON.parse(content);
    if (!results || !Array.isArray(results.frames)) return;

    // Predictions saved before the frame-range clamp (or from a pose file longer
    // than the video) can run far past the last frame. Trim them for display so
    // the uncertainty timeline stays aligned with the trackbar. Done in memory
    // only — the file on disk is left intact, since those extra predictions are
    // still valid for the part of the recording this clip was cut from.
    const vFrames = videoFrameCount();
    let trimmed = 0;
    if (vFrames > 0 && results.frames.length) {
      const kept = results.frames.filter((f) => f.index < vFrames);
      trimmed = results.frames.length - kept.length;
      if (trimmed > 0) results.frames = kept;
    }

    applyAlResults(results);
    const statusEl = document.getElementById("alStatus");
    if (statusEl) {
      statusEl.textContent = trimmed > 0
        ? `Loaded saved predictions: ${results.frames.length} frames ` +
          `(${trimmed} beyond the end of the video ignored)`
        : `Loaded saved predictions: ${results.frames.length} frames`;
    }
    if (trimmed > 0) {
      setStatus(
        `${trimmed} saved predictions fall past the end of this video — ` +
        `the pose file likely covers a longer recording.`
      );
    }
  } catch (e) {
    console.warn("[al] Failed to restore predictions", e);
  }
}

// Threshold slider
const alThresholdSlider = document.getElementById("alThreshold");
const alThresholdOutput = document.getElementById("alThresholdVal");
const alMinSegmentInput = document.getElementById("alMinSegment");
alThresholdSlider.addEventListener("input", () => {
  alThresholdOutput.textContent = parseFloat(alThresholdSlider.value).toFixed(2);
  alState.lowConfCursor = -1; // ranking changed → restart the walk
  renderUncertaintyOverlay();
  updateAlStats();
});

// Minimum consecutive-segment length filter
alMinSegmentInput.addEventListener("input", () => {
  alState.lowConfCursor = -1; // ranking changed → restart the walk
  renderUncertaintyOverlay();
  updateAlStats();
});

const getMinSegmentLength = () => Math.max(1, parseInt(alMinSegmentInput.value, 10) || 1);

// Runs of consecutive frames with uncertainty >= threshold, of length >= minLen.
// Returns [{ start, end, length }] (inclusive frame indices). Frames with no
// prediction (value < 0) break a run.
function getUncertainSegments(arr, threshold, minLen) {
  const segs = [];
  let runStart = -1;
  for (let i = 0; i <= arr.length; i++) {
    const uncertain = i < arr.length && arr[i] >= 0 && arr[i] >= threshold;
    if (uncertain && runStart === -1) {
      runStart = i;
    } else if (!uncertain && runStart !== -1) {
      const length = i - runStart;
      if (length >= minLen) segs.push({ start: runStart, end: i - 1, length });
      runStart = -1;
    }
  }
  return segs;
}

// Per-frame membership mask for qualifying segments.
function getUncertainMask(arr, threshold, minLen) {
  const mask = new Uint8Array(arr.length);
  for (const seg of getUncertainSegments(arr, threshold, minLen)) {
    for (let i = seg.start; i <= seg.end; i++) mask[i] = 1;
  }
  return mask;
}

// Show on timeline toggle
document.getElementById("alShowTimeline").addEventListener("change", () => {
  renderUncertaintyOverlay();
});

// Show which iteration a model came from — the thing that actually tells you
// how recent it is — rather than just a bare path.
function setAlCheckpointDisplay(path, iteration) {
  const el = document.getElementById("alCheckpointPath");
  if (!el) return;
  const iter = iteration || (path.match(/iteration-\d+/) || [])[0];
  const name = path.split(/[\\/]/).pop();
  el.textContent = iter ? `${name}  (${iter})` : name;
  el.title = path;
}

// Pick the most recently trained model in the project, so the Active Learning
// tab defaults to the newest rather than whatever was selected last.
async function selectNewestCheckpoint({ silent = false } = {}) {
  const pm = window.projectManager;
  if (!pm || !pm.isActive()) return null;
  try {
    const res = await backendFetch(
      `/api/training/checkpoints?search_dirs=${encodeURIComponent(pm.getProjectRoot())}`
    );
    const newest = (res.checkpoints || [])[0];   // backend sorts newest first
    if (!newest) return null;
    alState.checkpointPath = newest.path;
    setAlCheckpointDisplay(newest.path, newest.iteration);
    if (!silent) {
      setStatus(
        `Using newest model: ${newest.filename}` +
        (newest.iteration ? ` from ${newest.iteration}` : "") +
        (newest.macro_f1 ? ` (macro F1 ${newest.macro_f1.toFixed(3)})` : "")
      );
    }
    return newest;
  } catch {
    return null;   // backend down — leave whatever was selected
  }
}

document.getElementById("alUseNewest").addEventListener("click", () => {
  selectNewestCheckpoint().then((m) => {
    if (!m) setStatus("No trained models found in this project yet.");
  });
});

// What a trained model looks like on disk — .joblib for a random forest, .ckpt
// for a torch checkpoint. The shell shouldn't know, so it asks the pipeline.
function modelFileFilters() {
  const models = window.labelerPlugins && window.labelerPlugins.capability("models");
  const exts = (models && models.extensions) || ["*"];
  return [{ name: (models && models.label) || "Models", extensions: exts }];
}

// Select checkpoint
document.getElementById("alSelectCheckpoint").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectFile({
    title: "Select Model Checkpoint",
    filters: modelFileFilters(),
  });
  if (!res.canceled) {
    alState.checkpointPath = res.path;
    setAlCheckpointDisplay(res.path);
    // Record manually-selected checkpoints in the project manifest too.
    if (window.projectManager && window.projectManager.isActive()) {
      window.projectManager.addModel({
        name: res.path.split(/[\\/]/).pop(),
        path: res.path,
        created: new Date().toISOString().slice(0, 10),
        source: "selected",
      });
    }
  }
});
