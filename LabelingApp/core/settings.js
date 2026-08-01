// core/settings.js — Settings persistence and backend error reporting.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ═══════════════════════════════════════════════════════════════════════════
// Phase 6: Polish — localStorage persistence, auto-populate, error handling
// ═══════════════════════════════════════════════════════════════════════════

// ── Settings persistence ──────────────────────────────────────────────────

// Which localStorage slot this build writes to. Named by the host app so two
// labelers installed side by side don't overwrite each other's preferences.
const SETTINGS_KEY = (window.LABELER_APP && window.LABELER_APP.settingsKey) || "labeler_settings";

// The pipeline's own preferences, if a plugin is loaded. Merged into the same
// blob rather than kept separately, so one read/write covers both.
function pipelineSettings() {
  return (window.labelerPlugins && window.labelerPlugins.capability("settings")) || null;
}

function saveSettings() {
  const pipeline = pipelineSettings();
  let pipelineValues = {};
  if (pipeline && typeof pipeline.save === "function") {
    try { pipelineValues = pipeline.save() || {}; }
    catch (err) { console.warn("[labeler] pipeline settings save failed", err); }
  }

  const settings = {
    ...pipelineValues,
    binarizeSelected: [...binarizeSelected],
    // Active learning
    alThreshold: parseFloat(alThresholdSlider.value),
    alMinSegment: getMinSegmentLength(),
    lastCheckpointPath: alState.checkpointPath,
    // Layout
    playerCol,
    showHud: document.getElementById("showHud").checked,
  };
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch { /* quota exceeded or unavailable */ }
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);

    const setVal = (id, val) => { if (val !== undefined && val !== null) document.getElementById(id).value = val; };
    const setChecked = (id, val) => { if (val !== undefined) document.getElementById(id).checked = val; };

    // The pipeline restores its own keys out of the same blob. Do this before
    // the shell's, so anything the shell derives sees final pipeline state.
    const pipeline = pipelineSettings();
    if (pipeline && typeof pipeline.load === "function") {
      try { pipeline.load(s); }
      catch (err) { console.warn("[labeler] pipeline settings load failed", err); }
    }

    if (Array.isArray(s.binarizeSelected)) {
      binarizeSelected.clear();
      s.binarizeSelected.forEach((n) => binarizeSelected.add(n));
      renderBinarizeChecklist();
    }

    // Active learning
    if (s.alThreshold !== undefined) {
      alThresholdSlider.value = s.alThreshold;
      alThresholdOutput.textContent = s.alThreshold.toFixed(2);
    }
    if (s.alMinSegment !== undefined) alMinSegmentInput.value = s.alMinSegment;
    if (s.lastCheckpointPath) {
      alState.checkpointPath = s.lastCheckpointPath;
      setAlCheckpointDisplay(s.lastCheckpointPath);
    }

    // Layout
    if (s.playerCol !== undefined) setPlayerCol(s.playerCol);
    setChecked("showHud", s.showHud);
  } catch { /* corrupted or unavailable */ }
}

// Load on startup
loadSettings();

// Debounced save — saves 2s after last change
let _saveTimer = null;
function debouncedSave() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(saveSettings, 2000);
}

// Attach save listeners to all inputs in panels
document.querySelectorAll(".tab-panel input, .tab-panel select").forEach((el) => {
  el.addEventListener("input", debouncedSave);
  el.addEventListener("change", debouncedSave);
});

// ── Enhanced error handling for backend operations ────────────────────────

// Wrap backendFetch to show user-friendly errors
const _origBackendFetch = backendFetch;
// We can't reassign a function declaration, but the existing backendFetch
// already throws meaningful errors. Instead, add a global handler for
// unhandled promise rejections to catch backend errors.
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.message || String(e.reason);
  if (msg.includes("Backend") || msg.includes("fetch") || msg.includes("NetworkError")) {
    console.error("[Backend Error]", msg);
    // Don't spam alerts — the status indicator shows connection state
  }
});

// ── Backend health check improvements ─────────────────────────────────────
// The existing checkBackendHealth runs every 10s. Add a more aggressive
// initial check sequence: retry every 2s for the first 30s after app load.

let _backendReady = false;
async function initialBackendCheck() {
  for (let i = 0; i < 15; i++) {
    const ok = await checkBackendHealth();
    if (ok) {
      _backendReady = true;
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  // After 30s of trying, show a more detailed message
  backendStatusEl.textContent = "Backend: Failed to start — check Python/dependencies";
  backendStatusEl.className = "backend-status disconnected";
}
initialBackendCheck();
