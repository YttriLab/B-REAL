// core/backend.js — Backend transport (fetch + SSE), health polling, tab switching, video path/frame helpers.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ═══════════════════════════════════════════════════════════════════════════
// BACKEND INTEGRATION & NEW PANELS
// ═══════════════════════════════════════════════════════════════════════════

// Helper: set path display text + tooltip
function setPathDisplay(id, path) {
  const el = document.getElementById(id);
  el.textContent = path;
  el.title = path;
}

const BACKEND_PORT = (window.api && window.api.backendPort) || 8741;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

async function backendFetch(path, opts = {}) {
  const url = `${BACKEND_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status}: ${err}`);
  }
  return res.json();
}

function backendSSE(path, onMessage, onError) {
  const url = `${BACKEND_URL}${path}`;
  const src = new EventSource(url);
  src.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch (err) {
      console.error("SSE parse error", err);
    }
  };
  src.onerror = (e) => {
    src.close();
    if (onError) onError(e);
  };
  return src;
}

// ── Backend health check ──────────────────────────────────────────────────

const backendStatusEl = document.getElementById("backendStatus");

async function checkBackendHealth() {
  try {
    await backendFetch("/api/health");
    backendStatusEl.textContent = "Backend: Connected";
    backendStatusEl.className = "backend-status connected";
    return true;
  } catch {
    backendStatusEl.textContent = "Backend: Disconnected";
    backendStatusEl.className = "backend-status disconnected";
    return false;
  }
}

// Whether this build has a Python side at all (see app.config.json). Labeling
// never touches it — the player, bouts, CSV and project files are pure Electron
// — so a build without one is a complete application, not a broken one.
const BACKEND_ENABLED = !window.api || window.api.backendEnabled !== false;

if (BACKEND_ENABLED) {
  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);
} else {
  // Say nothing rather than "Disconnected", which reads as a fault.
  backendStatusEl.textContent = "";
  backendStatusEl.className = "backend-status";
}

// ── Tab switching ─────────────────────────────────────────────────────────

// Delegated, so it also covers tabs a pipeline mounts later (see
// core/panels.js). Binding each button directly would only have caught the two
// tabs index.html ships with.
document.querySelector(".tab-bar").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  showTab(btn.dataset.tab);
});

function showTab(tabName) {
  document.querySelectorAll(".tab-bar .tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.dataset.tab === tabName);
  });
  document.body.dataset.tab = tabName;
  updateContentLayout();
  // Pipeline-owned tabs refresh themselves through the plugin seam.
  try {
    if (window.labelerPlugins) {
      window.labelerPlugins.emit("onTabShown", { ...makeContext(), tab: tabName });
    }
  } catch (err) { console.warn("[labeler] onTabShown failed", err); }
}

// ── Video path / frame count ──────────────────────────────────────────────
function getVideoPath() {
  return currentFile && currentFile.path ? currentFile.path : null;
}

// How many frames the loaded video actually has. Everything the user reviews is
// indexed against this, so it's the ceiling for anything worth predicting.
// Returns 0 when unknown (no video loaded / duration not yet available).
function videoFrameCount() {
  if (webcodecsMode && webcodecsSamples.length) return webcodecsSamples.length;
  if (Number.isFinite(video.duration) && video.duration > 0) {
    const fps = Number(fpsInput.value) || 30;
    return Math.round(video.duration * fps);
  }
  return 0;
}
