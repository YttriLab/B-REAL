// core/layout.js — The plugin seam's context object, plus player/panel split sizing.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Pipeline plugin seam ──────────────────────────────────────────────────
// The shell below knows nothing about pose files, TSM models or RandomForests.
// A pipeline registers itself with window.labelerPlugins (see labeler-plugin.js)
// and the shell talks to it only through these two helpers. Keeping the arrow
// pointing this way is what lets the labeling shell become a standalone app.

// The state a plugin is allowed to see. Deliberately a snapshot of the things
// that are meaningful to any pipeline — not a handle on the shell's internals.
function makeContext() {
  return {
    videoName: currentVideoName,
    videoPath: getVideoPath(),
    frameIndex: currentFrameIndex,
    frameCount: videoFrameCount(),
    fps: Number(fpsInput.value) || 30,
    bouts,
    behaviors,
    projectManager: window.projectManager || null,
    setStatus,
  };
}

// Fire a lifecycle hook at every registered pipeline. Safe before any plugin
// has registered, and safe during module evaluation.
function emitPlugin(hookName) {
  try {
    if (window.labelerPlugins) window.labelerPlugins.emit(hookName, makeContext());
  } catch (err) {
    console.warn(`[labeler] hook "${hookName}" failed`, err);
  }
}

// ── Player / panel split sizing ───────────────────────────────────────────
// The player's width is a grid `fr` share (--player-col) weighed against the
// side panels' 1fr. Dragging the splitter rewrites that share, and it's
// persisted, so a layout you set up survives a restart. Height needs no control
// — the video shell flex-fills whatever the window leaves it.

const DEFAULT_PLAYER_COL = 2.6;
const MIN_PANE_PX = 260;      // neither pane may be dragged narrower than this
let playerCol = DEFAULT_PLAYER_COL;

function applyPlayerCol() {
  document.documentElement.style.setProperty("--player-col", `${playerCol}fr`);
}

function setPlayerCol(value) {
  if (!Number.isFinite(value)) return;
  playerCol = Math.min(8, Math.max(0.3, value));
  applyPlayerCol();
}

(() => {
  const splitter = document.getElementById("playerSplitter");
  if (!splitter) return;
  const playerEl = document.querySelector(".player");
  const panelsEl = document.querySelector(".side-panels");
  if (!playerEl || !panelsEl) return;
  let dragging = false;

  const onMove = (e) => {
    if (!dragging) return;
    // Convert the pointer position into an fr ratio between the two flexible
    // columns, measuring across the span they jointly occupy.
    const left = playerEl.getBoundingClientRect().left;
    const right = panelsEl.getBoundingClientRect().right;
    const total = right - left;
    if (total <= 0) return;
    const playerW = Math.max(MIN_PANE_PX, Math.min(e.clientX - left, total - MIN_PANE_PX));
    const panelsW = total - playerW;
    if (panelsW > 0) setPlayerCol(playerW / panelsW);
  };

  const stop = () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    debouncedSave();
  };

  splitter.addEventListener("pointerdown", (e) => {
    dragging = true;
    splitter.classList.add("dragging");
    // Suppress text selection and hold the resize cursor for the whole drag,
    // even when the pointer runs ahead of the splitter.
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    splitter.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  splitter.addEventListener("pointermove", onMove);
  splitter.addEventListener("pointerup", stop);
  splitter.addEventListener("pointercancel", stop);
  splitter.addEventListener("dblclick", () => {
    setPlayerCol(DEFAULT_PLAYER_COL);
    debouncedSave();
    setStatus("Player width reset.");
  });
})();
