// core/progress.js — Per-video label progress persistence and the video-list bookkeeping around it.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

const getProgressKey = (videoName) => `progress_${videoName}`;

let saveProgressTimeout = null;
const saveProgressDebounced = () => {
  if (saveProgressTimeout) {
    clearTimeout(saveProgressTimeout);
  }
  saveProgressTimeout = setTimeout(saveProgress, 500);
};

const saveProgress = () => {
  if (!currentVideoName) return;
  const progress = {
    frameIndex: currentFrameIndex,
    bouts: bouts,
    activeStarts: Array.from(activeStarts.entries())
  };
  try {
    localStorage.setItem(getProgressKey(currentVideoName), JSON.stringify(progress));
  } catch (error) {
    console.warn("Failed to save progress", error);
  }
  // Auto-save labels to project directory
  try {
    if (window.projectManager && window.projectManager.isActive() && currentVideoName) {
      const csv = generateTrainingLabelCsv(bouts);
      window.projectManager.saveVideoLabels(currentVideoName, csv).catch(() => {});
    }
  } catch { /* project manager not ready or generators not yet defined */ }
  // Update sidebar bout counts
  try { renderVideoSidebar(); } catch { /* sidebar not ready yet */ }
  // Rebuild misclassification overlay when bouts change
  try {
    if (alState && alState.results) {
      buildMisclassifiedArray();
      renderUncertaintyOverlay();
      updateAlStats();
    }
  } catch { /* alState not yet initialized */ }
};

const loadProgress = (videoName) => {
  try {
    const stored = localStorage.getItem(getProgressKey(videoName));
    if (!stored) return null;
    return JSON.parse(stored);
  } catch (error) {
    console.warn("Failed to load progress", error);
    return null;
  }
};

const restoreProgress = (videoName) => {
  const progress = loadProgress(videoName);
  if (!progress) return;
  
  if (Array.isArray(progress.bouts)) {
    bouts = progress.bouts;
  }
  if (Array.isArray(progress.activeStarts)) {
    activeStarts = new Map(progress.activeStarts);
  }
  renderBehaviors();
  renderBouts();
  
  // Restore frame position after a short delay to ensure video is ready
  if (typeof progress.frameIndex === "number" && progress.frameIndex > 0) {
    setTimeout(() => {
      seekToFrame(progress.frameIndex);
    }, 100);
  }
  
  setStatus(`Restored progress for ${videoName}.`);
};

// ── Video sidebar functions ───────────────────────────────────────────────

const saveVideoList = () => {
  const serializable = videoList.map(v => ({ name: v.name, path: v.path, isWebCodecs: !!v.isWebCodecs }));
  localStorage.setItem("videoList", JSON.stringify(serializable));
};

const getBoutCountForVideo = (videoName) => {
  if (videoName === currentVideoName) return bouts.length;
  const progress = loadProgress(videoName);
  return progress && Array.isArray(progress.bouts) ? progress.bouts.length : 0;
};

// Set the .content grid columns based on whether the video sidebar and the
// player are shown. The player is hidden on the Train tab (it never uses video).
const updateContentLayout = () => {
  const contentEl = document.querySelector(".content");
  if (!contentEl) return;
  // The player (and with it the splitter) is hidden on the Train tab, which has
  // no video. The track list has to match the number of visible children,
  // otherwise the leftovers wrap into implicit rows.
  const playerShown = document.body.dataset.tab !== "train";
  const splitter = document.getElementById("playerSplitter");
  if (splitter) splitter.style.display = playerShown ? "" : "none";
  contentEl.style.gridTemplateColumns = playerShown
    ? "auto var(--player-col) var(--splitter-w) 1fr"
    : "auto 1fr";
};
