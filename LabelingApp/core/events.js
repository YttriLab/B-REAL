// core/events.js — Top-level event listeners for the video sidebar.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Video sidebar event listeners ─────────────────────────────────────────

document.getElementById("addVideosBtn").addEventListener("click", async () => {
  if (!window.api || !window.api.selectFiles) {
    setStatus("File selection API unavailable.");
    return;
  }
  const result = await window.api.selectFiles({
    title: "Add Videos",
    filters: [
      { name: "Video Files", extensions: ["mp4", "webm", "avi", "mkv", "mov"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  if (result.canceled || !result.paths) return;

  for (const filePath of result.paths) {
    const name = filePath.split(/[\\/]/).pop();
    addVideoToList({ name, path: filePath });
  }

  // If no video is currently loaded, load the first new one
  if (!currentVideoName && videoList.length > 0) {
    switchToVideo(videoList[0].name);
  }
});

document.getElementById("toggleSidebar").addEventListener("click", () => {
  document.getElementById("videoSidebar").classList.toggle("collapsed");
});

// Startup work (first paint, de-duplication, reopening the last video) moved to
// core/boot.js — it reaches modules further down the load order.
