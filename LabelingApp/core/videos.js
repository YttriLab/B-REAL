// core/videos.js — The video sidebar, switching videos, playback controls, key actions, bout CSV.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

const renderVideoSidebar = () => {
  const sidebar = document.getElementById("videoSidebar");
  const listEl = document.getElementById("videoListEl");

  // Keep the sidebar (and its "+" Add button) visible even with no videos —
  // otherwise deleting the last video would hide the only way to add more.
  sidebar.classList.remove("hidden");
  updateContentLayout();

  listEl.innerHTML = "";
  if (videoList.length === 0) {
    const li = document.createElement("li");
    li.className = "video-list-empty muted";
    li.textContent = "No videos yet — click + to add.";
    listEl.appendChild(li);
    return;
  }
  videoList.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "video-list-item" + (entry.name === currentVideoName ? " active" : "");
    li.title = entry.path || entry.name;

    const nameSpan = document.createElement("span");
    nameSpan.className = "video-name";
    nameSpan.textContent = entry.name;

    const countSpan = document.createElement("span");
    countSpan.className = "bout-count";
    const count = getBoutCountForVideo(entry.name);
    countSpan.textContent = count > 0 ? `(${count})` : "";

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-btn";
    removeBtn.textContent = "\u00d7";
    removeBtn.title = "Remove video";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeVideoFromList(entry.name);
    });

    li.appendChild(nameSpan);
    li.appendChild(countSpan);
    li.appendChild(removeBtn);
    li.addEventListener("click", () => switchToVideo(entry.name));
    listEl.appendChild(li);
  });
};

const addVideoToList = (file) => {
  const name = file.name;
  const existing = videoList.find(v => v.name === name);
  if (existing) {
    // Already tracked — re-link to the newly chosen path if it changed.
    if (file.path && existing.path !== file.path) {
      existing.path = file.path;
      existing.isWebCodecs = useWebCodecsInput.checked;
      saveVideoList();
      renderVideoSidebar();
      try {
        if (window.projectManager && window.projectManager.isActive()) {
          window.projectManager.setVideoPath(name, file.path);
        }
      } catch { /* project manager not ready */ }
    }
    return;
  }
  videoList.push({
    name,
    path: file.path || "",
    isWebCodecs: useWebCodecsInput.checked
  });
  saveVideoList();
  renderVideoSidebar();
  // Add to project manifest
  try {
    if (window.projectManager && window.projectManager.isActive() && file.path) {
      window.projectManager.addVideoToManifest(file.path);
    }
  } catch { /* project manager not ready */ }
};

const removeVideoFromList = (videoName) => {
  videoList = videoList.filter(v => v.name !== videoName);
  saveVideoList();
  localStorage.removeItem(getProgressKey(videoName));
  // Remove from project manifest
  try {
    if (window.projectManager && window.projectManager.isActive()) {
      const id = window.projectManager.getVideoIdFromName(videoName);
      window.projectManager.removeVideoFromManifest(id);
    }
  } catch { /* project manager not ready */ }
  if (videoName === currentVideoName) {
    if (videoList.length > 0) {
      switchToVideo(videoList[0].name);
    } else {
      resetSession();
      currentVideoName = "";
      currentFile = null;
      if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute("src"); }
      renderVideoSidebar();
    }
  } else {
    renderVideoSidebar();
  }
};

const switchToVideo = (videoName) => {
  if (videoName === currentVideoName) {
    // Re-selecting the video that's currently loading means they changed their
    // mind back to it — drop any queued switch so we don't jump away when it's done.
    pendingVideoSwitch = null;
    return;
  }

  // Never start a second WebCodecs load while one is in flight — they share the
  // decoder/sample state and would corrupt each other (frames from different
  // videos mixed together, or a decoder error). Remember the latest requested
  // video and open it when the current load finishes. The user can keep clicking;
  // whichever they land on last is the one that opens.
  if (webcodecsLoading) {
    pendingVideoSwitch = videoName;
    setStatus(`Still loading ${currentVideoName || "a video"} — will open ${videoName} next.`);
    return;
  }

  // Save current progress before switching
  saveProgress();

  const entry = videoList.find(v => v.name === videoName);
  if (!entry) return;

  // Stop playback
  if (webcodecsMode && webcodecsPlaying) {
    togglePlayPause();
  } else if (!video.paused) {
    video.pause();
    playPauseButton.textContent = "Play";
  }

  // Build file-like object and load
  const fileObj = { name: entry.name, path: entry.path };
  currentFile = fileObj;

  if (entry.isWebCodecs && webcodecsSupported()) {
    useWebCodecsInput.checked = true;
    loadWebcodecsFile(fileObj);
  } else {
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(fileObj);
  }

  localStorage.setItem("lastActiveVideo", videoName);
  renderVideoSidebar();
  // Point the Pose panel (and the skeleton overlay) at this video's pose file.
  emitPlugin("onVideoSwitched");
};

const nextBehaviorKey = () => {
  const used = new Set(behaviors.map((behavior) => behavior.key));
  return KEY_ORDER.find((key) => !used.has(key));
};

const addBehavior = () => {
  const name = normalizeBehaviorName(behaviorNameInput.value);
  if (!name) {
    setStatus("Enter a behavior name before adding.");
    return;
  }

  const existing = findBehaviorByName(name);
  if (existing) {
    setStatus(`"${existing.name}" already exists (key ${existing.key}).`);
    behaviorNameInput.value = "";
    return;
  }

  const key = nextBehaviorKey();
  if (!key) {
    setStatus("All number keys are already assigned.");
    return;
  }

  behaviors.push({
    id: crypto.randomUUID(),
    name,
    key
  });

  behaviorNameInput.value = "";
  saveBehaviors();
  renderBehaviors();
  setStatus(`Assigned "${name}" to key ${key}.`);
};

const resetSession = () => {
  bouts = [];
  activeStarts = new Map();
  renderBehaviors();
  renderBouts();
  // Drop the previous video's predictions/overlays (alState may not be
  // initialized yet during early startup — guard against the TDZ).
  try { clearAlPredictions(); } catch { /* alState not ready */ }
};

const loadHtml5File = (file) => {
  if (video.src) {
    URL.revokeObjectURL(video.src);
  }
  currentVideoName = file.name;
  localStorage.setItem("lastActiveVideo", file.name);
  frameDurationMs = 1000 / (Number(fpsInput.value) || 30);
  lastFrameMediaTime = null;
  currentFrameIndex = 0;
  const filePath = file.path || "";
  const fileUrl = filePath ? toFileUrl(filePath) : "";
  if (fileUrl) {
    video.src = fileUrl;
  } else {
    video.src = URL.createObjectURL(file);
  }
  video.load();
  resetSession();
  restoreProgress(file.name);
  restorePredictions(file.name);
  emitPlugin("onVideoSwitched");
  setStatus(`Loaded ${file.name}`);
};

// Containers mp4box can't demux (it's ISO/MP4-only). The frame-accurate
// WebCodecs path needs these remuxed to MP4 first (see ensureRemuxedForWebcodecs).
// Backend ops (preprocess/AL/export) read the original via ffmpeg and need no remux.
const REMUX_EXTS = new Set(["mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "ogv"]);
const remuxCache = new Map(); // original path -> remuxed MP4 path (per session)

function needsRemux(name) {
  const m = /\.([^.]+)$/.exec(name || "");
  return m ? REMUX_EXTS.has(m[1].toLowerCase()) : false;
}

// Ensure an MP4 the WebCodecs path can decode exists for a non-MP4 source.
// Returns the MP4 path, or null if it can't/shouldn't proceed (user declined,
// no ffmpeg bridge, or remux failed). Stream-copy keeps frames identical so the
// decoded frame indices stay aligned with the backend's view of the original.
async function ensureRemuxedForWebcodecs(file) {
  if (!file || !file.path) return null;

  // Reuse within the session, and across sessions if the file is still on disk.
  const cached = remuxCache.get(file.path);
  if (cached && await fileExists(cached)) return cached;
  if (cached) remuxCache.delete(file.path);

  if (!window.api || typeof window.api.remuxVideo !== "function") {
    setWebcodecsStatus("This container needs conversion to MP4, but ffmpeg is unavailable.");
    return null;
  }

  const proceed = confirm(
    `"${file.name}" isn't an MP4. Frame-accurate (WebCodecs) mode can only ` +
    `demux MP4, so it needs converting.\n\n` +
    `Convert now? A fast stream-copy is used when possible (no re-encode, no ` +
    `quality loss); an MP4 is written next to the original and reused afterward.`
  );
  if (!proceed) return null;

  setWebcodecsStatus("Converting to MP4…");
  let res;
  try {
    res = await window.api.remuxVideo({ inputPath: file.path });
  } catch (e) {
    setWebcodecsStatus("Conversion failed: " + (e && e.message ? e.message : e));
    return null;
  }
  if (!res || !res.success) {
    const msg = (res && (res.error || res.stderr)) || "unknown error";
    setWebcodecsStatus("Conversion failed: " + String(msg).split("\n").pop());
    return null;
  }
  remuxCache.set(file.path, res.outputPath);
  return res.outputPath;
}

const loadWebcodecsFile = async (file) => {
  webcodecsOperationInProgress = true;
  currentVideoName = file.name;
  localStorage.setItem("lastActiveVideo", file.name);
  currentFrameIndex = 0;
  webcodecsDurationMs = 0;
  resetSession();
  webcodecsLoading = true;
  setStatus(`Loading ${file.name} with WebCodecs...`);
  setWebcodecsStatus("Loading...");

  // mp4box (the WebCodecs demuxer) is MP4-only. For MKV and other non-ISO
  // containers, decode a remuxed MP4 instead — but keep currentVideoName/
  // currentFile as the ORIGINAL so labels, progress, and backend ops (which read
  // the source via ffmpeg) stay keyed to it. Stream-copy keeps frame indices aligned.
  let decodeFile = file;
  if (needsRemux(file.name)) {
    const remuxedPath = await ensureRemuxedForWebcodecs(file);
    if (!remuxedPath) {
      // Declined or failed → best-effort HTML5 playback of the original.
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      useWebCodecsInput.checked = false;
      setWebcodecsMode(false);
      loadHtml5File(file);
      if (pendingVideoSwitch) {
        const next = pendingVideoSwitch;
        pendingVideoSwitch = null;
        if (next !== currentVideoName) switchToVideo(next);
      }
      return;
    }
    decodeFile = { name: file.name, path: remuxedPath };
  }

  let success = false;
  try {
    success = await initWebcodecsFromFile(decodeFile);
  } catch (err) {
    // A thrown/rejected load must NEVER wedge the loader. Without this catch the
    // rejection would skip the `webcodecsLoading = false` reset below, leaving the
    // flag stuck true — and the file-input change handler's `if (webcodecsLoading)
    // return;` guard would then silently ignore every subsequent video pick, so the
    // UI just sits on "Loading..." forever.
    console.error("[renderer] WebCodecs load threw:", err);
    setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
    success = false;
  } finally {
    webcodecsLoading = false;
  }
  console.log("[renderer] initWebcodecsFromFile returned:", success, "samples:", webcodecsSamples.length);
  if (!success) {
    webcodecsOperationInProgress = false;
    setStatus("Falling back to HTML5 mode.");
    if (!webcodecsStatus.textContent) {
      setWebcodecsStatus("WebCodecs failed.");
    }
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(file);
  } else {
    console.log("[renderer] WebCodecs success, enabling mode. Samples:", webcodecsSamples.length);
    setWebcodecsStatus("");
    setWebcodecsMode(true);
    webcodecsOperationInProgress = false;
    restoreProgress(file.name);
    restorePredictions(file.name);
    emitPlugin("onVideoSwitched");
    video.removeAttribute("src");
    video.load();
  }

  // Honor any video the user queued while this load was running (latest wins).
  // webcodecsLoading is already false, so this re-entrant switch starts a fresh,
  // non-overlapping load rather than racing the one that just finished.
  if (pendingVideoSwitch) {
    const next = pendingVideoSwitch;
    pendingVideoSwitch = null;
    if (next !== currentVideoName) switchToVideo(next);
  }
};

const handleVideoLoaded = () => {
  if (webcodecsMode) {
    return;
  }
  if (fpsLockInput.checked) {
    applyLockedFps();
  }
  video.playbackRate = Number(playbackSpeedInput.value) || 1;
  updateDurationDisplay();
  updateTrackbar();
  updateTimeDisplay();
  playPauseButton.textContent = "Play";
};

const togglePlayPause = () => {
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
    if (webcodecsPlaying) {
      stopWebcodecsPlayback();
    } else {
      startWebcodecsPlayback();
    }
    return;
  }

  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }

  if (video.paused) {
    video.play();
    suppressTimeUpdate = false;
    playPauseButton.textContent = "Pause";
  } else {
    video.pause();
    playPauseButton.textContent = "Play";
  }
};

const toggleFullscreen = async () => {
  if (!document.fullscreenElement) {
    await document.documentElement.requestFullscreen();
    fullscreenToggle.textContent = "Exit Fullscreen";
  } else {
    await document.exitFullscreen();
    fullscreenToggle.textContent = "Fullscreen";
  }
};

const stepFrame = (direction) => {
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
    stopWebcodecsPlayback();
    const newIndex = Math.max(
      0,
      Math.min(currentFrameIndex + direction, webcodecsSamples.length - 1)
    );
    if (newIndex === currentFrameIndex) {
      return; // Already at boundary
    }
    
    currentFrameIndex = newIndex;
    updateTimeDisplay();
    updateTrackbar();
    
    // If frame is already cached, render immediately
    if (webcodecsFrameCache.has(newIndex)) {
      renderWebcodecsFrame(newIndex);
    } else {
      // Need to decode
      decodeWebcodecsUpTo(newIndex);
    }
    return;
  }

  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }
  if (!Number.isFinite(video.duration)) {
    setStatus("Video is not ready yet.");
    return;
  }
  video.pause();
  playPauseButton.textContent = "Play";
  currentFrameIndex = Math.max(0, currentFrameIndex + direction);
  seekToFrame(currentFrameIndex);
};

const handleKeyAction = (key) => {
  // Check if video is loaded (either WebCodecs or HTML5)
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
  } else {
    if (!video.src) {
      setStatus("Load a video first.");
      return;
    }
    if (!Number.isFinite(video.duration)) {
      setStatus("Video time is not available yet.");
      return;
    }
  }

  const behavior = behaviors.find((item) => item.key === key);
  if (!behavior) {
    return;
  }

  if (!activeStarts.has(behavior.id)) {
    activeStarts.set(behavior.id, currentFrameIndex);
    setStatus(
      `Started ${behavior.name} at ${formatTime(
        currentFrameIndex * frameDurationMs
      )}.`
    );
    renderBehaviors();
    saveProgress();
    return;
  }

  const startFrame = activeStarts.get(behavior.id);
  activeStarts.delete(behavior.id);
  if (startFrame === currentFrameIndex) {
    setStatus(`Skipped ${behavior.name}: zero-length bout.`);
    renderBehaviors();
    saveProgress();
    return;
  }
  insertBoutSorted({
    behaviorId: behavior.id,
    behaviorName: behavior.name,
    startFrame,
    endFrame: currentFrameIndex
  });
  setStatus(
    `Ended ${behavior.name} at ${formatTime(
      currentFrameIndex * frameDurationMs
    )}.`
  );
  if (pauseOnLabelEndInput.checked) {
    if (webcodecsMode) {
      stopWebcodecsPlayback();
    } else {
      video.pause();
      playPauseButton.textContent = "Play";
    }
  }
  renderBehaviors();
  renderBouts();
  saveProgress();
};

const findOverlappingBouts = () => {
  for (let i = 0; i < bouts.length; i++) {
    for (let j = i + 1; j < bouts.length; j++) {
      const a = bouts[i];
      const b = bouts[j];
      // Bouts overlap if they share any frames
      if (a.startFrame < b.endFrame && b.startFrame < a.endFrame) {
        return [i, j];
      }
    }
  }
  return null;
};

const generateBoutsCsv = (boutsArr) => {
  const header = ["start", "end", "behavior"].join(",");
  const rows = boutsArr.map((bout) =>
    [
      csvEscape(formatTime(getFrameTimeMs(bout.startFrame))),
      csvEscape(formatTime(getFrameTimeMs(bout.endFrame))),
      csvEscape(bout.behaviorName)
    ].join(",")
  );
  return [header, ...rows].join("\n");
};

// Frame-indexed one-hot matrix — the canonical project label format consumed by
// the trainer (model.py::load_data). The `time` column is the NATIVE frame index
// (same index space as the extracted `{n}.png` files), and there is one 0/1
// column per behavior that has bouts. `load_data` takes argmax over each row to
// get the frame's class, so exactly one column is set per row.
//
// Only frames that fall inside a bout get a row (sparse — matching sparse frame
// extraction). Lead-in / gap frames are intentionally omitted: on disk they act
// only as triplet context for a labeled frame, never as labeled samples, so they
// must not appear in the label map (an all-zero row would otherwise argmax to the
// first behavior and mislabel the frame).
const generateTrainingLabelCsv = (boutsArr) => {
  // Behavior columns: names that actually have bouts, in first-appearance order.
  const behaviorNames = [];
  const seen = new Set();
  for (const bout of boutsArr) {
    if (!seen.has(bout.behaviorName)) {
      seen.add(bout.behaviorName);
      behaviorNames.push(bout.behaviorName);
    }
  }
  if (behaviorNames.length === 0) return "time\n"; // no labels yet

  // frame index → behavior column (bouts are non-overlapping; if any do overlap,
  // last-write-wins, which is fine for an argmax one-hot).
  const frameToCol = new Map();
  for (const bout of boutsArr) {
    const col = behaviorNames.indexOf(bout.behaviorName);
    for (let f = bout.startFrame; f <= bout.endFrame; f++) frameToCol.set(f, col);
  }

  const frames = [...frameToCol.keys()].sort((a, b) => a - b);
  const header = ["time", ...behaviorNames.map(csvEscape)].join(",");
  const rows = frames.map((f) => {
    const cols = new Array(behaviorNames.length).fill(0);
    cols[frameToCol.get(f)] = 1;
    return f + "," + cols.join(",");
  });
  return [header, ...rows].join("\n");
};

const exportCsv = async () => {
  if (bouts.length === 0) {
    setStatus("No bouts to export yet.");
    return;
  }

  const overlap = findOverlappingBouts();
  if (overlap) {
    const [i, j] = overlap;
    const boutA = bouts[i];
    const boutB = bouts[j];
    setStatus(
      `Cannot export: "${boutA.behaviorName}" (${formatTime(getFrameTimeMs(boutA.startFrame))}-${formatTime(getFrameTimeMs(boutA.endFrame))}) overlaps with "${boutB.behaviorName}" (${formatTime(getFrameTimeMs(boutB.startFrame))}-${formatTime(getFrameTimeMs(boutB.endFrame))}).`
    );
    return;
  }

  const content = generateBoutsCsv(bouts);

  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-bouts.csv`,
    content
  });

  if (result.canceled) {
    setStatus("Export canceled.");
  } else {
    setStatus(`CSV saved to ${result.filePath}`);
  }
};
