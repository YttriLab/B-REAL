// core/trackbar.js — Trackbar scrubbing, container-format handling, project import.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

let trackbarSeekTimeout = null;

// Seek to a frame from the trackbar, pausing any playback so scrubbing is
// authoritative. seekToFrame renders the frame in both modes (canvas for
// WebCodecs, <video> seek for HTML5) — no need to press play.
const trackbarSeekTo = (targetIndex) => {
  if (webcodecsMode) {
    if (webcodecsPlaying) stopWebcodecsPlayback();
  } else if (!video.paused) {
    video.pause();
    playPauseButton.textContent = "Play";
  }
  seekToFrame(targetIndex);
};

trackbar.addEventListener("input", () => {
  const targetIndex = Math.round(Number(trackbar.value) / frameDurationMs);
  currentFrameIndex = targetIndex;
  updateTimeDisplay();

  if (webcodecsMode) {
    // Instant feedback when the frame is already decoded; throttle the
    // heavier decode for uncached frames while the user is dragging.
    if (webcodecsFrameCache.has(targetIndex)) {
      renderWebcodecsFrame(targetIndex);
    }
    if (trackbarSeekTimeout) clearTimeout(trackbarSeekTimeout);
    trackbarSeekTimeout = setTimeout(() => {
      trackbarSeekTimeout = null;
      trackbarSeekTo(targetIndex);
    }, 40);
  } else {
    // Seeking a paused <video> updates the displayed frame immediately.
    trackbarSeekTo(targetIndex);
  }
});

// On release, land authoritatively on the final frame (covers any decode
// dropped by the drag throttle).
trackbar.addEventListener("change", () => {
  if (trackbarSeekTimeout) {
    clearTimeout(trackbarSeekTimeout);
    trackbarSeekTimeout = null;
  }
  const targetIndex = Math.round(Number(trackbar.value) / frameDurationMs);
  currentFrameIndex = targetIndex;
  updateTimeDisplay();
  trackbarSeekTo(targetIndex);
});

playPauseButton.addEventListener("click", togglePlayPause);
fullscreenToggle.addEventListener("click", () => {
  toggleFullscreen().catch((error) => {
    console.warn("Fullscreen failed", error);
  });
});
snapBackButton.addEventListener("click", () => {
  let targetFrame = 0;
  if (bouts.length > 0) {
    // Find the bout with the latest endFrame
    const lastBout = bouts.reduce((latest, bout) => 
      bout.endFrame > latest.endFrame ? bout : latest
    , bouts[0]);
    targetFrame = lastBout.endFrame;
  }
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs not ready.");
      return;
    }
    stopWebcodecsPlayback();
    currentFrameIndex = targetFrame;
    updateTimeDisplay();
    updateTrackbar();
    seekToFrame(targetFrame);
  } else {
    if (!video.src) {
      setStatus("Load a video first.");
      return;
    }
    video.currentTime = (targetFrame * frameDurationMs) / 1000;
    currentFrameIndex = targetFrame;
    updateTimeDisplay();
    updateTrackbar();
  }
  setStatus(`Snapped to frame ${targetFrame}.`);
});
playbackSpeedInput.addEventListener("input", () => {
  const nextRate = Number(playbackSpeedInput.value);
  if (Number.isFinite(nextRate) && nextRate > 0) {
    if (!webcodecsMode) {
      video.playbackRate = nextRate;
    } else if (webcodecsPlaying) {
      startWebcodecsPlayback();
    }
    setStatus(`Playback speed set to ${nextRate}x.`);
  }
});
fpsInput.addEventListener("input", () => {
  if (fpsLockInput.checked) {
    applyLockedFps();
    updateTimeDisplay();
    updateTrackbar();
    setStatus(`Locked FPS set to ${fpsInput.value}.`);
  }
});
fpsLockInput.addEventListener("change", () => {
  if (fpsLockInput.checked) {
    applyLockedFps();
    setStatus(`FPS locked at ${fpsInput.value}.`);
  } else {
    setStatus("FPS lock disabled.");
  }
  updateTimeDisplay();
  updateTrackbar();
  if (webcodecsMode && webcodecsPlaying) {
    startWebcodecsPlayback();
  }
});
useWebCodecsInput.addEventListener("change", () => {
  console.log("[renderer] useWebCodecsInput changed, checked:", useWebCodecsInput.checked);
  if (useWebCodecsInput.checked) {
    if (!webcodecsSupported()) {
      useWebCodecsInput.checked = false;
      const reason = webcodecsUnavailabilityReason();
      setStatus(
        reason
          ? `WebCodecs not supported; using HTML5 mode. ${reason}`
          : "WebCodecs not supported; using HTML5 mode."
      );
      setWebcodecsStatus(reason);
      return;
    }
    if (currentFile) {
      loadWebcodecsFile(currentFile);
    } else {
      setStatus("Load a video before enabling WebCodecs.");
      setWebcodecsStatus("Load a video first.");
      useWebCodecsInput.checked = false;
    }
  } else {
    console.log("[renderer] WebCodecs checkbox unchecked, disabling mode");
    setWebcodecsMode(false);
    setWebcodecsStatus("");
    if (currentFile) {
      loadHtml5File(currentFile);
    }
  }
});
addBehaviorButton.addEventListener("click", addBehavior);
behaviorNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    addBehavior();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) {
    return;
  }

  if (event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }

  if (/^\d$/.test(event.key)) {
    handleKeyAction(event.key);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    stepFrame(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    stepFrame(-1);
  } else if (event.key.toLowerCase() === "f") {
    event.preventDefault();
    toggleFullscreen().catch((error) => {
      console.warn("Fullscreen failed", error);
    });
  } else if (event.key === " ") {
    event.preventDefault();
    togglePlayPause();
  }
});

videoFileInput.addEventListener("change", (event) => {
  console.log("[renderer] videoFileInput changed, webcodecsLoading:", webcodecsLoading, "webcodecsMode:", webcodecsMode);
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  // Clear the input's value so picking the SAME file again still fires `change`
  // (needed to retry after a failed/wedged load, or to reload the same video).
  // `file` is already captured above, so this is safe.
  event.target.value = "";

  // Don't interrupt WebCodecs loading
  if (webcodecsLoading) {
    setStatus(`Still loading ${currentVideoName || "a video"} — wait for it to finish, then add this one.`);
    console.log("[renderer] Ignoring file change during WebCodecs loading");
    return;
  }

  currentFile = file;
  addVideoToList(file);
  console.log("[renderer] File selected:", file.name, "WebCodecs checked:", useWebCodecsInput.checked);
  if (useWebCodecsInput.checked) {
    if (!webcodecsSupported()) {
      useWebCodecsInput.checked = false;
      setStatus("WebCodecs not supported; using HTML5 mode.");
      setWebcodecsMode(false);
      loadHtml5File(file);
      return;
    }
    console.log("[renderer] Loading with WebCodecs");
    loadWebcodecsFile(file).catch((err) => {
      console.error("[renderer] loadWebcodecsFile rejected:", err);
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
    });
    return;
  }
  console.log("[renderer] Loading with HTML5");
  setWebcodecsMode(false);
  loadHtml5File(file);
});

video.addEventListener("loadeddata", () => {
  if (webcodecsMode || webcodecsLoading || useWebCodecsInput.checked) {
    return;
  }
  setStatus(`Video ready: ${currentVideoName || "loaded"}.`);
});

// ── Import Project (video + label CSV) ────────────────────────────────
document.getElementById("importProject").addEventListener("click", async () => {
  if (!window.api || !window.api.importProject) {
    setStatus("Import Project is unavailable.");
    return;
  }

  const result = await window.api.importProject();
  if (result.canceled) return;

  const videoPath = result.videoPath;
  const videoName = videoPath.split(/[\\/]/).pop();

  // Resolve label content (CSV text or XLSX binary)
  let labelContent;
  if (result.isXlsx) {
    if (!window.api.xlsxToCsv) { setStatus("XLSX support unavailable."); return; }
    labelContent = await window.api.xlsxToCsv(result.labelPath);
  } else {
    labelContent = result.labelContent;
  }
  if (!labelContent) { setStatus("Label file empty or unreadable."); return; }

  // Build a File-like object for the existing load functions
  const fileObj = { name: videoName, path: videoPath };
  currentFile = fileObj;
  addVideoToList(fileObj);

  // Load the video, then import labels once it's ready
  const importLabelsOnce = () => {
    parseCsvAndImport(labelContent, result.labelPath || "labels");
  };

  if (useWebCodecsInput.checked && webcodecsSupported()) {
    // For WebCodecs: wait for the load to complete, then import labels
    const origRestore = restoreProgress;
    loadWebcodecsFile(fileObj).then(() => {
      // restoreProgress already ran inside loadWebcodecsFile — now layer labels on top
      importLabelsOnce();
    }).catch((err) => {
      console.error("[renderer] importProject WebCodecs load rejected:", err);
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
      importLabelsOnce();
    });
  } else {
    // For HTML5: listen for loadeddata once, then import labels
    setWebcodecsMode(false);
    const onReady = () => {
      video.removeEventListener("loadeddata", onReady);
      importLabelsOnce();
    };
    video.addEventListener("loadeddata", onReady);
    loadHtml5File(fileObj);
  }
});

// Modal prompting the user how to handle an unsupported video format.
// Resolves to "choose" | "reencode" | null (dismissed).
function showFormatDialog(videoName) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const modal = document.createElement("div");
    modal.className = "modal";

    const title = document.createElement("h3");
    title.textContent = "Unsupported video format";

    const body = document.createElement("p");
    body.textContent =
      `"${videoName}" can't be played directly — its codec or container isn't supported ` +
      `(common with HEVC/H.265 and AVI). Choose a different video that's already MP4 (H.264) ` +
      `or WebM, or re-encode this file to H.264 MP4 with ffmpeg.`;

    const actions = document.createElement("div");
    actions.className = "modal-actions";

    const chooseBtn = document.createElement("button");
    chooseBtn.className = "primary";
    chooseBtn.textContent = "Choose Different Video";

    const reencodeBtn = document.createElement("button");
    reencodeBtn.className = "secondary";
    reencodeBtn.textContent = "Re-encode to MP4";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost";
    cancelBtn.textContent = "Cancel";

    const close = (result) => {
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
      resolve(result);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); close(null); }
    };

    chooseBtn.addEventListener("click", () => close("choose"));
    reencodeBtn.addEventListener("click", () => close("reencode"));
    cancelBtn.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
    // Capture phase so Escape closes the modal before global shortcuts see it.
    document.addEventListener("keydown", onKey, true);

    actions.append(chooseBtn, reencodeBtn, cancelBtn);
    modal.append(title, body, actions);
    overlay.append(modal);
    document.body.append(overlay);
    chooseBtn.focus();
  });
}

const reencodeCurrentVideo = async () => {
  if (!window.api || !window.api.reencodeVideo) {
    setStatus("Re-encode unavailable: ffmpeg integration not present.");
    return;
  }
  setStatus("Re-encoding with ffmpeg...");
  try {
    const result = await window.api.reencodeVideo({ inputPath: currentFile.path });
    if (result.success) {
      setStatus("Re-encoded successfully. Loading " + result.outputPath);
      const newName = result.outputPath.replace(/^.*[\\/]/, "");
      currentVideoName = newName;
      currentFile = { name: newName, path: result.outputPath };
      if (video.src) URL.revokeObjectURL(video.src);
      video.src = toFileUrl(result.outputPath);
      video.load();
      return;
    }
    const isMissing = /ENOENT|not found|not recognized/i.test(result.error || "");
    if (isMissing) {
      setStatus("ffmpeg not found. Install ffmpeg, or run manually: ffmpeg -i input -c:v libx264 -preset fast -crf 18 -c:a aac output.mp4");
    } else {
      setStatus("ffmpeg re-encode failed: " + (result.error || "unknown error"));
      console.error("[reencode] stderr:", result.stderr);
    }
  } catch (e) {
    setStatus("Re-encode error: " + (e.message || e));
  }
};

// Re-link the current (failing) video to a different file. Updates the stored
// path on the video-list entry and the project manifest, keeping the original
// name so its labels/progress stay attached, then reloads from the new path.
const relinkCurrentVideo = async () => {
  if (!window.api || !window.api.selectFile) {
    setStatus("File selection unavailable.");
    return;
  }
  const targetName = currentVideoName;
  const res = await window.api.selectFile({
    title: targetName ? `Locate video for "${targetName}"` : "Select Video",
    filters: [
      { name: "Video Files", extensions: ["mp4", "webm", "avi", "mkv", "mov"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (!res || res.canceled) return;
  const newPath = res.path;

  const entry = videoList.find(v => v.name === targetName);
  if (entry) {
    entry.path = newPath;
    saveVideoList();
    renderVideoSidebar();
    try {
      if (window.projectManager && window.projectManager.isActive()) {
        window.projectManager.setVideoPath(targetName, newPath);
      }
    } catch { /* project manager not ready */ }
    currentFile = { name: targetName, path: newPath };
  } else {
    // No tracked entry — treat as a fresh load.
    currentFile = { name: newPath.split(/[\\/]/).pop(), path: newPath };
    addVideoToList(currentFile);
  }

  setStatus(`Re-linked "${currentFile.name}". Loading…`);
  if (useWebCodecsInput.checked && webcodecsSupported()) {
    loadWebcodecsFile(currentFile);
  } else {
    setWebcodecsMode(false);
    loadHtml5File(currentFile);
  }
};

// If the video fails to load due to an unsupported codec (e.g. HEVC/H.265, AVI),
// prompt the user — never auto-re-encode.
video.addEventListener("error", async () => {
  const name = currentVideoName || "";
  const err = video.error;
  const code = err ? err.code : 0;
  const msg = err && err.message ? err.message : "";

  const isCodecError = code === 3 || code === 4
    || /\.avi$/i.test(name) || /\.av$/i.test(name)
    || (msg && /not supported|decode|format|codec/i.test(msg));

  if (isCodecError && currentFile && currentFile.path) {
    setStatus(`"${name}" is an unsupported format.`);
    const choice = await showFormatDialog(name);
    if (choice === "choose") {
      // Re-link the failing video to a correctly-formatted file (persists path).
      await relinkCurrentVideo();
    } else if (choice === "reencode") {
      await reencodeCurrentVideo();
    } else {
      setStatus("Load a supported video (MP4/H.264 or WebM), or re-encode it.");
    }
    return;
  }

  if (code === 2 || (msg && /network|abort/i.test(msg))) {
    setStatus("Video failed to load (network/abort). Try a local file or a different path.");
  } else if (isCodecError) {
    setStatus("Codec not supported. Re-encode with: ffmpeg -i input -c:v libx264 -preset fast -crf 18 -c:a aac output.mp4");
  } else {
    setStatus(`Video failed to load. Use MP4 (H.264) or WebM. ${msg ? `(${msg})` : ""}`);
  }
});

document.addEventListener("fullscreenchange", () => {
  document.body.classList.toggle("fullscreen", Boolean(document.fullscreenElement));
  fullscreenToggle.textContent = document.fullscreenElement
    ? "Exit Fullscreen"
    : "Fullscreen";
});

exportCsvButton.addEventListener("click", exportCsv);
importCsvButton.addEventListener("click", importCsv);
exportOneHotButton.addEventListener("click", exportOneHot);
generateLabeledVideoButton.addEventListener("click", () => generateLabeledVideo());

// The first paint of the behavior list, bout table and duration readout used to
// happen here. It now runs from core/boot.js, once every module is loaded —
// renderBouts reaches the HUD and the uncertainty overlay, which are declared
// further down the load order.
if (!webcodecsApiSupported()) {
  useWebCodecsInput.checked = false;
  useWebCodecsInput.disabled = true;
  const reason = webcodecsUnavailabilityReason();
  if (reason) {
    useWebCodecsInput.title = reason;
    if (modeControl) {
      modeControl.title = reason;
    }
    setWebcodecsStatus(reason);
  }
}

if (!window.api) {
  const ua = navigator.userAgent || "";
  const uaElectron = ua.includes("Electron");
  setStatus(
    `Preload API missing. ElectronUA=${uaElectron}. Ensure app is launched with \`npm start\` and use the Electron window (title: LabelingApp (Electron)).`
  );
}
