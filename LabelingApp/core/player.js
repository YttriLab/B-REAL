// core/player.js — Video decoding and transport: HTML5 + WebCodecs paths, frame indexing, seeking.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Multi-video sidebar ───────────────────────────────────────────────────
function loadVideoList() {
  try {
    const stored = localStorage.getItem("videoList");
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}
let videoList = loadVideoList();

let webcodecsMode = false;
let webcodecsReady = false;
let webcodecsLoading = false;
let webcodecsSamples = [];
let webcodecsKeyframes = [];
let webcodecsDecoder = null;
let webcodecsConfig = null;
let webcodecsDecodeQueue = [];
let webcodecsDecodedMax = -1;
let webcodecsFrameCache = new Map();
let webcodecsFrameOrder = [];
let webcodecsCacheLimit = 300;
let webcodecsDurationMs = 0;
let webcodecsPlaying = false;
let webcodecsTimer = null;
let webcodecsPendingRender = null;
let mp4boxWorker = null;
let mp4boxChunkId = 0;
let mp4boxChunkResolvers = new Map();
let webcodecsOperationInProgress = false;
// A WebCodecs load mutates shared decoder/sample globals, so only one can run at
// a time. If the user clicks another video mid-load, we stash the latest request
// here and open it once the in-flight load settles (latest pick wins).
let pendingVideoSwitch = null;

const setStatus = (message) => {
  statusEl.textContent = message;
};

const setWebcodecsStatus = (message) => {
  if (!webcodecsStatus) {
    return;
  }
  webcodecsStatus.textContent = message || "";
  webcodecsStatus.title = message || "";
};

const toFileUrl = (filePath) => {
  const normalized = filePath.replace(/\\/g, "/");
  if (/^[A-Za-z]:\//.test(normalized)) {
    return encodeURI(`file:///${normalized}`);
  }
  const prefix = normalized.startsWith("/") ? "" : "/";
  return encodeURI(`file://${prefix}${normalized}`);
};

const canvasContext = videoCanvas.getContext("2d");
const webcodecsApiSupported = () =>
  typeof VideoDecoder !== "undefined" &&
  typeof EncodedVideoChunk !== "undefined" &&
  typeof Worker !== "undefined";

const webcodecsSupported = () => webcodecsApiSupported();

const webcodecsUnavailabilityReason = () => {
  if (typeof VideoDecoder === "undefined") {
    return "WebCodecs is unavailable: VideoDecoder is not supported.";
  }
  if (typeof EncodedVideoChunk === "undefined") {
    return "WebCodecs is unavailable: EncodedVideoChunk is not supported.";
  }
  if (typeof Worker === "undefined") {
    return "WebCodecs is unavailable: Web Worker is not supported.";
  }
  try {
    // Worker will import mp4box for demuxing.
  } catch (error) {
    const message = error && error.message ? error.message : "Unknown error.";
    return `WebCodecs is unavailable: MP4 demuxer failed to load: ${message}`;
  }
  return "";
};

const formatTime = (milliseconds) => {
  const totalMs = Math.max(0, Math.round(milliseconds));
  const mins = Math.floor(totalMs / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}:${String(ms).padStart(3, "0")}`;
};

const parseTime = (timeString) => {
  // Parse "MM:SS:mmm", "MM:SS", or "HH:MM:SS:mmm" format
  const parts = timeString.trim().split(":");
  if (parts.length < 2 || parts.length > 4) return null;
  let hours = 0, mins, secs, ms = 0;
  if (parts.length === 4) {
    hours = parseInt(parts[0], 10);
    mins = parseInt(parts[1], 10);
    secs = parseInt(parts[2], 10);
    ms = parseInt(parts[3], 10);
  } else if (parts.length === 3) {
    mins = parseInt(parts[0], 10);
    secs = parseInt(parts[1], 10);
    ms = parseInt(parts[2], 10);
  } else {
    mins = parseInt(parts[0], 10);
    secs = parseInt(parts[1], 10);
  }
  if (isNaN(hours) || isNaN(mins) || isNaN(secs) || isNaN(ms)) return null;
  if (hours < 0 || mins < 0 || secs < 0 || ms < 0 || ms >= 1000) return null;
  return hours * 3600000 + mins * 60000 + secs * 1000 + ms;
};

const msToFrame = (ms) => {
  return Math.round(ms / frameDurationMs);
};

const editBoutRow = (boutIndex) => {
  const bout = bouts[boutIndex];
  if (!bout) return;
  
  const rows = boutsTable.querySelectorAll("tr");
  const row = rows[boutIndex];
  if (!row) return;
  
  const cells = row.querySelectorAll("td");
  const startCell = cells[0];
  const endCell = cells[1];
  const actionsCell = cells[3];
  
  // Replace start cell with input
  const startInput = document.createElement("input");
  startInput.type = "text";
  startInput.value = formatTime(getFrameTimeMs(bout.startFrame));
  startInput.style.width = "80px";
  startCell.textContent = "";
  startCell.appendChild(startInput);
  startCell.className = "";
  
  // Replace end cell with input
  const endInput = document.createElement("input");
  endInput.type = "text";
  endInput.value = formatTime(getFrameTimeMs(bout.endFrame));
  endInput.style.width = "80px";
  endCell.textContent = "";
  endCell.appendChild(endInput);
  endCell.className = "";
  
  // Replace actions with Save/Cancel
  actionsCell.innerHTML = "";
  const saveButton = document.createElement("button");
  saveButton.className = "ghost";
  saveButton.textContent = "Save";
  saveButton.addEventListener("click", () => {
    const startMs = parseTime(startInput.value);
    const endMs = parseTime(endInput.value);
    
    if (startMs === null) {
      setStatus("Invalid start time. Use MM:SS:mmm or MM:SS.");
      return;
    }
    if (endMs === null) {
      setStatus("Invalid end time. Use MM:SS:mmm or MM:SS.");
      return;
    }
    
    const newStartFrame = msToFrame(startMs);
    const newEndFrame = msToFrame(endMs);
    
    if (newStartFrame >= newEndFrame) {
      setStatus("Start time must be before end time.");
      return;
    }
    
    bout.startFrame = newStartFrame;
    bout.endFrame = newEndFrame;
    renderBouts();
    saveProgress();
    setStatus(`Updated bout to ${formatTime(getFrameTimeMs(newStartFrame))} - ${formatTime(getFrameTimeMs(newEndFrame))}.`);
  });
  
  const cancelButton = document.createElement("button");
  cancelButton.className = "ghost";
  cancelButton.textContent = "Cancel";
  cancelButton.addEventListener("click", () => {
    renderBouts();
  });
  
  actionsCell.appendChild(saveButton);
  actionsCell.appendChild(cancelButton);
  
  startInput.focus();
  startInput.select();
};

const updateFrameDuration = (mediaTime) => {
  if (fpsLockInput.checked) {
    return;
  }
  if (!Number.isFinite(mediaTime)) {
    return;
  }
  if (lastFrameMediaTime !== null) {
    const deltaMs = (mediaTime - lastFrameMediaTime) * 1000;
    if (deltaMs > 0.5 && deltaMs < 200) {
      frameDurationMs = deltaMs;
    }
  }
  lastFrameMediaTime = mediaTime;
};

const startFrameWatch = () => {
  if (webcodecsMode) {
    return;
  }
  if (typeof video.requestVideoFrameCallback !== "function") {
    return;
  }
  const onFrame = (_now, metadata) => {
    updateFrameDuration(metadata.mediaTime);
    currentFrameIndex = Math.round(
      (metadata.mediaTime * 1000) / frameDurationMs
    );
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
};

const applyLockedFps = () => {
  const fpsValue = Number(fpsInput.value);
  if (!Number.isFinite(fpsValue) || fpsValue <= 0) {
    return;
  }
  frameDurationMs = 1000 / fpsValue;
};

const updateDurationDisplay = () => {
  // Duration is the first point at which the video's frame count is knowable,
  // so record it against the project here (see setVideoFrameCount).
  try {
    const pm = window.projectManager;
    if (pm && pm.isActive() && currentVideoName) {
      pm.setVideoFrameCount(currentVideoName, videoFrameCount());
    }
  } catch { /* project manager or helpers not ready yet */ }

  if (webcodecsMode && webcodecsDurationMs) {
    durationEl.textContent = formatTime(webcodecsDurationMs);
    return;
  }
  if (Number.isFinite(video.duration)) {
    durationEl.textContent = formatTime(video.duration * 1000);
  } else {
    durationEl.textContent = "00:00:000";
  }
};

const updateTrackbar = () => {
  if (webcodecsMode) {
    trackbar.max = Math.floor(webcodecsDurationMs || 0);
    trackbar.value = Math.round(getFrameTimeMs(currentFrameIndex));
    return;
  }
  if (Number.isFinite(video.duration)) {
    trackbar.max = Math.floor(video.duration * 1000);
  } else {
    trackbar.max = 0;
  }
  trackbar.value = getVideoTimeMs();
};

const setWebcodecsMode = (enabled) => {
  console.log("[renderer] setWebcodecsMode called:", enabled, "opInProgress:", webcodecsOperationInProgress, "loading:", webcodecsLoading, "ready:", webcodecsReady, "samples:", webcodecsSamples.length);
  
  // Don't disable WebCodecs during an operation
  if (!enabled && webcodecsOperationInProgress) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - operation in progress");
    return;
  }
  
  // Don't disable WebCodecs if decoder exists (loading is in progress or complete)
  if (!enabled && webcodecsDecoder) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - decoder exists");
    return;
  }
  
  // Don't change mode during loading
  if (webcodecsLoading && !enabled) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) during loading");
    return;
  }
  
  // Don't disable WebCodecs if it's ready and working
  if (!enabled && webcodecsReady) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - WebCodecs is ready");
    return;
  }
  
  webcodecsMode = enabled;
  document.body.classList.toggle("webcodecs", webcodecsMode);
  if (webcodecsMode) {
    video.pause();
    playPauseButton.textContent = "Play";
  } else {
    resetWebcodecsState();
  }
};

const stopWebcodecsPlayback = () => {
  if (webcodecsTimer) {
    clearInterval(webcodecsTimer);
    webcodecsTimer = null;
  }
  webcodecsPlaying = false;
  // Clear the decode queue to prevent stale entries from interfering
  webcodecsDecodeQueue = [];
  playPauseButton.textContent = "Play";
};

const startWebcodecsPlayback = () => {
  if (!webcodecsReady) {
    setStatus("WebCodecs is not ready yet.");
    return;
  }
  const playbackRate = Number(playbackSpeedInput.value) || 1;
  const interval = Math.max(5, frameDurationMs / playbackRate);
  console.log("[renderer] Starting playback, interval:", interval, "ms, rate:", playbackRate);
  stopWebcodecsPlayback();
  webcodecsPlaying = true;
  playPauseButton.textContent = "Pause";
  
  let targetFrame = currentFrameIndex;
  webcodecsTimer = setInterval(() => {
    // Only advance target if decoder is keeping up
    if (!webcodecsDecoder || webcodecsDecoder.state !== "configured") {
      console.warn("[renderer] Decoder not ready, state:", webcodecsDecoder?.state);
      stopWebcodecsPlayback();
      return;
    }
    
    // Check if we've reached the end of available samples
    if (targetFrame + 1 >= webcodecsSamples.length) {
      // Check if we're truly at the end based on time
      const currentTimeMs = targetFrame * frameDurationMs;
      if (currentTimeMs >= webcodecsDurationMs - frameDurationMs) {
        console.log("[renderer] Reached end of video");
        stopWebcodecsPlayback();
        return;
      }
      // More samples should be coming, wait
      return;
    }
    
    // Don't advance too far ahead of what's been decoded
    // This prevents overwhelming the decoder
    const maxAhead = 5;
    if (targetFrame - webcodecsDecodedMax > maxAhead) {
      // Decoder is behind, wait for it to catch up
      return;
    }
    
    targetFrame += 1;
    decodeWebcodecsUpTo(targetFrame);
  }, interval);
};

const renderWebcodecsFrame = (frameIndex) => {
  const bitmap = webcodecsFrameCache.get(frameIndex);
  if (!bitmap) {
    console.warn("[renderer] No bitmap for frame:", frameIndex);
    return;
  }
  canvasContext.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  canvasContext.drawImage(bitmap, 0, 0, videoCanvas.width, videoCanvas.height);
  currentFrameIndex = frameIndex;
  updateTimeDisplay();
  updateTrackbar();
  saveProgressDebounced();
};

const cacheWebcodecsFrame = (frameIndex, bitmap) => {
  webcodecsFrameCache.set(frameIndex, bitmap);
  webcodecsFrameOrder.push(frameIndex);
  if (webcodecsFrameOrder.length > webcodecsCacheLimit) {
    const evictIndex = webcodecsFrameOrder.shift();
    const evictBitmap = webcodecsFrameCache.get(evictIndex);
    if (evictBitmap) {
      evictBitmap.close();
    }
    webcodecsFrameCache.delete(evictIndex);
  }
};

const handleDecodedFrame = (frame) => {
  const frameIndex = webcodecsDecodeQueue.shift();
  if (frameIndex === undefined) {
    frame.close();
    return;
  }
  createImageBitmap(frame)
    .then((bitmap) => {
      frame.close();
      cacheWebcodecsFrame(frameIndex, bitmap);
      
      if (webcodecsPlaying) {
        renderWebcodecsFrame(frameIndex);
      } else {
        // After caching this frame, check if the CURRENT target is now available
        // This handles the case where user stepped ahead while decode was in progress
        if (webcodecsFrameCache.has(currentFrameIndex)) {
          renderWebcodecsFrame(currentFrameIndex);
        }
      }
    })
    .catch((error) => {
      frame.close();
      console.warn("[renderer] Failed to create bitmap for frame", frameIndex, error);
    });
};

// Close and drop the active decoder, tolerating an already-closed one. A
// VideoDecoder auto-transitions to "closed" on error, and calling close() on a
// closed codec throws InvalidStateError ("Cannot call 'close' on a closed
// codec") — so always check state and swallow. Nulling the reference also lets
// setWebcodecsMode(false) proceed (it bails while a decoder object exists).
const closeWebcodecsDecoder = () => {
  if (webcodecsDecoder) {
    try {
      if (webcodecsDecoder.state !== "closed") webcodecsDecoder.close();
    } catch (e) {
      console.warn("[renderer] decoder close ignored:", e && e.message ? e.message : e);
    }
    webcodecsDecoder = null;
  }
};

const handleDecoderError = (error) => {
  console.error("[renderer] WebCodecs decoder error:", error);
  // The errored decoder is already closed; drop the stale reference first so the
  // HTML5 fallback's setWebcodecsMode(false) isn't blocked by "decoder exists".
  closeWebcodecsDecoder();
  setStatus("WebCodecs decoding failed; switching to HTML5 mode.");
  setWebcodecsStatus(`Decode error: ${error.message || error}`);
  stopWebcodecsPlayback();
  if (currentFile) {
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(currentFile);
  }
};

const recreateWebcodecsDecoder = () => {
  if (!webcodecsConfig) {
    return false;
  }
  closeWebcodecsDecoder();
  webcodecsDecoder = new VideoDecoder({
    output: handleDecodedFrame,
    error: handleDecoderError
  });
  webcodecsDecoder.configure(webcodecsConfig);
  return true;
};

const resetWebcodecsState = () => {
  console.log("[renderer] resetWebcodecsState called, had", webcodecsSamples.length, "samples");
  stopWebcodecsPlayback();
  webcodecsReady = false;
  webcodecsSamples = [];
  webcodecsKeyframes = [];
  webcodecsDecodeQueue = [];
  webcodecsDecodedMax = -1;
  webcodecsPendingRender = null;
  webcodecsDurationMs = 0;
  webcodecsConfig = null;
  webcodecsFrameOrder.forEach((index) => {
    const bitmap = webcodecsFrameCache.get(index);
    if (bitmap) {
      bitmap.close();
    }
  });
  webcodecsFrameCache.clear();
  webcodecsFrameOrder = [];
  closeWebcodecsDecoder();
  if (mp4boxWorker) {
    mp4boxWorker.terminate();
    mp4boxWorker = null;
  }
  canvasContext.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
};

const findNearestKeyframe = (targetIndex) => {
  for (let i = webcodecsKeyframes.length - 1; i >= 0; i -= 1) {
    if (webcodecsKeyframes[i] <= targetIndex) {
      return webcodecsKeyframes[i];
    }
  }
  return 0;
};

const decodeWebcodecsUpTo = (targetIndex) => {
  if (!webcodecsDecoder) {
    return;
  }
  
  if (webcodecsFrameCache.has(targetIndex)) {
    renderWebcodecsFrame(targetIndex);
    return;
  }

  // Determine if we need to reset the decoder
  let needsReset = false;
  let startIndex = webcodecsDecodedMax + 1;
  
  if (targetIndex <= webcodecsDecodedMax) {
    // Seeking backwards - need to reset
    needsReset = true;
    startIndex = findNearestKeyframe(targetIndex);
  } else if (targetIndex > webcodecsDecodedMax + 60) {
    // Seeking far forward - reset from keyframe near target instead of decoding everything
    needsReset = true;
    startIndex = findNearestKeyframe(targetIndex);
  }
  
  if (needsReset) {
    webcodecsDecodeQueue = [];
    webcodecsDecodedMax = startIndex - 1;
    webcodecsFrameOrder.forEach((index) => {
      const bitmap = webcodecsFrameCache.get(index);
      if (bitmap) {
        bitmap.close();
      }
    });
    webcodecsFrameCache.clear();
    webcodecsFrameOrder = [];
    recreateWebcodecsDecoder();
    if (!webcodecsDecoder) {
      return;
    }
  }
  
  // Check decoder queue - don't overwhelm it (check AFTER potential reset)
  if (webcodecsDecoder.decodeQueueSize > 10) {
    return;
  }

  webcodecsPendingRender = targetIndex;
  
  for (let i = webcodecsDecodedMax + 1; i <= targetIndex; i += 1) {
    const sample = webcodecsSamples[i];
    if (!sample || !sample.data || sample.data.byteLength === 0) {
      continue;
    }
    
    // Ensure we start with a keyframe after decoder reset
    if (webcodecsDecodedMax === -1 && !sample.is_sync) {
      continue;
    }
    
    try {
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? "key" : "delta",
        timestamp: Math.round((sample.cts * 1000000) / sample.timescale),
        duration: Math.round((sample.duration * 1000000) / sample.timescale),
        data: sample.data
      });
      webcodecsDecodeQueue.push(i);
      webcodecsDecoder.decode(chunk);
      webcodecsDecodedMax = i;
    } catch (err) {
      console.error(`[renderer] Decode error at sample ${i}:`, err.message);
      // If we hit a keyframe requirement error, reset decoder from nearest keyframe
      if (err.message && err.message.includes("key frame")) {
        const startIndex = findNearestKeyframe(i);
        webcodecsDecodeQueue = [];
        webcodecsDecodedMax = startIndex - 1;
        recreateWebcodecsDecoder();
      }
      break;
    }
  }
};

const buildDecoderConfig = (track) => {
  const config = {
    codec: track.codec,
    codedWidth: track.video.width,
    codedHeight: track.video.height
  };
  
  // Use the pre-extracted description from the worker (preferred)
  if (track.description) {
    config.description = track.description;
    console.log("[renderer] Using extracted description:", track.description.byteLength, "bytes");
  } else if (track.avcC && track.avcC.buffer) {
    config.description = track.avcC.buffer;
  } else if (track.hvcC && track.hvcC.buffer) {
    config.description = track.hvcC.buffer;
  } else if (track.avcC) {
    config.description = track.avcC;
  } else if (track.hvcC) {
    config.description = track.hvcC;
  }
  
  console.log("[renderer] Decoder config:", config.codec, config.codedWidth, "x", config.codedHeight, "description:", !!config.description);
  return config;
};

const configureWebcodecsDecoder = async (track) => {
  const config = buildDecoderConfig(track);
  
  // Check if description is required but missing (H.264/H.265)
  const needsDescription = config.codec.startsWith("avc") || config.codec.startsWith("hvc") || config.codec.startsWith("hev");
  if (needsDescription && !config.description) {
    console.error("[renderer] Codec", config.codec, "requires description but none was extracted");
    setWebcodecsStatus("Codec config extraction failed");
    return null;
  }
  
  let support = null;
  try {
    support = await VideoDecoder.isConfigSupported(config);
    console.log("[renderer] Config support check:", support?.supported, support?.config?.codec);
  } catch (error) {
    console.error("[renderer] isConfigSupported failed:", error);
    support = null;
  }
  
  if (!support || !support.supported) {
    // Try without description as fallback (won't work for H.264 but might for others)
    if (config.description) {
      console.warn("[renderer] Config not supported with description, trying without...");
      delete config.description;
      try {
        support = await VideoDecoder.isConfigSupported(config);
      } catch (error) {
        support = null;
      }
    }
  }
  
  if (!support || !support.supported) {
    console.error("[renderer] No supported config found for codec:", config.codec);
    return null;
  }

  webcodecsConfig = support.config || config;
  console.log("[renderer] Final config:", webcodecsConfig.codec, "description:", !!webcodecsConfig.description);
  return recreateWebcodecsDecoder() ? webcodecsConfig : null;
};

const createMp4Worker = () => {
  if (mp4boxWorker) {
    mp4boxWorker.terminate();
  }
  mp4boxWorker = new Worker("mp4box-worker.js", { type: "module" });
  mp4boxChunkResolvers = new Map();
  mp4boxChunkId = 0;
  return mp4boxWorker;
};

const initWebcodecsFromFile = async (file) => {
  resetWebcodecsState();
  if (!webcodecsSupported()) {
    setStatus("WebCodecs is not supported on this system.");
    setWebcodecsStatus("WebCodecs unsupported.");
    return false;
  }

  const filePath = file && file.path ? file.path : "";
  if (!filePath || !window.api || typeof window.api.getFileSize !== "function" || typeof window.api.readFileChunk !== "function") {
    setStatus("Failed to read video file.");
    setWebcodecsStatus("Video read failed: no file path or reader.");
    return false;
  }

  let selectedTrack = null;
  let mp4boxModule = null;

  try {
    setWebcodecsStatus("Loading MP4 parser...");
    mp4boxModule = await import("./node_modules/mp4box/dist/mp4box.all.js");
    console.log("[renderer-direct] mp4box module loaded, keys:", Object.keys(mp4boxModule).slice(0, 10).join(", "));
  } catch (importErr) {
    console.error("[renderer-direct] Failed to import mp4box:", importErr);
    setStatus("Failed to load MP4 parser.");
    setWebcodecsStatus("MP4 parser import failed: " + (importErr.message || importErr));
    return false;
  }

  const MP4Box = mp4boxModule.default || mp4boxModule;
  const DataStream = MP4Box.DataStream || mp4boxModule.DataStream;
  if (!MP4Box || typeof MP4Box.createFile !== "function") {
    setStatus("MP4 parser unavailable.");
    setWebcodecsStatus("MP4Box.createFile not found.");
    return false;
  }

  let mp4file = MP4Box.createFile();
  if (!mp4file) {
    setStatus("MP4 parser init failed.");
    return false;
  }
  console.log("[renderer-direct] mp4boxfile created");

  return new Promise(async (resolve) => {
    let resolved = false;
    const finish = (value) => {
      if (resolved) return;
      console.log("[renderer-direct] finish:", value);
      resolved = true;
      resolve(value);
    };

    mp4file.onError = (error) => {
      console.error("[renderer-direct] MP4Box error:", error);
      setStatus("MP4 parse error.");
      setWebcodecsStatus("MP4Box error: " + (error?.message || error));
      finish(false);
    };

    mp4file.onMoovStart = () => {
      console.log("[renderer-direct] moov atom parsing started");
      setWebcodecsStatus("MP4 moov atom found...");
    };

    mp4file.onReady = (info) => {
      console.log("[renderer-direct] onReady, tracks:", info.tracks.length);
      const vTrack = info.tracks.find((t) => t.type === "video");
      if (!vTrack) {
        setStatus("No video track found.");
        finish(false);
        return;
      }
      console.log("[renderer-direct] Video track:", vTrack.id, vTrack.codec, "samples in table:", vTrack.nb_samples);

      let description = null;
      try {
        const trak = mp4file.getTrackById(vTrack.id);
        if (trak?.mdia?.minf?.stbl?.stsd) {
          const entries = trak.mdia.minf.stbl.stsd.entries || [];
          for (const entry of entries) {
            const codecBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
            if (codecBox && DataStream) {
              const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
              codecBox.write(stream);
              if (stream.buffer.byteLength > 8) {
                description = new Uint8Array(stream.buffer, 8);
                console.log("[renderer-direct] description:", description.byteLength, "bytes");
              }
              break;
            }
          }
        }
      } catch (err) {
        console.warn("[renderer-direct] description extraction failed:", err);
      }

      selectedTrack = { ...vTrack, description };
      setWebcodecsStatus("Metadata ready, reading samples...");
      console.log("[renderer-direct] Track metadata captured");
    };

    try {
      const totalBytes = await window.api.getFileSize(filePath);
      if (!Number.isFinite(totalBytes) || totalBytes <= 0) {
        throw new Error("File size is " + totalBytes);
      }
      console.log("[renderer-direct] File size:", totalBytes);

      const chunkSize = 512 * 1024;
      let offset = 0;
      let chunkCount = 0;

      while (offset < totalBytes) {
        const length = Math.min(chunkSize, totalBytes - offset);
        const chunk = await window.api.readFileChunk(filePath, offset, length);
        const mp4Buffer = chunk instanceof ArrayBuffer ? chunk : (chunk && chunk.buffer);
        if (!mp4Buffer) throw new Error("Empty chunk at offset " + offset);

        mp4Buffer.fileStart = offset;
        const isLast = offset + length >= totalBytes;

        try {
          const nextOff = mp4file.appendBuffer(mp4Buffer, isLast);
          offset = typeof nextOff === "number" && nextOff > offset ? nextOff : offset + length;
        } catch (appendErr) {
          console.error("[renderer-direct] appendBuffer failed at offset", offset, ":", appendErr);
          throw appendErr;
        }

        chunkCount++;
        if (chunkCount % 50 === 0 || isLast) {
          const pct = Math.floor((offset / totalBytes) * 100);
          if (!resolved) setWebcodecsStatus("Reading MP4... " + pct + "%");
        }

        // Once we have the track metadata (onReady fired), stop feeding mp4box.
        if (selectedTrack) {
          console.log("[renderer-direct] Track metadata obtained after", chunkCount, "chunks");
          break;
        }

        if (chunkCount % 10 === 0) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      if (!selectedTrack) {
        setStatus("No video track found in file.");
        finish(false);
        return;
      }

      // Get sample metadata from mp4box, then discard it entirely.
      // We read sample data directly from the file using offset/size metadata.
      const trackId = selectedTrack.id;
      const rawSamples = mp4file.getTrackSamplesInfo(trackId);
      const totalSamples = rawSamples ? rawSamples.length : 0;
      console.log("[renderer-direct] Track has", totalSamples, "samples in metadata");

      // Copy only the metadata fields we need, then free mp4box
      const sampleMeta = [];
      for (let i = 0; i < totalSamples; i++) {
        const s = rawSamples[i];
        sampleMeta.push({
          offset: s.offset, size: s.size, duration: s.duration,
          cts: s.cts, dts: s.dts, timescale: s.timescale,
          is_sync: s.is_sync, number: s.number ?? i,
          track_id: s.track_id ?? trackId
        });
      }
      try { mp4file.stop(); } catch (_) {}

      if (totalSamples === 0) {
        setStatus("Video has no samples.");
        finish(false);
        return;
      }

      // Read compressed frame data directly from the file in batched regions
      let extracted = 0;
      const BATCH = 500;

      for (let batchStart = 0; batchStart < totalSamples; batchStart += BATCH) {
        const batchEnd = Math.min(batchStart + BATCH, totalSamples);
        const regionStart = sampleMeta[batchStart].offset;
        const regionEnd = sampleMeta[batchEnd - 1].offset + sampleMeta[batchEnd - 1].size;
        const regionSize = regionEnd - regionStart;

        let regionView = null;
        if (regionSize > 0 && regionSize < 50 * 1024 * 1024) {
          try {
            const regionBuf = await window.api.readFileChunk(filePath, regionStart, regionSize);
            regionView = new Uint8Array(regionBuf);
          } catch (e) {
            console.warn("[renderer-direct] Batch read failed, offset:", regionStart, "size:", regionSize);
          }
        }

        for (let i = batchStart; i < batchEnd; i++) {
          const si = sampleMeta[i];
          let dataCopy = null;

          if (regionView && si.offset >= regionStart && si.offset + si.size <= regionStart + regionView.byteLength) {
            const lo = si.offset - regionStart;
            dataCopy = new Uint8Array(si.size);
            dataCopy.set(regionView.subarray(lo, lo + si.size));
          } else {
            try {
              const buf = await window.api.readFileChunk(filePath, si.offset, si.size);
              dataCopy = new Uint8Array(buf);
            } catch (_) { continue; }
          }

          if (si.is_sync) webcodecsKeyframes.push(webcodecsSamples.length);
          webcodecsSamples.push({
            data: dataCopy, size: si.size, duration: si.duration,
            cts: si.cts, dts: si.dts, timescale: si.timescale,
            is_sync: si.is_sync, number: si.number,
            track_id: si.track_id, dataType: "Uint8Array"
          });
          extracted++;
        }

        regionView = null;

        // Configure decoder once after first batch
        if (!resolved && extracted >= 30 && webcodecsDecoder === null) {
          console.log("[renderer-direct] Configuring decoder with", extracted, "samples");
          try {
            const config = await configureWebcodecsDecoder(selectedTrack);
            if (config) {
              console.log("[renderer-direct] Decoder configured");
              finish(true);
            } else {
              setStatus("WebCodecs codec not supported.");
              finish(false);
              return;
            }
          } catch (cfgErr) {
            console.error("[renderer-direct] Decoder config error:", cfgErr);
            setStatus("Decoder config failed: " + (cfgErr.message || cfgErr));
            finish(false);
            return;
          }
        }

        const pct = Math.floor((batchEnd / totalSamples) * 100);
        setWebcodecsStatus("Loading frames... " + pct + "% (" + extracted + "/" + totalSamples + ")");
        await new Promise((r) => setTimeout(r, 0));
      }

      console.log("[renderer-direct] All samples loaded:", extracted, "/", totalSamples);
      setWebcodecsStatus("");

      if (!resolved && extracted > 0) {
        try {
          const config = await configureWebcodecsDecoder(selectedTrack);
          if (config) finish(true);
          else { setStatus("WebCodecs not supported."); finish(false); }
        } catch (e) { setStatus("Decoder failed."); finish(false); }
      } else if (!resolved) {
        setStatus("Could not read video samples.");
        finish(false);
      }

    } catch (error) {
      console.error("[renderer-direct] Error:", error);
      setStatus("Failed to read/parse video.");
      setWebcodecsStatus("Error: " + (error.message || error));
      finish(false);
    }
  }).then((success) => {
    console.log("[renderer] Promise resolved, success:", success, "selectedTrack:", !!selectedTrack, "samples:", webcodecsSamples.length);
    
    if (!success || !selectedTrack) {
      console.log("[renderer] Returning false due to !success or !selectedTrack");
      return false;
    }
    if (!webcodecsSamples.length) {
      setStatus("No samples extracted for WebCodecs.");
      setWebcodecsStatus("No samples extracted.");
      return false;
    }

    // Calculate duration from track metadata
    let durationSeconds = selectedTrack.duration / selectedTrack.timescale;
    
    // Fallback: calculate from last sample
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      const lastSample = webcodecsSamples[webcodecsSamples.length - 1];
      if (lastSample) {
        durationSeconds = (lastSample.cts + lastSample.duration) / lastSample.timescale;
      }
    }
    
    // Infer FPS from track timescale and sample duration
    if (!fpsLockInput.checked && durationSeconds > 0) {
      // Use timescale/duration ratio for accurate FPS
      const firstSample = webcodecsSamples[0];
      if (firstSample && firstSample.timescale && firstSample.duration) {
        const inferredFps = firstSample.timescale / firstSample.duration;
        if (inferredFps >= 1 && inferredFps <= 240) {
          fpsInput.value = inferredFps.toFixed(3);
          frameDurationMs = 1000 / inferredFps;
        } else {
          fpsInput.value = "30";
          frameDurationMs = 1000 / 30;
        }
      } else {
        fpsInput.value = "30";
        frameDurationMs = 1000 / 30;
      }
    } else if (fpsLockInput.checked) {
      applyLockedFps();
    }

    webcodecsDurationMs = durationSeconds * 1000;
    videoCanvas.width = selectedTrack.video.width;
    videoCanvas.height = selectedTrack.video.height;
    
    webcodecsReady = true;
    updateDurationDisplay();
    updateTrackbar();
    
    // Decode first frame
    try {
      if (webcodecsSamples.length === 0) {
        setStatus("No video samples available.");
        return false;
      }
      decodeWebcodecsUpTo(0);
      setStatus("WebCodecs ready.");
      setWebcodecsStatus("");
      return true;
    } catch (err) {
      console.error("[renderer] Failed to decode first frame:", err);
      setStatus("WebCodecs decode failed; falling back to HTML5.");
      setWebcodecsStatus(`Decode error: ${err.message || err}`);
      return false;
    }
  }).catch((err) => {
    console.error("[renderer] WebCodecs initialization failed:", err);
    setStatus("WebCodecs failed; falling back to HTML5.");
    setWebcodecsStatus(`Init error: ${err.message || err}`);
    return false;
  });
};

const csvEscape = (value) => {
  const stringValue = String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
};

const getVideoTimeMs = () => Math.round(video.currentTime * 1000);

const getFrameTimeMs = (frameIndex) => {
  // Use frame index * duration for consistent monotonic time display
  // Note: Videos with B-frames have non-monotonic CTS in decode order,
  // so using actual CTS would cause time to jump around when stepping frames
  return frameIndex * frameDurationMs;
};

const updateTimeDisplay = () => {
  currentTimeEl.textContent = formatTime(getFrameTimeMs(currentFrameIndex));
  updateAlFrameInfo();
};

const seekHtml5ToMs = (milliseconds) => {
  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }
  const target = Math.min(
    Math.max(0, milliseconds / 1000),
    Number.isFinite(video.duration) ? video.duration : Infinity
  );
  video.currentTime = target;
  updateTimeDisplay();
  updateTrackbar();
};

const seekToMs = (milliseconds) => {
  if (webcodecsMode) {
    const targetIndex = Math.round(milliseconds / frameDurationMs);
    seekToFrame(targetIndex);
    return;
  }
  seekHtml5ToMs(milliseconds);
};

const seekToFrame = (frameIndex) => {
  const maxIndex = webcodecsSamples.length > 0 ? webcodecsSamples.length - 1 : 0;
  const safeIndex = Math.max(0, Math.min(frameIndex, maxIndex));
  currentFrameIndex = safeIndex;
  if (webcodecsMode) {
    decodeWebcodecsUpTo(safeIndex);
    return;
  }
  pendingSeekFrame = safeIndex;
  pendingSeekTime = (safeIndex * frameDurationMs) / 1000;
  suppressTimeUpdate = true;
  seekHtml5ToMs(safeIndex * frameDurationMs);
};
