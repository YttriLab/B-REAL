// core/csv.js — CSV import/export, one-hot label files, labeled-video generation.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── CSV line parser (shared) ────────────────────────────────────────────
const parseCsvLine = (line) => {
  const fields = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') { current += '"'; i++; }
      else if (ch === '"') { inQuotes = false; }
      else { current += ch; }
    } else {
      if (ch === '"') { inQuotes = true; }
      else if (ch === ',') { fields.push(current.trim()); current = ""; }
      else { current += ch; }
    }
  }
  fields.push(current.trim());
  return fields;
};

// ── Parse CSV content and import into bouts/behaviors ─────────────────
const parseCsvAndImport = (content, sourceName = "CSV") => {
  const lines = content.split(/\r?\n/).filter((l) => l.trim());
  if (lines.length < 2) {
    setStatus("CSV file is empty or has no data rows.");
    return 0;
  }

  const headerFields = parseCsvLine(lines[0]);
  const headerLower = lines[0].toLowerCase();

  // Detect format: bout CSV ("start,end,behavior") vs frame-level label matrix
  const isBoutFormat = /start/.test(headerLower) && /end/.test(headerLower) && /behav/i.test(headerLower);
  const isFrameMatrix = /time/.test(headerLower) || (!isNaN(parseFloat(headerFields[0])) && headerFields.length > 3);

  let imported = 0;

  if (isBoutFormat) {
    // --- Bout format: start, end, behavior ---
    const dataLines = lines.slice(1);
    let skipped = 0;
    for (const line of dataLines) {
      const fields = parseCsvLine(line);
      if (fields.length < 3) { console.warn("[import] Row skipped (need 3 columns, got " + fields.length + "):", line); skipped++; continue; }

      const startMs = parseTime(fields[0]);
      const endMs = parseTime(fields[1]);
      const behaviorName = fields[2];

      if (startMs === null || endMs === null || !behaviorName) { console.warn("[import] Row skipped (parse failed) start='" + fields[0] + "' end='" + fields[1] + "' behavior='" + fields[2] + "' → startMs=" + startMs + " endMs=" + endMs); skipped++; continue; }

      const startFrame = msToFrame(startMs);
      const endFrame = msToFrame(endMs);
      if (startFrame === endFrame) { console.warn("[import] Row skipped (zero-length) start='" + fields[0] + "' end='" + fields[1] + "' → frame " + startFrame); skipped++; continue; }

      let behavior = findBehaviorByName(behaviorName);
      if (!behavior) {
        const key = KEY_ORDER[behaviors.length] || "";
        behavior = { id: Date.now() + "-" + Math.random(), name: normalizeBehaviorName(behaviorName), key };
        behaviors.push(behavior);
      }

      insertBoutSorted({
        behaviorId: behavior.id,
        behaviorName: behavior.name,
        startFrame: Math.min(startFrame, endFrame),
        endFrame: Math.max(startFrame, endFrame)
      });
      imported++;
    }
    const skippedMsg = skipped > 0 ? ` (${skipped} rows skipped)` : "";
    setStatus(`Imported ${imported} bouts from ${sourceName}${skippedMsg}`);

  } else if (isFrameMatrix) {
    // --- Frame-level label matrix: time, behavior1, behavior2, ... ---
    // First column is frame index, remaining columns are 0/1 per behavior.
    const hasTextHeader = isNaN(parseFloat(headerFields[0]));
    const behaviorNames = hasTextHeader
      ? headerFields.slice(1)
      : headerFields.slice(1).map((_, i) => "Behavior " + (i + 1));
    const dataLines = hasTextHeader ? lines.slice(1) : lines;

    console.log("[import] Detected frame-level label matrix with", behaviorNames.length, "behaviors:", behaviorNames.join(", "));

    // Parse all rows into per-behavior frame arrays to find contiguous runs
    const numBehaviors = behaviorNames.length;
    const active = new Array(numBehaviors).fill(false);
    const startFrames = new Array(numBehaviors).fill(0);
    let prevFrame = null;

    const closeBout = (b, endFrame) => {
      let behavior = findBehaviorByName(behaviorNames[b]);
      if (!behavior) {
        const key = KEY_ORDER[behaviors.length] || "";
        behavior = { id: Date.now() + "-" + Math.random(), name: normalizeBehaviorName(behaviorNames[b]), key };
        behaviors.push(behavior);
      }
      insertBoutSorted({
        behaviorId: behavior.id,
        behaviorName: behavior.name,
        startFrame: startFrames[b],
        endFrame
      });
      imported++;
    };

    for (const line of dataLines) {
      const fields = parseCsvLine(line);
      const frameIdx = parseInt(fields[0], 10);
      if (isNaN(frameIdx)) continue;

      // Sparse matrices (only labeled frames present) have no explicit "off"
      // row at a bout's end — a gap in the frame index implies the run ended at
      // the previous present frame. Close any still-open bouts at that boundary.
      if (prevFrame !== null && frameIdx > prevFrame + 1) {
        for (let b = 0; b < numBehaviors; b++) {
          if (active[b]) { active[b] = false; closeBout(b, prevFrame); }
        }
      }
      prevFrame = frameIdx;

      for (let b = 0; b < numBehaviors; b++) {
        const val = parseFloat(fields[b + 1]);
        const on = val >= 0.5;

        if (on && !active[b]) {
          active[b] = true;
          startFrames[b] = frameIdx;
        } else if (!on && active[b]) {
          active[b] = false;
          closeBout(b, frameIdx - 1);
        }
      }
    }

    // Close any still-active bouts at the last present frame
    const lastFields = parseCsvLine(dataLines[dataLines.length - 1]);
    const lastFrame = parseInt(lastFields[0], 10);
    for (let b = 0; b < numBehaviors; b++) {
      if (active[b] && !isNaN(lastFrame)) closeBout(b, lastFrame);
    }

    setStatus(`Imported ${imported} bouts (from frame-level labels) from ${sourceName}`);

  } else {
    setStatus("Unrecognized CSV format. Expected 'start,end,behavior' columns or a frame-level label matrix.");
    return 0;
  }

  if (imported > 0) {
    mergeDuplicateBehaviors();
    saveBehaviors();
    renderBehaviors();
    renderBouts();
    saveProgress();
  }
  return imported;
};

// ── Import CSV via file dialog ────────────────────────────────────────
const importCsv = async () => {
  if (!window.api || !window.api.openCsv) {
    setStatus("Label import is unavailable.");
    return;
  }
  const result = await window.api.openCsv();
  if (result.canceled) return;
  let content;
  if (result.isXlsx) {
    if (!window.api.xlsxToCsv) { setStatus("XLSX support unavailable."); return; }
    content = await window.api.xlsxToCsv(result.filePath);
  } else {
    content = result.content;
  }
  if (!content) { setStatus("Import canceled or file empty."); return; }
  parseCsvAndImport(content, result.filePath || "Labels");
};

const exportOneHot = async () => {
  if (bouts.length === 0) {
    setStatus("No bouts to export.");
    return;
  }

  // Determine frame range: 0 to max endFrame across all bouts
  let maxFrame = 0;
  for (const bout of bouts) {
    if (bout.endFrame > maxFrame) maxFrame = bout.endFrame;
  }

  // Collect unique behavior names in the order they first appear
  const behaviorNames = [];
  const seen = new Set();
  for (const bout of bouts) {
    if (!seen.has(bout.behaviorName)) {
      seen.add(bout.behaviorName);
      behaviorNames.push(bout.behaviorName);
    }
  }

  const fillBehavior = (fillBehaviorInput.value || "").trim();
  if (fillBehavior && !behaviorNames.includes(fillBehavior)) {
    behaviorNames.push(fillBehavior);
  }
  const fillIdx = fillBehavior ? behaviorNames.indexOf(fillBehavior) : -1;

  const minFrame = fillIdx >= 0
    ? bouts.reduce((m, b) => Math.min(m, b.startFrame), Infinity)
    : 0;

  // Build one-hot rows from minFrame..maxFrame (matching fill_missing_frames_onehot range)
  const header = ["time", ...behaviorNames.map(csvEscape)].join(",");
  const rows = [];
  for (let f = minFrame; f <= maxFrame; f++) {
    const cols = new Array(behaviorNames.length).fill(0);
    let hasLabel = false;
    for (const bout of bouts) {
      if (f >= bout.startFrame && f <= bout.endFrame) {
        const idx = behaviorNames.indexOf(bout.behaviorName);
        if (idx >= 0) { cols[idx] = 1; hasLabel = true; }
      }
    }
    if (!hasLabel && fillIdx >= 0) cols[fillIdx] = 1;
    rows.push(f + "," + cols.join(","));
  }
  const content = [header, ...rows].join("\n");

  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-onehot.csv`,
    content
  });

  if (result.canceled) {
    setStatus("Export canceled.");
  } else {
    setStatus(`One-hot CSV saved to ${result.filePath} (${maxFrame + 1} frames, ${behaviorNames.length} behaviors)`);
  }
};

// Burn behavior labels into a copy of the video with ffmpeg drawtext.
// `segments` defaults to the hand-labeled bouts; the Active Learning tab passes
// model predictions instead (A-SOID's `create_labeled_vid`), which is the same
// rendering job over a different source of intervals.
const generateLabeledVideo = async (segments = null) => {
  const source = segments || bouts;
  if (source.length === 0) {
    setStatus(segments ? "No predictions to overlay." : "No bouts to overlay.");
    return;
  }

  const videoPath = currentFile && currentFile.path ? currentFile.path : "";
  if (!videoPath) {
    setStatus("No video loaded. Load a video first.");
    return;
  }

  if (!window.api || !window.api.saveVideoDialog) {
    setStatus("File save is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const saveResult = await window.api.saveVideoDialog({
    defaultPath: `${baseName}-${segments ? "predicted" : "labeled"}.mp4`
  });

  if (saveResult.canceled || !saveResult.filePath) {
    setStatus("Labeled video canceled.");
    return;
  }

  const outputPath = saveResult.filePath;
  const fps = parseFloat(fpsInput.value) || 30;

  // Build ffmpeg drawtext filters for each interval
  const filters = source.map((bout) => {
    const startSec = (bout.startFrame * frameDurationMs) / 1000;
    const endSec = (bout.endFrame * frameDurationMs) / 1000;
    const escaped = bout.behaviorName.replace(/'/g, "'\\''").replace(/:/g, "\\:");
    return `drawtext=text='${escaped}':fontsize=36:fontcolor=white:borderw=2:bordercolor=black:x=w*0.7:y=h*0.15:enable='between(t,${startSec.toFixed(4)},${endSec.toFixed(4)})'`;
  });

  const filterStr = filters.join(",");

  if (!window.api.runFfmpeg) {
    const cmd = `ffmpeg -i "${videoPath}" -vf "${filterStr}" -c:v libx264 -preset fast -crf 18 -c:a aac "${outputPath}"`;
    setStatus("ffmpeg not available in app. Run this command manually:");
    console.log("[labeled-video] ffmpeg command:\n" + cmd);
    try { await navigator.clipboard.writeText(cmd); } catch (_) {}
    setStatus("ffmpeg command copied to clipboard. Paste into a terminal to generate the labeled video.");
    return;
  }

  setStatus("Generating labeled video...");
  try {
    const result = await window.api.runFfmpeg({
      inputPath: videoPath,
      outputPath,
      filterStr,
      fps
    });
    if (result.success) {
      setStatus(`Labeled video saved to ${outputPath}`);
    } else {
      setStatus("ffmpeg failed: " + (result.error || "unknown error"));
      console.error("[labeled-video] ffmpeg stderr:", result.stderr);
    }
  } catch (err) {
    setStatus("ffmpeg error: " + (err.message || err));
  }
};

video.addEventListener("loadedmetadata", handleVideoLoaded);
video.addEventListener("loadeddata", startFrameWatch);
video.addEventListener("timeupdate", () => {
  if (webcodecsMode) {
    return;
  }
  if (!suppressTimeUpdate) {
    currentFrameIndex = Math.round(
      (video.currentTime * 1000) / frameDurationMs
    );
  }
  updateTimeDisplay();
  updateTrackbar();
});
video.addEventListener("ended", () => {
  if (webcodecsMode) {
    return;
  }
  playPauseButton.textContent = "Play";
});

video.addEventListener("seeked", () => {
  if (webcodecsMode) {
    return;
  }
  if (pendingSeekFrame === null) {
    return;
  }
  const delta = Math.abs(video.currentTime - (pendingSeekTime || 0));
  if (delta > (frameDurationMs / 1000) * 0.5) {
    setStatus("Frame seek limited by codec; landed on nearest frame.");
  }
  suppressTimeUpdate = false;
  pendingSeekFrame = null;
  pendingSeekTime = null;
  updateTimeDisplay();
  updateTrackbar();
});
