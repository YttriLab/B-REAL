// core/timeline.js — The uncertainty / misclassification timeline canvas.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Uncertainty timeline canvas ───────────────────────────────────────────

const uncertaintyCanvas = document.getElementById("uncertaintyCanvas");
const ucCtx = uncertaintyCanvas.getContext("2d");

function renderUncertaintyOverlay() {
  const showOnTimeline = document.getElementById("alShowTimeline").checked;
  const arr = alState.uncertaintyArray;

  if (!showOnTimeline || !arr) {
    uncertaintyCanvas.classList.remove("visible");
    return;
  }
  uncertaintyCanvas.classList.add("visible");

  const W = uncertaintyCanvas.clientWidth;
  const H = 20;
  uncertaintyCanvas.width = W;
  uncertaintyCanvas.height = H;

  const N = arr.length;
  if (N === 0 || W === 0) return;

  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  // Only frames inside qualifying segments get the threshold marker.
  const mask = getUncertainMask(arr, threshold, minLen);

  for (let px = 0; px < W; px++) {
    const startFrame = Math.floor((px / W) * N);
    const endFrame = Math.min(Math.floor(((px + 1) / W) * N), N);

    // Get max uncertainty in this pixel's frame range
    let maxU = -1;
    let inSegment = false;
    for (let i = startFrame; i < endFrame; i++) {
      if (arr[i] >= 0 && arr[i] > maxU) maxU = arr[i];
      if (mask[i]) inSegment = true;
    }

    if (maxU < 0) {
      // No data — draw grey
      ucCtx.fillStyle = "#d4d8e3";
    } else {
      // HSL: green (120) to red (0) based on uncertainty
      const hue = (1 - maxU) * 120;
      ucCtx.fillStyle = `hsl(${hue}, 85%, 50%)`;
    }
    ucCtx.fillRect(px, 0, 1, H);

    // Draw threshold marker only for qualifying segments
    if (inSegment) {
      ucCtx.fillStyle = "rgba(0, 0, 0, 0.3)";
      ucCtx.fillRect(px, 0, 1, 3);
    }

    // Draw misclassification marker (magenta tick at bottom)
    const misArr = alState.misclassifiedArray;
    if (misArr) {
      let hasMis = false;
      for (let i = startFrame; i < endFrame && i < misArr.length; i++) {
        if (misArr[i] === 1) { hasMis = true; break; }
      }
      if (hasMis) {
        ucCtx.fillStyle = "rgba(200, 0, 200, 0.85)";
        ucCtx.fillRect(px, H - 4, 1, 4);
      }
    }
  }
}

// Click on uncertainty canvas to seek
uncertaintyCanvas.addEventListener("click", (e) => {
  if (!alState.uncertaintyArray) return;
  const rect = uncertaintyCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = x / rect.width;
  const frameIndex = Math.floor(pct * alState.uncertaintyArray.length);
  seekToFrame(frameIndex);
});

// Redraw on resize
window.addEventListener("resize", () => {
  if (alState.uncertaintyArray) renderUncertaintyOverlay();
});

// Jump to the start of the next qualifying uncertain segment
document.getElementById("alJumpNext").addEventListener("click", () => {
  if (!alState.uncertaintyArray) { alert("Run inference first."); return; }
  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  const segments = getUncertainSegments(alState.uncertaintyArray, threshold, minLen);

  if (segments.length === 0) {
    alert(`No uncertain segments of ${minLen}+ consecutive frames above threshold.`);
    return;
  }

  // Next segment starting after the current frame, else wrap to the first.
  const next = segments.find((s) => s.start > currentFrameIndex) || segments[0];
  seekToFrame(next.start);
  setStatus(`Uncertain segment: frames ${next.start}–${next.end} (${next.length} frames).`);
});

// Rank qualifying uncertain segments worst-first (by their single lowest-
// confidence frame). Each entry: { start, end, length, peakIndex, peakUnc }.
// With min-segment-length = 1 each entry is a single frame, so this covers both
// "lowest-confidence frame" and "lowest-confidence frame set".
function rankLowConfidenceSegments() {
  const arr = alState.uncertaintyArray;
  if (!arr) return [];
  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  const ranked = getUncertainSegments(arr, threshold, minLen).map((s) => {
    let peakIndex = s.start;
    let peakUnc = arr[s.start];
    for (let i = s.start + 1; i <= s.end; i++) {
      if (arr[i] > peakUnc) { peakUnc = arr[i]; peakIndex = i; }
    }
    return { ...s, peakIndex, peakUnc };
  });
  // Worst (highest uncertainty / lowest confidence) first.
  ranked.sort((a, b) => b.peakUnc - a.peakUnc);
  return ranked;
}

// Jump to the next-lowest-confidence frame set, walking the ranking worst-first.
document.getElementById("alJumpLowestConf").addEventListener("click", () => {
  if (!alState.uncertaintyArray) { alert("Run inference first."); return; }
  const ranked = rankLowConfidenceSegments();
  if (ranked.length === 0) {
    const minLen = getMinSegmentLength();
    alert(`No uncertain segments of ${minLen}+ consecutive frames above threshold.`);
    return;
  }

  // Advance the worst-first cursor (wraps at the end).
  alState.lowConfCursor = (alState.lowConfCursor + 1) % ranked.length;
  const item = ranked[alState.lowConfCursor];
  seekToFrame(item.peakIndex);

  // Prefer the model's actual max-probability as "confidence"; fall back to
  // 1 - normalized-entropy if the per-frame prediction vector isn't available.
  const f = alState.frameByIndex ? alState.frameByIndex.get(item.peakIndex) : null;
  const conf = f && Array.isArray(f.predictions) && f.predictions.length
    ? Math.max(...f.predictions)
    : 1 - item.peakUnc;
  const where = item.length > 1
    ? `frames ${item.start}–${item.end}, worst frame ${item.peakIndex}`
    : `frame ${item.peakIndex}`;
  setStatus(
    `Lowest confidence #${alState.lowConfCursor + 1}/${ranked.length}: ${where} ` +
    `(confidence ${conf.toFixed(2)}).`
  );
});

// Jump to next misclassified frame
document.getElementById("alJumpNextMisclassified").addEventListener("click", () => {
  if (!alState.misclassifiedArray) { alert("Run inference first."); return; }
  const arr = alState.misclassifiedArray;
  const start = currentFrameIndex + 1;

  // Search forward
  for (let i = start; i < arr.length; i++) {
    if (arr[i] === 1) {
      seekToFrame(i);
      return;
    }
  }
  // Wrap around
  for (let i = 0; i < start && i < arr.length; i++) {
    if (arr[i] === 1) {
      seekToFrame(i);
      return;
    }
  }
  alert("No misclassified frames found.");
});

// Export model predictions as a frame-indexed one-hot matrix — the SAME format
// as labels (generateTrainingLabelCsv), so they can be re-imported as labels.
const generatePredictionsCsv = () => {
  const frames = alState.results.frames;
  const classNames = alState.results.class_names || [];
  const header = ["time", ...classNames.map(csvEscape)].join(",");
  const rows = frames.map((f) => {
    const cols = new Array(classNames.length).fill(0);
    if (f.predicted_class >= 0 && f.predicted_class < classNames.length) {
      cols[f.predicted_class] = 1;
    }
    return f.index + "," + cols.join(",");
  });
  return [header, ...rows].join("\n");
};

// Collapse per-frame predictions into contiguous same-class intervals, in the
// same {startFrame, endFrame, behaviorName} shape as a bout — so the labeled
// video renderer, which already knows how to burn bouts in, takes them as-is.
// "other" runs are skipped: it means "no behavior", not a behavior to caption.
function predictionSegments() {
  if (!alState.results || !alState.results.frames.length) return [];
  const classNames = alState.results.class_names || [];
  const frames = alState.results.frames;

  const segments = [];
  let run = null;
  for (const f of frames) {
    const name = classNames[f.predicted_class] ?? `class ${f.predicted_class}`;
    // Break the run on a class change OR a gap in frame indices (scoped
    // inference returns only the requested ranges).
    if (run && run.behaviorName === name && f.index === run.endFrame + 1) {
      run.endFrame = f.index;
    } else {
      if (run && run.behaviorName !== "other") segments.push(run);
      run = { startFrame: f.index, endFrame: f.index, behaviorName: name };
    }
  }
  if (run && run.behaviorName !== "other") segments.push(run);
  return segments;
}

document.getElementById("alPredictedVideo").addEventListener("click", async () => {
  if (!alState.results || !alState.results.frames.length) {
    alert("Run inference first.");
    return;
  }
  const segments = predictionSegments();
  if (segments.length === 0) {
    setStatus("Every frame was predicted 'other' — nothing to caption.");
    return;
  }
  setStatus(`Rendering ${segments.length} predicted segments...`);
  await generateLabeledVideo(segments);
});

document.getElementById("alExportPredictions").addEventListener("click", async () => {
  if (!alState.results || !alState.results.frames.length) {
    alert("Run inference first.");
    return;
  }
  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }
  const content = generatePredictionsCsv();
  const baseName = currentVideoName ? currentVideoName.replace(/\.[^/.]+$/, "") : "video";
  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-predictions.csv`,
    content,
  });
  setStatus(result.canceled ? "Export canceled." : `Predictions saved to ${result.filePath}`);
});
