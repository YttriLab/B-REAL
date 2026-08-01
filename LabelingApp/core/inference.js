// core/inference.js — Inference scope selection and the model's label space.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Inference scope (entire video / frame range / labeled frames) ──────────
const alScopeSelect = document.getElementById("alScope");
const alRangeRow = document.getElementById("alRangeRow");
const alLabeledHint = document.getElementById("alLabeledHint");

function updateAlScopeVisibility() {
  const scope = alScopeSelect.value;
  alRangeRow.style.display = scope === "range" ? "" : "none";
  alLabeledHint.style.display = scope === "labeled" ? "" : "none";
}
alScopeSelect.addEventListener("change", updateAlScopeVisibility);
updateAlScopeVisibility();

document.getElementById("alRangeStartCur").addEventListener("click", () => {
  document.getElementById("alRangeStart").value = currentFrameIndex;
});
document.getElementById("alRangeEndCur").addEventListener("click", () => {
  document.getElementById("alRangeEnd").value = currentFrameIndex;
});

// Resolve the scope selection into frame_ranges for the backend.
// Returns { ranges } where ranges is null (entire video) or a list of
// inclusive [start, end] pairs, or { error } if the selection is invalid.
function resolveAlFrameRanges() {
  const scope = alScopeSelect.value;
  if (scope === "all") {
    // "Entire video" means exactly that — not the entire *pose file*, which may
    // be much longer (a full session's tracking against a short review clip).
    // Bounding it here keeps the uncertainty timeline aligned with the trackbar
    // and stops inference spending minutes on frames that can't be looked at.
    const vFrames = videoFrameCount();
    return { ranges: vFrames > 0 ? [[0, vFrames - 1]] : null };
  }

  if (scope === "range") {
    const start = parseInt(document.getElementById("alRangeStart").value, 10);
    const end = parseInt(document.getElementById("alRangeEnd").value, 10);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return { error: "Enter a valid start and end frame." };
    }
    const lo = Math.max(0, Math.min(start, end));
    const hi = Math.max(start, end);
    if (hi < lo) return { error: "End frame must be ≥ start frame." };
    return { ranges: [[lo, hi]] };
  }

  if (scope === "labeled") {
    if (!bouts.length) {
      return { error: "No labeled frames on this video. Add bouts first, or pick another scope." };
    }
    const ranges = bouts.map((b) => [b.startFrame, b.endFrame]);
    return { ranges };
  }

  return { ranges: null };
}

// Run inference on this video's pose file. Body parts and bin width come from
// the checkpoint itself, so the feature space always matches how it was trained.
document.getElementById("alRunInference").addEventListener("click", async () => {
  if (!alState.checkpointPath) { alert("Select a checkpoint first."); return; }

  // The shell doesn't know what this pipeline needs to run — it asks.
  const inference = window.labelerPlugins.capability("inference");
  if (!inference) { alert("No pipeline is registered to run inference."); return; }
  const ctx = makeContext();
  const problem = inference.validate ? inference.validate(ctx) : null;
  if (problem) { alert(problem); return; }

  const scopeResult = resolveAlFrameRanges();
  if (scopeResult.error) { alert(scopeResult.error); return; }
  const frameRanges = scopeResult.ranges;

  document.getElementById("alRunInference").disabled = true;
  document.getElementById("alStopInference").disabled = false;
  document.getElementById("alStatus").textContent = "Starting...";

  try {
    const res = await backendFetch("/api/inference/run", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_path: alState.checkpointPath,
        frame_ranges: frameRanges,
        ...inference.buildRequest(ctx),
      }),
    });
    alState.taskId = res.task_id;

    alState.sseSource = backendSSE(
      `/api/inference/status/${res.task_id}`,
      async (data) => {
        const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
        document.getElementById("alProgress").value = pct;
        document.getElementById("alStatus").textContent = `${data.status}: ${data.progress}/${data.total}`;

        if (data.status === "completed") {
          alState.sseSource?.close();
          document.getElementById("alRunInference").disabled = false;
          document.getElementById("alStopInference").disabled = true;

          // Fetch results
          try {
            const results = await backendFetch(`/api/inference/results/${alState.taskId}`);
            applyAlResults(results);
            persistPredictions();
            document.getElementById("alStatus").textContent = `Done: ${results.frames.length} frames`;
          } catch (e) {
            document.getElementById("alStatus").textContent = `Error fetching results: ${e.message}`;
          }
        } else if (data.status === "error" || data.status === "cancelled") {
          alState.sseSource?.close();
          document.getElementById("alRunInference").disabled = false;
          document.getElementById("alStopInference").disabled = true;
          if (data.error) document.getElementById("alStatus").textContent = `Error: ${data.error}`;
        }
      },
      () => {
        document.getElementById("alRunInference").disabled = false;
        document.getElementById("alStopInference").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("alStatus").textContent = `Error: ${e.message}`;
    document.getElementById("alRunInference").disabled = false;
    document.getElementById("alStopInference").disabled = true;
  }
});

// Stop inference
document.getElementById("alStopInference").addEventListener("click", async () => {
  if (alState.taskId) {
    try {
      await backendFetch(`/api/inference/stop/${alState.taskId}`, { method: "POST" });
    } catch { /* ignore */ }
  }
});

// Build dense uncertainty array from sparse results
function buildUncertaintyArray() {
  if (!alState.results || !alState.results.frames.length) {
    alState.uncertaintyArray = null;
    return;
  }
  const frames = alState.results.frames;
  const maxIndex = frames[frames.length - 1].index;
  const arr = new Float32Array(maxIndex + 1);
  // Fill with -1 (no data) then overwrite with actual values
  arr.fill(-1);
  for (const f of frames) {
    arr[f.index] = f.uncertainty;
  }
  alState.uncertaintyArray = arr;
}

// ── Model label space (binarized vs multi-class) ──────────────────────────────
// A model trained with a positive class predicts in a COLLAPSED label space:
// load_data sets class_names = ['other', <positive>]. Bouts keep their original
// fine-grained names, so comparing the two raw strings falsely reports correct
// predictions as errors (a 'walking' bout the model rightly calls 'other', or a
// union member 'groom' it rightly calls 'positive'). Map the label first.

// Training config recorded on the manifest entry for the active checkpoint
// (carries `binarize` / `label_union`). Null when the checkpoint isn't tracked.
function getAlModelConfig() {
  try {
    const pm = window.projectManager;
    if (!pm || !pm.isActive() || !pm.getModels || !alState.checkpointPath) return null;
    const entry = pm.getModels().find((m) => m.path === alState.checkpointPath);
    return (entry && entry.config) || null;
  } catch {
    return null;
  }
}

// Translate a bout's behavior name into the class this model would predict for
// it. Returns null when the mapping can't be determined — callers must NOT flag
// those frames, since a label-space gap is indistinguishable from a real error.
function mapLabelToModelClass(behaviorName, classNames, cfg) {
  if (!behaviorName || !classNames || !classNames.length) return null;
  if (classNames.includes(behaviorName)) return behaviorName; // multi-class: direct

  // Binarized models look like ['other', <positive>].
  const positiveName =
    classNames.length === 2 && classNames[0] === "other" ? classNames[1] : null;
  if (!positiveName) return null;

  // Which original behaviors make up the positive class?
  let members = null;
  if (cfg && cfg.label_union && Array.isArray(cfg.label_union[positiveName])) {
    members = cfg.label_union[positiveName];   // >=2 behaviors unioned under a synthetic name
  } else if (cfg && cfg.binarize === positiveName) {
    members = [positiveName];                  // single behavior binarized
  } else if (behaviors.some((b) => b.name === positiveName)) {
    members = [positiveName];                  // no recorded config, but a real behavior
  }
  // Synthetic union name with no recorded membership → unknowable.
  if (!members) return null;
  return members.includes(behaviorName) ? positiveName : "other";
}

// Build dense misclassification array by comparing predictions to labeled bouts
function buildMisclassifiedArray() {
  if (!alState.results || !alState.results.frames.length) {
    alState.misclassifiedArray = null;
    return;
  }
  const frames = alState.results.frames;
  const classNames = alState.results.class_names || [];
  const cfg = getAlModelConfig();
  const maxIndex = frames[frames.length - 1].index;
  const arr = new Int8Array(maxIndex + 1); // 0 = correct / no label, 1 = misclassified

  for (const f of frames) {
    const predictedName = classNames[f.predicted_class] || "";
    // Find the bout that covers this frame index
    const bout = bouts.find(
      (b) => f.index >= b.startFrame && f.index <= b.endFrame
    );
    if (!bout || !predictedName) continue;
    // Compare in the model's label space, not the raw behavior name.
    const truth = mapLabelToModelClass(bout.behaviorName, classNames, cfg);
    if (truth !== null && truth !== predictedName) {
      arr[f.index] = 1;
    }
  }
  alState.misclassifiedArray = arr;
}

// Update stats display
function updateAlStats() {
  if (!alState.results) {
    document.getElementById("alStats").style.display = "none";
    return;
  }
  document.getElementById("alStats").style.display = "grid";
  const frames = alState.results.frames;
  const uncertainties = frames.map((f) => f.uncertainty);
  const threshold = parseFloat(alThresholdSlider.value);

  const sorted = [...uncertainties].sort((a, b) => a - b);
  const mean = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
  const median = sorted[Math.floor(sorted.length / 2)];

  // Filter to qualifying consecutive segments (>= min length).
  const minLen = getMinSegmentLength();
  const segments = alState.uncertaintyArray
    ? getUncertainSegments(alState.uncertaintyArray, threshold, minLen)
    : [];
  const framesInSegments = segments.reduce((sum, s) => sum + s.length, 0);

  document.getElementById("alTotalFrames").textContent = frames.length;
  document.getElementById("alMeanUncertainty").textContent = mean.toFixed(3);
  document.getElementById("alMedianUncertainty").textContent = median.toFixed(3);
  document.getElementById("alAboveThreshold").textContent =
    `${framesInSegments} (${((framesInSegments / frames.length) * 100).toFixed(1)}%)`;
  document.getElementById("alSegments").textContent =
    `${segments.length} (≥${minLen} frame${minLen === 1 ? "" : "s"})`;

  // Misclassified count
  const misArr = alState.misclassifiedArray;
  if (misArr) {
    let misCount = 0;
    for (let i = 0; i < misArr.length; i++) { if (misArr[i] === 1) misCount++; }
    document.getElementById("alMisclassified").textContent = `${misCount} (${((misCount / frames.length) * 100).toFixed(1)}%)`;
  } else {
    document.getElementById("alMisclassified").textContent = "-";
  }

  document.getElementById("alClassNames").textContent = (alState.results.class_names || []).join(", ");
}
