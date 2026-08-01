// core/active-learning.js — Active-learning state and the in-place relabel strip.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// ── Active Learning Panel ─────────────────────────────────────────────────

const alState = {
  checkpointPath: null,
  taskId: null,
  sseSource: null,
  results: null, // { frames: [...], class_names: [...] }
  uncertaintyArray: null, // Float32Array of normalized entropy per frame index
  misclassifiedArray: null, // Int8Array: 1 = misclassified, 0 = correct or no label
  frameByIndex: null, // Map<frameIndex, frameResult> for O(1) per-frame lookups
  lowConfCursor: -1, // walk position for "Jump to Lowest Confidence" (worst-first)
};

// ── Relabel strip (AL pane) ───────────────────────────────────────────────
// Surfaces the behavior→key legend as clickable buttons plus this frame's
// predicted class vs. current label, so the correct-in-place loop stays in the
// Active Learning tab. Labeling reuses handleKeyAction (the same path as the
// number keys), so bouts edited here are identical to Label-tab bouts.
function renderAlBehaviorButtons() {
  const container = document.getElementById("alBehaviorButtons");
  if (!container) return;
  container.innerHTML = "";

  if (behaviors.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No behaviors yet — add some in the Label tab.";
    container.appendChild(empty);
    return;
  }

  behaviors.forEach((behavior) => {
    const btn = document.createElement("button");
    btn.className = "al-behavior-btn";
    const recording = activeStarts.has(behavior.id);
    if (recording) btn.classList.add("recording");

    const badge = document.createElement("span");
    badge.className = "key-badge";
    badge.textContent = behavior.key;

    const name = document.createElement("span");
    name.textContent = behavior.name;

    btn.appendChild(badge);
    btn.appendChild(name);
    btn.title = recording
      ? `Recording ${behavior.name} — click to end the bout on this frame`
      : `Start a ${behavior.name} bout on this frame`;
    btn.addEventListener("click", () => handleKeyAction(behavior.key));
    container.appendChild(btn);
  });
}

// What the model says about the current frame vs. what the human labeled.
// Shared by the Active Learning panel's text readout and the on-video HUD so
// the two can never disagree about the same frame.
function currentFrameInfo() {
  let classNames = [];
  let cfg = null;
  let predName = null;
  let uncertainty = null;
  let hasPrediction = false;
  let hasResults = false;

  // Every read of alState stays inside this guard: updateAlFrameInfo runs
  // during module evaluation (via renderBehaviors/renderBouts), which is before
  // `const alState` is initialized, so an unguarded touch is a TDZ crash.
  try {
    if (alState && alState.results && alState.frameByIndex) {
      hasResults = true;
      classNames = alState.results.class_names || [];
      cfg = getAlModelConfig();
      const f = alState.frameByIndex.get(currentFrameIndex);
      if (f) {
        hasPrediction = true;
        predName = classNames[f.predicted_class] ?? `class ${f.predicted_class}`;
        if (typeof f.uncertainty === "number") uncertainty = f.uncertainty;
      }
    }
  } catch { /* alState not initialized yet */ }

  // Current label = the bout covering this frame, if any, translated into the
  // model's label space (a binarized model can't predict the raw behavior name).
  const bout = bouts.find(
    (b) => currentFrameIndex >= b.startFrame && currentFrameIndex <= b.endFrame
  );
  const truth = bout ? mapLabelToModelClass(bout.behaviorName, classNames, cfg) : null;
  // A real disagreement, in the same label space only (null truth = the two
  // aren't comparable, so it is never flagged).
  const mismatch = !!(bout && predName && truth !== null && truth !== predName);

  return { classNames, cfg, predName, uncertainty, hasPrediction, hasResults, bout, truth, mismatch };
}

// Uncertainty is normalized Shannon entropy in [0,1]; confidence is its
// complement, which is what a reader actually wants on screen.
function confidenceFrom(uncertainty) {
  if (typeof uncertainty !== "number") return null;
  return Math.max(0, Math.min(1, 1 - uncertainty));
}

function updateAlFrameInfo() {
  const frameEl = document.getElementById("alCurFrame");
  const predEl = document.getElementById("alCurPredicted");
  const labelEl = document.getElementById("alCurLabel");
  if (!frameEl || !predEl || !labelEl) return;

  frameEl.textContent = currentFrameIndex;

  const info = currentFrameInfo();

  let predText = "—";
  if (info.hasPrediction) {
    predText = info.uncertainty !== null
      ? `${info.predName} (unc ${info.uncertainty.toFixed(2)})`
      : info.predName;
  } else if (info.hasResults) {
    predText = "(no prediction)";
  }
  predEl.textContent = predText;

  // Show "walking -> other" when binarization collapses the label, so it's clear
  // why a differently-named bout is not a disagreement.
  labelEl.textContent = !info.bout
    ? "(none)"
    : (info.truth && info.truth !== info.bout.behaviorName
        ? `${info.bout.behaviorName} → ${info.truth}`
        : info.bout.behaviorName);

  predEl.classList.toggle("al-pred-mismatch", info.mismatch);

  renderVideoHud(info);
}
