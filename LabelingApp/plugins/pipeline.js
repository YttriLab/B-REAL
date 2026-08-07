// plugins/realtime-tsm.js — The RealtimeASOID pipeline: background subtraction,
// crop tuning, extracted frames, and a TSM convolutional network over them.
//
// Registered against the shell through window.labelerPlugins (see
// labeler-plugin.js). Nothing in core/ may name anything defined here — the
// shell is shared with A-SOID Desktop and with the standalone labeler, and
// `npm run check` enforces it.

// ── Panel markup ──────────────────────────────────────────────────────────
// The tabs that configure this pipeline. They live here rather than in
// index.html so the shell's page is the same file for every pipeline — and so
// deleting this plugin removes its UI with it. Mounted first, because the panel
// code below queries these elements while it loads.
mountPanels([
  {
    tab: "preprocess",
    label: "Preprocess",
    after: "label",
    html: `
<section class="panel-card">
  <h2>Preprocessing</h2>

  <h3>Background</h3>
  <div class="form-row">
    <button id="ppComputeBg">Compute from Video</button>
    <button id="ppSelectBg">Select Image</button>
    <span id="ppBgPath" class="path-display">No background set</span>
  </div>
  <div class="form-row">
    <label><span>Num frames for BG:</span>
      <input id="ppBgNumFrames" type="number" value="500" min="10" max="5000" />
    </label>
  </div>

  <h3>Image</h3>
  <p class="hint">Original frame: <span id="ppFrameSize">—</span> · Processed: <span id="ppResizedSize">—</span> <span class="muted">(preview to measure)</span></p>
  <div class="slider-grid">
    <label>
      <span>Resize whole frame: <output id="ppResizeVal">100%</output></span>
      <input id="ppResize" type="range" min="10" max="100" value="100" step="5" />
    </label>
  </div>

  <h3>Crop Parameters</h3>
  <div class="slider-grid">
    <label>
      <span>Box Size: <output id="ppBoxSizeVal">220</output></span>
      <input id="ppBoxSize" type="range" min="10" max="500" value="220" />
    </label>
    <label>
      <span>Diff Threshold: <output id="ppDiffThreshVal">5</output></span>
      <input id="ppDiffThresh" type="range" min="1" max="50" value="5" />
    </label>
    <label>
      <span>Diff Mode:</span>
      <select id="ppDiffMode">
        <option value="directional">Directional (darker than bg)</option>
        <option value="absolute">Absolute (|frame − bg|)</option>
      </select>
    </label>
    <label>
      <span>Downsample: <output id="ppDownsampleVal">1</output></span>
      <input id="ppDownsample" type="range" min="1" max="4" value="1" />
    </label>
    <label>
      <span>Open1 Kernel: <output id="ppOpen1Val">4</output></span>
      <input id="ppOpen1" type="range" min="1" max="30" value="4" />
    </label>
    <label>
      <span>Close Kernel: <output id="ppCloseVal">15</output></span>
      <input id="ppClose" type="range" min="1" max="30" value="15" />
    </label>
    <label>
      <span>Open2 Kernel: <output id="ppOpen2Val">10</output></span>
      <input id="ppOpen2" type="range" min="1" max="30" value="10" />
    </label>
    <label>
      <span>Smoothing: <output id="ppSmoothingVal">0.6</output></span>
      <input id="ppSmoothing" type="range" min="0" max="0.95" step="0.05" value="0.6" />
    </label>
    <label>
      <span>Dead-band (px): <output id="ppDeadbandVal">2</output></span>
      <input id="ppDeadband" type="range" min="0" max="20" step="1" value="2" />
    </label>
    <label>
      <span>Min Blob Area: <output id="ppMinAreaVal">0</output></span>
      <input id="ppMinArea" type="range" min="0" max="1000" step="10" value="0" />
    </label>
  </div>
  <p class="hint">Smoothing/dead-band steady the crop center so the classifier sees the animal's motion, not box jitter. Step frames below to watch the box settle.</p>

  <h3>Preview</h3>
  <div class="form-row">
    <button id="ppPreview">Preview Current Frame</button>
    <button id="ppPrevFrame" class="ghost" title="Previous frame">◀ Prev</button>
    <span class="muted">Frame <output id="ppPreviewFrame">—</output></span>
    <button id="ppNextFrame" class="ghost" title="Next frame">Next ▶</button>
  </div>
  <div id="ppPreviewArea" class="preview-area">
    <div><canvas id="ppPreviewBbox" width="320" height="240"></canvas><span>Original + BBox</span></div>
    <div><canvas id="ppPreviewMask" width="320" height="240"></canvas><span>Mask</span></div>
    <div><canvas id="ppPreviewCrop" width="220" height="220"></canvas><span>Cropped</span></div>
  </div>

  <h3>Run Preprocessing</h3>
  <p class="hint">Only labeled frames are extracted, saved into the project at <code>frames/&lt;video&gt;/</code>.</p>
  <div class="form-row">
    <button id="ppRun" class="primary">Run Preprocessing</button>
    <button id="ppStop" class="danger" disabled>Stop</button>
  </div>
  <div class="progress-area">
    <progress id="ppProgress" value="0" max="100"></progress>
    <span id="ppStatus">Idle</span>
  </div>
</section>
`,
  },
  {
    tab: "train",
    label: "Train",
    after: "preprocess",
    html: `
<section class="panel-card">
  <h2>Training</h2>

  <h3>Data Sources</h3>
  <div id="trainVideoChecklist" class="video-checklist"></div>
  <div id="trainDataSources" class="data-sources"></div>
  <div class="form-row">
    <button id="trainAddSource">Add External Source&hellip;</button>
  </div>

  <h3>Model</h3>
  <div class="form-grid">
    <label><span>Variant:</span>
      <select id="trainVariant">
        <option value="b0" selected>EfficientNet-B0</option>
        <option value="b1">EfficientNet-B1</option>
        <option value="b2">EfficientNet-B2</option>
        <option value="b3">EfficientNet-B3</option>
      </select>
    </label>
    <label><span>Frames per group:</span>
      <input id="trainFramesPerGroup" type="number" value="3" min="2" max="10" />
    </label>
    <label><span>TSM Stages:</span>
      <input id="trainTsmStages" type="text" value="1,2,3,4,5,6,7" placeholder="e.g. 5,6,7" />
    </label>
    <label><span>Temporal Pool:</span>
      <select id="trainTemporalPool">
        <option value="last" selected>Last</option>
        <option value="avg">Average</option>
        <option value="max">Max</option>
      </select>
    </label>
    <label title="Causal shift: temporal mixing uses past frames only (never future), matching real-time/online inference on a live rig.">
      <span>Unidirectional (causal) TSM:</span>
      <input id="trainUnidirectional" type="checkbox" />
    </label>
    <label><span>LR:</span>
      <input id="trainLr" type="number" value="0.0001" step="0.00001" />
    </label>
    <label><span>Weight Decay:</span>
      <input id="trainWeightDecay" type="number" value="0.001" step="0.0001" />
    </label>
    <label><span>Dropout:</span>
      <input id="trainDropout" type="number" value="0.2" min="0" max="0.9" step="0.05" />
    </label>
    <label><span>Max Epochs:</span>
      <input id="trainMaxEpochs" type="number" value="100" min="1" />
    </label>
    <label><span>Early Stop Patience:</span>
      <input id="trainPatience" type="number" value="20" min="1" />
    </label>
  </div>

  <h3>Data Split</h3>
  <div class="form-grid">
    <label><span>Train %:</span>
      <input id="trainSplitTrain" type="number" value="0.7" min="0" max="1" step="0.05" />
    </label>
    <label><span>Val %:</span>
      <input id="trainSplitVal" type="number" value="0.15" min="0" max="1" step="0.05" />
    </label>
    <label><span>Test %:</span>
      <input id="trainSplitTest" type="number" value="0.15" min="0" max="1" step="0.05" />
    </label>
  </div>

  <h3>Options</h3>
  <label><span>Positive class (binarize):</span></label>
  <div id="trainBinarizeList" class="behavior-checklist"></div>
  <p class="hint">Tick one or more behaviors to train a binary classifier (checked = positive, rest = "other"). Leave all unchecked for multi-class.</p>
  <div class="form-grid">
    <label class="checkbox-label">
      <input id="trainBoundarySmoothing" type="checkbox" checked />
      <span>Boundary Smoothing</span>
    </label>
    <label><span>Epsilon:</span>
      <input id="trainBsEpsilon" type="number" value="0.3" min="0" max="1" step="0.05" />
    </label>
    <label><span>Window:</span>
      <input id="trainBsWindow" type="number" value="5" min="1" max="20" />
    </label>
  </div>

  <h3>Weights &amp; Biases</h3>
  <div class="form-grid">
    <label class="checkbox-label">
      <input id="trainUseWandb" type="checkbox" />
      <span>Log to W&amp;B</span>
    </label>
    <label><span>Project:</span>
      <input id="trainWandbProject" type="text" value="realtime-asoid" placeholder="W&B project name" />
    </label>
    <label><span>Run Name:</span>
      <input id="trainWandbRunName" type="text" value="" placeholder="auto-generated if empty" />
    </label>
    <label><span>API Key:</span>
      <input id="trainWandbApiKey" type="password" value="" placeholder="leave empty if logged in" />
    </label>
  </div>

  <h3>Export to TorchScript</h3>
  <p class="hint">Convert a checkpoint to a single <code>.pt</code> (Bonsai / real-time rigs). Uses the TSM stages set above.</p>
  <div class="form-row">
    <button id="trainExportSelectCkpt">Select Checkpoint</button>
    <span id="trainExportCkptPath" class="path-display">Last trained best checkpoint</span>
  </div>
  <div class="form-grid">
    <label><span>Output name:</span>
      <input id="trainExportName" type="text" value="tsm_model" placeholder="tsm_model" />
    </label>
  </div>
  <div class="form-row">
    <button id="trainExportRun">Export .pt</button>
    <span id="trainExportStatus" class="muted"></span>
  </div>

  <div class="form-row" style="margin-top:16px;">
    <button id="trainStart" class="primary">Start Training</button>
    <button id="trainStop" class="danger" disabled>Stop</button>
    <button id="benchInfer" title="Time the optimized (TorchScript) inference latency for this architecture — no training needed">Benchmark inference</button>
    <label style="display:inline-flex;align-items:center;gap:4px;font-weight:normal;">
      <input id="benchFp16" type="checkbox" /> FP16
    </label>
  </div>
  <div id="benchResult" class="metrics-display" style="display:none;"></div>
  <div class="progress-area">
    <progress id="trainProgress" value="0" max="100"></progress>
    <span id="trainStatus">Idle</span>
  </div>
  <div id="trainMetrics" class="metrics-display" style="display:none;">
    <div>Epoch: <span id="trainEpoch">-</span> / <span id="trainTotalEpochs">-</span></div>
    <div>Train Loss: <span id="trainLossVal">-</span></div>
    <div>Val Loss: <span id="trainValLoss">-</span></div>
    <div>Best Val Loss: <span id="trainBestValLoss">-</span></div>
    <div>Val MCC: <span id="trainValMcc">-</span></div>
    <div>Best Val MCC: <span id="trainBestValMcc">-</span></div>
    <div>Stagnation (val MCC): <span id="trainStagnation">-</span></div>
    <div>Train Acc: <span id="trainAccVal">-</span></div>
    <div>Val Acc: <span id="trainValAcc">-</span></div>
    <div>LR: <span id="trainLrVal">-</span></div>
  </div>
  <pre id="trainLog" class="train-log" style="display:none;"></pre>
</section>
`,
  },
]);

// This pipeline decodes and crops the video on the fly, so its inference needs
// a frame grouping the shell knows nothing about. It goes in the slot the
// Active Learning tab leaves open.
mountSlot("alPipelineOptions", `
<p class="hint">Inference runs on the current video on the fly — no frame directory needed.</p>
<div class="form-row">
  <label><span>Frames per group:</span>
    <input id="alFramesPerGroup" type="number" value="3" min="2" max="10" />
  </label>
</div>
`);

// ── Preprocessing Panel ───────────────────────────────────────────────────

const ppState = {
  bgPath: null,
  taskId: null,
  sseSource: null,
  bgByVideo: {}, // videoName -> background image path (auto-loaded on switch)
  cropByVideo: {}, // videoName -> crop/preprocess params (auto-loaded on switch)
  frameW: null,  // original source frame size (from last preview)
  frameH: null,
  previewFrame: null,   // frame index currently shown in the preview
  previewCenter: null,  // smoothed crop center [cx,cy] (small space), chained on step
};

// Remember a background for a video (persisted in settings; also recorded on the
// project manifest when a project is open).
function rememberVideoBackground(videoName, absPath) {
  if (!videoName || !absPath) return;
  ppState.bgByVideo[videoName] = absPath;
  try { saveSettings(); } catch { /* settings not ready */ }
  if (window.projectManager && window.projectManager.isActive()) {
    try { window.projectManager.setVideoBackground(videoName, absPath); } catch { /* ignore */ }
  }
}

function resolveVideoBackground(videoName) {
  if (!videoName) return null;
  if (window.projectManager && window.projectManager.isActive()) {
    const p = window.projectManager.getVideoBackgroundAbs(videoName);
    if (p) return p;
  }
  return ppState.bgByVideo[videoName] || null;
}

// Remember the crop/preprocess params that generated a video's frames. Saved in
// settings (survives restarts) and on the project manifest (travels with the
// project/.rlt), mirroring rememberVideoBackground.
function rememberVideoCropParams(videoName, params) {
  if (!videoName || !params) return;
  ppState.cropByVideo[videoName] = params;
  try { saveSettings(); } catch { /* settings not ready */ }
  if (window.projectManager && window.projectManager.isActive()) {
    try { window.projectManager.setVideoPreprocessing(videoName, params); } catch { /* ignore */ }
  }
}

function resolveVideoCropParams(videoName) {
  if (!videoName) return null;
  if (window.projectManager && window.projectManager.isActive()) {
    const p = window.projectManager.getVideoPreprocessing(videoName);
    if (p) return p;
  }
  return ppState.cropByVideo[videoName] || null;
}

// Load a video's saved crop params into the preprocess panel (if any exist).
function applyVideoCropParams(videoName) {
  const p = resolveVideoCropParams(videoName);
  if (p) applyCropParams(p);
}

// Load the background that belongs to a video (if any) into the preprocess panel.
async function applyVideoBackground(videoName) {
  const p = resolveVideoBackground(videoName);
  if (p && await fileExists(p)) {
    ppState.bgPath = p;
    setPathDisplay("ppBgPath", p);
  } else {
    ppState.bgPath = null;
    const el = document.getElementById("ppBgPath");
    if (el) el.textContent = "No background";
  }
}

// Slider value displays
["ppBoxSize", "ppDiffThresh", "ppDownsample", "ppOpen1", "ppClose", "ppOpen2", "ppSmoothing", "ppDeadband", "ppMinArea"].forEach((id) => {
  const slider = document.getElementById(id);
  const output = document.getElementById(id + "Val");
  if (slider && output) {
    slider.addEventListener("input", () => { output.textContent = slider.value; });
  }
});

// Resize slider shows a percent and live-updates the processed-size readout.
(() => {
  const slider = document.getElementById("ppResize");
  const output = document.getElementById("ppResizeVal");
  if (!slider) return;
  slider.addEventListener("input", () => {
    if (output) output.textContent = `${slider.value}%`;
    updateResizedSizeDisplay();
  });
})();

function getCropParams() {
  return {
    box_size: parseInt(document.getElementById("ppBoxSize").value),
    difference_threshold: parseInt(document.getElementById("ppDiffThresh").value),
    difference_mode: document.getElementById("ppDiffMode").value,
    downsample_factor: parseInt(document.getElementById("ppDownsample").value),
    open1_kernel: parseInt(document.getElementById("ppOpen1").value),
    close_kernel: parseInt(document.getElementById("ppClose").value),
    open2_kernel: parseInt(document.getElementById("ppOpen2").value),
    resize_factor: (parseInt(document.getElementById("ppResize").value) || 100) / 100,
    smoothing: parseFloat(document.getElementById("ppSmoothing").value),
    deadband: parseFloat(document.getElementById("ppDeadband").value),
    min_contour_area: parseFloat(document.getElementById("ppMinArea").value),
  };
}

// Inverse of getCropParams: push a saved params object into the panel controls
// (and refresh their value labels). Missing fields are left at their current
// values so older/partial records don't wipe the panel.
function applyCropParams(p) {
  if (!p) return;
  const setSlider = (id, valId, val) => {
    const el = document.getElementById(id);
    const out = document.getElementById(valId);
    if (el && val !== undefined && val !== null) { el.value = val; if (out) out.textContent = val; }
  };
  setSlider("ppBoxSize", "ppBoxSizeVal", p.box_size);
  setSlider("ppDiffThresh", "ppDiffThreshVal", p.difference_threshold);
  if (p.difference_mode) document.getElementById("ppDiffMode").value = p.difference_mode;
  setSlider("ppDownsample", "ppDownsampleVal", p.downsample_factor);
  setSlider("ppOpen1", "ppOpen1Val", p.open1_kernel);
  setSlider("ppClose", "ppCloseVal", p.close_kernel);
  setSlider("ppOpen2", "ppOpen2Val", p.open2_kernel);
  if (p.smoothing !== undefined) setSlider("ppSmoothing", "ppSmoothingVal", p.smoothing);
  if (p.deadband !== undefined) setSlider("ppDeadband", "ppDeadbandVal", p.deadband);
  if (p.min_contour_area !== undefined) setSlider("ppMinArea", "ppMinAreaVal", p.min_contour_area);
  if (p.resize_factor) {
    const pct = Math.round(p.resize_factor * 100);
    const el = document.getElementById("ppResize");
    const out = document.getElementById("ppResizeVal");
    if (el) el.value = pct;
    if (out) out.textContent = `${pct}%`;
  }
  updateResizedSizeDisplay();
}

// Update the "Processed" size readout from the stored original frame size and
// the current resize slider (so it reacts live without re-running a preview).
function updateResizedSizeDisplay() {
  const el = document.getElementById("ppResizedSize");
  if (!el) return;
  const factor = (parseInt(document.getElementById("ppResize").value) || 100) / 100;
  if (ppState.frameW && ppState.frameH) {
    el.textContent = `${Math.round(ppState.frameW * factor)} × ${Math.round(ppState.frameH * factor)}`;
  } else {
    el.textContent = "—";
  }
}

function getVideoPath() {
  return currentFile && currentFile.path ? currentFile.path : null;
}

// Compute background
document.getElementById("ppComputeBg").addEventListener("click", async () => {
  const videoPath = getVideoPath();
  if (!videoPath) { alert("Load a video first."); return; }
  const numFrames = parseInt(document.getElementById("ppBgNumFrames").value) || 500;
  document.getElementById("ppBgPath").textContent = "Computing...";
  const body = { video_path: videoPath, num_frames: numFrames };
  // When a project is open, write the background into the project's backgrounds/ dir.
  if (window.projectManager && window.projectManager.isActive()) {
    body.output_dir = window.projectManager.getBackgroundsDir();
  }
  try {
    const res = await backendFetch("/api/preprocess/background", {
      method: "POST",
      body: JSON.stringify(body),
    });
    ppState.bgPath = res.path;
    setPathDisplay("ppBgPath", res.path);
    rememberVideoBackground(currentVideoName, res.path);
  } catch (e) {
    document.getElementById("ppBgPath").textContent = `Error: ${e.message}`;
  }
});

// Select background image
document.getElementById("ppSelectBg").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectFile({
    title: "Select Background Image",
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "bmp"] }],
  });
  if (!res.canceled) {
    ppState.bgPath = res.path;
    setPathDisplay("ppBgPath", res.path);
    rememberVideoBackground(currentVideoName, res.path);
  }
});

// Preview a specific frame. When `chain` is true, the smoothed center from the
// last previewed frame is carried in so stepping forward shows the temporal
// smoothing settle (a single preview otherwise has no history).
async function runPreview(frameIndex, { chain = false } = {}) {
  const videoPath = getVideoPath();
  if (!videoPath) { alert("Load a video first."); return; }
  if (!ppState.bgPath) { alert("Compute or select a background first."); return; }
  frameIndex = Math.max(0, Math.round(frameIndex));

  try {
    const res = await backendFetch("/api/preprocess/preview", {
      method: "POST",
      body: JSON.stringify({
        video_path: videoPath,
        bg_path: ppState.bgPath,
        frame_index: frameIndex,
        crop_params: getCropParams(),
        prev_center: chain ? ppState.previewCenter : null,
      }),
    });

    ppState.previewFrame = frameIndex;
    ppState.previewCenter = res.center || null;
    const frameOut = document.getElementById("ppPreviewFrame");
    if (frameOut) frameOut.textContent = frameIndex;
    // Keep the main player in sync so you see the frame in context.
    try { if (typeof seekToFrame === "function") seekToFrame(frameIndex); } catch { /* ignore */ }

    // Update the frame-size readouts from the original (source) dims.
    if (res.frame_width && res.frame_height) {
      ppState.frameW = res.frame_width;
      ppState.frameH = res.frame_height;
      document.getElementById("ppFrameSize").textContent =
        `${res.frame_width} × ${res.frame_height}`;
    }
    if (res.resized_width && res.resized_height) {
      document.getElementById("ppResizedSize").textContent =
        `${res.resized_width} × ${res.resized_height}`;
    } else {
      updateResizedSizeDisplay();
    }

    // Draw base64 images to canvases
    const pairs = [
      ["ppPreviewBbox", res.original_bbox],
      ["ppPreviewMask", res.mask],
      ["ppPreviewCrop", res.cropped],
    ];
    for (const [canvasId, b64] of pairs) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
      img.src = `data:image/jpeg;base64,${b64}`;
    }
  } catch (e) {
    alert(`Preview failed: ${e.message}`);
  }
}

// "Preview Current Frame" — resets the smoothing chain at the player's frame.
document.getElementById("ppPreview").addEventListener("click", () => {
  ppState.previewCenter = null;
  runPreview(currentFrameIndex, { chain: false });
});

// Step forward (chains the smoothed center so you watch the box settle) / back.
document.getElementById("ppNextFrame").addEventListener("click", () => {
  const base = ppState.previewFrame != null ? ppState.previewFrame : currentFrameIndex;
  runPreview(base + 1, { chain: true });
});
document.getElementById("ppPrevFrame").addEventListener("click", () => {
  const base = ppState.previewFrame != null ? ppState.previewFrame : currentFrameIndex;
  ppState.previewCenter = null; // stepping back breaks the forward chain
  runPreview(base - 1, { chain: false });
});

// Run preprocessing — frames always go into the active project's frames/<id>/.
// Merge the current video's bouts into inclusive [start, end] native-index
// ranges, each extended by `leadIn` frames so the clip context exists on disk.
function computeLabeledFrameRanges(boutList, leadIn) {
  if (!Array.isArray(boutList) || boutList.length === 0) return [];
  const raw = boutList
    .filter((b) => Number.isFinite(b.startFrame) && Number.isFinite(b.endFrame))
    .map((b) => [
      Math.max(0, Math.min(b.startFrame, b.endFrame) - leadIn),
      Math.max(b.startFrame, b.endFrame),
    ])
    .sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of raw) {
    const last = merged[merged.length - 1];
    if (last && s <= last[1] + 1) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  return merged;
}

document.getElementById("ppRun").addEventListener("click", async () => {
  const videoPath = getVideoPath();
  if (!videoPath) { alert("Load a video first."); return; }

  const pm = window.projectManager;
  if (!pm || !pm.isActive()) { alert("Open a project first."); return; }
  const outputDir = pm.getProjectRoot() + "/frames";
  const outputName = pm.getVideoIdFromName(currentVideoName);

  // Only labeled regions are extracted (training needs nothing else; AL reads
  // the video on the fly). Extend each bout by frames_per_group-1 for clip context.
  const leadIn = (parseInt(document.getElementById("trainFramesPerGroup").value, 10) || 3) - 1;
  const frameRanges = computeLabeledFrameRanges(bouts, leadIn);
  if (frameRanges.length === 0) {
    alert("No labeled bouts for this video — label some frames first (only labeled frames are extracted).");
    return;
  }

  document.getElementById("ppRun").disabled = true;
  document.getElementById("ppStop").disabled = false;
  document.getElementById("ppStatus").textContent = "Starting...";

  // These are the params that produce this video's frames — record them against
  // the video so switching back restores them and re-runs stay consistent.
  const cropParams = getCropParams();
  rememberVideoCropParams(currentVideoName, cropParams);

  try {
    const res = await backendFetch("/api/preprocess/run", {
      method: "POST",
      body: JSON.stringify({
        video_path: videoPath,
        output_dir: outputDir,
        crop_params: cropParams,
        frame_ranges: frameRanges,
        bg_path: ppState.bgPath,
        output_name: outputName,
      }),
    });
    ppState.taskId = res.task_id;

    // Subscribe to SSE
    ppState.sseSource = backendSSE(
      `/api/preprocess/status/${res.task_id}`,
      (data) => {
        const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
        document.getElementById("ppProgress").value = pct;
        document.getElementById("ppStatus").textContent = `${data.status}: ${data.progress}/${data.total}`;
        if (data.status === "completed" || data.status === "error" || data.status === "cancelled") {
          ppState.sseSource?.close();
          document.getElementById("ppRun").disabled = false;
          document.getElementById("ppStop").disabled = true;
          if (data.status === "error") {
            document.getElementById("ppStatus").textContent = `Error: ${data.error}`;
          } else if (data.output_dir) {
            document.getElementById("ppStatus").textContent = `Done: ${data.output_dir}`;
            // Newly-created frames may make this video trainable — refresh the list.
            renderTrainVideoChecklist();
          }
        }
      },
      () => {
        document.getElementById("ppRun").disabled = false;
        document.getElementById("ppStop").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("ppStatus").textContent = `Error: ${e.message}`;
    document.getElementById("ppRun").disabled = false;
    document.getElementById("ppStop").disabled = true;
  }
});

// Stop preprocessing
document.getElementById("ppStop").addEventListener("click", async () => {
  if (ppState.taskId) {
    try {
      await backendFetch(`/api/preprocess/stop/${ppState.taskId}`, { method: "POST" });
    } catch { /* ignore */ }
  }
});

// ── Training Panel ────────────────────────────────────────────────────────

const trainState = {
  dataSources: [],        // manually-added external sources
  selectedVideos: new Set(), // project video names checked for training
  taskId: null,
  sseSource: null,
  lastBestCheckpoint: null, // best .ckpt from the most recent completed run
  exportCkptPath: null,     // user-picked checkpoint for TorchScript export
};

// Render the project videos as a checklist; checked videos become training
// data sources (frames + labels resolved from the project).
async function renderTrainVideoChecklist() {
  const container = document.getElementById("trainVideoChecklist");
  if (!container) return;
  container.innerHTML = "";

  const pm = window.projectManager;
  if (!pm || !pm.isActive()) {
    container.innerHTML =
      '<p class="muted">Open a project to pick videos here, or use “Add External Source”.</p>';
    return;
  }

  const manifest = pm.getManifest();
  const vids = (manifest && manifest.videos) || [];
  if (vids.length === 0) {
    container.innerHTML = '<p class="muted">No videos in this project yet.</p>';
    return;
  }

  for (const v of vids) {
    const name = v.path ? v.path.split(/[\\/]/).pop() : v.id;
    const framesDir = pm.getFramesDir(name);
    const labelPath = pm.getVideoLabelsAbs(name);

    const row = document.createElement("label");
    row.className = "checkbox-row";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = trainState.selectedVideos.has(name);
    cb.addEventListener("change", () => {
      if (cb.checked) trainState.selectedVideos.add(name);
      else trainState.selectedVideos.delete(name);
    });

    const text = document.createElement("span");
    text.className = "checklist-name";
    text.textContent = name;

    const status = document.createElement("span");
    status.className = "muted checklist-status";
    status.textContent = "checking…";

    row.append(cb, text, status);
    container.appendChild(row);

    // Annotate availability of frames + labels asynchronously.
    Promise.all([fileExists(framesDir), fileExists(labelPath)]).then(([hasFrames, hasLabels]) => {
      status.textContent = `${hasLabels ? "✓" : "✗"} labels · ${hasFrames ? "✓" : "✗"} frames`;
      if (!hasFrames || !hasLabels) {
        row.title = "Missing " +
          [!hasLabels ? "labels" : null, !hasFrames ? "frames (run Preprocess)" : null]
            .filter(Boolean).join(" and ");
      }
    });
  }
}

// Add data source
document.getElementById("trainAddSource").addEventListener("click", async () => {
  if (!window.api) return;

  const frameRes = await window.api.selectDirectory();
  if (frameRes.canceled) return;

  const labelRes = await window.api.selectFile({
    title: "Select Label CSV",
    filters: [{ name: "CSV", extensions: ["csv"] }],
  });
  if (labelRes.canceled) return;

  trainState.dataSources.push({ frameDir: frameRes.path, labelPath: labelRes.path });
  renderTrainDataSources();
});

function renderTrainDataSources() {
  const container = document.getElementById("trainDataSources");
  container.innerHTML = "";
  trainState.dataSources.forEach((src, idx) => {
    const row = document.createElement("div");
    row.className = "data-source-row";
    row.innerHTML = `
      <div class="paths">
        <span title="${src.frameDir}">Frames: ${src.frameDir}</span>
        <span title="${src.labelPath}">Labels: ${src.labelPath}</span>
      </div>
      <button data-idx="${idx}">Remove</button>
    `;
    row.querySelector("button").addEventListener("click", () => {
      trainState.dataSources.splice(idx, 1);
      renderTrainDataSources();
    });
    container.appendChild(row);
  });
}

// Start training
document.getElementById("trainStart").addEventListener("click", async () => {
  // Combine checked project videos with any manually-added external sources.
  const pm = window.projectManager;
  const projectSources = [];
  if (pm && pm.isActive()) {
    for (const name of trainState.selectedVideos) {
      const frameDir = pm.getFramesDir(name);
      const labelPath = pm.getVideoLabelsAbs(name);
      if (!frameDir || !labelPath) continue;
      // Regenerate the on-disk label file from the canonical (frame-indexed)
      // bouts right before training. This guarantees the trainer reads the
      // frame-indexed one-hot format `load_data` expects — and migrates any
      // label file still saved in the old timestamp bout format.
      const videoBouts = name === currentVideoName
        ? bouts
        : ((loadProgress(name) || {}).bouts || []);
      if (videoBouts.length > 0) {
        try { await pm.saveVideoLabels(name, generateTrainingLabelCsv(videoBouts)); }
        catch { /* fall back to whatever is on disk */ }
      }
      projectSources.push({ name, frameDir, labelPath });
    }
  }
  const sources = [...projectSources, ...trainState.dataSources];
  if (sources.length === 0) {
    alert("Select at least one project video, or add an external data source.");
    return;
  }

  // Pre-flight: confirm the labeled frames are actually preprocessed. load_data
  // silently drops labeled frames whose PNGs (or clip context) are missing, so
  // warn before launching a run that would quietly ignore part of the labels.
  const framesPerGroup = parseInt(document.getElementById("trainFramesPerGroup").value) || 3;
  try {
    const cov = await backendFetch("/api/training/check-coverage", {
      method: "POST",
      body: JSON.stringify({
        frame_dirs: sources.map((s) => s.frameDir),
        label_paths: sources.map((s) => s.labelPath),
        frames_per_group: framesPerGroup,
      }),
    });
    const problems = [];
    (cov.datasets || []).forEach((d, i) => {
      const name = (sources[i] && sources[i].name)
        || String(d.frame_dir).split(/[\\/]/).filter(Boolean).pop()
        || `source ${i + 1}`;
      if (d.error) problems.push(`• ${name}: ${d.error}`);
      else if (d.not_extracted > 0) {
        problems.push(`• ${name}: ${d.not_extracted} of ${d.labeled} labeled frames not preprocessed`);
      }
    });
    if (problems.length) {
      const proceed = confirm(
        "Some labeled frames aren't preprocessed and will be IGNORED during training:\n\n" +
        problems.join("\n") +
        `\n\nTraining would use ${cov.total_ready} of ${cov.total_labeled} labeled frames.\n` +
        "Run Preprocess on those videos to include them.\n\n" +
        "Proceed with training anyway?"
      );
      if (!proceed) {
        document.getElementById("trainStatus").textContent =
          "Cancelled — run Preprocess on the labeled frames first.";
        return;
      }
    }
  } catch (e) {
    // Never hard-block training because the check itself failed; just log it.
    console.warn("[train] coverage check failed", e);
  }

  const tsmStagesStr = document.getElementById("trainTsmStages").value;
  const tsmStages = tsmStagesStr.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));

  // Positive class from the binarize checklist:
  //   0 checked → multi-class; 1 → binarize that behavior;
  //   ≥2 → union them into one positive label (label_union), then binarize it.
  const positive = behaviors.map((b) => b.name).filter((n) => binarizeSelected.has(n));
  let binarize = null;
  let labelUnion = null;
  if (positive.length === 1) {
    binarize = positive[0];
  } else if (positive.length >= 2) {
    const existing = new Set(behaviors.map((b) => b.name));
    let unionName = "positive";
    while (existing.has(unionName)) unionName += "_";
    binarize = unionName;
    labelUnion = { [unionName]: positive };
  }

  const config = {
    label_paths: sources.map((s) => s.labelPath),
    frame_dirs: sources.map((s) => s.frameDir),
    frames_per_group: framesPerGroup,
    binarize: binarize,
    label_union: labelUnion,
    train_percent: parseFloat(document.getElementById("trainSplitTrain").value),
    val_percent: parseFloat(document.getElementById("trainSplitVal").value),
    test_percent: parseFloat(document.getElementById("trainSplitTest").value),
    num_frames: framesPerGroup,
    variant: document.getElementById("trainVariant").value,
    shift_fraction: 0.125,
    unidirectional: document.getElementById("trainUnidirectional").checked,
    pretrained: true,
    tsm_stages: tsmStages,
    dropout: parseFloat(document.getElementById("trainDropout").value),
    lr: parseFloat(document.getElementById("trainLr").value),
    weight_decay: parseFloat(document.getElementById("trainWeightDecay").value),
    temporal_pool: document.getElementById("trainTemporalPool").value,
    max_epochs: parseInt(document.getElementById("trainMaxEpochs").value),
    early_stopping_patience: parseInt(document.getElementById("trainPatience").value),
    boundary_smoothing_enabled: document.getElementById("trainBoundarySmoothing").checked,
    boundary_smoothing_epsilon: parseFloat(document.getElementById("trainBsEpsilon").value),
    boundary_smoothing_window: parseInt(document.getElementById("trainBsWindow").value),
    use_wandb: document.getElementById("trainUseWandb").checked,
    project_name: document.getElementById("trainWandbProject").value.trim() || "realtime-asoid",
    run_name: document.getElementById("trainWandbRunName").value.trim() || undefined,
    wandb_api_key: document.getElementById("trainWandbApiKey").value.trim() || undefined,
  };

  // If a project is open, save checkpoints into it so the model path is stable
  // and can be recorded in the manifest (otherwise it lands in a temp dir).
  if (window.projectManager && window.projectManager.isActive()) {
    config.checkpoint_dir = window.projectManager.getCheckpointsDir();
  }

  document.getElementById("trainStart").disabled = true;
  document.getElementById("trainStop").disabled = false;
  document.getElementById("trainStatus").textContent = "Starting...";
  document.getElementById("trainMetrics").style.display = "grid";
  const trainLogEl = document.getElementById("trainLog");
  trainLogEl.textContent = "";
  trainLogEl.style.display = "block";
  const appendTrainLog = (line) => {
    trainLogEl.textContent += (trainLogEl.textContent ? "\n" : "") + line;
    trainLogEl.scrollTop = trainLogEl.scrollHeight;
  };

  try {
    const res = await backendFetch("/api/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    });
    trainState.taskId = res.task_id;

    trainState.sseSource = backendSSE(
      `/api/training/status/${res.task_id}`,
      (data) => {
        // Verbose startup/progress lines from the subprocess.
        if (data.message) {
          appendTrainLog(data.message);
          document.getElementById("trainStatus").textContent = data.message;
        } else {
          document.getElementById("trainStatus").textContent = data.status || "training";
        }
        if (data.epoch !== undefined) {
          document.getElementById("trainEpoch").textContent = data.epoch + 1;
          document.getElementById("trainTotalEpochs").textContent = data.total_epochs || "-";
          const pct = data.total_epochs ? Math.round(((data.epoch + 1) / data.total_epochs) * 100) : 0;
          document.getElementById("trainProgress").value = pct;
          // Per-epoch summary line in the log (skip the final completion event,
          // which carries no train_loss).
          if (data.train_loss !== undefined && data.status === "training") {
            const parts = [
              `epoch ${data.epoch + 1}/${data.total_epochs || "?"}`,
              `train_loss ${data.train_loss.toFixed(4)}`,
              `val_loss ${data.val_loss.toFixed(4)}`,
            ];
            if (data.val_mcc !== undefined)
              parts.push(`val_mcc ${data.val_mcc.toFixed(4)}`);
            // stagnation now tracks val_mcc (the early-stopping metric)
            if (data.stagnation_epochs !== undefined)
              parts.push(`stagnation ${data.stagnation_epochs}/${data.patience}`);
            appendTrainLog(parts.join("  "));
          }
        }
        if (data.train_loss !== undefined) document.getElementById("trainLossVal").textContent = data.train_loss.toFixed(4);
        if (data.val_loss !== undefined) document.getElementById("trainValLoss").textContent = data.val_loss.toFixed(4);
        if (typeof data.best_val_loss === "number" && isFinite(data.best_val_loss))
          document.getElementById("trainBestValLoss").textContent = data.best_val_loss.toFixed(4);
        if (data.val_mcc !== undefined) document.getElementById("trainValMcc").textContent = data.val_mcc.toFixed(4);
        if (typeof data.best_val_mcc === "number" && isFinite(data.best_val_mcc))
          document.getElementById("trainBestValMcc").textContent = data.best_val_mcc.toFixed(4);
        if (data.stagnation_epochs !== undefined) {
          const patience = data.patience !== undefined ? data.patience : "-";
          document.getElementById("trainStagnation").textContent = `${data.stagnation_epochs} / ${patience}`;
        }
        if (data.train_acc !== undefined) document.getElementById("trainAccVal").textContent = (data.train_acc * 100).toFixed(1) + "%";
        if (data.val_acc !== undefined) document.getElementById("trainValAcc").textContent = (data.val_acc * 100).toFixed(1) + "%";
        if (data.lr !== undefined) document.getElementById("trainLrVal").textContent = data.lr.toExponential(2);

        if (data.status === "completed" || data.status === "error" || data.status === "cancelled") {
          trainState.sseSource?.close();
          document.getElementById("trainStart").disabled = false;
          document.getElementById("trainStop").disabled = true;
          if (data.status === "error") {
            const detail = (data.error || "").trim();
            const statusEl = document.getElementById("trainStatus");
            statusEl.textContent = detail
              ? `Error: ${detail.split("\n").pop()}`
              : "Error (no details returned by backend)";
            statusEl.title = detail || "";
            if (detail) {
              console.error("[training] subprocess failed:\n" + detail);
              appendTrainLog("\nERROR:\n" + detail);
            }
          } else if (data.status === "cancelled") {
            appendTrainLog("\nCancelled.");
          }
          if (data.best_checkpoint) {
            document.getElementById("trainStatus").textContent = `Done! Best: ${data.best_checkpoint}`;
            trainState.lastBestCheckpoint = data.best_checkpoint;
            // Auto-record the trained checkpoint into the project manifest so it
            // travels with the .rlt as a reference (re-linkable on import).
            if (data.status === "completed" && window.projectManager && window.projectManager.isActive()) {
              window.projectManager.addModel({
                name: data.best_checkpoint.split(/[\\/]/).pop(),
                path: data.best_checkpoint,
                created: new Date().toISOString().slice(0, 10),
                val_mcc: typeof data.best_val_mcc === "number" ? data.best_val_mcc : null,
                val_loss: typeof data.best_val_loss === "number" ? data.best_val_loss : null,
                config: {
                  variant: config.variant,
                  binarize: config.binarize,
                  // Needed to map original behavior names into this model's
                  // collapsed label space when scoring Active Learning
                  // predictions (see mapLabelToModelClass).
                  label_union: config.label_union,
                  num_frames: config.num_frames,
                  tsm_stages: config.tsm_stages,
                  unidirectional: config.unidirectional,
                },
              });
            }
          }
        }
      },
      () => {
        document.getElementById("trainStart").disabled = false;
        document.getElementById("trainStop").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("trainStatus").textContent = `Error: ${e.message}`;
    document.getElementById("trainStart").disabled = false;
    document.getElementById("trainStop").disabled = true;
  }
});

// Stop training
document.getElementById("trainStop").addEventListener("click", async () => {
  if (trainState.taskId) {
    try {
      await backendFetch(`/api/training/stop/${trainState.taskId}`, { method: "POST" });
    } catch { /* ignore */ }
  }
});

// ── Inference benchmark (pre-training compute budget) ───────────────────────
// Times the optimized (TorchScript) forward pass for the CURRENT architecture
// without training it — latency depends on architecture + input shape, not on
// trained weights — so users can decide if a variant fits their rig's frame rate.
document.getElementById("benchInfer").addEventListener("click", async () => {
  const btn = document.getElementById("benchInfer");
  const out = document.getElementById("benchResult");
  const tsmStages = document.getElementById("trainTsmStages").value
    .split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));
  const body = {
    variant: document.getElementById("trainVariant").value,
    num_frames: parseInt(document.getElementById("trainFramesPerGroup").value, 10) || 3,
    tsm_stages: tsmStages,
    temporal_pool: document.getElementById("trainTemporalPool").value,
    unidirectional: document.getElementById("trainUnidirectional").checked,
    precision: document.getElementById("benchFp16").checked ? "fp16" : "fp32",
  };

  btn.disabled = true;
  out.style.display = "block";
  out.textContent = "Building + tracing model, timing…";
  try {
    const r = await backendFetch("/api/training/benchmark-inference", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const fps30 = (r.clips_per_sec / 30).toFixed(1);
    const fps60 = (r.clips_per_sec / 60).toFixed(1);
    out.innerHTML = "";
    const rows = [
      `Architecture: ${r.variant}, T=${r.num_frames}, stages [${(r.tsm_stages || []).join(",")}], pool ${r.temporal_pool}`,
      `Device: ${r.device_name} (${r.precision.toUpperCase()})`,
      `Latency: ${r.ms_per_clip.toFixed(2)} ms/clip  →  ${r.clips_per_sec.toFixed(1)} clips/s`,
      `Real-time headroom: ${fps30}× at 30 fps · ${fps60}× at 60 fps`,
    ];
    if (r.note) rows.push(`Note: ${r.note}`);
    for (const line of rows) {
      const div = document.createElement("div");
      div.textContent = line;
      out.appendChild(div);
    }
  } catch (e) {
    out.textContent = `Benchmark failed: ${e.message || e}`;
  } finally {
    btn.disabled = false;
  }
});

// ── TorchScript export ─────────────────────────────────────────────────────

document.getElementById("trainExportSelectCkpt").addEventListener("click", async () => {
  if (!window.api || !window.api.selectFile) return;
  const res = await window.api.selectFile({
    title: "Select Checkpoint (.ckpt)",
    filters: [{ name: "Checkpoint", extensions: ["ckpt"] }],
  });
  if (res.canceled) return;
  trainState.exportCkptPath = res.path;
  document.getElementById("trainExportCkptPath").textContent = res.path;
});

document.getElementById("trainExportRun").addEventListener("click", async () => {
  const statusEl = document.getElementById("trainExportStatus");
  const checkpointPath = trainState.exportCkptPath || trainState.lastBestCheckpoint;
  if (!checkpointPath) {
    statusEl.textContent = "No checkpoint — train a model or pick a .ckpt first.";
    return;
  }

  // Save into the project's models/exports when a project is open; otherwise ask.
  const pm = window.projectManager;
  let outputDir = pm && pm.isActive() ? pm.getExportsDir() : null;
  if (!outputDir) {
    const dirRes = await window.api.selectDirectory({ title: "Select output folder for .pt" });
    if (dirRes.canceled) return;
    outputDir = dirRes.path;
  }

  const outputName = (document.getElementById("trainExportName").value || "tsm_model").trim();
  const tsmStages = document.getElementById("trainTsmStages").value
    .split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));

  const btn = document.getElementById("trainExportRun");
  btn.disabled = true;
  statusEl.textContent = "Exporting…";
  try {
    const res = await backendFetch("/api/training/export-torchscript", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_path: checkpointPath,
        output_name: outputName,
        output_dir: outputDir,
        tsm_stages: tsmStages.length ? tsmStages : null,
      }),
    });
    statusEl.textContent = `Exported: ${res.output_path}`;
    // Record the export in the project manifest so it travels with the .rlt.
    if (pm && pm.isActive() && pm.addModel) {
      pm.addModel({
        name: res.output_path.split(/[\\/]/).pop(),
        path: res.output_path,
        created: new Date().toISOString().slice(0, 10),
        kind: "torchscript",
      });
    }
  } catch (e) {
    statusEl.textContent = `Export failed: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
});


// ── Register this pipeline with the shell ─────────────────────────────────
// Everything above is TSM-specific: background subtraction, crop tuning,
// extracted frame directories, a torch checkpoint. None of it belongs to the
// labeling shell, and nothing in core/ names any of it. The shell reaches this
// file only through the seam below, which is what lets the same shell run
// A-SOID Desktop's pose + random forest pipeline — or no pipeline at all, as a
// standalone labeler.
window.labelerPlugins.register({
  id: "realtime-tsm",

  hooks: {
    // A different video is active: load its saved background and crop box.
    onVideoSwitched: (ctx) => {
      applyVideoBackground(ctx.videoName);
      applyVideoCropParams(ctx.videoName);
    },
    // A pipeline tab became visible: its video checklist may be stale.
    onTabShown: (ctx) => {
      if (ctx.tab === "train") renderTrainVideoChecklist();
    },
    onProjectOpened: () => {
      renderTrainVideoChecklist();
      selectNewestCheckpoint({ silent: true });
    },
  },

  // What a trained model looks like on disk, for the shell's file pickers.
  models: { label: "Checkpoints", extensions: ["ckpt", "pt"] },

  // Turn the shell's context into this backend's inference request. The
  // response must come back in the standard predictions shape:
  //   { frames: [{ index, predicted_class, uncertainty }], class_names: [...] }
  // indexed by native video frame — the same shape A-SOID Desktop returns,
  // which is why the entire Active Learning panel is shared code.
  inference: {
    buildRequest: (ctx) => {
      const pm = ctx.projectManager;
      const bgPath = pm && pm.isActive()
        ? pm.getVideoBackgroundAbs(ctx.videoName)
        : ppState.bgPath;
      return {
        video_path: ctx.videoPath,
        bg_path: bgPath || null,
        crop_params: getCropParams(),
        frames_per_group: parseInt(document.getElementById("alFramesPerGroup").value) || 3,
      };
    },
    validate: (ctx) => (ctx.videoPath ? null : "Load a video first."),
  },

  // The shell owns one settings blob in localStorage; the pipeline contributes
  // its own keys to it rather than the shell knowing what a crop box is.
  // Keys must not collide with the shell's (playerCol, showHud, alThreshold…).
  settings: {
    save: () => ({
      cropParams: getCropParams(),
      bgNumFrames: parseInt(document.getElementById("ppBgNumFrames").value),
      lastBgPath: ppState.bgPath,
      bgByVideo: ppState.bgByVideo,
      cropByVideo: ppState.cropByVideo,
      trainVariant: document.getElementById("trainVariant").value,
      trainFramesPerGroup: parseInt(document.getElementById("trainFramesPerGroup").value),
      trainTsmStages: document.getElementById("trainTsmStages").value,
      trainTemporalPool: document.getElementById("trainTemporalPool").value,
      trainUnidirectional: document.getElementById("trainUnidirectional").checked,
      trainLr: parseFloat(document.getElementById("trainLr").value),
      trainWeightDecay: parseFloat(document.getElementById("trainWeightDecay").value),
      trainDropout: parseFloat(document.getElementById("trainDropout").value),
      trainMaxEpochs: parseInt(document.getElementById("trainMaxEpochs").value),
      trainPatience: parseInt(document.getElementById("trainPatience").value),
      trainSplitTrain: parseFloat(document.getElementById("trainSplitTrain").value),
      trainSplitVal: parseFloat(document.getElementById("trainSplitVal").value),
      trainSplitTest: parseFloat(document.getElementById("trainSplitTest").value),
      trainBoundarySmoothing: document.getElementById("trainBoundarySmoothing").checked,
      trainBsEpsilon: parseFloat(document.getElementById("trainBsEpsilon").value),
      trainBsWindow: parseInt(document.getElementById("trainBsWindow").value),
      trainDataSources: trainState.dataSources,
      alFramesPerGroup: parseInt(document.getElementById("alFramesPerGroup").value),
    }),

    load: (s) => {
      const setVal = (id, v) => { if (v !== undefined && v !== null) document.getElementById(id).value = v; };
      const setChecked = (id, v) => { if (v !== undefined) document.getElementById(id).checked = v; };

      // Global last-used crop; per-video overrides apply on switch.
      if (s.cropParams) applyCropParams(s.cropParams);
      if (s.cropByVideo && typeof s.cropByVideo === "object") ppState.cropByVideo = s.cropByVideo;
      if (s.bgByVideo && typeof s.bgByVideo === "object") ppState.bgByVideo = s.bgByVideo;
      if (s.bgNumFrames) setVal("ppBgNumFrames", s.bgNumFrames);
      if (s.lastBgPath) {
        ppState.bgPath = s.lastBgPath;
        setPathDisplay("ppBgPath", s.lastBgPath);
      }

      setVal("trainVariant", s.trainVariant);
      setVal("trainFramesPerGroup", s.trainFramesPerGroup);
      setVal("trainTsmStages", s.trainTsmStages);
      setVal("trainTemporalPool", s.trainTemporalPool);
      setChecked("trainUnidirectional", s.trainUnidirectional);
      setVal("trainLr", s.trainLr);
      setVal("trainWeightDecay", s.trainWeightDecay);
      setVal("trainDropout", s.trainDropout);
      setVal("trainMaxEpochs", s.trainMaxEpochs);
      setVal("trainPatience", s.trainPatience);
      setVal("trainSplitTrain", s.trainSplitTrain);
      setVal("trainSplitVal", s.trainSplitVal);
      setVal("trainSplitTest", s.trainSplitTest);
      setChecked("trainBoundarySmoothing", s.trainBoundarySmoothing);
      setVal("trainBsEpsilon", s.trainBsEpsilon);
      setVal("trainBsWindow", s.trainBsWindow);
      if (s.alFramesPerGroup) setVal("alFramesPerGroup", s.alFramesPerGroup);

      if (s.trainDataSources && s.trainDataSources.length) {
        trainState.dataSources = s.trainDataSources;
        renderTrainDataSources();
      }
    },
  },
});
