// core/panels.js — Mounting pipeline tabs into the shell's tab bar.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.
//
// index.html ships only the tabs the shell owns — Label and Active Learning.
// Everything a pipeline needs to configure itself (A-SOID's Pose/Train/Predict,
// RealtimeASOID's Preprocess/Train/Predict) is markup the pipeline supplies and
// mounts through here, so the same index.html serves either one, or neither.

// Insert a pipeline's tabs. Called from the top of a plugin file, before that
// file queries any of its own elements.
//
//   mountPanels([{ tab: "pose", label: "Pose", after: "label", html: "..." }])
//
// `after` is the data-tab of the tab this one should follow; omit it to append.
// The panel markup is the tab body only — the wrapping .tab-panel div is added
// here so a plugin cannot get the shell's class names wrong.
function mountPanels(panels) {
  const bar = document.querySelector(".tab-bar");
  const content = document.querySelector(".side-panels");
  if (!bar || !content) {
    console.warn("[labeler] no tab bar to mount panels into");
    return;
  }

  for (const p of panels) {
    if (!p || !p.tab) continue;
    if (content.querySelector(`.tab-panel[data-tab="${p.tab}"]`)) {
      console.warn(`[labeler] tab "${p.tab}" already mounted`);
      continue;
    }

    const btn = document.createElement("button");
    btn.className = "tab";
    btn.dataset.tab = p.tab;
    btn.textContent = p.label || p.tab;

    const panel = document.createElement("div");
    panel.className = "tab-panel";
    panel.dataset.tab = p.tab;
    panel.innerHTML = p.html || "";

    const afterBtn = p.after ? bar.querySelector(`.tab[data-tab="${p.after}"]`) : null;
    const afterPanel = p.after ? content.querySelector(`.tab-panel[data-tab="${p.after}"]`) : null;

    if (afterBtn) afterBtn.after(btn); else bar.appendChild(btn);
    if (afterPanel) afterPanel.after(panel); else content.appendChild(panel);
  }
}

// Fill one of the shell's slots — currently #alPipelineOptions, the gap in the
// Active Learning panel where a pipeline puts the controls its own inference
// needs. Silently does nothing if the slot isn't in this build's index.html, so
// a plugin can offer options a stripped-down shell doesn't have a place for.
function mountSlot(slotId, html) {
  const slot = document.getElementById(slotId);
  if (!slot) return false;
  slot.innerHTML = html;
  return true;
}
