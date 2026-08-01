// core/behaviors.js — Behaviors and bouts: rendering, editing, name normalization, merging.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

// The saved behavior list, restored the moment its loader exists. core/state.js
// declares `behaviors` but cannot fill it — loadBehaviors is declared here, and
// function hoisting does not cross script boundaries.
behaviors = loadBehaviors();

const renderBehaviors = () => {
  behaviorList.innerHTML = "";

  if (behaviors.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No behaviors yet.";
    behaviorList.appendChild(empty);
    updateMergeButton();
    renderBinarizeChecklist();
    renderAlBehaviorButtons();
    updateAlFrameInfo();
    return;
  }

  behaviors.forEach((behavior) => {
    const row = document.createElement("div");
    row.className = "behavior-row";
    if (activeStarts.has(behavior.id)) {
      row.classList.add("behavior-active");
    }

    const mergeCheck = document.createElement("input");
    mergeCheck.type = "checkbox";
    mergeCheck.className = "merge-check";
    mergeCheck.title = "Select to merge";
    mergeCheck.checked = selectedBehaviorIds.has(behavior.id);
    mergeCheck.addEventListener("change", () => {
      if (mergeCheck.checked) selectedBehaviorIds.add(behavior.id);
      else selectedBehaviorIds.delete(behavior.id);
      updateMergeButton();
    });

    const keyBadge = document.createElement("span");
    keyBadge.className = "key-badge";
    keyBadge.textContent = behavior.key;

    const name = document.createElement("span");
    name.className = "behavior-name";
    name.textContent = behavior.name;
    name.title = "Click to rename";
    name.style.cursor = "pointer";
    name.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "text";
      input.value = behavior.name;
      input.className = "behavior-name-edit";
      input.style.width = Math.max(80, name.offsetWidth + 10) + "px";

      const commit = () => {
        const newName = normalizeBehaviorName(input.value);
        if (!newName) { renderBehaviors(); return; }
        const clash = behaviors.find(
          (b) => b !== behavior && behaviorNameKey(b.name) === behaviorNameKey(newName)
        );
        if (clash) {
          setStatus(`"${clash.name}" already exists; rename canceled.`);
          renderBehaviors();
          return;
        }
        if (newName !== behavior.name) {
          behavior.name = newName;
          for (const bout of bouts) {
            if (bout.behaviorId === behavior.id) bout.behaviorName = newName;
          }
          saveBehaviors();
          saveProgress();
          renderBouts();
        }
        renderBehaviors();
      };

      input.addEventListener("blur", commit);
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); input.blur(); }
        else if (e.key === "Escape") { input.value = behavior.name; input.blur(); }
      });

      name.replaceWith(input);
      input.focus();
      input.select();
    });

    const activeIndicator = document.createElement("span");
    activeIndicator.className = "active-indicator";
    activeIndicator.textContent = activeStarts.has(behavior.id)
      ? "Recording"
      : "";

    const removeButton = document.createElement("button");
    removeButton.className = "ghost";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", () => {
      behaviors = behaviors.filter((item) => item.id !== behavior.id);
      activeStarts.delete(behavior.id);
      selectedBehaviorIds.delete(behavior.id);
      saveBehaviors();
      renderBehaviors();
      renderBouts();
    });

    row.appendChild(mergeCheck);
    row.appendChild(keyBadge);
    row.appendChild(name);
    row.appendChild(activeIndicator);
    row.appendChild(removeButton);
    behaviorList.appendChild(row);
  });
  updateMergeButton();
  renderBinarizeChecklist();
  renderAlBehaviorButtons();
  updateAlFrameInfo();
};

// Render the Train-tab "Positive class (binarize)" checklist from the current
// behaviors. Tracked by behavior NAME (stable across project reloads, and the
// name is what the training payload needs).
function renderBinarizeChecklist() {
  const container = document.getElementById("trainBinarizeList");
  if (!container) return;
  container.innerHTML = "";

  if (behaviors.length === 0) {
    container.innerHTML = '<p class="muted">No behaviors yet.</p>';
    return;
  }
  // Drop names that no longer exist (renamed/removed/merged behaviors).
  const names = new Set(behaviors.map((b) => b.name));
  for (const n of [...binarizeSelected]) if (!names.has(n)) binarizeSelected.delete(n);

  for (const b of behaviors) {
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = binarizeSelected.has(b.name);
    cb.addEventListener("change", () => {
      if (cb.checked) binarizeSelected.add(b.name);
      else binarizeSelected.delete(b.name);
    });
    const span = document.createElement("span");
    span.textContent = b.name;
    row.append(cb, span);
    container.appendChild(row);
  }
}

const renderBouts = () => {
  boutsTable.innerHTML = "";

  if (bouts.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "muted";
    cell.textContent = "No bouts recorded.";
    row.appendChild(cell);
    boutsTable.appendChild(row);
    return;
  }

  bouts.forEach((bout, index) => {
    const row = document.createElement("tr");

    const startCell = document.createElement("td");
    startCell.textContent = formatTime(getFrameTimeMs(bout.startFrame));
    startCell.className = "time-link";
    startCell.title = "Jump to start";
    startCell.addEventListener("click", () => {
      seekToFrame(bout.startFrame);
    });

    const endCell = document.createElement("td");
    endCell.textContent = formatTime(getFrameTimeMs(bout.endFrame));
    endCell.className = "time-link";
    endCell.title = "Jump to end";
    endCell.addEventListener("click", () => {
      seekToFrame(bout.endFrame);
    });

    const behaviorCell = document.createElement("td");
    behaviorCell.textContent = bout.behaviorName;
    behaviorCell.className = "time-link";
    behaviorCell.title = "Click to reassign behavior";
    behaviorCell.addEventListener("click", () => {
      const select = document.createElement("select");
      for (const b of behaviors) {
        const opt = document.createElement("option");
        opt.value = b.id;
        opt.textContent = b.name;
        if (b.id === bout.behaviorId) opt.selected = true;
        select.appendChild(opt);
      }
      const commit = () => {
        const chosen = behaviors.find((b) => b.id === select.value);
        if (chosen && chosen.id !== bout.behaviorId) {
          bout.behaviorId = chosen.id;
          bout.behaviorName = chosen.name;
          saveProgress();
        }
        renderBouts();
      };
      select.addEventListener("change", commit);
      select.addEventListener("blur", () => renderBouts());
      behaviorCell.textContent = "";
      behaviorCell.appendChild(select);
      select.focus();
    });

    const actionsCell = document.createElement("td");
    const editButton = document.createElement("button");
    editButton.className = "ghost";
    editButton.textContent = "Edit";
    editButton.addEventListener("click", () => {
      editBoutRow(index);
    });
    const removeButton = document.createElement("button");
    removeButton.className = "ghost";
    removeButton.textContent = "Delete";
    removeButton.addEventListener("click", () => {
      bouts.splice(index, 1);
      renderBouts();
      saveProgress();
    });
    actionsCell.appendChild(editButton);
    actionsCell.appendChild(removeButton);

    row.appendChild(startCell);
    row.appendChild(endCell);
    row.appendChild(behaviorCell);
    row.appendChild(actionsCell);
    boutsTable.appendChild(row);
  });

  // The bout covering the current frame may have just changed, so refresh the
  // label readout (panel + on-video HUD). Without this, closing a bout on a
  // paused frame leaves the overlay showing the pre-edit state until you scrub.
  updateAlFrameInfo();
};

const saveBehaviors = () => {
  localStorage.setItem("behaviors", JSON.stringify(behaviors));
  try {
    if (window.projectManager && window.projectManager.isActive()) {
      window.projectManager.updateBehaviors(behaviors);
    }
  } catch { /* project manager not ready */ }
};

function loadBehaviors() {
  try {
    const stored = localStorage.getItem("behaviors");
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn("Failed to load behaviors", error);
    return [];
  }
}

// ── Behavior-name normalization & de-duplication ──────────────────────────
// Trim ends and collapse internal whitespace runs so "face  groom",
// "face groom ", and "face groom" are treated as one behavior.
const normalizeBehaviorName = (s) =>
  String(s == null ? "" : s).trim().replace(/\s+/g, " ");

// Canonical matching key: normalized + case-insensitive.
const behaviorNameKey = (s) => normalizeBehaviorName(s).toLowerCase();

const findBehaviorByName = (name) => {
  const key = behaviorNameKey(name);
  return behaviors.find((b) => behaviorNameKey(b.name) === key);
};

// Merge existing behaviors that share a canonical name: keep the first, remap
// every bout (current video + all stored per-video progress) to it, and clean
// up display names. Returns the number of duplicate behaviors removed.
const mergeDuplicateBehaviors = () => {
  const survivors = new Map(); // canonical key -> survivor behavior
  const idMap = new Map();     // duplicate id -> survivor id
  const kept = [];
  let renamed = false;

  for (const b of behaviors) {
    const key = behaviorNameKey(b.name);
    const survivor = survivors.get(key);
    if (survivor) {
      idMap.set(b.id, survivor.id);
    } else {
      const norm = normalizeBehaviorName(b.name);
      if (norm !== b.name) { b.name = norm; renamed = true; }
      survivors.set(key, b);
      kept.push(b);
    }
  }

  if (idMap.size === 0 && !renamed) return 0;

  behaviors = kept;
  const nameById = new Map(kept.map((b) => [b.id, b.name]));

  const remap = (list) => {
    let changed = false;
    for (const bout of list) {
      const sid = idMap.get(bout.behaviorId);
      if (sid) {
        bout.behaviorId = sid;
        bout.behaviorName = nameById.get(sid) || bout.behaviorName;
        changed = true;
      } else if (nameById.has(bout.behaviorId)) {
        const nn = nameById.get(bout.behaviorId);
        if (bout.behaviorName !== nn) { bout.behaviorName = nn; changed = true; }
      }
    }
    return changed;
  };

  remap(bouts);

  // Rewrite every video's stored progress so merges apply project-wide.
  try {
    const vids = (typeof videoList !== "undefined" && Array.isArray(videoList)) ? videoList : [];
    for (const v of vids) {
      const prog = loadProgress(v.name);
      if (prog && Array.isArray(prog.bouts) && remap(prog.bouts)) {
        localStorage.setItem(getProgressKey(v.name), JSON.stringify(prog));
      }
    }
  } catch { /* ignore */ }

  saveBehaviors();
  return idMap.size;
};

// Reassign every bout from `fromIds` to `target` (id + name), across the current
// video and all stored per-video progress.
const reassignBoutsToBehavior = (fromIds, target) => {
  const fromSet = new Set(fromIds);
  const apply = (list) => {
    let changed = false;
    for (const bout of list) {
      if (fromSet.has(bout.behaviorId)) {
        bout.behaviorId = target.id;
        bout.behaviorName = target.name;
        changed = true;
      }
    }
    return changed;
  };
  apply(bouts);
  try {
    const vids = (typeof videoList !== "undefined" && Array.isArray(videoList)) ? videoList : [];
    for (const v of vids) {
      const prog = loadProgress(v.name);
      if (prog && Array.isArray(prog.bouts) && apply(prog.bouts)) {
        localStorage.setItem(getProgressKey(v.name), JSON.stringify(prog));
      }
    }
  } catch { /* ignore */ }
};

// Destructively merge `ids` into the behavior `targetId`: reassign their bouts
// to the target and remove the others.
const mergeBehaviors = (ids, targetId) => {
  const target = behaviors.find((b) => b.id === targetId);
  if (!target) return;
  const fromIds = ids.filter((id) => id !== targetId && behaviors.some((b) => b.id === id));
  if (fromIds.length === 0) return;

  reassignBoutsToBehavior(fromIds, target);
  const removeSet = new Set(fromIds);
  behaviors = behaviors.filter((b) => !removeSet.has(b.id));
  for (const id of removeSet) { activeStarts.delete(id); selectedBehaviorIds.delete(id); }
  selectedBehaviorIds.delete(targetId);

  saveBehaviors();
  renderBehaviors();
  renderBouts();
  saveProgress();
  setStatus(`Merged ${fromIds.length + 1} behaviors into "${target.name}".`);
};

// Show/hide the "Merge selected" button based on how many behaviors are ticked.
function updateMergeButton() {
  const btn = document.getElementById("mergeBehaviors");
  if (!btn) return;
  const count = behaviors.filter((b) => selectedBehaviorIds.has(b.id)).length;
  btn.classList.toggle("hidden", count < 2);
  btn.textContent = count >= 2 ? `Merge ${count} selected` : "Merge selected";
}

// Modal to pick which selected behavior to keep. Resolves to its id, or null.
function chooseMergeTarget(selected) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const modal = document.createElement("div");
    modal.className = "modal";

    const title = document.createElement("h3");
    title.textContent = "Merge behaviors";
    const body = document.createElement("p");
    body.textContent =
      `Combine ${selected.length} behaviors into one. Their bouts are reassigned ` +
      `to the kept behavior, and the others are removed. This can't be undone.`;

    const label = document.createElement("label");
    label.textContent = "Keep: ";
    const select = document.createElement("select");
    for (const b of selected) {
      const opt = document.createElement("option");
      opt.value = b.id;
      opt.textContent = `${b.name} (key ${b.key})`;
      select.appendChild(opt);
    }
    label.appendChild(select);

    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const mergeBtn = document.createElement("button");
    mergeBtn.className = "primary";
    mergeBtn.textContent = "Merge";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost";
    cancelBtn.textContent = "Cancel";

    const close = (val) => {
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
      resolve(val);
    };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(null); } };
    mergeBtn.addEventListener("click", () => close(select.value));
    cancelBtn.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
    document.addEventListener("keydown", onKey, true);

    actions.append(mergeBtn, cancelBtn);
    modal.append(title, body, label, actions);
    overlay.append(modal);
    document.body.append(overlay);
    select.focus();
  });
}

document.getElementById("mergeBehaviors").addEventListener("click", async () => {
  const selected = behaviors.filter((b) => selectedBehaviorIds.has(b.id));
  if (selected.length < 2) return;
  const targetId = await chooseMergeTarget(selected);
  if (!targetId) return;
  mergeBehaviors(selected.map((b) => b.id), targetId);
});
