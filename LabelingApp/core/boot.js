// core/boot.js — Startup, and nothing else. Loaded last.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.
//
// When the shell was one 6,000-line file, these calls could sit anywhere: a
// single script hoists all its function declarations, so the first paint could
// happen halfway down and still reach code defined at the bottom. Split into
// separate scripts, hoisting stops at the file boundary — the first paint
// reaches the HUD, the uncertainty overlay and the project loader, all of which
// load after it. So all startup work happens here, once everything exists.
//
// The rule this file enforces: modules declare, boot executes.

document.body.dataset.tab = "label";   // matches the default active tab

// Active Learning is the one shell tab that needs a model behind it — every
// control in it either lists checkpoints or runs inference. With no pipeline
// registered there is nothing it could do, so drop it rather than leave a tab
// of dead buttons. Every other tab works with no backend at all.
if (!window.labelerPlugins || !window.labelerPlugins.capability("inference")) {
  for (const el of document.querySelectorAll('[data-tab="active-learning"]')) el.remove();
  const uc = document.getElementById("uncertaintyCanvas");
  if (uc) uc.style.display = "none";
}

renderBehaviors();
renderBouts();
updateDurationDisplay();
renderVideoSidebar();
updateContentLayout();

// Fold together duplicate behaviors (spacing/case variants) left by older saved
// data, and repaint if anything actually merged.
{
  const merged = mergeDuplicateBehaviors();
  if (merged > 0) {
    renderBehaviors();
    renderBouts();
    setStatus(`Merged ${merged} duplicate behavior${merged === 1 ? "" : "s"}.`);
  }
}

// Reopen whatever video was last active, if it is still on disk.
{
  const lastActive = localStorage.getItem("lastActiveVideo");
  const entry = lastActive && videoList.find((v) => v.name === lastActive);
  if (entry && entry.path) switchToVideo(lastActive);
}
