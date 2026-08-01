// labeler-plugin.js — the seam between the labeling shell and a pipeline.
//
// The shell (video player, behaviors, bouts, project manifest, timeline
// overlays, the active-learning relabel loop) is identical across
// ASOiD-Desktop and RealtimeASOID; measured, 91 of 125 divergent hunks in
// renderer.js sit in one region — the tabs that configure and run a pipeline.
// Everything else differs by a handful of lines.
//
// This module defines what a pipeline has to supply so the shell can stay
// pipeline-agnostic and eventually ship as its own standalone labeling app.
//
// A plugin is a plain object:
//
//   {
//     id: "asoid-pose",
//     hooks: {
//       onVideoLoaded(ctx)    // a video finished loading (frame count known)
//       onVideoSwitched(ctx)  // the active video changed
//       onProjectOpened(ctx)  // a project was opened or closed
//     },
//     overlays: [ { id, draw(ctx2d, ctx) } ],   // drawn over the video frame
//     inference: { buildRequest(ctx) },          // -> backend payload; the
//                                                //    response must be the
//                                                //    standard predictions shape
//     training:  { buildRequest(ctx) },
//   }
//
// `ctx` is the shell context (see makeContext in renderer.js): the current
// video, frame index, bouts, behaviors and project manager. Hooks are optional —
// a plugin supplies only what it needs, and the standalone labeler supplies
// none of it.
//
// THE CONTRACT THAT MAKES THIS WORK: inference must resolve to
//     { frames: [{ index, predicted_class, uncertainty }], class_names: [...] }
// indexed by native video frame. Both pipelines already produce exactly this —
// it is why the entire Active Learning panel ported between them unchanged —
// so the shell can own the heatmap, the misclassification ticks, the jump
// navigation and the on-video HUD without knowing what kind of model ran.

(function () {
  "use strict";

  const plugins = [];

  function register(plugin) {
    if (!plugin || !plugin.id) {
      console.warn("[labeler] ignoring plugin with no id", plugin);
      return;
    }
    if (plugins.some((p) => p.id === plugin.id)) {
      console.warn(`[labeler] plugin "${plugin.id}" already registered`);
      return;
    }
    plugins.push(plugin);
  }

  // Fire a lifecycle hook on every plugin that implements it. Never lets one
  // plugin's failure break the shell — a broken pose overlay must not stop the
  // user from labeling.
  function emit(hookName, ctx) {
    for (const p of plugins) {
      const fn = p.hooks && p.hooks[hookName];
      if (typeof fn !== "function") continue;
      try {
        fn(ctx);
      } catch (err) {
        console.warn(`[labeler] plugin "${p.id}" hook "${hookName}" failed`, err);
      }
    }
  }

  // Every registered canvas overlay, in registration order.
  function overlays() {
    return plugins.flatMap((p) => (Array.isArray(p.overlays) ? p.overlays : []));
  }

  // First plugin providing the named capability wins. Two pipelines are never
  // active at once, so this is a lookup rather than a merge.
  function capability(name) {
    for (const p of plugins) {
      if (p[name]) return p[name];
    }
    return null;
  }

  window.labelerPlugins = {
    register,
    emit,
    overlays,
    capability,
    list: () => plugins.map((p) => p.id),
  };
})();
