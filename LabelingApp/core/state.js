// core/state.js — DOM handles, key order, and the label/player state every other module reads.
//
// Part of the labeling shell. Loaded as a classic script in the order listed
// at the bottom of index.html; every module shares one global scope, so this
// split is a reorganization, not a rewrite.

const video = document.getElementById("video");
const videoShell = document.getElementById("videoShell");
const videoCanvas = document.getElementById("videoCanvas");
const trackbar = document.getElementById("trackbar");
const playPauseButton = document.getElementById("playPause");
const snapBackButton = document.getElementById("snapBack");
const fullscreenToggle = document.getElementById("fullscreenToggle");
const playbackSpeedInput = document.getElementById("playbackSpeed");
const pauseOnLabelEndInput = document.getElementById("pauseOnLabelEnd");
const fpsInput = document.getElementById("fpsInput");
const fpsLockInput = document.getElementById("fpsLock");
const useWebCodecsInput = document.getElementById("useWebCodecs");
const modeControl = document.getElementById("modeControl");
const webcodecsStatus = document.getElementById("webcodecsStatus");
const currentTimeEl = document.getElementById("currentTime");
const durationEl = document.getElementById("duration");
const videoFileInput = document.getElementById("videoFile");
const behaviorNameInput = document.getElementById("behaviorName");
const addBehaviorButton = document.getElementById("addBehavior");
const behaviorList = document.getElementById("behaviorList");
const boutsTable = document.getElementById("boutsTable");
const exportCsvButton = document.getElementById("exportCsv");
const importCsvButton = document.getElementById("importCsv");
const exportOneHotButton = document.getElementById("exportOneHot");
const fillBehaviorInput = document.getElementById("fillBehavior");
const generateLabeledVideoButton = document.getElementById("generateLabeledVideo");
const statusEl = document.getElementById("status");

const KEY_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

// index.html carries placeholder copy marked data-app="name|tagline|gate"; the
// host fills it in from window.LABELER_APP. Keeping the page free of any one
// app's name is what lets the same file ship with either pipeline.
for (const el of document.querySelectorAll("[data-app]")) {
  const text = (window.LABELER_APP || {})[el.dataset.app];
  if (text) el.textContent = text;
}

// Populated by core/behaviors.js as soon as loadBehaviors() exists. Function
// declarations hoist within a script but not across them, so this can no longer
// be initialized here the way it was when everything lived in one file.
let behaviors = [];
let bouts = [];
let activeStarts = new Map();
const selectedBehaviorIds = new Set(); // behaviors ticked for merging
const binarizeSelected = new Set();    // behavior NAMES forming the positive class (Train tab)

const insertBoutSorted = (bout) => {
  let i = bouts.length;
  while (i > 0 && bouts[i - 1].startFrame > bout.startFrame) i--;
  bouts.splice(i, 0, bout);
};
let currentVideoName = "";
let frameDurationMs = 1000 / 30;
let lastFrameMediaTime = null;
let currentFrameIndex = 0;
let pendingSeekFrame = null;
let pendingSeekTime = null;
let suppressTimeUpdate = false;
let currentFile = null;
