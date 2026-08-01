#!/usr/bin/env node
// Two invariants that keep the labeling shell shareable. Both are invisible to
// `node --check` — the files parse fine and the app dies at runtime — so they
// are worth a script rather than a code-review habit.
//
//   1. SEAM. Nothing in core/ may name anything declared in plugins/. The shell
//      has to be runnable with plugins/ deleted; that is what lets the same
//      core/ serve A-SOID Desktop, RealtimeASOID and a standalone labeler.
//
//   2. LOAD ORDER. The shell used to be one 6,000-line script, where every
//      function declaration hoisted over the whole file. Split into separate
//      <script> tags, hoisting stops at each file boundary: anything a file
//      runs during evaluation — and everything that transitively calls — must
//      already be declared. This is what core/boot.js exists to satisfy.
//
// Usage: node tools/check-shell.mjs [LabelingApp dir]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as acorn from "acorn";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(here, "..");

// Load order is the source of truth, and index.html is where it is declared —
// so read it from there rather than keeping a second copy in sync.
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const files = [...html.matchAll(/<script src="((?:core|plugins)\/[^"]+)"><\/script>/g)].map((m) => m[1]);
if (files.length === 0) {
  console.error("no core/ or plugins/ scripts found in index.html");
  process.exit(1);
}

const declIndex = new Map();   // name -> index of the file declaring it
const bodyOf = new Map();      // name -> function node, for top-level functions
const asts = [];

const declare = (name, i) => { if (!declIndex.has(name)) declIndex.set(name, i); };

files.forEach((f, i) => {
  const src = fs.readFileSync(path.join(root, f), "utf8");
  const ast = acorn.parse(src, { ecmaVersion: 2023, locations: true });
  asts.push({ f, i, src, ast });
  for (const node of ast.body) {
    if (node.type === "FunctionDeclaration" && node.id) {
      declare(node.id.name, i);
      bodyOf.set(node.id.name, node);
    } else if (node.type === "ClassDeclaration" && node.id) {
      declare(node.id.name, i);
    } else if (node.type === "VariableDeclaration") {
      for (const d of node.declarations) {
        const collect = (p) => {
          if (!p) return;
          if (p.type === "Identifier") {
            declare(p.name, i);
            if (d.init && (d.init.type === "FunctionExpression" || d.init.type === "ArrowFunctionExpression")) {
              bodyOf.set(p.name, d.init);
            }
          } else if (p.type === "ObjectPattern") p.properties.forEach((q) => collect(q.value || q.argument));
          else if (p.type === "ArrayPattern") p.elements.forEach(collect);
        };
        collect(d.id);
      }
    }
  }
});

let failures = 0;
const fail = (msg) => { failures++; console.log("  " + msg); };

// ── 1. Seam ────────────────────────────────────────────────────────────────

const pluginOwned = new Map();   // name -> plugin file
for (const { f, i, ast } of asts) {
  if (!f.startsWith("plugins/")) continue;
  for (const [name, idx] of declIndex) if (idx === i) pluginOwned.set(name, f);
}

console.log(`seam: ${pluginOwned.size} symbols owned by plugins/`);
for (const { f, src } of asts) {
  if (!f.startsWith("core/")) continue;
  for (const [name, origin] of pluginOwned) {
    const re = new RegExp(`\\b${name.replace(/[$]/g, "\\$")}\\b`, "g");
    let m;
    while ((m = re.exec(src))) {
      fail(`${f}:${src.slice(0, m.index).split("\n").length}  names '${name}' from ${origin}`);
    }
  }
}

// ── 2. Load order ──────────────────────────────────────────────────────────

// Callbacks the callee invokes synchronously, so their bodies run during
// evaluation. addEventListener / setTimeout / promise handlers genuinely don't.
const IMMEDIATE = new Set(["forEach", "map", "filter", "reduce", "some", "every",
                           "find", "findIndex", "flatMap", "sort"]);

const skipKey = (n, key) =>
  key === "loc" || key === "start" || key === "end" || key === "type" || key.startsWith("__") ||
  (n.type === "MemberExpression" && key === "property" && !n.computed) ||
  (n.type === "Property" && key === "key" && !n.computed) ||
  (n.type === "MethodDefinition" && key === "key" && !n.computed);

function markImmediate(node) {
  const c = node.callee;
  if (c.type === "FunctionExpression" || c.type === "ArrowFunctionExpression") c.__runsNow = true;
  const m = c.type === "MemberExpression" && !c.computed && c.property.type === "Identifier"
    ? c.property.name : null;
  if (m && IMMEDIATE.has(m)) {
    for (const a of node.arguments) {
      if (a.type === "FunctionExpression" || a.type === "ArrowFunctionExpression") a.__runsNow = true;
    }
  }
}

// What runs while this script is being evaluated: top level, IIFEs, and
// synchronously-invoked callbacks. Passing a function is not calling it.
function scanEvaluated(node, read, called, deferred) {
  if (!node || typeof node.type !== "string") return;
  if (node.type === "FunctionDeclaration") return;
  if (node.type === "CallExpression") {
    markImmediate(node);
    if (!deferred && node.callee.type === "Identifier") called.add(node.callee.name);
  }
  if (node.type === "Identifier") { if (!deferred) read.add(node.name); return; }
  for (const key of Object.keys(node)) {
    if (skipKey(node, key)) continue;
    const child = node[key];
    const nowDeferred = deferred ||
      ((node.type === "FunctionExpression" || node.type === "ArrowFunctionExpression") &&
        key === "body" && !node.__runsNow);
    if (Array.isArray(child)) child.forEach((c) => scanEvaluated(c, read, called, nowDeferred));
    else if (child && typeof child === "object") scanEvaluated(child, read, called, nowDeferred);
  }
}

const cache = new Map();
function bodyRefs(fn) {
  if (cache.has(fn)) return cache.get(fn);
  const read = new Set(), called = new Set();
  const walk = (n) => {
    if (!n || typeof n.type !== "string") return;
    if (n.type === "CallExpression" && n.callee.type === "Identifier") called.add(n.callee.name);
    if (n.type === "Identifier") { read.add(n.name); return; }
    for (const key of Object.keys(n)) {
      if (skipKey(n, key)) continue;
      const child = n[key];
      if (Array.isArray(child)) child.forEach(walk);
      else if (child && typeof child === "object") walk(child);
    }
  };
  walk(fn.body);
  const out = { read, called };
  cache.set(fn, out);
  return out;
}

console.log(`load order: ${files.length} scripts`);
for (const entry of asts) {
  const read = new Set(), called = new Set();
  for (const stmt of entry.ast.body) scanEvaluated(stmt, read, called, false);

  const seen = new Set();
  const queue = [...called];
  while (queue.length) {
    const name = queue.pop();
    if (seen.has(name)) continue;
    seen.add(name);
    const fn = bodyOf.get(name);
    if (!fn) continue;
    const r = bodyRefs(fn);
    for (const x of r.read) read.add(x);
    for (const x of r.called) { read.add(x); if (!seen.has(x)) queue.push(x); }
  }

  const reported = new Set();
  for (const name of read) {
    const j = declIndex.get(name);
    if (j === undefined || j <= entry.i || reported.has(name)) continue;
    reported.add(name);
    fail(`${entry.f}  needs '${name}' at load time, declared later in ${files[j]}`);
  }
}

// ── 3. Project manager API ─────────────────────────────────────────────────
//
// project-manager.js is host-owned: its manifest schema is pipeline-specific
// (pose files and feature sets for one, backgrounds and frame dirs for
// another). But core/ calls into it, so the subset core/ uses is a contract the
// host has to satisfy. A missing method is a runtime crash in a rarely-taken
// branch, which is exactly the kind of thing that survives a manual port.

const pmPath = path.join(root, "project-manager.js");
if (fs.existsSync(pmPath)) {
  const pmSrc = fs.readFileSync(pmPath, "utf8");
  const wanted = new Set();
  for (const { f, src } of asts) {
    if (!f.startsWith("core/")) continue;
    for (const m of src.matchAll(/(?:projectManager|\bpm)\.([A-Za-z0-9_]+)\s*\(/g)) wanted.add(m[1]);
  }
  const missing = [...wanted].filter((n) => !new RegExp(`\\b${n}\\b\\s*[:(]`).test(pmSrc));
  console.log(`project manager: core/ calls ${wanted.size} methods`);
  for (const n of missing) fail(`project-manager.js  does not define '${n}', which core/ calls`);
}

console.log(failures ? `\n${failures} problem(s)` : "\nshell OK: seam holds, load order clean, project manager complete");
process.exit(failures ? 1 : 0);
