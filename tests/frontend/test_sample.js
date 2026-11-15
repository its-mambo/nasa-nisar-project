"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function page() {
  const nodes = new Map();
  const el = id => {
    if (!nodes.has(id)) nodes.set(id, {
      hidden:false, disabled:true, textContent:"", dataset:{}, firstChild:{},
      addEventListener() {},
    });
    return nodes.get(id);
  };
  const window = {};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../../frontend/features/samples/sample.js"), "utf8"), {
    document:{getElementById:el}, window, Intl, Date,
  });
  return {el, sample:window.NisarSample};
}

test("Bundled metadata enables Explore and supplies both dates and the preview", () => {
  const root = path.join(__dirname, "../../samples/judges");
  const {revision} = JSON.parse(fs.readFileSync(path.join(root, "current.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(path.join(root, revision, "manifest.json"), "utf8"));
  const {el, sample} = page();
  sample.configure({model_enabled:false, sample:{...manifest.sample, available:true, preview_url:`/sample-preview.png?v=${revision}`}});
  assert.equal(el("openSample").disabled, false);
  assert.match(el("sampleBefore").textContent, /21 June 2026/);
  assert.match(el("sampleAfter").textContent, /13 September 2026/);
  assert.equal(el("samplePreview").hidden, false);
  assert.equal(el("samplePreview").src, `/sample-preview.png?v=${revision}`);
});

test("A failed bundle shows the server error beside the disabled Explore button", () => {
  const {el, sample} = page();
  const error = "Cached sample checksum mismatch: result.json";
  sample.configure({model_enabled:true, sample:{available:false, error}});
  assert.equal(el("openSample").disabled, true);
  assert.equal(el("sampleStatus").textContent, error);
  assert.equal(el("sampleBefore").textContent, "Unavailable");
  assert.equal(el("samplePreview").hidden, true);
  assert.match(el("sampleModeNote").textContent, /complete checkout/);
});
