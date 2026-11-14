"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {createPreferences,crsName,captureViewport,restoreViewport} = require("../../frontend/shared/map/explorer.js");
const {metrics,sectorCsv} = require("../../frontend/features/results/results-evidence.js");
const {composition} = require("../../frontend/features/reporting/report-view.js");
const {isResultRevision} = require("../../frontend/shared/map/explorer.js");

test("Live and immutable sample revisions are accepted without allowing arbitrary paths", () => {
  for (const revision of ["12", "sample-0123456789abcdef"]) assert.equal(isResultRevision(revision), true);
  for (const revision of [null, "sample-bad", "../result", "sample-0123456789abcdef/extra"]) assert.equal(isResultRevision(revision), false);
});

test("Report composition uses all grid cells and keeps missing data out of unflagged counts",() => {
  const c=composition({changed_pixels:50,valid_pixels:100,roi:{model_grid_width_px:20,model_grid_height_px:10}});
  assert.equal(c.total,200);
  assert.deepEqual(c.parts.map(p=>p.count),[50,50,100]);
  assert.equal(c.parts.reduce((sum,p)=>sum+p.count,0),c.total);
  assert.equal(composition({}),null);
  assert.equal(composition({changed_pixels:11,valid_pixels:10,roi:{model_grid_width_px:10,model_grid_height_px:10}}),null);
});

test("Results and Change viewer share independent flags and dates for the same result",() => {
  const stored = new Map();
  const storage = () => ({getItem:k=>stored.get(k),setItem:(k,v)=>stored.set(k,v)});
  const results = createPreferences(storage), viewer = createPreferences(storage);
  results.save(3,{grayscale:true,showChange:false,date:"t1"});
  assert.deepEqual(viewer.read(3),{grayscale:true,showChange:false,date:"t1"});
  viewer.save(3,{showChange:true,viewerMode:"split",split:73});
  assert.deepEqual(results.read(3),{grayscale:true,showChange:true,date:"t1",viewerMode:"split",split:73});
  assert.deepEqual(results.read(4),{});
});
test("Invalid browser storage is ignored and blocked storage falls back to memory",() => {
  const corrupt = createPreferences(() => ({getItem:()=>'{"grayscale":"true","date":"bad","split":900}'}));
  assert.deepEqual(corrupt.read(2),{split:100});
  const blocked = createPreferences(() => {throw new Error("blocked");});
  blocked.save(2,{grayscale:false,showChange:true});
  assert.deepEqual(blocked.read(2),{grayscale:false,showChange:true});
});
test("The same geographic view is restored across differently sized canvases",() => {
  const captured = captureViewport({width:1000,height:1000,w:500,h:500,scale:1,fitScale:.5,pixel:()=>[350,650]});
  assert.deepEqual(captured,{zoom:2,x:.35,y:.65});
  const target = {width:1000,height:1000,w:800,h:600,fitScale:.6,maxScale:()=>8,clamp(){},invalidate(){}};
  restoreViewport(target,captured);
  assert.equal(target.scale,1.2);
  assert.equal(target.x,-20);
  assert.equal(target.y,-480);
});
test("Map reference names cover supported northern/southern UTM and polar grids",() => {
  assert.equal(crsName(32640),"WGS 84 / UTM zone 40N");
  assert.equal(crsName(32741),"WGS 84 / UTM zone 41S");
  assert.match(crsName(3413),/Polar Stereographic North/);
});
test("Coverage uses valid model cells and distinguishes missing data from zero change",() => {
  const result={valid_pixels:60,changed_pixels:0,roi:{model_grid_width_px:10,model_grid_height_px:10,model_spacing_m:20}};
  assert.deepEqual(metrics(result),{cells:100,coveragePct:60,validArea:.024,flaggedArea:0,unflaggedArea:.024});
  assert.equal(metrics({}).coveragePct,null);
  assert.equal(metrics({}).flaggedArea,null);
});
test("Sector export preserves signed values, empty samples, and provenance",() => {
  const csv = sectorCsv({threshold:.15,roi:{epsg:32640,model_spacing_m:20},paired_sector_deltas:[{zone:'upper "right"',flagged:{cells:50,median_delta_hh_hv_db:[-4.39,.78]},unflagged:{cells:2,median_delta_hh_hv_db:null}}]});
  assert.match(csv,/"upper ""right""","HH","50","2","-4.39","","","0.15","20","32640"/);
  assert.match(csv,/"HV","50","2","0.78"/);
});
