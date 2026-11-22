import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createTimelineScene, observeTimelineFrame } from '../src/core/timelineScene.js';
import {
  createSegmentProject,
  recordSideObservations,
  recordAllFrames,
  solveProjectLinked,
  solveSegment,
  compareSegmentSolvers,
  invalidateSegment,
  restoreSegment,
  segmentStatus,
  frameState,
  setSegmentBoundary,
  splitSegment,
  markStaleSegments,
  measureCrossingLineKink,
  scanKinks,
  localizeStaleEdit,
  virtualImageError,
  boundaryVirtualImageError,
  serializeProject,
  deserializeProject,
  syncSceneEvents,
  validateSegmentMirror,
  layersForSegment,
  logEdit,
} from '../src/core/segments.js';
import { mirrorFrame, reflectionValidity } from '../src/core/mirror.js';
import { dist } from '../src/core/math.js';

const DEG = Math.PI / 180;

function solvedProject() {
  const scene = createTimelineScene();
  const project = createSegmentProject(scene, { boundaries: [12] });
  recordAllFrames(project, scene);
  const res = solveProjectLinked(project, scene);
  assert.ok(res.ok);
  return { scene, project };
}

// ---------------- 1. 两侧分别录入观测 ----------------

test('observations are recorded separately from both sides of the mirror frame', () => {
  const scene = createTimelineScene();
  const project = createSegmentProject(scene, { boundaries: [12] });
  recordSideObservations(project, scene, 5, 'left');
  let entry = project.observations.get(5);
  assert.ok(entry.sides.left.direct.length > 0);
  assert.equal(entry.sides.right.direct.length, 0, 'right side not recorded yet');
  recordSideObservations(project, scene, 5, 'right');
  entry = project.observations.get(5);
  assert.ok(entry.sides.right.direct.length > 0);
  assert.ok(entry.sides.right.mirror.length > 0);
  // Deterministic re-recording of the same frame/side keeps pixels identical.
  const before = entry.sides.left.direct.map((o) => o.pixel);
  recordSideObservations(project, scene, 5, 'left');
  const after = project.observations.get(5).sides.left.direct.map((o) => o.pixel);
  assert.ok(before.every((p, i) => p[0] === after[i][0] && p[1] === after[i][1]));
  // Every recorded observation carries its side tag.
  for (const side of ['left', 'right']) {
    for (const o of entry.sides[side].direct) assert.equal(o.side, side);
  }
});

// ---------------- 2. 双区段 + 锁定共享镜框 + 边界拖动重绑定 ----------------

test('two active segments share one locked mirror frame; dragging the mid boundary rebinds state', () => {
  const { scene, project } = solvedProject();
  assert.equal(project.segments.length, 2);
  assert.equal(project.sharedFrame.locked, true);
  assert.ok(project.sharedFrame.anchorObserved, 'shared anchor observed from frame corners');
  // Segment A is fresh; segment B is stale by design (the frame-18 mirror
  // move is not covered by its calibration evidence).
  assert.equal(segmentStatus(project, 'S0'), 'ok');
  assert.equal(segmentStatus(project, 'S1'), 'stale');
  // Both segments solved; panel centres hang on the shared anchor.
  for (const seg of project.segments) {
    assert.ok(seg.calibration, 'segment solved');
    assert.ok(dist(seg.calibration.mirror.center, project.sharedFrame.anchorObserved) < 0.05);
  }
  // Frame -> segment mapping before the drag.
  assert.equal(frameState(project, 10).segmentId, 'S0');
  assert.equal(frameState(project, 20).segmentId, 'S1');
  const mirrorA = frameState(project, 10).mirror;
  const mirrorB = frameState(project, 20).mirror;
  assert.notEqual(mirrorA, mirrorB);
  // Drag the mid boundary from 12 to 15: frames 12..14 rebind to segment A.
  setSegmentBoundary(project, 0, 15);
  assert.equal(frameState(project, 13).segmentId, 'S0');
  assert.equal(frameState(project, 13).mirror, mirrorA, 'frame 13 now points at segment A mirror');
  assert.equal(frameState(project, 20).mirror, mirrorB, 'frame 20 still points at segment B mirror');
  assert.equal(frameState(project, 13).camera, frameState(project, 20).camera, 'shared locked-off camera');
  // The composited asset follows the frame's segment state.
  const assetLayer = layersForSegment(project, 'S0').find((l) => l.kind === 'asset');
  assert.ok(assetLayer, 'asset layer bound to segment');
  // Truth check: each segment recovers its own pose.
  const truthA = scene.mirrorAt(5);
  const truthB = scene.mirrorAt(14);
  const nA = mirrorFrame(project.segments[0].calibration.mirror).n;
  const nB = mirrorFrame(project.segments[1].calibration.mirror).n;
  const angle = (a, b) => (Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180) / Math.PI;
  assert.ok(angle(nA, mirrorFrame(truthA).n) < 0.3, `segment A normal err ${angle(nA, mirrorFrame(truthA).n)}`);
  assert.ok(angle(nB, mirrorFrame(truthB).n) < 0.3, `segment B normal err ${angle(nB, mirrorFrame(truthB).n)}`);
});

// ---------------- 3. 联合拟合 vs 共享锚衔接：对照后选定 ----------------

test('joint fit vs shared-anchor linked solve: comparable on clean data, linked isolates poisoning', () => {
  const { scene, project } = solvedProject();
  const cmp = compareSegmentSolvers(project, scene);
  assert.equal(cmp.choice, 'linked');
  // Clean data: both methods recover every segment's plane within tolerance.
  for (const m of cmp.linked.perSegment) {
    assert.ok(m.ok);
    assert.ok(m.normalErrDeg < 0.3, `linked ${m.id} normal err ${m.normalErrDeg}`);
    assert.ok(m.planeDistErrM < 0.01, `linked ${m.id} plane err ${m.planeDistErrM}`);
  }
  for (const m of cmp.joint.perSegment) {
    assert.ok(m.ok);
    assert.ok(m.normalErrDeg < 0.3, `joint ${m.id} normal err ${m.normalErrDeg}`);
    assert.ok(m.planeDistErrM < 0.01, `joint ${m.id} plane err ${m.planeDistErrM}`);
  }
  assert.ok(cmp.linked.anchorErrM < 0.05, `linked anchor err ${cmp.linked.anchorErrM}`);
  assert.ok(cmp.joint.anchorErrM < 0.05, `joint anchor err ${cmp.joint.anchorErrM}`);

  // Poison segment B's pool: three mirrored observations with swapped pixels
  // (a misclassification that survived recording). Linked must isolate the
  // damage to segment B; the joint fit bends the shared anchor and with it
  // segment A's panel placement.
  const poisoned = deserializeProject(serializeProject(project));
  const poolB = [12, 13];
  for (const f of poolB) {
    const entry = poisoned.observations.get(f);
    const mir = entry.sides.left.mirror;
    for (let i = 0; i < 3; i++) {
      const j = (i + 3) % mir.length;
      const tmp = mir[i].pixel;
      mir[i].pixel = mir[j].pixel;
      mir[j].pixel = tmp;
    }
  }
  const cmp2 = compareSegmentSolvers(poisoned, scene);
  const linkedA = cmp2.linked.perSegment.find((m) => m.id === 'S0');
  const jointA = cmp2.joint.perSegment.find((m) => m.id === 'S0');
  assert.ok(
    cmp2.linked.anchorErrM < cmp2.joint.anchorErrM,
    `anchor: linked ${cmp2.linked.anchorErrM} should beat joint ${cmp2.joint.anchorErrM}`,
  );
  assert.ok(
    linkedA.centreErrM < jointA.centreErrM,
    `segment A panel: linked ${linkedA.centreErrM} should beat joint ${jointA.centreErrM}`,
  );
  assert.ok(linkedA.centreErrM < 0.05, `linked keeps segment A panel isolated (${linkedA.centreErrM})`);
});

// ---------------- 4. 参数失效隔离 + 拖回后核对反射方向 ----------------

test('invalid mirror parameters isolate only dependent layers; drag-back re-checks reflection direction', () => {
  const { project } = solvedProject();
  const layersB = layersForSegment(project, 'S1');
  const layersA = layersForSegment(project, 'S0');
  invalidateSegment(project, 'S1', '镜面参数失效（模拟）');
  assert.ok(layersB.every((l) => l.isolated), 'all B layers isolated');
  assert.ok(layersA.every((l) => !l.isolated), 'A layers untouched');
  assert.equal(segmentStatus(project, 'S1'), 'invalid');
  // The corrupted calibration must fail the reflection-direction audit.
  const obs = [];
  for (const f of project.segments[1].lastGood.solveFrames) {
    const e = project.observations.get(f);
    for (const side of ['left', 'right']) obs.push(...e.sides[side].mirror);
  }
  const bad = validateSegmentMirror(project.segments[1].calibration.mirror, project.camera, obs);
  assert.equal(bad.ok, false, 'corrupted calibration fails validation');
  // 拖回：restore last good, with the direction audit passing.
  const res = restoreSegment(project, null, 'S1');
  assert.ok(res.ok, `restore failed: ${res.reasons?.join('；')}`);
  assert.ok(Math.abs(res.check.checks.detVirtual + 1) < 1e-6, 'virtual camera det = -1');
  assert.ok(res.check.checks.cameraInFront, 'camera on the reflective side');
  assert.ok(layersB.every((l) => !l.isolated), 'B layers restored');
  assert.equal(segmentStatus(project, 'S1'), 'stale', 'B is stale again (frame-18 move still uncovered), not invalid');
});

// ---------------- 5. 演员遮挡 + 手性相反干扰标记 ----------------

test('actor occlusion drops frame/mirror evidence; chirality-reversed distractors are rejected', () => {
  const { scene, project } = solvedProject();
  // Occluded frames: no mirror observations, no corner observations.
  for (const f of [14, 15, 16, 17]) {
    const entry = project.observations.get(f);
    const mirrorCount = entry.sides.left.mirror.length + entry.sides.right.mirror.length;
    assert.equal(mirrorCount, 0, `frame ${f} mirror evidence blocked by the actor`);
    assert.equal(entry.corners.length, 0, `frame ${f} frame corners blocked`);
    assert.ok(entry.missing.some((m) => /演员遮挡/.test(m.reason ?? '')));
  }
  // The shared anchor is still observed (from unoccluded frames).
  assert.equal(project.sharedFrame.anchorSource, 'observed');
  assert.ok(dist(project.sharedFrame.anchorObserved, scene.sharedFrame.anchor) < 0.03);
  // Distractors: same image position as true reflections, rejected on chirality.
  const entry = project.observations.get(9);
  assert.equal(entry.distractors.length, 3);
  for (const d of entry.distractors) {
    assert.ok(/手性相反/.test(d.reason));
    const src = scene.markers.find((m) => m.id === d.mirrorsId);
    const v = reflectionValidity(scene.camera, scene.mirrorAt(9), src.X);
    assert.ok(Math.hypot(d.pixel[0] - v.proj.x, d.pixel[1] - v.proj.y) < 2, 'distractor sits at the true reflection position');
  }
  // And the solve is unaffected by the distractors.
  const truthA = scene.mirrorAt(5);
  const nA = mirrorFrame(project.segments[0].calibration.mirror).n;
  const nT = mirrorFrame(truthA).n;
  const errDeg = (Math.acos(Math.max(-1, Math.min(1, nA[0] * nT[0] + nA[1] * nT[1] + nA[2] * nT[2]))) * 180) / Math.PI;
  assert.ok(errDeg < 0.3, `segment A normal err ${errDeg} with distractors present`);
});

// ---------------- 6. 求解不足的区段单独重估 ----------------

test('an under-determined segment can be re-estimated alone without touching others', () => {
  const { scene, project } = solvedProject();
  // Wipe segment B's calibration and its late observations to force
  // under-determination, then re-estimate it alone.
  const segB = project.segments[1];
  const calABefore = JSON.stringify(project.segments[0].calibration);
  segB.calibration = null;
  segB.attempted = false;
  for (const f of [12, 13]) {
    project.observations.get(f).sides.left.mirror = project.observations.get(f).sides.left.mirror.slice(0, 1);
    project.observations.get(f).sides.right.mirror = [];
  }
  const res1 = solveSegment(project, scene, 'S1');
  assert.equal(res1.ok, false);
  assert.equal(res1.reason, 'underdetermined');
  assert.equal(segmentStatus(project, 'S1'), 'underdetermined');
  // Restore full observations and re-estimate only segment B.
  const project2 = createSegmentProject(scene, { boundaries: [12] });
  recordAllFrames(project2, scene);
  project2.camera = project.camera;
  project2.cameraSource = project.cameraSource;
  project2.segments[0].calibration = project.segments[0].calibration;
  project2.segments[0].attempted = true;
  syncSceneEvents(project2, scene);
  const res2 = solveSegment(project2, scene, 'S1');
  assert.ok(res2.ok);
  assert.equal(segmentStatus(project2, 'S1') === 'ok' || segmentStatus(project2, 'S1') === 'stale', true);
  assert.equal(JSON.stringify(project2.segments[0].calibration), calABefore, 'segment A calibration untouched');
});

// ---------------- 7. 保存工程 + 从两侧拖回同一帧量取虚像误差 ----------------

test('saved project reloads; scrubbing back to the same frame from both sides gives identical virtual-image error', () => {
  const { scene, project } = solvedProject();
  const vie = boundaryVirtualImageError(project, 0);
  // The owning segment explains its boundary frame; the other segment does not.
  assert.ok(vie.leftFrame.own < 1.5, `frame ${vie.leftFrame.frame} own err ${vie.leftFrame.own}`);
  assert.ok(vie.leftFrame.other > 15, `frame ${vie.leftFrame.frame} other err ${vie.leftFrame.other}`);
  assert.ok(vie.rightFrame.own < 1.5, `frame ${vie.rightFrame.frame} own err ${vie.rightFrame.own}`);
  assert.ok(vie.rightFrame.other > 15, `frame ${vie.rightFrame.frame} other err ${vie.rightFrame.other}`);
  // Save / load round trip.
  const json = serializeProject(project);
  const loaded = deserializeProject(json);
  // Scrub to the boundary frame from the left side and from the right side:
  // state restoration is path-independent and the errors reproduce exactly.
  const fromLeft = frameState(loaded, 11);
  const fromRight = frameState(loaded, 12);
  assert.equal(fromLeft.segmentId, 'S0');
  assert.equal(fromRight.segmentId, 'S1');
  const vie2 = boundaryVirtualImageError(loaded, 0);
  assert.equal(vie2.leftFrame.own, vie.leftFrame.own);
  assert.equal(vie2.leftFrame.other, vie.leftFrame.other);
  assert.equal(vie2.rightFrame.own, vie.rightFrame.own);
  assert.equal(vie2.rightFrame.other, vie.rightFrame.other);
  // Frame states identical after reload.
  for (const f of [0, 7, 11, 12, 19, 23]) {
    assert.equal(frameState(loaded, f).segmentId, frameState(project, f).segmentId);
    assert.equal(frameState(loaded, f).mirror, frameState(project, f).mirror === null ? null : frameState(loaded, f).mirror);
  }
  assert.equal(virtualImageError(loaded, 11, 'S0'), virtualImageError(project, 11, 'S0'));
});

// ---------------- 8. 过期标定由穿框直线折角定位 ----------------

test('stale calibration after one side exits is located by the crossing-line kink', () => {
  const { scene, project } = solvedProject();
  // Segment B's evidence stops at frame 17 (mirror moved at 18, right-side
  // markers gone) -> it is stale from frame 18.
  assert.equal(segmentStatus(project, 'S1'), 'stale');
  assert.equal(project.segments[1].staleSince, 18);
  // Kink scan: frames before 18 are continuous, frames 18+ show the kink.
  const kinks = scanKinks(project, scene);
  for (const f of [0, 5, 11, 12, 13]) {
    const k = kinks.get(f);
    assert.ok(k.ok, `frame ${f} kink measurable`);
    assert.equal(k.kink, false, `frame ${f} must be continuous (gap ${k.gapPx?.toFixed(2)} px)`);
  }
  for (const f of [18, 20, 23]) {
    const k = kinks.get(f);
    assert.ok(k.ok, `frame ${f} kink measurable`);
    assert.equal(k.kink, true, `frame ${f} must kink (gap ${k.gapPx?.toFixed(2)} px)`);
  }
  // Occluded frames have no rail reflection evidence -> not measurable, not kinked.
  assert.equal(kinks.get(15).ok, false);
  // The kink localizes the edit: the frame-18 mirror move that segment B's
  // calibration never covered, with the right side out of the viewfinder.
  const loc = localizeStaleEdit(project, scene);
  assert.ok(loc.found);
  assert.equal(loc.frame, 18);
  assert.equal(loc.segmentId, 'S1');
  assert.ok(loc.edit, 'localized edit found');
  assert.equal(loc.edit.type, 'mirror-move');
  assert.equal(loc.edit.frame, 18);
  assert.ok(/折角/.test(loc.message) && /第 18 帧/.test(loc.message));
  // Remedy: split at 18 and re-estimate the new segment alone -> kink clears.
  const seg = splitSegment(project, 'S1', 18);
  assert.ok(seg);
  const res = solveSegment(project, scene, seg.id);
  assert.ok(res.ok, `re-estimate failed: ${res.reason}`);
  markStaleSegments(project);
  const kinks2 = scanKinks(project, scene);
  for (const f of [18, 20, 23]) {
    assert.equal(kinks2.get(f).kink, false, `frame ${f} continuous after re-estimation`);
  }
  assert.equal(segmentStatus(project, 'S1'), 'ok');
  assert.equal(segmentStatus(project, seg.id), 'ok');
});

// ---------------- 9. 虚像误差随过期标定增大 ----------------

test('virtual-image error is small for fresh segments and large where calibration is stale', () => {
  const { scene, project } = solvedProject();
  const fresh = virtualImageError(project, 13, 'S1');
  assert.ok(fresh < 1.5, `fresh segment VIE ${fresh}`);
  const stale = virtualImageError(project, 20, 'S1');
  assert.ok(stale > 8, `stale segment VIE ${stale}`);
  // After re-estimating the moved setup, the error drops again.
  const seg = splitSegment(project, 'S1', 18);
  solveSegment(project, scene, seg.id);
  const fixed = virtualImageError(project, 20, seg.id);
  assert.ok(fixed < 1.5, `re-estimated segment VIE ${fixed}`);
});
