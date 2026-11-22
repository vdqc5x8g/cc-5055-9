import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createDefaultScene } from '../src/core/scene.js';
import { runPipeline } from '../src/core/pipeline.js';
import { projectPoint } from '../src/core/camera.js';
import { dist } from '../src/core/math.js';
import { reflectionValidity } from '../src/core/mirror.js';

const DEG = Math.PI / 180;
const CONSTRAINT = { center: [0, 1.02, 0], yaw: 0, tilt: 0.6 * DEG, width: 1.5, height: 2.0 };

function solve(overrides = {}) {
  const scene = createDefaultScene();
  return runPipeline({
    scene,
    observationOptions: { noisePx: 0.5, seed: 11, ...(overrides.observationOptions ?? {}) },
    mirrorConstraint: overrides.mirrorConstraint === undefined ? CONSTRAINT : overrides.mirrorConstraint,
    refit: overrides.refit ?? true,
    assetPlacement: overrides.assetPlacement,
  });
}

test('full pipeline solves camera and mirror within ground-truth tolerance', () => {
  const r = solve();
  assert.notEqual(r.cameraSource, 'truth-fallback');
  assert.ok(r.grading.cameraCenterErrorM < 0.02, `centre err ${r.grading.cameraCenterErrorM}`);
  assert.ok(r.grading.cameraRotationErrorDeg < 0.5, `rot err ${r.grading.cameraRotationErrorDeg}`);
  assert.ok(r.grading.focalErrorPx < 10, `focal err ${r.grading.focalErrorPx}`);
  assert.ok(r.grading.mirrorNormalErrorDeg < 0.4, `normal err ${r.grading.mirrorNormalErrorDeg}`);
  assert.ok(r.grading.mirrorPlaneDistanceErrorM < 0.01, `plane err ${r.grading.mirrorPlaneDistanceErrorM}`);
  assert.ok(r.branches.best.rmsPx < 2, `reflection rms ${r.branches.best.rmsPx}`);
  assert.equal(r.branches.best.detVirtual, -1);
});

test('chirality/depth branches separate the correct reflection from wrong models', () => {
  const r = solve();
  const byKey = Object.fromEntries(r.branches.results.map((b) => [b.key, b]));
  const refl = byKey.reflection.rmsPx;

  assert.equal(byKey.reflection.validModel, true);
  assert.equal(byKey['proper-flip'].validModel, false);
  assert.equal(byKey['direct-dup'].validModel, false);

  // The correct branch must explain the pixels far better than the alternatives.
  assert.ok(!Number.isFinite(byKey['direct-dup'].rmsPx) || byKey['direct-dup'].rmsPx > 20);
  assert.ok(byKey['proper-flip'].rmsPx > 100, `proper-flip rms ${byKey['proper-flip'].rmsPx}`);
  assert.ok(byKey['depth-0.75'].rmsPx > 20 * refl, `depth-0.75 rms ${byKey['depth-0.75'].rmsPx}`);
  assert.ok(byKey['depth-1.25'].rmsPx > 20 * refl, `depth-1.25 rms ${byKey['depth-1.25'].rmsPx}`);
});

test('horizontal-flip composite disagrees strongly with the true reflection', () => {
  const r = solve();
  assert.ok(r.flipCheck.meanDiscrepancyPx > 100, `flip discrepancy ${r.flipCheck.meanDiscrepancyPx}`);
});

test('missing direct board: mirror augmentation still recovers the camera', () => {
  const scene = createDefaultScene();
  const boardIds = scene.markers.filter((m) => m.group === 'board').map((m) => m.id);
  const r = solve({ observationOptions: { missingDirect: boardIds } });
  assert.equal(r.direct.length, 2, 'only floor + ledge direct markers remain');
  assert.ok(r.cameraSource.includes('mirror-augmented'), `source ${r.cameraSource}`);
  assert.ok(r.grading.cameraCenterErrorM < 0.06, `centre err ${r.grading.cameraCenterErrorM}`);
  assert.ok(r.grading.cameraRotationErrorDeg < 1.5, `rot err ${r.grading.cameraRotationErrorDeg}`);
  assert.ok(r.grading.mirrorNormalErrorDeg < 0.5);
  assert.equal(r.branches.best.detVirtual, -1);
});

test('no mirror constraint: plane still solved, panel provenance flagged as non-constraint', () => {
  const r = solve({ mirrorConstraint: null });
  assert.notEqual(r.cameraSource, 'truth-fallback');
  assert.ok(r.mirrorSolve.ok);
  assert.equal(r.mirrorSolve.panelSource, 'hit-centroid');
  assert.ok(r.diagnostics.some((d) => /尺度|场景单位/.test(d.message)), 'must warn about metric evidence');
  assert.ok(r.constraints.some((c) => c.key === 'mirror-plane'));
  assert.ok(r.grading.mirrorNormalErrorDeg < 0.4);
});

test('valid observations never fall outside the mirror panel', () => {
  const r = solve();
  for (const o of r.mirrorObs) {
    const v = reflectionValidity(r.camera, r.mirror, o.X);
    assert.ok(v.inBounds, `${o.id} reflection outside panel but used`);
  }
});

test('asset placed by ground-ray sits on y=0 and its contact projects to the picked pixel', () => {
  const scene = createDefaultScene();
  const truth = scene.cameraTruth;
  const contactXZ = [0.42, 0.95];
  const px = projectPoint(truth, [contactXZ[0], 0, contactXZ[1]]);
  const r = solve({ assetPlacement: { screenPoint: [px.x, px.y] } });
  assert.ok(Math.abs(r.asset.contact[1]) < 1e-9, 'contact must lie on the ground');
  assert.ok(dist(r.asset.contact, [contactXZ[0], 0, contactXZ[1]]) < 0.02, `contact err ${dist(r.asset.contact, [contactXZ[0], 0, contactXZ[1]])}`);
  const reproj = projectPoint(r.camera, r.asset.contact);
  assert.ok(Math.hypot(reproj.x - px.x, reproj.y - px.y) < 5, 'contact should reproject near the pick');
  assert.equal(r.reflectionCheck.ok, true);
});
