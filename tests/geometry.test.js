import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createDefaultScene, observe, directObservations, mirrorObservations } from '../src/core/scene.js';
import {
  makeMirror,
  mirrorFrame,
  reflectPoint,
  virtualCamera,
  reflectionHit,
  reflectionValidity,
  insideMirrorBounds,
  projectReflection,
  reflectionMatrix,
  normalizeMirror,
} from '../src/core/mirror.js';
import { projectPoint, cameraFromLookAt } from '../src/core/camera.js';
import { dltCamera, decomposeCameraMatrix } from '../src/core/dlt.js';
import { mat3Mul, mat3T, rotationErrorDeg, dot, sub, dist } from '../src/core/math.js';

const scene = createDefaultScene();
const DEG = Math.PI / 180;

// ------------------------- virtual camera / reflection algebra -------------------------

test('virtual camera is a separate improper camera (det = -1), not co-located markers', () => {
  const vc = virtualCamera(scene.cameraTruth, scene.mirror);
  const det =
    vc.R[0][0] * (vc.R[1][1] * vc.R[2][2] - vc.R[1][2] * vc.R[2][1]) -
    vc.R[0][1] * (vc.R[1][0] * vc.R[2][2] - vc.R[1][2] * vc.R[2][0]) +
    vc.R[0][2] * (vc.R[1][0] * vc.R[2][1] - vc.R[1][1] * vc.R[2][0]);
  assert.ok(Math.abs(det + 1) < 1e-9, `det(R_virtual) = ${det}, expected -1`);

  // Its centre is the reflected real centre, not the real centre.
  assert.ok(dist(vc.C, scene.cameraTruth.C) > 1, 'virtual centre must be mirrored across the plane');
  const reflectedC = reflectPoint(scene.mirror, scene.cameraTruth.C);
  assert.ok(dist(vc.C, reflectedC) < 1e-9);
});

test('virtual-camera projection equals projecting the reflected point through the real camera', () => {
  const vc = virtualCamera(scene.cameraTruth, scene.mirror);
  let maxErr = 0;
  for (const m of scene.markers) {
    const a = projectPoint(vc, m.X);
    const b = projectPoint(scene.cameraTruth, reflectPoint(scene.mirror, m.X));
    assert.ok(a && b);
    maxErr = Math.max(maxErr, Math.hypot(a.x - b.x, a.y - b.y));
  }
  assert.ok(maxErr < 1e-6, `max pixel discrepancy ${maxErr}`);
});

test('mirror reflection is an involution and preserves the plane', () => {
  const m = makeMirror({ center: [0.1, 1.0, -0.2], yaw: 12 * DEG, tilt: -4 * DEG });
  const X = [0.6, 1.4, 1.1];
  const Y = reflectPoint(m, X);
  const back = reflectPoint(m, Y);
  assert.ok(dist(back, X) < 1e-12);
  // Midpoint lies on the plane.
  const mid = [(X[0] + Y[0]) / 2, (X[1] + Y[1]) / 2, (X[2] + Y[2]) / 2];
  const { n } = mirrorFrame(m);
  assert.ok(Math.abs(dot(n, sub(mid, m.center))) < 1e-12);
  // Reflection matrix is symmetric and orthogonal with det = -1.
  const S = reflectionMatrix(m);
  assert.ok(Math.abs(S[0][1] - S[1][0]) < 1e-12);
  const StS = mat3Mul(mat3T(S), S);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(StS[i][j] - (i === j ? 1 : 0)) < 1e-12);
  }
});

test('reflection hit point lies on the finite panel and on the glass plane', () => {
  const cam = scene.cameraTruth;
  const m = scene.mirror;
  const X = [0.2, 1.1, 1.6];
  const v = reflectionValidity(cam, m, X);
  assert.ok(v.ok, v.reasons.join(';'));
  assert.ok(insideMirrorBounds(m, v.H, 1e-6));
  const { n } = mirrorFrame(m);
  assert.ok(Math.abs(dot(n, sub(v.H, m.center))) < 1e-9);
  const pr = projectReflection(cam, m, X);
  assert.ok(pr.H && dist(pr.H, v.H) < 1e-9);
});

// ------------------------- mirror panel observability -------------------------

test('in-plane panel slide is unobservable from reflections (same plane => same image)', () => {
  const a = makeMirror({ center: [0, 1.02, 0], yaw: 6 * DEG, tilt: 1 * DEG, width: 1.5, height: 2 });
  const { u } = mirrorFrame(a);
  const b = { ...a, center: [a.center[0] + u[0] * 0.37, a.center[1] + u[1] * 0.37, a.center[2] + u[2] * 0.37] };
  const vca = virtualCamera(scene.cameraTruth, a);
  const vcb = virtualCamera(scene.cameraTruth, b);
  for (const X of [[0.2, 1.1, 1.4], [-0.5, 0.8, 2.0], [0.9, 1.6, 0.7]]) {
    const pa = projectPoint(vca, X);
    const pb = projectPoint(vcb, X);
    assert.ok(pa && pb);
    assert.ok(Math.hypot(pa.x - pb.x, pa.y - pb.y) < 1e-9, 'plane slide must not move the reflected image');
  }
});

test('off-mirror reflection is rejected and never becomes a valid observation', () => {
  const obs = observe(scene, { noisePx: 0.5, seed: 3 });
  const validIds = new Set(mirrorObservations(obs).map((o) => o.id));
  const outside = obs.rejected.filter((r) => r.kind === 'mirror' && /镜面矩形之外|超出画面/.test(r.reason));
  assert.ok(outside.length > 0, 'expected some reflections to fall outside the panel/frame');
  for (const r of outside) {
    assert.ok(!validIds.has(r.id), `${r.id} must not be used as a valid reflection`);
  }
  // And a point largely off the glass gets a clear rejection reason.
  const far = [4.5, 1.0, 1.0];
  const v = reflectionValidity(scene.cameraTruth, scene.mirror, far);
  assert.equal(v.ok, false);
  assert.ok(v.reasons.length > 0);
});

// ------------------------- DLT camera recovery -------------------------

test('DLT recovers a physically meaningful camera from direct observations', () => {
  const obs = observe(scene, { noisePx: 0, seed: 1 });
  const direct = directObservations(obs);
  assert.ok(direct.length >= 12);
  const { P, singularValues } = dltCamera(direct.map((o) => o.X), direct.map((o) => o.pixel));
  const smallest = Math.min(...singularValues.map(Math.abs));
  assert.ok(smallest < 1e-6, `null space should be small at zero noise, got ${smallest}`);
  const dec = decomposeCameraMatrix(P, { width: scene.width, height: scene.height });
  assert.ok(dist(dec.C, scene.cameraTruth.C) < 1e-3, `centre error ${dist(dec.C, scene.cameraTruth.C)}`);
  assert.ok(Math.abs(dec.f - scene.cameraTruth.f) / scene.cameraTruth.f < 1e-3);
  assert.ok(rotationErrorDeg(dec.R, scene.cameraTruth.R) < 0.1);
});

test('cameraFromLookAt produces an orthonormal right-handed rotation', () => {
  const cam = cameraFromLookAt({ position: [1, 2, 3], target: [0, 1, 0], upHint: [0, 1, 0], f: 900, width: 1280, height: 720 });
  const RtR = mat3Mul(mat3T(cam.R), cam.R);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(RtR[i][j] - (i === j ? 1 : 0)) < 1e-12);
  }
  const det =
    cam.R[0][0] * (cam.R[1][1] * cam.R[2][2] - cam.R[1][2] * cam.R[2][1]) -
    cam.R[0][1] * (cam.R[1][0] * cam.R[2][2] - cam.R[1][2] * cam.R[2][0]) +
    cam.R[0][2] * (cam.R[1][0] * cam.R[2][1] - cam.R[1][1] * cam.R[2][0]);
  assert.ok(Math.abs(det - 1) < 1e-12);
});

// ------------------------- constraint normalisation -------------------------

test('partial mirror constraints never produce NaN geometry', () => {
  const m = normalizeMirror({ center: [0, 1.0, 0] }, { yaw: 0, tilt: 0, width: 1.5, height: 2 });
  const q = reflectionValidity(scene.cameraTruth, m, [0.2, 1.1, 1.4]);
  assert.ok(Number.isFinite(q.camSide));
  for (const v of m.center) assert.ok(Number.isFinite(v));
  assert.ok(Number.isFinite(m.yaw) && Number.isFinite(m.tilt));
});
