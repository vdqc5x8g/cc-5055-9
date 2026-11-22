import { createDefaultScene, observe, directObservations, mirrorObservations } from '../src/core/scene.js';
import { virtualCamera } from '../src/core/mirror.js';
import { projectPoint } from '../src/core/camera.js';
import { dltCamera, decomposeCameraMatrix } from '../src/core/dlt.js';
import { rotationErrorDeg } from '../src/core/math.js';

const scene = createDefaultScene();
const obs = observe(scene, { noisePx: 0 });
console.log('markers', scene.markers.length);
console.log('direct ok', directObservations(obs).length, 'mirror ok', mirrorObservations(obs).length, 'rejected', obs.rejected.length);

// Check virtual camera projection equals reflected-point projection.
const vc = virtualCamera(scene.cameraTruth, scene.mirror);
let maxDiff = 0;
for (const m of scene.markers) {
  const a = projectPoint(vc, m.X);
  if (!a) continue;
  const { reflectPoint } = await import('../src/core/mirror.js');
  const b = projectPoint(scene.cameraTruth, reflectPoint(scene.mirror, m.X));
  maxDiff = Math.max(maxDiff, Math.hypot(a.x - b.x, a.y - b.y));
}
console.log('virtual-camera vs reflected projection max diff px', maxDiff.toExponential(2));

// DLT on direct observations only.
const direct = directObservations(obs);
const { P } = dltCamera(direct.map((o) => o.X), direct.map((o) => o.pixel));
let errs = [];
for (const o of direct) {
  const p = [0, 1, 2].map((i) => P[i][0] * o.X[0] + P[i][1] * o.X[1] + P[i][2] * o.X[2] + P[i][3]);
  errs.push(Math.hypot(p[0] / p[2] - o.pixel[0], p[1] / p[2] - o.pixel[1]));
}
console.log('DLT direct rms px', Math.sqrt(errs.reduce((s, e) => s + e * e, 0) / errs.length).toExponential(2));
const dec = decomposeCameraMatrix(P, { width: 1280, height: 720 });
console.log('recovered C', dec.C.map((v) => v.toFixed(4)), 'true', scene.cameraTruth.C.map((v) => v.toFixed(4)));
console.log('recovered f', dec.f.toFixed(2), 'true', scene.cameraTruth.f);
console.log('R err deg', rotationErrorDeg(dec.R, scene.cameraTruth.R).toFixed(4));
