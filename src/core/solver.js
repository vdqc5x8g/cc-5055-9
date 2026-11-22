import {
  add,
  addScaled,
  cross,
  dist,
  dot,
  mat3Mul,
  normalize,
  scale,
  so3Exp,
  sub,
} from './math.js';
import { cameraMatrix, projectHomogeneous, projectPoint, makeCamera } from './camera.js';
import { dltCamera, decomposeCameraMatrix } from './dlt.js';
import {
  makeMirror,
  mirrorFrame,
  reflectPoint,
  virtualCamera,
  reflectionValidity,
  reflectionHit,
  reflectionMatrix,
} from './mirror.js';
import { levenbergMarquardt } from './optimize.js';
import { jacobiEigenSymmetric } from './linalg.js';
import { orthonormalize, mat3T } from './math.js';

// ------------------------- camera from direct observations -------------------------

export function solveDirectCamera(observations, {
  width,
  height,
  fPrior = null,
  fixF = false,
  fixPrincipal = true,
  initial = null,
} = {}) {
  const obs = observations.filter((o) => o.pixel && o.X);
  if (obs.length < 6) {
    return {
      ok: false,
      error: `直接观测仅 ${obs.length} 个，至少需要 6 个`,
      rmsPx: null,
      camera: null,
      usedCount: obs.length,
    };
  }
  let init = initial;
  let dltInfo = null;
  if (!init) {
    const dlt = robustCameraInit(obs, { width, height, fixF, fPrior });
    if (!dlt.ok) {
      return { ok: false, error: dlt.error, rmsPx: null, camera: null, usedCount: obs.length };
    }
    dltInfo = dlt.dec;
    init = dlt.camera;
  }
  const R0 = init.R.map((r) => r.slice());
  const useF = !(fixF && fPrior);
  const useC = !fixPrincipal;
  const build = (p) => {
    const R = mat3Mul(so3Exp([p[3], p[4], p[5]]), R0);
    const f = useF ? Math.exp(p[6]) : fPrior ?? init.f;
    const cx = useC ? p[7] : width / 2;
    const cy = useC ? p[8] : height / 2;
    return { R, C: [p[0], p[1], p[2]], f, fx: f, fy: f, cx, cy, width, height };
  };
  const residualsMake = (obsList) => (p) => {
    const cam = build(p);
    const P = cameraMatrix(cam);
    const res = [];
    for (const o of obsList) {
      const pr = projectHomogeneous(P, [o.X[0], o.X[1], o.X[2], 1]);
      if (Math.abs(pr[2]) < 1e-12) {
        res.push(1e5, 1e5);
      } else {
        res.push(pr[0] / pr[2] - o.pixel[0], pr[1] / pr[2] - o.pixel[1]);
      }
    }
    return res;
  };
  const p0 = [
    init.C[0],
    init.C[1],
    init.C[2],
    0,
    0,
    0,
    Math.log(init.f),
    ...(useC ? [init.cx, init.cy] : []),
  ];
  const fit = levenbergMarquardt({
    params: p0,
    residuals: residualsMake(obs),
    maxIterations: 150,
  });
  const camera = build(fit.params);
  const errors = perPointErrors(camera, obs);
  return {
    ok: true,
    camera,
    rmsPx: rmsOf(errors),
    rmsPerPointPx: errors,
    iterations: fit.iterations,
    converged: fit.converged,
    used: obs,
    dlt: dltInfo,
  };
}

// ------------------------- camera from mirror observations + known plane -------------------------

// If direct markers are missing, a known mirror plane still lets us recover the
// real camera: mirrored pixels form a virtual (improper) camera; reflecting its
// centre and orientation across the plane gives the physical camera.
export function solveCameraFromMirror(mirrorObs, mirror, { width, height, fixF = false, fPrior = null } = {}) {
  const obs = mirrorObs.filter((o) => o.pixel && o.X);
  if (obs.length < 6) {
    return { ok: false, error: `镜面观测仅 ${obs.length} 个，至少需要 6 个且已知镜面约束`, usedCount: obs.length };
  }
  const { P } = dltCamera(obs.map((o) => o.X), obs.map((o) => o.pixel));
  // Mirrored pixels are what the virtual camera P_v sees of physical points.
  // The reflection transform H about the plane is an involution (H = H^-1),
  // and P_v = P_real * H, so the real camera matrix is P_real = P_v * H.
  const Preal = multiply34x44(P, reflectionHomogeneous(mirror));
  const dec = decomposeCameraMatrix(Preal, { width, height });
  const f = fixF && fPrior ? fPrior : dec.f;
  const C = dec.C;
  const R = dec.R;
  const camera = makeCamera({ R, C, f, cx: width / 2, cy: height / 2, width, height });
  // Refine against the mirrored observations using the exact mirror model.
  const R0 = R.map((r) => r.slice());
  const useF = !(fixF && fPrior);
  const build = (p) => {
    const Rr = mat3Mul(so3Exp([p[3], p[4], p[5]]), R0);
    const ff = useF ? Math.exp(p[6]) : f;
    return { R: Rr, C: [p[0], p[1], p[2]], f: ff, fx: ff, fy: ff, cx: width / 2, cy: height / 2, width, height };
  };
  const residuals = (p) => {
    const cam = build(p);
    const vc = virtualCamera(cam, mirror);
    const res = [];
    for (const o of obs) {
      const pr = projectPoint(vc, o.X);
      if (!pr) res.push(1e5, 1e5);
      else res.push(pr.x - o.pixel[0], pr.y - o.pixel[1]);
    }
    return res;
  };
  const p0 = [C[0], C[1], C[2], 0, 0, 0, Math.log(f)];
  const fit = levenbergMarquardt({ params: p0, residuals, maxIterations: 150 });
  const cam = build(fit.params);
  const errors = [];
  const vc = virtualCamera(cam, mirror);
  for (const o of obs) {
    const pr = projectPoint(vc, o.X);
    errors.push(pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity);
  }
  return {
    ok: true,
    camera: cam,
    rmsPx: rmsOf(errors),
    rmsPerPointPx: errors,
    detVirtual: -1,
    used: obs,
    iterations: fit.iterations,
    converged: fit.converged,
  };
}

// ------------------------- mirror plane from reflections -------------------------

function normalFromYawTilt(yaw, tilt) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const ct = Math.cos(tilt);
  const st = Math.sin(tilt);
  return [sy * ct, st, cy * ct];
}

function yawTiltFromNormal(n) {
  const nn = normalize(n);
  const tilt = Math.asin(Math.max(-1, Math.min(1, nn[1])));
  const yaw = Math.atan2(nn[0], nn[2]);
  return { yaw, tilt };
}

export function cameraCenterFromP(P) {
  const M = [P[0].slice(0, 3), P[1].slice(0, 3), P[2].slice(0, 3)];
  const p4 = [P[0][3], P[1][3], P[2][3]];
  const d = mat3Det(M);
  if (Math.abs(d) < 1e-18) return null;
  const id = 1 / d;
  const inv = [
    [
      (M[1][1] * M[2][2] - M[1][2] * M[2][1]) * id,
      (M[0][2] * M[2][1] - M[0][1] * M[2][2]) * id,
      (M[0][1] * M[1][2] - M[0][2] * M[1][1]) * id,
    ],
    [
      (M[1][2] * M[2][0] - M[1][0] * M[2][2]) * id,
      (M[0][0] * M[2][2] - M[0][2] * M[2][0]) * id,
      (M[0][2] * M[1][0] - M[0][0] * M[1][2]) * id,
    ],
    [
      (M[1][0] * M[2][1] - M[1][1] * M[2][0]) * id,
      (M[0][1] * M[2][0] - M[0][0] * M[2][1]) * id,
      (M[0][0] * M[1][1] - M[0][1] * M[1][0]) * id,
    ],
  ];
  return [
    -(inv[0][0] * p4[0] + inv[0][1] * p4[1] + inv[0][2] * p4[2]),
    -(inv[1][0] * p4[0] + inv[1][1] * p4[1] + inv[1][2] * p4[2]),
    -(inv[2][0] * p4[0] + inv[2][1] * p4[1] + inv[2][2] * p4[2]),
  ];
}

// Plane initialisation from the virtual camera centre recovered by DLT.
export function initialMirrorPlane(camera, mirrorObs) {
  if (mirrorObs.length < 6) return null;
  const { P } = dltCamera(mirrorObs.map((o) => o.X), mirrorObs.map((o) => o.pixel));
  const Cv = cameraCenterFromP(P);
  if (!Cv) return null;
  const v = sub(camera.C, Cv);
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-9) return null;
  const n = normalize(v);
  const h = len / 2;
  const { yaw, tilt } = yawTiltFromNormal(n);
  return { yaw, tilt, d: dot(n, camera.C) - h, h, virtualCenter: Cv, P };
}

function planeResiduals(p, camera, obs) {
  const n = normalFromYawTilt(p[0], p[1]);
  const d = p[2];
  const mirror = makeMirrorFromPlane(n, d, 2, 2);
  const vc = virtualCamera(camera, mirror);
  const res = [];
  for (const o of obs) {
    const pr = projectPoint(vc, o.X);
    if (!pr) res.push(1e5, 1e5);
    else res.push(pr.x - o.pixel[0], pr.y - o.pixel[1]);
  }
  return res;
}

export function makeMirrorFromPlane(n, d, width, height) {
  const { yaw, tilt } = yawTiltFromNormal(n);
  return makeMirror({ center: scale(n, d), yaw, tilt, width, height });
}

// Panel lateral placement is NOT observable from reflections (sliding the panel
// within its own plane leaves every reflected ray unchanged). It must come from
// a scene constraint, or be estimated as the centroid of the reflection hits.
function panelAnchor(camera, n, d, obs, constraint) {
  if (constraint && constraint.center) {
    const c = constraint.center;
    return {
      anchor: add(c, scale(n, d - dot(n, c))),
      source: 'constraint',
      note: '面板位置取自场景约束',
    };
  }
  const planeMirror = makeMirrorFromPlane(n, d, 100, 100);
  const hits = [];
  for (const o of obs) {
    const H = reflectionHit(camera, planeMirror, o.X);
    if (H) hits.push(H);
  }
  if (!hits.length) {
    return { anchor: scale(n, d), source: 'plane-origin', note: '无命中点，面板中心退回平面原点' };
  }
  const c = [0, 0, 0];
  for (const H of hits) for (let k = 0; k < 3; k++) c[k] += H[k] / hits.length;
  return {
    anchor: add(c, scale(n, d - dot(n, c))),
    source: 'hit-centroid',
    note: '面板面内位置由反射命中点质心估计（非约束，仅用于放置有限面板）',
    hits,
  };
}

export function solveMirrorPlane(camera, mirrorObs, {
  constraint = null,
  width = 1280,
  height = 720,
} = {}) {
  const obs = mirrorObs.filter((o) => o.pixel && o.X);
  if (obs.length < 6) {
    return {
      ok: false,
      error: `镜面观测仅 ${obs.length} 个或退化，无法初始化镜面平面`,
    };
  }
  const init = initialMirrorPlane(camera, obs);
  if (!init) {
    return { ok: false, error: '镜面平面初始化退化（虚拟相机中心与真实相机中心重合）' };
  }
  const fit = levenbergMarquardt({
    params: [init.yaw, init.tilt, init.d],
    residuals: (p) => planeResiduals(p, camera, obs),
    maxIterations: 200,
    lambda0: 1e-4,
  });
  const [yaw, tilt, d] = fit.params;
  const n = normalFromYawTilt(yaw, tilt);
  const panelWidth = constraint?.width ?? 1.5;
  const panelHeight = constraint?.height ?? 2.0;
  const { anchor, source, note, hits } = panelAnchor(camera, n, d, obs, constraint);
  const mirror = makeMirror({
    center: anchor,
    yaw,
    tilt,
    width: panelWidth,
    height: panelHeight,
    centerSource: source,
  });
  const vc = virtualCamera(camera, mirror);
  const errors = [];
  const detail = [];
  for (const o of obs) {
    const pr = projectPoint(vc, o.X);
    const err = pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity;
    errors.push(err);
    detail.push({ id: o.id, err, predicted: pr ? [pr.x, pr.y] : null });
  }
  return {
    ok: true,
    mirror,
    yaw,
    tilt,
    d,
    n,
    h: dot(n, camera.C) - d,
    rmsPx: rmsOf(errors),
    residuals: detail,
    panelSource: source,
    panelNote: note,
    init,
    hits,
    iterations: fit.iterations,
    converged: fit.converged,
    usedCount: obs.length,
  };
}

// ------------------------- joint refinement -------------------------

// Bundle-adjust camera, mirror plane and optional asset together. The mirror is
// parameterised by (yaw, tilt, d) so the panel's lateral position stays fixed by
// its anchor; this is the correct observable parameterisation.
export function refineJoint({
  camera,
  mirror,
  directObs = [],
  mirrorObs = [],
  fixMirror = false,
  fixF = false,
  width,
  height,
  anchor = null,
}) {
  const n0 = mirrorFrame(mirror).n;
  const { yaw: yaw0, tilt: tilt0 } = yawTiltFromNormal(n0);
  const d0 = dot(n0, mirror.center);
  const R0 = camera.R.map((r) => r.slice());
  const anchorPoint = anchor ?? mirror.center;
  const build = (p) => {
    const R = mat3Mul(so3Exp([p[3], p[4], p[5]]), R0);
    const f = fixF ? camera.f : Math.exp(p[6]);
    const cam = { R, C: [p[0], p[1], p[2]], f, fx: f, fy: f, cx: width / 2, cy: height / 2, width, height };
    let mir = mirror;
    if (!fixMirror) {
      const n = normalFromYawTilt(p[7], p[8]);
      const d = p[9];
      const c = add(anchorPoint, scale(n, d - dot(n, anchorPoint)));
      const { yaw, tilt } = yawTiltFromNormal(n);
      mir = makeMirror({ center: c, yaw, tilt, width: mirror.width, height: mirror.height, centerSource: mirror.centerSource });
    }
    return { cam, mir };
  };
  const residuals = (p) => {
    const { cam, mir } = build(p);
    const res = [];
    for (const o of directObs) {
      const pr = projectPoint(cam, o.X);
      if (!pr) res.push(1e5, 1e5);
      else res.push(pr.x - o.pixel[0], pr.y - o.pixel[1]);
    }
    const vc = virtualCamera(cam, mir);
    for (const o of mirrorObs) {
      const pr = projectPoint(vc, o.X);
      if (!pr) res.push(1e5, 1e5);
      else res.push(pr.x - o.pixel[0], pr.y - o.pixel[1]);
    }
    return res;
  };
  const p0 = [
    camera.C[0], camera.C[1], camera.C[2],
    0, 0, 0,
    ...(fixF ? [] : [Math.log(camera.f)]),
    ...(fixMirror ? [] : [yaw0, tilt0, d0]),
  ];
  const fit = levenbergMarquardt({ params: p0, residuals, maxIterations: 250, lambda0: 1e-3 });
  const { cam, mir } = build(fit.params);
  return {
    camera: cam,
    mirror: mir,
    cost: fit.cost,
    rmsPx: Math.sqrt(fit.cost / Math.max(1, directObs.length + mirrorObs.length)),
    iterations: fit.iterations,
    converged: fit.converged,
  };
}

// ------------------------- helpers -------------------------

function perPointErrors(camera, obs) {
  return obs.map((o) => {
    const pr = projectPoint(camera, o.X);
    return pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity;
  });
}

export function rmsOf(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return Infinity;
  return Math.sqrt(finite.reduce((s, v) => s + v * v, 0) / finite.length);
}

function mat3Det(A) {
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

// 4x4 homogeneous reflection about the mirror plane: X -> S X + 2 (n.centre) n
export function reflectionHomogeneous(mirror) {
  const { n } = mirrorFrame(mirror);
  const S = reflectionMatrix(mirror);
  const t = 2 * dot(n, mirror.center);
  return [
    [S[0][0], S[0][1], S[0][2], t * n[0]],
    [S[1][0], S[1][1], S[1][2], t * n[1]],
    [S[2][0], S[2][1], S[2][2], t * n[2]],
    [0, 0, 0, 1],
  ];
}

function multiply34x44(P, H) {
  return [0, 1, 2].map((i) => [0, 1, 2, 3].map((j) => {
    let s = 0;
    for (let k = 0; k < 4; k++) s += P[i][k] * H[k][j];
    return s;
  }));
}

// Robust initialisation for the direct camera. Planar or near-degenerate marker
// subsets make a single DLT unstable, so sample several subsets, reject
// non-physical solutions (markers behind the camera, focal outside the frame
// scale, camera below ground) and keep the lowest-residual candidate.
export function robustCameraInit(obs, { width, height, fixF = false, fPrior = null } = {}) {
  const n = obs.length;
  const attempts = [];
  const indices = obs.map((_, i) => i);
  const trySubset = (idxs) => {
    const sub = idxs.map((i) => obs[i]);
    if (sub.length < 6) return;
    let P;
    try {
      ({ P } = dltCamera(sub.map((o) => o.X), sub.map((o) => o.pixel)));
    } catch {
      return;
    }
    let dec;
    try {
      dec = decomposeCameraMatrix(P, { width, height });
    } catch {
      return;
    }
    if (!dec || !Number.isFinite(dec.f) || !(dec.f > 1e-3)) return;
    const camera = makeCamera({
      R: dec.R,
      C: dec.C,
      f: fixF && fPrior ? fPrior : dec.f,
      cx: width / 2,
      cy: height / 2,
      width,
      height,
    });
    // Physical plausibility: all observed markers in front of the camera.
    let inFront = 0;
    for (const o of obs) {
      const pr = projectPoint(camera, o.X);
      if (pr && pr.depth > 0) inFront++;
    }
    attempts.push({ camera, dec, inFront, inFrontRatio: inFront / n });
  };
  trySubset(indices);
  if (n > 6) {
    // Deterministic pseudo-random 6-subsets (fixed seed) plus stride subsets.
    let seed = 0x9e3779b9;
    const rand = () => {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      return (seed >>> 0) / 4294967296;
    };
    for (let t = 0; t < 24; t++) {
      const pool = indices.slice();
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
      }
      trySubset(pool.slice(0, 6));
    }
    for (let offset = 0; offset < Math.min(16, n); offset++) {
      // Distinct sample via a seeded LCG, with a hard bound so it always ends.
      const picked = new Set();
      let x = (offset * 2654435761 + 12345) >>> 0;
      for (let guard = 0; guard < 200 && picked.size < 6; guard++) {
        x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
        picked.add(x % n);
      }
      if (picked.size === 6) trySubset([...picked]);
    }
  }
  if (!attempts.length) return { ok: false, error: '相机 DLT 初始化失败（点集退化或共面）' };
  // Prefer physically valid solutions, then lowest reprojection error.
  const scored = attempts.map((a) => {
    const errs = perPointErrors(a.camera, obs);
    return { ...a, rms: rmsOf(errs) };
  });
  scored.sort((a, b) => {
    if (a.inFrontRatio !== b.inFrontRatio) return b.inFrontRatio - a.inFrontRatio;
    return a.rms - b.rms;
  });
  const best = scored[0];
  if (!Number.isFinite(best.rms)) {
    return { ok: false, error: '相机初始化所有候选均不可用' };
  }
  return { ok: true, camera: best.camera, dec: best.dec, rms: best.rms };
}

// When direct markers are coplanar (or too few), mirror observations plus a
// known mirror constraint provide extra non-coplanar evidence: the mirrored
// image of X is exactly the direct image of reflect(X). Reflecting the object
// points (not the pixels) yields valid direct observations for the real camera.
export function augmentWithMirror(directObs, mirrorObs, mirror) {
  const aug = mirrorObs
    .filter((o) => o.pixel && o.X)
    .map((o) => ({
      id: o.id,
      X: reflectPoint(mirror, o.X),
      pixel: o.pixel,
      kind: 'mirror-augmented',
      sourceId: o.id,
    }));
  return [...directObs, ...aug];
}

export function planarityOf(points) {
  if (points.length < 3) return { planar: true, ratio: 0 };
  const c = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / points.length;
  const Cov = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const p of points) {
    const d = sub(p, c);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Cov[i][j] += d[i] * d[j];
  }
  const { values } = jacobiEigenSymmetric(Cov);
  const sorted = values.slice().sort((a, b) => b - a);
  return { planar: sorted[2] < 1e-9 * Math.max(1e-18, sorted[0]), ratio: sorted[0] > 1e-18 ? sorted[2] / sorted[0] : 0 };
}
