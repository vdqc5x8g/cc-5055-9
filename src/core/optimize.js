import {
  add,
  addScaled,
  cross,
  cross as x3,
  dist,
  dot,
  mat3Mul,
  mat3Vec,
  normalize,
  scale,
  so3Exp,
  sub,
} from './math.js';
import { cameraMatrix, projectHomogeneous } from './camera.js';

// Levenberg-Marquardt for a residual function r(p) -> array of numbers.
// J is computed by central differences; small systems make this adequate and
// keeps the solver dependency-free.
export function levenbergMarquardt({
  params,
  residuals,
  maxIterations = 80,
  lambda0 = 1e-3,
  tolerance = 1e-10,
  stepEps = 1e-7,
  onIteration = null,
}) {
  let p = params.slice();
  let r = residuals(p);
  let cost = dot(r, r);
  let lambda = lambda0;
  for (let iter = 0; iter < maxIterations; iter++) {
    const n = p.length;
    const m = r.length;
    const J = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      const h = stepEps * Math.max(1, Math.abs(p[j]));
      const trial = p.slice();
      trial[j] += h;
      const rp = residuals(trial);
      trial[j] -= 2 * h;
      const rm = residuals(trial);
      for (let i = 0; i < m; i++) J[i][j] = (rp[i] - rm[i]) / (2 * h);
    }
    // Normal equations: (J^T J + lambda diag) dp = -J^T r
    const JtJ = Array.from({ length: n }, () => new Array(n).fill(0));
    const Jtr = new Array(n).fill(0);
    for (let i = 0; i < m; i++) {
      const ri = r[i];
      for (let a = 0; a < n; a++) {
        Jtr[a] += J[i][a] * ri;
        for (let b = a; b < n; b++) JtJ[a][b] += J[i][a] * J[i][b];
      }
    }
    for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) JtJ[a][b] = JtJ[b][a];
    let improved = false;
    for (let attempt = 0; attempt < 10; attempt++) {
      const A = JtJ.map((row, i) =>
        row.map((v, j) => (i === j ? v + lambda * Math.max(1e-12, Math.abs(v)) : v)),
      );
      const dp = solveSPD(A, Jtr.map((v) => -v));
      if (!dp) {
        lambda *= 10;
        continue;
      }
      const candidate = p.map((v, i) => v + dp[i]);
      const rc = residuals(candidate);
      const costC = dot(rc, rc);
      if (costC < cost) {
        p = candidate;
        r = rc;
        const improvement = cost - costC;
        cost = costC;
        lambda = Math.max(1e-12, lambda * 0.3);
        improved = true;
        if (onIteration) onIteration(iter, cost, lambda);
        if (improvement < tolerance * (1 + cost)) {
          return { params: p, residuals: r, cost, iterations: iter + 1, converged: true };
        }
        break;
      }
      lambda *= 10;
    }
    if (!improved) {
      return { params: p, residuals: r, cost, iterations: iter + 1, converged: false };
    }
  }
  return { params: p, residuals: r, cost, iterations: maxIterations, converged: true };
}

function solveSPD(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let k = 0; k < n; k++) {
    let piv = k;
    let best = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > best) {
        best = Math.abs(M[i][k]);
        piv = i;
      }
    }
    if (best < 1e-16) return null;
    if (piv !== k) {
      const t = M[k];
      M[k] = M[piv];
      M[piv] = t;
    }
    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      if (f === 0) continue;
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// Parameter vector for a camera in the mirror tool:
//   C (3), rotation increment w (3), log f (1), and optionally cx, cy.
// Rotation is R = so3Exp(w) * R0, so the DLT estimate supplies R0.
export function cameraParamCount(mode) {
  return mode === 'f' ? 7 : mode === 'fc' ? 9 : 6;
}

export function unpackCamera(p, R0, width, height, mode = 'f') {
  const C = [p[0], p[1], p[2]];
  const w = [p[3], p[4], p[5]];
  const R = mat3Mul(so3Exp(w), R0);
  const f = mode === 'f' || mode === 'fc' ? Math.exp(p[6]) : 1150;
  const cx = mode === 'fc' ? p[7] : width / 2;
  const cy = mode === 'fc' ? p[8] : height / 2;
  return { R, C, f, fx: f, fy: f, cx, cy, width, height };
}

export function cameraResiduals(p, R0, width, height, mode, observations) {
  const cam = unpackCamera(p, R0, width, height, mode);
  const P = cameraMatrix(cam);
  const res = [];
  for (const o of observations) {
    const pr = projectHomogeneous(P, [o.X[0], o.X[1], o.X[2], 1]);
    if (Math.abs(pr[2]) < 1e-12) {
      res.push(1e4, 1e4);
      continue;
    }
    res.push(pr[0] / pr[2] - o.pixel[0], pr[1] / pr[2] - o.pixel[1]);
  }
  return res;
}

export function initialRotationFromDLT(camEstimate) {
  return camEstimate.R.map((r) => r.slice());
}
