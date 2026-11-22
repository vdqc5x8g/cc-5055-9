import {
  mat3Det,
  mat3Identity,
  mat3Mul,
  mat3T,
  orthonormalize,
} from './math.js';
import { jacobiEigenSymmetric } from './linalg.js';

function normalize3D(points) {
  const n = points.length;
  const c = [0, 0, 0];
  for (const p of points) {
    for (let k = 0; k < 3; k++) c[k] += p[k] / n;
  }
  let meanDist = 0;
  for (const p of points) meanDist += Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) / n;
  const s = meanDist > 1e-12 ? Math.sqrt(3) / meanDist : 1;
  const T = [
    [s, 0, 0, -s * c[0]],
    [0, s, 0, -s * c[1]],
    [0, 0, s, -s * c[2]],
    [0, 0, 0, 1],
  ];
  return { T, s, c };
}

function normalize2D(pixels) {
  const n = pixels.length;
  let cu = 0;
  let cv = 0;
  for (const p of pixels) {
    cu += p[0] / n;
    cv += p[1] / n;
  }
  let meanDist = 0;
  for (const p of pixels) meanDist += Math.hypot(p[0] - cu, p[1] - cv) / n;
  const s = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;
  const T = [
    [s, 0, -s * cu],
    [0, s, -s * cv],
    [0, 0, 1],
  ];
  const Tinv = [
    [1 / s, 0, cu],
    [0, 1 / s, cv],
    [0, 0, 1],
  ];
  return { T, Tinv, s, c: [cu, cv] };
}

function applyT(T, X) {
  return [
    T[0][0] * X[0] + T[0][1] * X[1] + T[0][2] * X[2] + T[0][3],
    T[1][0] * X[0] + T[1][1] * X[1] + T[1][2] * X[2] + T[1][3],
    T[2][0] * X[0] + T[2][1] * X[1] + T[2][2] * X[2] + T[2][3],
  ];
}

// Normalized direct linear transform: world points + pixels -> 3x4 camera matrix.
export function dltCamera(worldPoints, pixels) {
  const n = worldPoints.length;
  if (n < 6) throw new Error('DLT requires at least 6 correspondences');
  const n3 = normalize3D(worldPoints);
  const n2 = normalize2D(pixels);
  const rows = [];
  const W = worldPoints.map((X) => applyT(n3.T, X));
  const Pm = pixels.map((p) => [(p[0] - n2.c[0]) * n2.s, (p[1] - n2.c[1]) * n2.s]);
  for (let i = 0; i < n; i++) {
    const [X, Y, Z] = W[i];
    const [u, v] = Pm[i];
    rows.push([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]);
    rows.push([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]);
  }
  const AtA = Array.from({ length: 12 }, () => new Array(12).fill(0));
  for (const row of rows) {
    for (let i = 0; i < 12; i++) {
      for (let j = i; j < 12; j++) AtA[i][j] += row[i] * row[j];
    }
  }
  for (let i = 0; i < 12; i++) for (let j = 0; j < i; j++) AtA[i][j] = AtA[j][i];
  const { values, vectors } = jacobiEigenSymmetric(AtA);
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  const pn = vectors.map((row) => row[best]);
  const Pm3 = [
    [pn[0], pn[1], pn[2], pn[3]],
    [pn[4], pn[5], pn[6], pn[7]],
    [pn[8], pn[9], pn[10], pn[11]],
  ];
  // P = Tinv2 * Pn * T3
  const tmp = [0, 1, 2].map((i) =>
    [0, 1, 2, 3].map((j) => {
      let s = 0;
      for (let k = 0; k < 4; k++) s += Pm3[i][k] * n3.T[k][j];
      return s;
    }),
  );
  const P = [0, 1, 2].map((i) => {
    const A = n2.Tinv[i];
    return [0, 1, 2, 3].map((j) => A[0] * tmp[0][j] + A[1] * tmp[1][j] + A[2] * tmp[2][j]);
  });
  return { P, singularValues: values, nullVector: pn };
}

function givensRight(M, p, q, c, s) {
  // Returns M * G with G[p][p]=c, G[q][q]=c, G[p][q]=s, G[q][p]=-s.
  const n = M.length;
  const out = M.map((row) => row.slice());
  for (let r = 0; r < n; r++) {
    const x = M[r][p];
    const y = M[r][q];
    out[r][p] = c * x - s * y;
    out[r][q] = s * x + c * y;
  }
  return out;
}

// RQ decomposition M = K * R, K upper triangular with positive diagonal,
// R orthogonal. Uses right Givens rotations.
function rq3(M) {
  let K = M.map((row) => row.slice());
  let G = mat3Identity();
  const zeroLower = (row, col) => {
    const r = Math.hypot(K[row][col], K[row][col + 1]);
    if (r < 1e-18) return;
    const c = K[row][col + 1] / r;
    const s = K[row][col] / r;
    K = givensRight(K, col, col + 1, c, s);
    G = givensRight(G, col, col + 1, c, s);
  };
  zeroLower(2, 0);
  zeroLower(2, 1);
  zeroLower(1, 0);
  // Negative diagonal handling: K D * D^T R with D = diag(signs).
  const D = [
    [K[0][0] < 0 ? -1 : 1, 0, 0],
    [0, K[1][1] < 0 ? -1 : 1, 0],
    [0, 0, K[2][2] < 0 ? -1 : 1],
  ];
  K = mat3Mul(K, D);
  const R = mat3Mul(D, mat3T(G));
  return { K, R };
}

// Decompose P [3x4] into K, R and C (camera centre).
// P is homogeneous; its sign is arbitrary. `desiredDet` selects which sign of P
// to accept, allowing recovery of an improper (mirrored) virtual camera when
// desiredDet = -1.
export function decomposeCameraMatrix(P, { width = 1920, height = 1080, desiredDet = 1 } = {}) {
  let best = null;
  for (const sgn of [1, -1]) {
    const p4 = [sgn * P[0][3], sgn * P[1][3], sgn * P[2][3]];
    const M = [0, 1, 2].map((i) => [sgn * P[i][0], sgn * P[i][1], sgn * P[i][2]]);
    const { K, R } = rq3(M);
    const det = mat3Det(R);
    if (!best) best = { K, R, M, p4 };
    if (Math.sign(det) === Math.sign(desiredDet)) {
      best = { K, R, M, p4 };
      break;
    }
  }
  const { K: Kraw, R, M, p4 } = best;
  // P is only defined up to scale; force K[2][2] = 1 for a metric decomposition.
  const kz = Math.abs(Kraw[2][2]) > 1e-15 ? Kraw[2][2] : 1;
  const K = Kraw.map((row) => row.map((v) => v / kz));
  const p4n = [p4[0] / kz, p4[1] / kz, p4[2] / kz];
  const Mn = M.map((row) => row.map((v) => v / kz));
  const Minv = invert3(Mn);
  if (!Minv) throw new Error('Singular camera matrix');
  const C = [
    -(Minv[0][0] * p4n[0] + Minv[0][1] * p4n[1] + Minv[0][2] * p4n[2]),
    -(Minv[1][0] * p4n[0] + Minv[1][1] * p4n[1] + Minv[1][2] * p4n[2]),
    -(Minv[2][0] * p4n[0] + Minv[2][1] * p4n[1] + Minv[2][2] * p4n[2]),
  ];
  const f = (K[0][0] + K[1][1]) / 2;
  return {
    R: orthonormalize(R),
    C,
    f,
    fEstimate: f,
    cx: Number.isFinite(K[0][2]) && Math.abs(K[0][2]) < 1e5 ? K[0][2] : width / 2,
    cy: Number.isFinite(K[1][2]) && Math.abs(K[1][2]) < 1e5 ? K[1][2] : height / 2,
    K,
    width,
    height,
  };
}

function invert3(A) {
  const d = mat3Det(A);
  if (Math.abs(d) < 1e-18) return null;
  const id = 1 / d;
  return [
    [
      (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id,
      (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id,
      (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id,
    ],
    [
      (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id,
      (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id,
      (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id,
    ],
    [
      (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id,
      (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id,
      (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id,
    ],
  ];
}

// Triangulate a 3D point from two camera matrices (diagnostics only).
export function triangulate(P1, P2, x1, x2) {
  const A = [
    [x1[0] * P1[2][0] - P1[0][0], x1[0] * P1[2][1] - P1[0][1], x1[0] * P1[2][2] - P1[0][2], x1[0] * P1[2][3] - P1[0][3]],
    [x1[1] * P1[2][0] - P1[1][0], x1[1] * P1[2][1] - P1[1][1], x1[1] * P1[2][2] - P1[1][2], x1[1] * P1[2][3] - P1[1][3]],
    [x2[0] * P2[2][0] - P2[0][0], x2[0] * P2[2][1] - P2[0][1], x2[0] * P2[2][2] - P2[0][2], x2[0] * P2[2][3] - P2[0][3]],
    [x2[1] * P2[2][0] - P2[1][0], x2[1] * P2[2][1] - P2[1][1], x2[1] * P2[2][2] - P2[1][2], x2[1] * P2[2][3] - P2[1][3]],
  ];
  const AtA = Array.from({ length: 4 }, () => new Array(4).fill(0));
  for (const row of A) for (let i = 0; i < 4; i++) for (let j = i; j < 4; j++) AtA[i][j] += row[i] * row[j];
  for (let i = 0; i < 4; i++) for (let j = 0; j < i; j++) AtA[i][j] = AtA[j][i];
  const { values, vectors } = jacobiEigenSymmetric(AtA);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] < values[best]) best = i;
  const v = vectors.map((row) => row[best]);
  if (Math.abs(v[3]) < 1e-12) return null;
  return [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
}

export function reprojectionErrorPx(P, X, pixel) {
  const Xh = [X[0], X[1], X[2], 1];
  const val = [0, 1, 2].map(
    (i) => P[i][0] * Xh[0] + P[i][1] * Xh[1] + P[i][2] * Xh[2] + P[i][3],
  );
  if (Math.abs(val[2]) < 1e-12) return Infinity;
  return Math.hypot(val[0] / val[2] - pixel[0], val[1] / val[2] - pixel[1]);
}
