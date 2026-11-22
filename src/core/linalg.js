import { cross, dot, mat3Det, mat3Mul, mat3T, normalize } from './math.js';

export function jacobiEigenSymmetric(Ain, maxSweeps = 80) {
  const n = Ain.length;
  const A = Ain.map((r) => r.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: A.map((r, i) => r[i]), vectors: V };
}

export function symmetricEigenSorted(A) {
  const { values, vectors } = jacobiEigenSymmetric(A);
  const order = values.map((v, i) => i).sort((i, j) => values[j] - values[i]);
  return {
    values: order.map((i) => values[i]),
    // vectors is returned as columns: vectors[row][colIndex].
    vectors: order.map((idx) => vectors.map((row) => row[idx])),
  };
}

// SVD of a 3x3 matrix: A = U * diag(S) * V^T, with U and V columns orthogonal.
export function svd3(A) {
  const AtA = mat3Mul(mat3T(A), A);
  const { values, vectors: cols } = symmetricEigenSorted(AtA); // cols[i] = i-th eigenvector
  const S = values.map((v) => Math.sqrt(Math.max(0, v)));
  const V = [0, 1, 2].map((row) => [0, 1, 2].map((i) => cols[i][row]));
  const Ucols = [null, null, null];
  const eps = 1e-12 * (S[0] || 1);
  for (let i = 0; i < 3; i++) {
    if (S[i] > eps) {
      const Av = [
        A[0][0] * V[0][i] + A[0][1] * V[1][i] + A[0][2] * V[2][i],
        A[1][0] * V[0][i] + A[1][1] * V[1][i] + A[1][2] * V[2][i],
        A[2][0] * V[0][i] + A[2][1] * V[1][i] + A[2][2] * V[2][i],
      ];
      Ucols[i] = normalize([Av[0] / S[i], Av[1] / S[i], Av[2] / S[i]]);
    }
  }
  const known = Ucols.map((u, i) => (u ? i : -1)).filter((i) => i >= 0);
  if (known.length >= 1) {
    const u0 = Ucols[known[0]];
    const ref = Math.abs(u0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let ua = normalize(cross(u0, ref));
    let ub = normalize(cross(u0, ua));
    if (known.length >= 2) {
      const u1 = Ucols[known[1]];
      ua = normalize(cross(u0, u1));
      if (Math.hypot(...ua) < 1e-9) ua = ub;
      ub = normalize(cross(u0, ua));
    }
    let idx = 0;
    const missing = [0, 1, 2].filter((i) => !Ucols[i]);
    for (const i of missing) {
      Ucols[i] = idx === 0 ? ua : ub;
      idx++;
    }
  } else {
    Ucols[0] = [1, 0, 0];
    Ucols[1] = [0, 1, 0];
    Ucols[2] = [0, 0, 1];
  }
  const U = [0, 1, 2].map((row) => [Ucols[0][row], Ucols[1][row], Ucols[2][row]]);
  return { U, S, V };
}

// Nearest proper rotation to M (polar decomposition). If det(UV^T) < 0,
// the last singular direction is flipped so the result is a rotation.
export function polarRotation(M) {
  const { U, V } = svd3(M);
  const Ut = mat3T(U);
  const Vit = mat3T(V);
  // R = U * V^T
  let R = mat3Mul(U, Vit);
  if (mat3Det(R) < 0) {
    const U2 = U.map((row) => [row[0], row[1], -row[2]]);
    R = mat3Mul(U2, Vit);
  }
  return R;
}

// Rigid transform src -> dst: dst ≈ R src + t, R proper rotation.
export function rigidFromCorrespondences(src, dst) {
  const n = src.length;
  const cs = [0, 0, 0];
  const cd = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      cs[k] += src[i][k] / n;
      cd[k] += dst[i][k] / n;
    }
  }
  const H = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    const s = [src[i][0] - cs[0], src[i][1] - cs[1], src[i][2] - cs[2]];
    const d = [dst[i][0] - cd[0], dst[i][1] - cd[1], dst[i][2] - cd[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] += d[r] * s[c];
  }
  const { U, V } = svd3(H);
  const Ut = mat3T(U);
  const Vt = mat3T(V);
  let R = mat3Mul(U, Vt);
  if (mat3Det(R) < 0) {
    const U2 = U.map((row) => [row[0], row[1], -row[2]]);
    R = mat3Mul(U2, Vt);
  }
  const t = [
    cd[0] - (R[0][0] * cs[0] + R[0][1] * cs[1] + R[0][2] * cs[2]),
    cd[1] - (R[1][0] * cs[0] + R[1][1] * cs[1] + R[1][2] * cs[2]),
    cd[2] - (R[2][0] * cs[0] + R[2][1] * cs[1] + R[2][2] * cs[2]),
  ];
  return { R, t };
}

// Planarity of a 3D point set via the smallest eigenvalue of its covariance.
export function planarity(points) {
  if (points.length < 3) return { planar: false, ratio: Infinity, normal: null, centroid: null };
  const c = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / points.length;
  const C = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const p of points) {
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += d[i] * d[j];
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] /= points.length;
  const { values, vectors } = symmetricEigenSorted(C);
  const normal = vectors[2];
  const ratio = values[0] > 1e-18 ? values[2] / values[0] : Infinity;
  return { planar: values[2] < 1e-8 * Math.max(1e-30, values[0]), ratio, normal, centroid: c, values };
}

// Real roots of a polynomial given highest-to-lowest coefficients.
// Durand-Kerner for degree <= 4 (used by the P3P solver).
export function realPolynomialRoots(coeffs) {
  let c = coeffs.slice();
  while (c.length > 1 && Math.abs(c[0]) < 1e-14) c.shift();
  const deg = c.length - 1;
  if (deg < 1) return [];
  if (deg === 1) return [-c[1] / c[0]];
  const lead = c[0];
  const norm = c.map((v) => v / lead);
  const radius = 1 + Math.max(...norm.slice(1).map((v) => Math.abs(v)));
  let roots = [];
  for (let k = 0; k < deg; k++) {
    const ang = (2 * Math.PI * k) / deg + 0.4;
    roots.push({ re: radius * Math.cos(ang), im: radius * Math.sin(ang) });
  }
  const evalPoly = (z) => {
    let re = 0;
    let im = 0;
    for (let i = 0; i < norm.length; i++) {
      const nr = re * z.re - im * z.im + norm[i];
      const ni = re * z.im + im * z.re;
      re = nr;
      im = ni;
    }
    return { re, im };
  };
  const cdiv = (a, b) => {
    const d = b.re * b.re + b.im * b.im;
    return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
  };
  for (let iter = 0; iter < 200; iter++) {
    let maxDelta = 0;
    const next = roots.map((z, i) => {
      let denom = { re: 1, im: 0 };
      for (let j = 0; j < deg; j++) {
        if (j === i) continue;
        const diff = { re: z.re - roots[j].re, im: z.im - roots[j].im };
        denom = {
          re: denom.re * diff.re - denom.im * diff.im,
          im: denom.re * diff.im + denom.im * diff.re,
        };
      }
      const val = evalPoly(z);
      const delta = cdiv(val, denom);
      maxDelta = Math.max(maxDelta, Math.hypot(delta.re, delta.im));
      return { re: z.re - delta.re, im: z.im - delta.im };
    });
    roots = next;
    if (maxDelta < 1e-13) break;
  }
  return roots
    .filter((z) => Math.abs(z.im) < 1e-7 * (1 + Math.abs(z.re)))
    .map((z) => z.re)
    .sort((a, b) => a - b);
}

export function polyMul(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  return out;
}

export function polyAddScaled(a, b, s) {
  const n = Math.max(a.length, b.length);
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const av = a[a.length - 1 - i] || 0;
    const bv = b[b.length - 1 - i] || 0;
    out[n - 1 - i] = av + s * bv;
  }
  return out;
}

export function evalPoly(coeffs, x) {
  let v = 0;
  for (const c of coeffs) v = v * x + c;
  return v;
}

export function dotSafe(a, b) {
  return dot(a, b);
}
