// Small, dependency-free 3D math helpers used by both the browser tool and the CLI.
// Vectors are [x, y, z]; matrices are 3x3 nested arrays (row-major).

export const EPS = 1e-12;
export const DEG = Math.PI / 180;
export const RAD = 180 / Math.PI;

export function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

export function v3(x = 0, y = 0, z = 0) {
  return [x, y, z];
}

export function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function addScaled(a, b, s) {
  return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
}

export function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function length(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize(a) {
  const l = length(a);
  if (l < 1e-15) return [0, 0, 0];
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function dist(a, b) {
  return length(sub(a, b));
}

export function lerp(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function mean(points) {
  if (!points.length) return [0, 0, 0];
  const s = points.reduce((acc, p) => add(acc, p), [0, 0, 0]);
  return scale(s, 1 / points.length);
}

export function mat3Identity() {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

export function mat3Diag(a, b, c) {
  return [
    [a, 0, 0],
    [0, b, 0],
    [0, 0, c],
  ];
}

export function mat3Mul(a, b) {
  const c = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return c;
}

export function mat3MulMany(...mats) {
  return mats.reduce((acc, m) => mat3Mul(acc, m), mat3Identity());
}

export function mat3Vec(a, x) {
  return [
    a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
    a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
    a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
  ];
}

export function mat3T(a) {
  return [
    [a[0][0], a[1][0], a[2][0]],
    [a[0][1], a[1][1], a[2][1]],
    [a[0][2], a[1][2], a[2][2]],
  ];
}

export function mat3Det(a) {
  return (
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  );
}

export function mat3FromAxisAngle(axis, angle) {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const C = 1 - c;
  return [
    [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
  ];
}

export function rotationAngle(R) {
  const tr = clamp((R[0][0] + R[1][1] + R[2][2] - 1) / 2, -1, 1);
  return Math.acos(tr);
}

// Rotation vector (axis * angle) from an SO(3) matrix. Stable near 0 and pi.
export function so3Log(R) {
  const angle = rotationAngle(R);
  if (angle < 1e-9) {
    return [0, 0, 0];
  }
  if (Math.PI - angle < 1e-6) {
    // Near pi: recover axis from the symmetric part.
    const xx = Math.sqrt(Math.max(0, (R[0][0] + 1) / 2));
    const yy = Math.sqrt(Math.max(0, (R[1][1] + 1) / 2));
    const zz = Math.sqrt(Math.max(0, (R[2][2] + 1) / 2));
    let axis = [xx, yy, zz];
    if (R[0][1] < 0) axis[1] = -axis[1];
    if (R[0][2] < 0) axis[2] = -axis[2];
    axis = normalize(axis);
    return scale(axis, angle);
  }
  const k = angle / (2 * Math.sin(angle));
  return [k * (R[2][1] - R[1][2]), k * (R[0][2] - R[2][0]), k * (R[1][0] - R[0][1])];
}

export function rotationErrorDeg(Ra, Rb) {
  return rotationAngle(mat3Mul(mat3T(Ra), Rb)) * RAD;
}

// Exponential map from so(3) to SO(3).
export function so3Exp(w) {
  const theta = Math.hypot(w[0], w[1], w[2]);
  if (theta < 1e-12) {
    return [
      [1, -w[2], w[1]],
      [w[2], 1, -w[0]],
      [-w[1], w[0], 1],
    ];
  }
  return mat3FromAxisAngle(w, theta);
}

export function orthonormalize(R) {
  // Modified Gram-Schmidt on rows.
  let r0 = normalize(R[0]);
  let r1 = sub(R[1], scale(r0, dot(r0, R[1])));
  r1 = normalize(r1);
  const r2 = cross(r0, r1);
  return [r0, r1, r2];
}

export function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let k = 0; k < n; k++) {
    let piv = k;
    let best = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      const a = Math.abs(M[i][k]);
      if (a > best) {
        best = a;
        piv = i;
      }
    }
    if (best < 1e-14) return null;
    if (piv !== k) {
      const tmp = M[k];
      M[k] = M[piv];
      M[piv] = tmp;
    }
    const d = M[k][k];
    for (let j = k; j <= n; j++) M[k][j] /= d;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = M[i][k];
      if (f === 0) continue;
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }
  return M.map((row) => row[n]);
}

export function sumSq(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return s;
}

export function rms(a, dof = 0) {
  if (!a.length) return 0;
  return Math.sqrt(sumSq(a) / Math.max(1, a.length - dof));
}

export function median(values) {
  if (!values.length) return 0;
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
