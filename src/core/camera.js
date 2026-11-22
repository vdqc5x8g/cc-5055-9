import {
  add,
  addScaled,
  cross,
  dist,
  dot,
  mat3Mul,
  normalize,
  scale,
  sub,
  cross as x3,
} from './math.js';

// Camera convention: x right, y down, z forward (computer-vision style).
// Focal length is expressed in pixel units; principal point in pixels.

export function makeCamera({ R, C, f, cx, cy, width, height }) {
  return {
    R,
    C,
    f,
    cx,
    cy,
    width,
    height,
    fx: f,
    fy: f,
  };
}

export function cameraFromLookAt({ position, target, upHint = [0, 1, 0], f, width, height }) {
  const z = normalize(sub(target, position));
  let x = cross(z, upHint);
  if (Math.hypot(x[0], x[1], x[2]) < 1e-6) {
    x = cross(z, [1, 0, 0]);
  }
  x = normalize(x);
  let y = cross(z, x);
  y = normalize(y);
  return makeCamera({
    R: [x, y, z],
    C: position.slice(),
    f,
    cx: width / 2,
    cy: height / 2,
    width,
    height,
  });
}

export function cameraCenter(C) {
  return C.slice();
}

export function cloneCamera(cam) {
  return {
    R: cam.R.map((r) => r.slice()),
    C: cam.C.slice(),
    f: cam.f,
    cx: cam.cx,
    cy: cam.cy,
    width: cam.width,
    height: cam.height,
    fx: cam.fx,
    fy: cam.fy,
  };
}

// Xc = R * (Xw - C)
export function worldToCamera(cam, X) {
  const d = sub(X, cam.C);
  return [dot(cam.R[0], d), dot(cam.R[1], d), dot(cam.R[2], d)];
}

export function cameraToWorld(cam, Xc) {
  return add(cam.C, [
    cam.R[0][0] * Xc[0] + cam.R[1][0] * Xc[1] + cam.R[2][0] * Xc[2],
    cam.R[0][1] * Xc[0] + cam.R[1][1] * Xc[1] + cam.R[2][1] * Xc[2],
    cam.R[0][2] * Xc[0] + cam.R[1][2] * Xc[1] + cam.R[2][2] * Xc[2],
  ]);
}

export function projectPoint(cam, X) {
  const Xc = worldToCamera(cam, X);
  const depth = Xc[2];
  if (!(depth > 1e-9)) return null;
  return {
    x: cam.cx + (cam.f * Xc[0]) / depth,
    y: cam.cy + (cam.f * Xc[1]) / depth,
    depth,
    Xc,
  };
}

export function unprojectPixel(cam, px, py, depth = 1) {
  const dirCam = [(px - cam.cx) / cam.f, (py - cam.cy) / cam.f, 1];
  const dirWorld = [
    cam.R[0][0] * dirCam[0] + cam.R[1][0] * dirCam[1] + cam.R[2][0] * dirCam[2],
    cam.R[0][1] * dirCam[0] + cam.R[1][1] * dirCam[1] + cam.R[2][1] * dirCam[2],
    cam.R[0][2] * dirCam[0] + cam.R[1][2] * dirCam[1] + cam.R[2][2] * dirCam[2],
  ];
  const n = normalize(dirWorld);
  return { origin: cam.C.slice(), dir: n, point: addScaled(cam.C, n, depth) };
}

export function rayDirection(cam, px, py) {
  return unprojectPixel(cam, px, py, 1).dir;
}

// Camera matrix (3x4) in the world-to-image form: P = K [R | -RC]
export function cameraMatrix(cam) {
  const r0 = [0, 1, 2].map((j) => cam.f * cam.R[0][j] + cam.cx * cam.R[2][j]);
  const r1 = [0, 1, 2].map((j) => cam.f * cam.R[1][j] + cam.cy * cam.R[2][j]);
  const r2 = cam.R[2].slice();
  const t = [dot(r0, cam.C), dot(r1, cam.C), dot(r2, cam.C)];
  return [r0.concat([-t[0]]), r1.concat([-t[1]]), r2.concat([-t[2]])];
}

// Pinhole projection of a 4-vector, returns homogeneous pixel coords {u,v,w}.
export function projectHomogeneous(P, Xh) {
  return [
    P[0][0] * Xh[0] + P[0][1] * Xh[1] + P[0][2] * Xh[2] + P[0][3] * Xh[3],
    P[1][0] * Xh[0] + P[1][1] * Xh[1] + P[1][2] * Xh[2] + P[1][3] * Xh[3],
    P[2][0] * Xh[0] + P[2][1] * Xh[1] + P[2][2] * Xh[2] + P[2][3] * Xh[3],
  ];
}

export function pixelToNormalized(cam, px, py) {
  return [(px - cam.cx) / cam.f, (py - cam.cy) / cam.f];
}
