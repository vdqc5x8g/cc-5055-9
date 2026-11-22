import {
  add,
  addScaled,
  cross,
  dot,
  mat3Diag,
  mat3FromAxisAngle,
  mat3Identity,
  mat3Mul,
  mat3MulMany,
  mat3T,
  normalize,
  scale,
  sub,
} from './math.js';
import { projectPoint } from './camera.js';

// A plane mirror panel. `center` is the panel centre, yaw rotates the normal
// from +Z towards +X, tilt rotates it from +Z towards +Y. The reflective face
// is the +normal side (front). Local axes: u (width, right), v (height, up).
export function makeMirror({ center = [0, 1, 0], yaw = 0, tilt = 0, width = 1.3, height = 1.9, centerSource = 'constraint' } = {}) {
  return { center: center.slice(), yaw, tilt, width, height, centerSource };
}

export function mirrorFrame(mirror) {
  const cy = Math.cos(mirror.yaw);
  const sy = Math.sin(mirror.yaw);
  const ct = Math.cos(mirror.tilt);
  const st = Math.sin(mirror.tilt);
  // R = Ry(yaw) * Rx(-tilt)? Build explicitly so tilt rotates the normal up.
  const n = [sy * ct, st, cy * ct];
  const u = [cy, 0, -sy];
  const v = cross(n, u); // [ -sy*st, ct, -cy*st ] -> mostly +Y
  return { n, u, v };
}

export function mirrorNormal(mirror) {
  return mirrorFrame(mirror).n;
}

export function frontSign(mirror, point) {
  const { n } = mirrorFrame(mirror);
  return dot(n, sub(point, mirror.center));
}

export function reflectPoint(mirror, X) {
  const { n } = mirrorFrame(mirror);
  const s = dot(n, sub(X, mirror.center));
  return sub(X, scale(n, 2 * s));
}

// Reflection matrix about the mirror plane.
export function reflectionMatrix(mirror) {
  const { n } = mirrorFrame(mirror);
  const N = [
    [n[0] * n[0], n[0] * n[1], n[0] * n[2]],
    [n[1] * n[0], n[1] * n[1], n[1] * n[2]],
    [n[2] * n[0], n[2] * n[1], n[2] * n[2]],
  ];
  const I = mat3Identity();
  const S = [
    [I[0][0] - 2 * N[0][0], I[0][1] - 2 * N[0][1], I[0][2] - 2 * N[0][2]],
    [I[1][0] - 2 * N[1][0], I[1][1] - 2 * N[1][1], I[1][2] - 2 * N[1][2]],
    [I[2][0] - 2 * N[2][0], I[2][1] - 2 * N[2][1], I[2][2] - 2 * N[2][2]],
  ];
  return S;
}

export function reflectVector(mirror, d) {
  const { n } = mirrorFrame(mirror);
  return sub(d, scale(n, 2 * dot(n, d)));
}

// The virtual camera is modelled separately from the real one:
//   C' = reflect(C) (a point reflection about the finite mirror plane),
//   R' = R S,  det(R') = -det(R)  (improper / mirrored handedness)
// Its projection of a world point equals the real camera's projection of the
// reflected point; it is never used as a second set of physical markers.
export function virtualCamera(cam, mirror) {
  const S = reflectionMatrix(mirror);
  const Cv = reflectPoint(mirror, cam.C);
  return {
    R: mat3Mul(cam.R, S),
    C: Cv,
    f: cam.f,
    fx: cam.f,
    fy: cam.f,
    cx: cam.cx,
    cy: cam.cy,
    width: cam.width,
    height: cam.height,
    det: -1,
    isMirrored: true,
  };
}

// Pixel of the reflected image of a physical point, computed through the
// virtual camera. Returns diagnostics including the reflection hit point.
export function projectReflection(cam, mirror, X) {
  const vc = virtualCamera(cam, mirror);
  const proj = projectPoint(vc, X);
  if (!proj) return null;
  const H = reflectionHit(cam, mirror, X);
  return { ...proj, H, virtual: vc, depthVirtual: proj.depth };
}

// Point where the line from the mirrored camera centre to X intersects the
// mirror plane. This is the physical reflection point on the glass.
export function reflectionHit(cam, mirror, X) {
  const Cv = reflectPoint(mirror, cam.C);
  const { n } = mirrorFrame(mirror);
  const denom = dot(n, sub(X, Cv));
  if (Math.abs(denom) < 1e-9) return null;
  const t = dot(n, sub(mirror.center, Cv)) / denom;
  if (!(t > 0 && t < 1)) return null;
  return addScaled(Cv, sub(X, Cv), t);
}

export function mirrorLocalCoords(mirror, point) {
  const { u, v } = mirrorFrame(mirror);
  const d = sub(point, mirror.center);
  return { s: dot(d, u), t: dot(d, v) };
}

export function insideMirrorBounds(mirror, point, margin = 0) {
  const { s, t } = mirrorLocalCoords(mirror, point);
  return (
    Math.abs(s) <= mirror.width / 2 - margin &&
    Math.abs(t) <= mirror.height / 2 - margin
  );
}

// Full geometric validity of a reflected observation of a physical point.
// Checks: both camera and point on the reflective side, hit point exists,
// hit point lies on the finite panel, and the reflected pixel is in frame.
export function reflectionValidity(cam, mirror, X, { margin = 0.006, requireInFrame = true } = {}) {
  const { n } = mirrorFrame(mirror);
  const camSide = dot(n, sub(cam.C, mirror.center));
  const objSide = dot(n, sub(X, mirror.center));
  const reasons = [];
  let ok = true;
  if (camSide <= 1e-6) {
    ok = false;
    reasons.push('相机位于镜面背面');
  }
  if (objSide <= 1e-6) {
    ok = false;
    reasons.push('物体位于镜面背面（或贴面）');
  }
  if (camSide * objSide < 0) {
    ok = false;
    reasons.push('相机与物体分居镜面两侧，无同一反射路径');
  }
  const H = ok ? reflectionHit(cam, mirror, X) : null;
  if (ok && !H) {
    ok = false;
    reasons.push('反射路径未与镜面相交');
  }
  const inBounds = H ? insideMirrorBounds(mirror, H, margin) : false;
  if (ok && !inBounds) {
    ok = false;
    reasons.push('反射点落在镜面矩形之外');
  }
  const vc = virtualCamera(cam, mirror);
  const proj = projectPoint(vc, X);
  if (ok && requireInFrame) {
    if (!proj) {
      ok = false;
      reasons.push('镜中像在虚拟相机后方');
    } else if (
      proj.x < 0 ||
      proj.x > cam.width ||
      proj.y < 0 ||
      proj.y > cam.height
    ) {
      ok = false;
      reasons.push('镜中像超出画面');
    }
  }
  return {
    ok,
    reasons,
    H,
    proj,
    inBounds,
    camSide,
    objSide,
    chirality: proj ? -1 : null,
    virtualCamera: vc,
  };
}

// Mirror pose from yaw/tilt/offset along the normal, keeping the panel centre
// projected on its plane at the same lateral position (in-plane slide is not
// observable from reflected point correspondences alone).
export function mirrorWithPlane(mirror, yaw, tilt, offsetAlongNormal) {
  const base = makeMirror({ ...mirror, yaw, tilt });
  const { n } = mirrorFrame(base);
  const center = addScaled(base.center, n, offsetAlongNormal);
  return { ...base, center };
}

export function mirrorPlaneOffset(mirror) {
  const { n } = mirrorFrame(mirror);
  return dot(n, mirror.center);
}

export function intersectRayPlane(origin, dir, mirror) {
  const { n } = mirrorFrame(mirror);
  const denom = dot(n, dir);
  if (Math.abs(denom) < 1e-12) return null;
  const t = dot(n, sub(mirror.center, origin)) / denom;
  if (t <= 0) return null;
  return { t, point: addScaled(origin, dir, t) };
}

export function mirrorQuad(mirror) {
  const { u, v } = mirrorFrame(mirror);
  const hw = mirror.width / 2;
  const hh = mirror.height / 2;
  return [
    addScaled(addScaled(mirror.center, u, -hw), v, hh),
    addScaled(addScaled(mirror.center, u, hw), v, hh),
    addScaled(addScaled(mirror.center, u, hw), v, -hh),
    addScaled(addScaled(mirror.center, u, -hw), v, -hh),
  ];
}

export function mirrorCorners(mirror) {
  return mirrorQuad(mirror);
}

// Accept a partial mirror constraint (centre, size, optional yaw/tilt) and fill
// defaults so every downstream routine sees a complete, finite mirror.
export function normalizeMirror(mirror, defaults = {}) {
  const base = {
    center: [0, 1.0, 0],
    yaw: 0,
    tilt: 0,
    width: 1.5,
    height: 2.0,
    ...defaults,
    ...(mirror || {}),
  };
  const num = (v, d) => (Number.isFinite(v) ? v : d);
  const c = Array.isArray(base.center) ? base.center : [0, 1, 0];
  return {
    center: [num(c[0], 0), num(c[1], 1), num(c[2], 0)],
    yaw: num(base.yaw, 0),
    tilt: num(base.tilt, 0),
    width: num(base.width, 1.5),
    height: num(base.height, 2.0),
    centerSource: base.centerSource ?? 'constraint',
  };
}
