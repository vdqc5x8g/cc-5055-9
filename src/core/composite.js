import { projectPoint } from './camera.js';
import { reflectionValidity } from './mirror.js';
import { boxVertices, boxFaces } from './asset.js';

// A composite frame consists of:
//   - direct layer: the virtual asset as seen directly
//   - mirror layer: the same asset reflected about the solved plane, drawn only
//     where a physical reflection path exists (hit point inside the panel and
//     in front of the camera, asset on the reflective side).
// The mirror layer is produced by projecting the *reflected geometry* through
// the real camera (equivalently the virtual camera), not by flipping pixels.
export function buildCompositeFrame({ camera, mirror, asset, style = {} }) {
  const direct = projectAsset(camera, asset, style);
  const validity = reflectionValidity(camera, mirror, asset.contact);
  const mirrorShape = projectAsset(camera, { ...asset, yaw: asset.yaw }, style, mirror);
  const visible = validity.ok;
  return {
    camera,
    mirror,
    asset,
    direct,
    mirrorShape,
    mirrorVisible: visible,
    mirrorReasons: validity.reasons,
    hitPoint: validity.H,
    detVirtual: -1,
    style,
  };
}

function projectAsset(camera, asset, style, mirror = null) {
  const verts = boxVertices(asset).map((v) => (mirror ? reflectPointLocal(mirror, v) : v));
  const projected = verts.map((v) => projectPoint(camera, v));
  const faces = boxFaces().map((f) => ({
    idx: f.idx,
    points: f.idx.map((i) => projected[i]),
    normal: f.normal,
  }));
  const valid = projected.every((p) => p && p.depth > 0);
  return {
    vertices: verts,
    projected,
    faces,
    valid,
    depthOrder: sortFacesByDepth(projected, boxFaces()),
  };
}

// Painter's ordering for a convex box: sum of vertex depths per face.
function sortFacesByDepth(projected, faces) {
  return faces
    .map((f, i) => ({
      i,
      depth: f.idx.reduce((s, k) => s + (projected[k]?.depth ?? Infinity), 0),
    }))
    .sort((a, b) => b.depth - a.depth)
    .map((e) => e.i);
}

function reflectPointLocal(mirror, X) {
  // local copy to avoid circular import of mirror helpers in the renderer
  const cy = Math.cos(mirror.yaw);
  const sy = Math.sin(mirror.yaw);
  const ct = Math.cos(mirror.tilt);
  const st = Math.sin(mirror.tilt);
  const n = [sy * ct, st, cy * ct];
  const d = (X[0] - mirror.center[0]) * n[0] + (X[1] - mirror.center[1]) * n[1] + (X[2] - mirror.center[2]) * n[2];
  return [X[0] - 2 * d * n[0], X[1] - 2 * d * n[1], X[2] - 2 * d * n[2]];
}

// Signed area of a projected polygon (for culling / orientation checks).
export function polygonArea(points) {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const q = points[(i + 1) % points.length];
    if (!p || !q) return 0;
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

// Face orientation test: a box face is front-facing if its outward normal
// points toward the camera. Used to avoid drawing back faces over front ones.
export function boxFaceIsFront(asset, faceIndex, camera) {
  const verts = boxVertices(asset);
  const face = boxFaces()[faceIndex];
  const a = verts[face.idx[0]];
  const b = verts[face.idx[1]];
  const c = verts[face.idx[2]];
  const normal = cross(sub(b, a), sub(c, a));
  const centre = [0, 0, 0];
  for (const i of face.idx) for (let k = 0; k < 3; k++) centre[k] += verts[i][k] / 4;
  const toCam = sub(camera.C, centre);
  return dot(normal, toCam) > 0;
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
