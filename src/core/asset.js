import { mat3Mul, mat3T, add, cross, dot, normalize, scale, sub } from './math.js';
import { makeMirror, mirrorFrame, reflectPoint, reflectionValidity, reflectionHit } from './mirror.js';
import { projectPoint } from './camera.js';

// Build the 8 corners of an oriented box from a bottom-centre contact frame.
export function boxVertices({ position, yaw, scale: sc }) {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const ux = [c, 0, -s];
  const uz = [s, 0, c];
  const [hw, h, hd] = sc;
  const base = [position[0], position[1], position[2]];
  const out = [];
  for (const sy of [0, 1]) {
    for (const su of [-1, 1]) {
      for (const sz of [-1, 1]) {
        out.push([
          base[0] + ux[0] * hw * su + uz[0] * hd * sz,
          base[1] + h * sy,
          base[2] + ux[2] * hw * su + uz[2] * hd * sz,
        ]);
      }
    }
  }
  // Order: bottom (y=0) indices 0..3, top (y=1) indices 4..7
  return out;
}

export function boxFaces() {
  return [
    { idx: [0, 1, 3, 2], normal: 'bottom' },
    { idx: [4, 6, 7, 5], normal: 'top' },
    { idx: [0, 4, 5, 1], normal: 'front' },
    { idx: [2, 3, 7, 6], normal: 'back' },
    { idx: [0, 2, 6, 4], normal: 'left' },
    { idx: [1, 5, 7, 3], normal: 'right' },
  ];
}

// Place the virtual asset using a ground-plane constraint at a nominated screen
// point. Scale is supplied by the user (or a known marker height); the tool
// never invents an absolute distance without such evidence.
export function placeAssetOnGround({ camera, asset, groundY = 0, screenPoint = null, groundPoint = null }) {
  if (groundPoint) {
    return {
      ...asset,
      contact: [groundPoint[0], groundY, groundPoint[2]],
      position: [groundPoint[0], groundY, groundPoint[2]],
      placement: 'ground-constraint',
    };
  }
  if (!screenPoint) {
    return { ...asset, placement: 'asset-default' };
  }
  // Ray through the pixel intersected with the ground plane y = groundY.
  const dirCam = [
    (screenPoint[0] - camera.cx) / camera.f,
    (screenPoint[1] - camera.cy) / camera.f,
    1,
  ];
  const dir = normalize([
    camera.R[0][0] * dirCam[0] + camera.R[1][0] * dirCam[1] + camera.R[2][0] * dirCam[2],
    camera.R[0][1] * dirCam[0] + camera.R[1][1] * dirCam[1] + camera.R[2][1] * dirCam[2],
    camera.R[0][2] * dirCam[0] + camera.R[1][2] * dirCam[1] + camera.R[2][2] * dirCam[2],
  ]);
  if (Math.abs(dir[1]) < 1e-6) return { ...asset, placement: 'ray-parallel-ground' };
  const t = (groundY - camera.C[1]) / dir[1];
  if (t <= 0) return { ...asset, placement: 'ray-away-from-ground' };
  const p = add(camera.C, scale(dir, t));
  return {
    ...asset,
    contact: [p[0], groundY, p[2]],
    position: [p[0], groundY, p[2]],
    placement: 'screen-ray-ground',
  };
}

// Mirror image of the asset about the solved plane, used both for rendering the
// reflected object and for grading its image-space position.
export function mirrorsAsset(asset, mirror) {
  const vertices = boxVertices(asset).map((v) => reflectPoint(mirror, v));
  return { ...asset, vertices, mirrored: true };
}

export function assetProjection(camera, asset) {
  const verts = boxVertices(asset);
  const projected = verts.map((v) => projectPoint(camera, v));
  return { verts, projected };
}

export function assetContactValidity(camera, asset) {
  const p = projectPoint(camera, asset.contact);
  return {
    contactWorld: asset.contact,
    contactPixel: p ? [p.x, p.y] : null,
    onGround: Math.abs(asset.contact[1] - 0) < 1e-6,
  };
}

// Reflection path of the asset's contact point: the reflected image must land
// at the projection of the mirrored contact point, and the ray must hit the
// panel inside its bounds. This is the correctness check for the composite.
export function assetReflectionCheck(camera, mirror, asset) {
  const mirrored = mirrorsAsset(asset, mirror);
  const contactMirror = reflectPoint(mirror, asset.contact);
  const projMirror = projectPoint(camera, contactMirror);
  const validity = reflectionValidity(camera, mirror, asset.contact);
  const H = reflectionHit(camera, mirror, asset.contact);
  return {
    contactMirror,
    contactMirrorPixel: projMirror ? [projMirror.x, projMirror.y] : null,
    hitPoint: H,
    inBounds: validity.inBounds,
    ok: validity.ok,
    reasons: validity.reasons,
    detVirtual: -1,
  };
}

// Compare composite reflection against a naive horizontal flip of the direct
// sprite (the anti-pattern the tool must not claim as correct). Returns the
// mean discrepancy between the flipped direct image and the true reflected
// image, measured at the box vertices.
export function flipVsReflection(camera, mirror, asset) {
  const verts = boxVertices(asset);
  const directPx = verts.map((v) => projectPoint(camera, v));
  const reflPx = verts.map((v) => projectPoint(camera, reflectPoint(mirror, v)));
  const cx = camera.width / 2;
  const flipped = directPx.map((p) => (p ? [2 * cx - p.x, p.y] : null));
  let sum = 0;
  let n = 0;
  const pairs = [];
  for (let i = 0; i < verts.length; i++) {
    if (!reflPx[i] || !flipped[i]) continue;
    const d = Math.hypot(reflPx[i].x - flipped[i][0], reflPx[i].y - flipped[i][1]);
    sum += d;
    n++;
    pairs.push({ vertex: i, reflected: [reflPx[i].x, reflPx[i].y], flipped: flipped[i], delta: d });
  }
  return {
    meanDiscrepancyPx: n ? sum / n : Infinity,
    pairs,
    explanation: '水平翻转只是把直接画面镜像，两个端点一般无法同时落在真实反射位置；差值即空间关系错误的量度。',
  };
}
