import { projectPoint, unprojectPixel } from './camera.js';
import { mirrorFrame, mirrorQuad, reflectPoint, reflectionHit, reflectionValidity, virtualCamera } from './mirror.js';
import { boxVertices, boxFaces, assetProjection } from './asset.js';
import { polygonArea } from './composite.js';

// All drawing is in CSS pixels with the camera's native resolution mapped onto
// the canvas viewport. No pixel-flip trick is used: the mirror layer projects
// the reflected 3D geometry through the solved camera.
export function drawScene(ctx, view, {
  camera,
  mirror,
  asset,
  observations = null,
  showDirectRays = true,
  showMirrorRays = true,
  showMirrorFrame = true,
  showTruth = false,
  truthMirror = null,
  selectedMarker = null,
  hoveredMarker = null,
  rays = [],
  showLabels = true,
} = {}) {
  const { width: W, height: H } = view;
  clear(ctx, W, H);
  if (!camera || !mirror) return;

  const cam = fitCamera(camera, W, H);

  // Background: a neutral studio wall with a floor line, drawn in world space.
  drawBackdrop(ctx, cam, W, H);

  const mirrorVisible = drawMirror(ctx, cam, mirror, showMirrorFrame);

  // Observed physical markers.
  if (observations) {
    drawObservations(ctx, cam, observations, { selectedMarker, hoveredMarker, showLabels });
  }

  // Reflected layer, clipped to the mirror quad.
  if (asset && mirrorVisible) {
    drawReflectedAsset(ctx, cam, mirror, asset);
  }
  if (asset) {
    drawAsset(ctx, cam, asset);
  }

  // Reflection hit points and mirror-validity markers.
  if (mirror && asset) {
    drawAssetReflectionOverlay(ctx, cam, mirror, asset);
  }

  if (showTruth && truthMirror) {
    drawMirrorOutline(ctx, cam, truthMirror, '#2f9e44', true);
  }

  // Direct and mirror ray paths for the selected marker.
  for (const r of rays) {
    drawRay(ctx, cam, r, r.color);
  }
}

function clear(ctx, W, H) {
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#11151c');
  g.addColorStop(1, '#0a0c10');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

// Camera view transform: map native camera resolution onto the viewport.
function fitCamera(camera, W, H) {
  const sx = W / camera.width;
  const sy = H / camera.height;
  const s = Math.max(sx, sy); // cover
  const ox = (W - camera.width * s) / 2;
  const oy = (H - camera.height * s) / 2;
  return {
    camera,
    project(X) {
      const p = projectPoint(camera, X);
      if (!p) return null;
      return { x: ox + p.x * s, y: oy + p.y * s, depth: p.depth, native: p };
    },
    scale: s,
    viewport: { ox, oy, s },
  };
}

function drawBackdrop(ctx, cam, W, H) {
  // Floor plane y = 0, near-to-far quad.
  const corners = [
    [-4, 0, -2.5],
    [4, 0, -2.5],
    [4, 0, 5],
    [-4, 0, 5],
  ].map((p) => cam.project(p));
  if (corners.every(Boolean)) {
    pathPoly(ctx, corners);
    ctx.fillStyle = '#171c24';
    ctx.fill();
    ctx.strokeStyle = '#232b36';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  // Back wall plane z = -2.5.
  const wall = [
    [-4, 0, -2.5],
    [4, 0, -2.5],
    [4, 3.2, -2.5],
    [-4, 3.2, -2.5],
  ].map((p) => cam.project(p));
  if (wall.every(Boolean)) {
    pathPoly(ctx, wall);
    ctx.fillStyle = '#121720';
    ctx.fill();
  }
  // Floor grid.
  ctx.save();
  ctx.strokeStyle = 'rgba(90,110,135,0.10)';
  for (let x = -4; x <= 4; x += 0.5) {
    const a = cam.project([x, 0, -2.5]);
    const b = cam.project([x, 0, 5]);
    if (a && b) line(ctx, a, b);
  }
  for (let z = -2.5; z <= 5; z += 0.5) {
    const a = cam.project([-4, 0, z]);
    const b = cam.project([4, 0, z]);
    if (a && b) line(ctx, a, b);
  }
  ctx.restore();
}

function drawMirror(ctx, cam, mirror, showFrame) {
  const quad = mirrorQuad(mirror).map((p) => cam.project(p));
  if (!quad.every(Boolean)) return false;
  ctx.save();
  pathPoly(ctx, quad);
  const g = ctx.createLinearGradient(quad[0].x, quad[0].y, quad[2].x, quad[2].y);
  g.addColorStop(0, 'rgba(120,180,210,0.16)');
  g.addColorStop(0.5, 'rgba(150,205,230,0.24)');
  g.addColorStop(1, 'rgba(100,150,185,0.14)');
  ctx.fillStyle = g;
  ctx.fill();
  if (showFrame) {
    ctx.strokeStyle = 'rgba(150,220,245,0.85)';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Normal indicator from the panel centre.
    const c = cam.project(mirror.center);
    const { n } = mirrorFrame(mirror);
    const tip = cam.project([
      mirror.center[0] + n[0] * 0.45,
      mirror.center[1] + n[1] * 0.45,
      mirror.center[2] + n[2] * 0.45,
    ]);
    if (c && tip) {
      line(ctx, c, tip, 'rgba(255,205,90,0.9)', 2);
      dot(ctx, tip, 4, '#ffcd5a');
      if (showFrame) {
        ctx.fillStyle = 'rgba(255,205,90,0.95)';
        ctx.font = '12px ui-monospace, monospace';
        ctx.fillText('n', tip.x + 6, tip.y - 4);
      }
    }
  }
  ctx.restore();
  return true;
}

function drawMirrorOutline(ctx, cam, mirror, color, dashed) {
  const quad = mirrorQuad(mirror).map((p) => cam.project(p));
  if (!quad.every(Boolean)) return;
  ctx.save();
  pathPoly(ctx, quad);
  ctx.setLineDash(dashed ? [6, 5] : []);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function drawObservations(ctx, cam, observations, { selectedMarker, hoveredMarker, showLabels }) {
  ctx.save();
  for (const o of observations.observations ?? observations) {
    if (!o.pixel) continue;
    const p = cam.project(o.X);
    if (!p) continue;
    const isMirror = o.kind === 'mirror';
    const isSel = selectedMarker === o.id;
    const isHover = hoveredMarker === o.id;
    const color = isMirror ? 'rgba(120,215,255,0.95)' : 'rgba(255,170,120,0.98)';
    if (isMirror) {
      // Marked with a ring to distinguish reflected identity from direct.
      ctx.beginPath();
      ctx.arc(p.x, p.y, isSel ? 7 : 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = isSel ? 2.4 : 1.5;
      ctx.stroke();
    } else {
      dot(ctx, p, isSel ? 5 : 3.2, color);
    }
    if (isSel || isHover || (showLabels && isSel)) {
      ctx.fillStyle = color;
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText(`${o.id}${isMirror ? "'" : ''}`, p.x + 8, p.y - 6);
    }
  }
  ctx.restore();
}

function drawAsset(ctx, cam, asset) {
  const verts = boxVertices(asset);
  const proj = verts.map((v) => cam.project(v));
  if (!proj.every(Boolean)) return;
  drawBox(ctx, proj, asset, false);
  // Contact point on the floor with a vertical drop line.
  const contact = cam.project(asset.contact);
  if (contact) {
    dot(ctx, contact, 4, '#ffd43b');
    const corners = boxVertices(asset).filter((v) => Math.abs(v[1] - asset.contact[1]) < 1e-9)
      .map((v) => cam.project(v)).filter(Boolean);
    if (corners.length) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255,212,59,0.7)';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (const c of corners) {
        ctx.moveTo(contact.x, contact.y);
        ctx.lineTo(c.x, c.y);
      }
      ctx.stroke();
      ctx.restore();
    }
  }
}

function drawReflectedAsset(ctx, cam, mirror, asset) {
  const quad = mirrorQuad(mirror).map((p) => cam.project(p));
  if (!quad.every(Boolean)) return;
  const mirroredVerts = boxVertices(asset).map((v) => reflectPoint(mirror, v));
  const proj = mirroredVerts.map((v) => cam.project(v));
  if (!proj.every(Boolean)) return;
  ctx.save();
  pathPoly(ctx, quad);
  ctx.clip();
  drawBox(ctx, proj, asset, true);
  ctx.restore();
}

// Draw a box from projected corner points. Face ordering uses mean depth.
function drawBox(ctx, proj, asset, mirrored) {
  const faces = boxFaces();
  const order = faces
    .map((f, i) => ({ i, depth: f.idx.reduce((s, k) => s + (proj[k]?.depth ?? 1e9), 0) / 4 }))
    .sort((a, b) => b.depth - a.depth);
  const baseColor = mirrored ? [110, 180, 235] : [235, 150, 95];
  for (const { i } of order) {
    const f = faces[i];
    const pts = f.idx.map((k) => proj[k]);
    if (!pts.every(Boolean)) continue;
    pathPoly(ctx, pts);
    const shade = faceShade(f.normal);
    const alpha = mirrored ? 0.34 : 0.9;
    ctx.fillStyle = `rgba(${baseColor[0] * shade | 0},${baseColor[1] * shade | 0},${baseColor[2] * shade | 0},${alpha})`;
    ctx.fill();
    ctx.strokeStyle = mirrored ? 'rgba(160,220,255,0.55)' : 'rgba(255,205,170,0.9)';
    ctx.lineWidth = mirrored ? 1 : 1.6;
    ctx.stroke();
  }
}

function faceShade(normal) {
  const table = { top: 1.15, bottom: 0.55, front: 0.95, back: 0.7, left: 0.8, right: 0.9 };
  return table[normal] ?? 0.9;
}

function drawAssetReflectionOverlay(ctx, cam, mirror, asset) {
  const camera = cam.camera;
  const validity = reflectionValidity(camera, mirror, asset.contact);
  const H = validity.H ?? reflectionHit(camera, mirror, asset.contact);
  if (!H) return;
  const p = cam.project(H);
  if (!p) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
  ctx.strokeStyle = validity.ok ? 'rgba(105,219,124,0.95)' : 'rgba(255,107,107,0.95)';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = 'rgba(105,219,124,0.9)';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('H', p.x + 8, p.y + 14);
  ctx.restore();
}

function drawRay(ctx, cam, r, color) {
  const a = cam.project(r.from);
  const b = cam.project(r.to);
  if (!a || !b) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  if (r.dashed) ctx.setLineDash([5, 4]);
  line(ctx, a, b);
  ctx.restore();
}

export function computeRays({ camera, mirror, marker, observation }) {
  const rays = [];
  if (!marker || !observation) return rays;
  // Direct ray: camera centre -> object -> image plane.
  const directHit = observation.observations?.find?.((o) => o.id === marker.id && o.kind === 'direct');
  if (directHit?.pixel) {
    rays.push({ from: camera.C, to: marker.X, color: 'rgba(255,170,120,0.85)' });
  }
  const mirrorHit = observation.observations?.find?.((o) => o.id === marker.id && o.kind === 'mirror');
  if (mirrorHit?.pixel) {
    const vc = virtualCamera(camera, mirror);
    const H = reflectionHit(camera, mirror, marker.X);
    rays.push({ from: vc.C, to: marker.X, color: 'rgba(120,215,255,0.45)', dashed: true });
    if (H) {
      rays.push({ from: camera.C, to: H, color: 'rgba(120,215,255,0.95)' });
      rays.push({ from: H, to: marker.X, color: 'rgba(120,215,255,0.95)' });
      rays.push({ from: H, to: vc.C, color: 'rgba(120,215,255,0.35)', dashed: true });
    }
  }
  return rays;
}

function pathPoly(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
}

function line(ctx, a, b, color = null, w = 1) {
  ctx.save();
  if (color) ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function dot(ctx, p, r, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
