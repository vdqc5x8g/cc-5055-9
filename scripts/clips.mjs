import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { createDefaultScene } from '../src/core/scene.js';
import { buildMarkers, observe, directObservations, mirrorObservations } from '../src/core/scene.js';
import { projectPoint } from '../src/core/camera.js';
import {
  mirrorQuad,
  mirrorFrame,
  reflectPoint,
  reflectionValidity,
  virtualCamera,
} from '../src/core/mirror.js';
import { boxVertices, boxFaces, placeAssetOnGround } from '../src/core/asset.js';
import { runPipeline } from '../src/core/pipeline.js';
import { DEG, dist } from '../src/core/math.js';
import { createRaster } from './raster.mjs';
import { encodePNG } from './png.mjs';

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'clips');
const WIDTH = 960;
const HEIGHT = 540;
const NOISE_PX = 0.5;

// Marker identity colours per physical group (the plate is synthetic footage;
// codes stay readable in the picture and are mirrored in the manifest).
const GROUP_COLORS = {
  board: [232, 189, 118],
  floor: [158, 214, 164],
  ledge: [214, 158, 214],
};

function scaledProject(cam, X, S) {
  const p = projectPoint(cam, X);
  return p ? { x: p.x * S, y: p.y * S, depth: p.depth } : null;
}

function makeFrameScene(base, { camera, mirror }) {
  return {
    ...base,
    cameraTruth: camera,
    mirror,
    width: base.width,
    height: base.height,
    markers: buildMarkers(mirror),
    virtualAsset: base.virtualAsset,
  };
}

// Draw an identifiable marker patch: 5 binary bars (5-bit id) plus an
// asymmetric orientation wedge in the corner.
function drawMarker(raster, cx, cy, size, id, rgb, alpha = 255) {
  const s = Math.max(4, Math.round(size));
  const x0 = Math.round(cx - s / 2);
  const y0 = Math.round(cy - s / 2);
  raster.fillPolygon(
    [
      { x: x0, y: y0 },
      { x: x0 + s, y: y0 },
      { x: x0 + s, y: y0 + s },
      { x: x0, y: y0 + s },
    ],
    [18, 22, 28, alpha],
  );
  raster.strokePolygon(
    [
      { x: x0 + 0.5, y: y0 + 0.5 },
      { x: x0 + s - 0.5, y: y0 + 0.5 },
      { x: x0 + s - 0.5, y: y0 + s - 0.5 },
      { x: x0 + 0.5, y: y0 + s - 0.5 },
    ],
    [rgb[0], rgb[1], rgb[2], alpha],
    1,
  );
  // 5-bit id as vertical bars across the lower two thirds.
  const pad = Math.max(1, s * 0.12);
  const innerW = s - pad * 2;
  const barW = innerW / 5;
  const barY0 = y0 + s * 0.32;
  const barH = s * 0.56;
  for (let b = 0; b < 5; b++) {
    if (!((id >> b) & 1)) continue;
    const bx = x0 + pad + b * barW;
    raster.fillPolygon(
      [
        { x: bx + barW * 0.18, y: barY0 },
        { x: bx + barW * 0.82, y: barY0 },
        { x: bx + barW * 0.82, y: barY0 + barH },
        { x: bx + barW * 0.18, y: barY0 + barH },
      ],
      [rgb[0], rgb[1], rgb[2], alpha],
    );
  }
  // Orientation wedge in the top-left corner.
  raster.fillPolygon(
    [
      { x: x0 + 1, y: y0 + 1 },
      { x: x0 + s * 0.42, y: y0 + 1 },
      { x: x0 + 1, y: y0 + s * 0.42 },
    ],
    [rgb[0], rgb[1], rgb[2], alpha],
  );
}

function drawBackdrop(raster, cam, S) {
  raster.clear([15, 18, 24]);
  // Back wall.
  const wall = [
    [-5, 0, -2.6],
    [5, 0, -2.6],
    [5, 3.4, -2.6],
    [-5, 3.4, -2.6],
  ].map((p) => scaledProject(cam, p, S));
  if (wall.every(Boolean)) raster.fillPolygon(wall, [21, 26, 34]);
  // Floor.
  const floor = [
    [-5, 0, -2.6],
    [5, 0, -2.6],
    [5, 0, 6],
    [-5, 0, 6],
  ].map((p) => scaledProject(cam, p, S));
  if (floor.every(Boolean)) raster.fillPolygon(floor, [29, 34, 43]);
  for (let x = -5; x <= 5; x += 0.5) {
    const a = scaledProject(cam, [x, 0, -2.6], S);
    const b = scaledProject(cam, [x, 0, 6], S);
    if (a && b) raster.line(a, b, [44, 53, 66, 255], 1);
  }
  for (let z = -2.5; z <= 6; z += 0.5) {
    const a = scaledProject(cam, [-5, 0, z], S);
    const b = scaledProject(cam, [5, 0, z], S);
    if (a && b) raster.line(a, b, [44, 53, 66, 255], 1);
  }
}

function drawMirrorPanel(raster, cam, mirror, S) {
  const quad = mirrorQuad(mirror).map((p) => scaledProject(cam, p, S));
  if (!quad.every(Boolean)) return false;
  raster.fillPolygon(quad, [52, 78, 96, 255]);
  raster.pushClipRect(
    Math.min(...quad.map((p) => p.x)),
    Math.min(...quad.map((p) => p.y)),
    Math.max(...quad.map((p) => p.x)),
    Math.max(...quad.map((p) => p.y)),
  );
  raster.fillPolygon(quad, [74, 112, 137, 255]);
  raster.strokePolygon(quad, [150, 196, 219, 255], 2);
  raster.popClip();
  // Glass highlight line.
  const a = scaledProject(cam, quadWorldPoint(mirror, -0.45, 0.85), S);
  const b = scaledProject(cam, quadWorldPoint(mirror, 0.15, -0.9), S);
  if (a && b) raster.line(a, b, [180, 214, 232, 90], 3);
  return true;
}

function quadWorldPoint(mirror, s, t) {
  const { u, v } = mirrorFrame(mirror);
  return [
    mirror.center[0] + u[0] * s * mirror.width + v[0] * t * mirror.height * 0.5,
    mirror.center[1] + u[1] * s * mirror.width + v[1] * t * mirror.height * 0.5,
    mirror.center[2] + u[2] * s * mirror.width + v[2] * t * mirror.height * 0.5,
  ];
}

function markerPixelSize(cam, X, sizeM, S) {
  const p = projectPoint(cam, X);
  if (!p) return 8;
  return Math.max(6, (cam.f * sizeM * S) / p.depth);
}

function drawPlate(raster, cam, mirror, obs, S, markerSize) {
  drawBackdrop(raster, cam, S);
  drawMirrorPanel(raster, cam, mirror, S);
  const quad = mirrorQuad(mirror).map((p) => scaledProject(cam, p, S));

  // Reflected marker images, clipped to the finite glass panel. Only pixels
  // that are valid reflected observations are drawn (same evidence the solver
  // receives).
  if (quad.every(Boolean)) {
    raster.pushClipRect(
      Math.min(...quad.map((p) => p.x)),
      Math.min(...quad.map((p) => p.y)),
      Math.max(...quad.map((p) => p.x)),
      Math.max(...quad.map((p) => p.y)),
    );
    for (const o of obs.observations) {
      if (o.kind !== 'mirror' || !o.pixel) continue;
      const p = { x: o.pixel[0] * S, y: o.pixel[1] * S };
      const color = GROUP_COLORS[o.group] ?? [200, 220, 235];
      const tint = [Math.round(color[0] * 0.72 + 40), Math.round(color[1] * 0.72 + 50), Math.round(color[2] * 0.72 + 65)];
      drawMarker(raster, p.x, p.y, markerPixelSize(cam, o.X, markerSize, S), markerId(o.id), tint, 225);
    }
    raster.popClip();
  }
  // Direct marker patches (they sit in front of the glass and occlude it).
  for (const o of obs.observations) {
    if (o.kind !== 'direct' || !o.pixel) continue;
    const color = GROUP_COLORS[o.group] ?? [230, 230, 230];
    drawMarker(raster, o.pixel[0] * S, o.pixel[1] * S, markerPixelSize(cam, o.X, markerSize, S), markerId(o.id), color, 255);
  }
}

function markerId(id) {
  return parseInt(String(id).replace(/[^0-9]/g, ''), 10) || 0;
}

// Diagnostic overlay drawn with the SOLVED camera + mirror: predicted reflection
// positions, solved panel outline, solved normal, and asset contact.
function drawCheckOverlay(raster, solved, frameObs, S, markerSize, truthAsset) {
  const cam = solved.camera;
  const mirror = solved.mirror;
  const quad = mirrorQuad(mirror).map((p) => scaledProject(cam, p, S));
  if (quad.every(Boolean)) {
    raster.strokePolygon(quad, [105, 219, 124, 255], 2);
    // Normal arrow.
    const { n } = mirrorFrame(mirror);
    const c = scaledProject(cam, mirror.center, S);
    const tip = scaledProject(
      cam,
      [mirror.center[0] + n[0] * 0.5, mirror.center[1] + n[1] * 0.5, mirror.center[2] + n[2] * 0.5],
      S,
    );
    if (c && tip) {
      raster.line(c, tip, [255, 205, 90, 255], 2);
      raster.disc(tip.x, tip.y, 3, [255, 205, 90, 255]);
    }
  }
  // Predicted vs observed reflection pixels.
  const vc = virtualCamera(cam, mirror);
  for (const o of frameObs.observations) {
    if (o.kind !== 'mirror' || !o.pixel) continue;
    const pr = projectPoint(vc, o.X);
    if (!pr) continue;
    const obsPx = { x: o.pixel[0] * S, y: o.pixel[1] * S };
    const predPx = { x: pr.x * S, y: pr.y * S };
    raster.line(obsPx, predPx, [255, 212, 59, 220], 1);
    const ringR = markerPixelSize(cam, o.X, markerSize, S) * 0.62;
    raster.ring(predPx.x, predPx.y, ringR, [105, 219, 124, 255], 1.5);
  }
  // Asset contact: truth ring + solved cross.
  const trueContact = scaledProject(cam, truthAsset.contact, S);
  const solvedContact = scaledProject(cam, solved.asset.contact, S);
  if (trueContact) raster.ring(trueContact.x, trueContact.y, 9, [105, 219, 124, 255], 2);
  if (solvedContact) {
    raster.line({ x: solvedContact.x - 6, y: solvedContact.y }, { x: solvedContact.x + 6, y: solvedContact.y }, [255, 107, 107, 255], 2);
    raster.line({ x: solvedContact.x, y: solvedContact.y - 6 }, { x: solvedContact.x, y: solvedContact.y + 6 }, [255, 107, 107, 255], 2);
  }
}

// ------------------------- clips -------------------------

function clipFrames() {
  const frames = [];
  // Clip 1: the glass panel pans and yaws while the camera and physical
  // markers stay put. Reflections shift, several leave the panel → they must be
  // rejected as tie-points.
  const n1 = 16;
  for (let i = 0; i < n1; i++) {
    const t = i / (n1 - 1);
    frames.push({
      clip: 'mirror-arc',
      index: i,
      mirror: {
        center: [-0.20 + 0.40 * t, 1.02, 0.0],
        yaw: (-7 + 14 * t) * DEG,
        tilt: (0.4 + 1.6 * t) * DEG,
        width: 1.5,
        height: 2.0,
      },
      camera: null, // use scene truth
      missingDirect: [],
      assetContact: [0.42, 0, 0.95],
    });
  }
  // Clip 2: direct board occluded for part of the shot. Only floor + ledge
  // remain visible; the mirror constraint must let the camera be recovered.
  const n2 = 10;
  const boardIds = Array.from({ length: 16 }, (_, i) => `M${String(i + 1).padStart(2, '0')}`);
  for (let i = 0; i < n2; i++) {
    const t = i / (n2 - 1);
    frames.push({
      clip: 'missing-direct',
      index: i,
      mirror: {
        center: [0.04 * Math.sin(t * Math.PI), 1.02, 0.0],
        yaw: 1.5 * DEG * Math.sin(t * Math.PI),
        tilt: 0.8 * DEG,
        width: 1.5,
        height: 2.0,
      },
      camera: null,
      missingDirect: i >= 2 ? boardIds : [],
      assetContact: [0.42, 0, 0.95],
    });
  }
  return frames;
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const base = createDefaultScene();
  const S = WIDTH / base.width;
  const markerSize = 0.11; // metres; large enough to be readable in the plate
  const frames = clipFrames();
  const records = [];
  let failures = 0;

  for (let f = 0; f < frames.length; f++) {
    const spec = frames[f];
    const camera = spec.camera ?? base.cameraTruth;
    const mirror = { ...base.mirror, ...spec.mirror, centerSource: 'constraint' };
    const scene = makeFrameScene(base, { camera, mirror });
    const seed = 100 + f;
    const obs = observe(scene, { noisePx: NOISE_PX, seed, missingDirect: spec.missingDirect });

    // Plate (synthetic footage with direct markers + mirror reflections).
    const raster = createRaster(WIDTH, HEIGHT);
    drawPlate(raster, camera, mirror, obs, S, markerSize);
    const file = `${spec.clip}-${String(spec.index).padStart(2, '0')}.png`;
    await writeFile(path.join(OUT, file), encodePNG(WIDTH, HEIGHT, raster.rgb));

    // Solve frame with the known panel constraint (a scene constraint, not
    // the answer: the plane orientation/depth is still estimated).
    const constraint = {
      center: mirror.center,
      yaw: mirror.yaw,
      tilt: mirror.tilt,
      width: mirror.width,
      height: mirror.height,
    };
    const screenPoint = projectPoint(camera, spec.assetContact);
    const t0 = performance.now();
    const result = runPipeline({
      scene,
      observationOptions: { noisePx: NOISE_PX, seed, missingDirect: spec.missingDirect },
      mirrorConstraint: constraint,
      refit: true,
      assetPlacement: { screenPoint: [screenPoint.x, screenPoint.y] },
    });
    const ms = performance.now() - t0;

    // Check overlay written on top of the plate.
    const overlay = createRaster(WIDTH, HEIGHT);
    overlay.rgb.set(raster.rgb);
    drawCheckOverlay(overlay, result, obs, S, markerSize, {
      contact: [spec.assetContact[0], 0, spec.assetContact[2]],
    });
    const checkFile = `check-${spec.clip}-${String(spec.index).padStart(2, '0')}.png`;
    await writeFile(path.join(OUT, checkFile), encodePNG(WIDTH, HEIGHT, overlay.rgb));

    // Per-frame ground truth for every marker (direct + reflection) and the
    // solved comparison.
    const truthMarkers = scene.markers.map((m) => {
      const validity = reflectionValidity(camera, mirror, m.X);
      const mirroredX = reflectPoint(mirror, m.X);
      return {
        id: m.id,
        group: m.group,
        X: m.X,
        mirroredX,
        directPixel: obs.observations.find((o) => o.id === m.id && o.kind === 'direct' && o.pixel)?.pixel ?? null,
        mirrorPixel: validity.ok && validity.proj ? [validity.proj.x, validity.proj.y] : null,
        mirrorValid: validity.ok,
        mirrorReasons: validity.reasons,
        hitPoint: validity.H,
      };
    });

    const validMirror = mirrorObservations(obs);
    const invalidUsed = validMirror.filter((o) => !reflectionValidity(result.camera, result.mirror, o.X).ok);
    const contactErrorM = dist(result.asset.contact, [spec.assetContact[0], 0, spec.assetContact[2]]);
    const checks = {
      projectionRmsPx: result.branches.best.rmsPx,
      cameraCenterErrorM: result.grading.cameraCenterErrorM,
      cameraRotationErrorDeg: result.grading.cameraRotationErrorDeg,
      focalErrorPx: result.grading.focalErrorPx,
      mirrorNormalErrorDeg: result.grading.mirrorNormalErrorDeg,
      mirrorPlaneDistanceErrorM: result.grading.mirrorPlaneDistanceErrorM,
      contactErrorM,
      reflectionDet: result.branches.best.detVirtual,
      reflectionValidForAsset: result.reflectionCheck.ok,
      validReflectionCount: validMirror.length,
      rejectedReflectionCount: obs.rejected.filter((r) => r.kind === 'mirror').length,
      invalidReflectionUsed: invalidUsed.length,
      cameraSource: result.cameraSource,
      directCount: directObservations(obs).length,
    };

    const ok =
      checks.projectionRmsPx < 2.5 &&
      checks.cameraCenterErrorM < 0.05 &&
      checks.cameraRotationErrorDeg < 1.5 &&
      checks.mirrorNormalErrorDeg < 0.6 &&
      checks.mirrorPlaneDistanceErrorM < 0.01 &&
      checks.contactErrorM < 0.02 &&
      checks.reflectionDet === -1 &&
      checks.invalidReflectionUsed === 0 &&
      checks.validReflectionCount >= 6 &&
      checks.cameraSource !== 'truth-fallback';

    if (!ok) failures++;
    records.push({
      clip: spec.clip,
      index: spec.index,
      file,
      checkFile,
      seed,
      elapsedMs: Number(ms.toFixed(1)),
      ok,
      truth: {
        camera: { C: camera.C, R: camera.R, f: camera.f, cx: camera.cx, cy: camera.cy, width: camera.width, height: camera.height },
        mirror: { center: mirror.center, yaw: mirror.yaw, tilt: mirror.tilt, width: mirror.width, height: mirror.height },
        markers: truthMarkers,
        asset: { contact: [spec.assetContact[0], 0, spec.assetContact[2]], screenPoint: [screenPoint.x, screenPoint.y] },
      },
      solved: {
        camera: { C: result.camera.C, R: result.camera.R, f: result.camera.f },
        mirror: { center: result.mirror.center, yaw: result.mirror.yaw, tilt: result.mirror.tilt, width: result.mirror.width, height: result.mirror.height },
        assetContact: result.asset.contact,
        panelSource: result.mirrorSolve?.panelSource ?? null,
        cameraSource: result.cameraSource,
        rejectedMirror: obs.rejected.filter((r) => r.kind === 'mirror').map((r) => ({ id: r.id, reason: r.reason })),
        diagnostics: result.diagnostics.map((d) => ({ level: d.level, message: d.message })),
      },
      checks,
    });
    process.stdout.write(
      `${file}  refl ${fmt(checks.projectionRmsPx)} px  cam ${fmt(checks.cameraCenterErrorM, 4)} m/` +
      `${fmt(checks.cameraRotationErrorDeg, 3)}°  n ${fmt(checks.mirrorNormalErrorDeg, 3)}°  ` +
      `plane ${fmt(checks.mirrorPlaneDistanceErrorM, 4)} m  contact ${fmt(checks.contactErrorM, 4)} m  ` +
      `valid ${checks.validReflectionCount}  rej ${checks.rejectedReflectionCount}  ${ok ? 'OK' : 'FAIL'}\n`,
    );
  }

  const byClip = {};
  for (const r of records) {
    const c = (byClip[r.clip] ??= { frames: 0, ok: 0, worst: {} });
    c.frames++;
    if (r.ok) c.ok++;
    for (const [k, v] of Object.entries(r.checks)) {
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      c.worst[k] = Math.max(c.worst[k] ?? -Infinity, v);
    }
  }

  const manifest = {
    generator: 'scripts/clips.mjs',
    description: '合成片段：每帧含可直接观测的标记与镜中反射，且带镜面/相机真值。镜面在 first clip 中移动，second clip 中直接标板被遮挡。',
    width: WIDTH,
    height: HEIGHT,
    cameraNative: { width: base.width, height: base.height },
    noisePx: NOISE_PX,
    markerSizeM: markerSize,
    clips: {
      'mirror-arc': '镜面面板沿 X 平移并偏航；反射点会移出有限面板并被剔除。',
      'missing-direct': '直接标板在第 3 帧起被遮挡，仅剩地面/台座直接标记，依赖镜面约束增广求解相机。',
    },
    frameCount: records.length,
    failures,
    summary: byClip,
    frames: records,
  };
  await writeFile(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));
  console.log(`\nWrote ${records.length} frames x2 (plate + check) and manifest.json to ${OUT}`);
  if (failures) {
    console.error(`${failures} frame(s) failed the ground-truth checks.`);
    process.exitCode = 1;
  } else {
    console.log('All frames passed the ground-truth checks.');
  }
}

function fmt(v, d = 3) {
  return Number.isFinite(v) ? v.toFixed(d) : '—';
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
