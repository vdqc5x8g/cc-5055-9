import { cameraFromLookAt, projectPoint } from './camera.js';
import {
  makeMirror,
  virtualCamera,
  reflectionValidity,
  mirrorFrame,
} from './mirror.js';
import { add, gaussian, mulberry32, sub, dot } from './math.js';

// Default studio scene. World units are metres. Y is up, the mirror faces -Z.
export function createDefaultScene() {
  const mirror = makeMirror({
    center: [0, 1.02, 0.0],
    yaw: 0,
    tilt: 0.01,
    width: 1.5,
    height: 2.0,
  });
  const cameraTruth = cameraFromLookAt({
    position: [1.05, 1.5, 2.15],
    target: [-0.45, 1.0, 0.15],
    upHint: [0, 1, 0],
    f: 1150,
    width: 1280,
    height: 720,
  });
  return {
    name: '正厅平面镜',
    mirror,
    cameraTruth,
    width: 1280,
    height: 720,
    markers: buildMarkers(mirror),
    virtualAsset: buildVirtualAsset(mirror),
    noisePx: 0.5,
  };
}

// Marker identities are shared between direct and mirrored observations. The
// mirror image must never be treated as a second physical co-located marker.
export function buildMarkers(mirror) {
  // A physical target board facing the camera plus a floor ruler. Both sit
  // between the camera and the mirror so their reflections are also visible.
  const groups = [
    {
      id: 'board',
      label: '后墙标板',
      origin: [-0.62, 1.05, 0.55],
      du: [0.5, 0, 0],
      dv: [0, 0.9, 0],
      nu: 4,
      nv: 4,
    },
    {
      id: 'floor',
      label: '地面标尺',
      origin: [-0.95, 0.05, 1.15],
      du: [1.35, 0, 0],
      dv: [0, 0, -0.85],
      nu: 4,
      nv: 2,
    },
    {
      id: 'ledge',
      label: '镜侧台座',
      origin: [0.82, 0.72, 0.72],
      du: [0.32, 0, 0],
      dv: [0, 0.5, 0],
      nu: 2,
      nv: 2,
    },
  ];
  const markers = [];
  let idx = 0;
  for (const g of groups) {
    for (let iv = 0; iv < g.nv; iv++) {
      for (let iu = 0; iu < g.nu; iu++) {
        const t = g.nu > 1 ? iu / (g.nu - 1) : 0.5;
        const s = g.nv > 1 ? iv / (g.nv - 1) : 0.5;
        const X = [
          g.origin[0] + g.du[0] * t + g.dv[0] * s,
          g.origin[1] + g.du[1] * t + g.dv[1] * s,
          g.origin[2] + g.du[2] * t + g.dv[2] * s,
        ];
        markers.push({
          id: `M${String(idx + 1).padStart(2, '0')}`,
          group: g.id,
          groupLabel: g.label,
          X,
          size: 0.035,
          seenDirect: true,
          seenMirror: true,
        });
        idx++;
      }
    }
  }
  return markers;
}

// A "virtual" CG object whose placement the tool must solve for and then
// verify against the mirror. The truth is used only for grading.
export function buildVirtualAsset(mirror) {
  const base = [0.42, 0.0, 0.95];
  return {
    name: '虚拟方盒',
    position: base,
    yaw: -0.35,
    scale: [0.22, 0.34, 0.22],
    // Contact frame is the bottom centre; y = 0 is the floor.
    contact: [base[0], 0, base[2]],
  };
}

export function cameraOnGrid(scene, overrides = {}) {
  const W = overrides.width ?? scene.width;
  const H = overrides.height ?? scene.height;
  const pos = overrides.position ?? scene.cameraTruth.C;
  const target = overrides.target ?? add(pos, [
    scene.cameraTruth.R[2][0],
    scene.cameraTruth.R[2][1],
    scene.cameraTruth.R[2][2],
  ]);
  return cameraFromLookAt({
    position: pos,
    target,
    upHint: [0, 1, 0],
    f: overrides.f ?? scene.cameraTruth.f,
    width: W,
    height: H,
  });
}

function inFrame(cam, px, margin = 0) {
  return (
    px.x >= -margin &&
    px.x <= cam.width + margin &&
    px.y >= -margin &&
    px.y <= cam.height + margin
  );
}

export function observe(scene, {
  camera = scene.cameraTruth,
  noisePx = scene.noisePx,
  seed = 7,
  mirror = scene.mirror,
  missingDirect = [],
  extraDirectMissingFraction = 0,
  hideMirrorOutside = true,
} = {}) {
  const rand = mulberry32(seed);
  const vc = virtualCamera(camera, mirror);
  const observations = [];
  const rejected = [];
  const missingSet = new Set(missingDirect);
  for (const m of scene.markers) {
    const directProj = projectPoint(camera, m.X);
    const directVisible = directProj && inFrame(camera, directProj);
    const directMissing =
      missingSet.has(m.id) ||
      (extraDirectMissingFraction > 0 && rand() < extraDirectMissingFraction);
    if (directMissing) {
      observations.push({
        id: m.id,
        kind: 'direct',
        X: m.X,
        pixel: null,
        status: 'missing',
        reason: '直接标记缺失（遮挡/污损）',
      });
    } else if (!directVisible) {
      rejected.push({
        id: m.id,
        kind: 'direct',
        reason: '直接观测不在画面内',
      });
    } else {
      const pixel = jitter(directProj, noisePx, rand);
      observations.push({
        id: m.id,
        kind: 'direct',
        X: m.X,
        pixel,
        truePixel: [directProj.x, directProj.y],
        status: 'ok',
      });
    }

    const validity = reflectionValidity(camera, mirror, m.X);
    if (!validity.ok) {
      if (hideMirrorOutside) {
        rejected.push({
          id: m.id,
          kind: 'mirror',
          reason: validity.reasons.join('；'),
          H: validity.H,
          detail: validity,
        });
      } else {
        // Explicitly show why an off-mirror point is not used: the pixel may
        // exist mathematically but is not a valid reflected observation.
        rejected.push({
          id: m.id,
          kind: 'mirror',
          reason: `已排除：${validity.reasons.join('；')}`,
          H: validity.H,
          detail: validity,
        });
      }
      continue;
    }
    const jittered = jitter(validity.proj, noisePx, rand);
    observations.push({
      id: m.id,
      kind: 'mirror',
      X: m.X,
      pixel: jittered,
      truePixel: [validity.proj.x, validity.proj.y],
      H: validity.H,
      status: 'ok',
    });
  }
  return { observations, rejected, virtualCamera: vc, camera, mirror };
}

function jitter(proj, sigma, rand) {
  return [proj.x + sigma * gaussian(rand), proj.y + sigma * gaussian(rand)];
}

export function directObservations(obs) {
  return obs.observations.filter((o) => o.kind === 'direct' && o.pixel);
}

export function mirrorObservations(obs) {
  return obs.observations.filter((o) => o.kind === 'mirror' && o.pixel);
}

export function correspondencesById(obs) {
  const byId = new Map();
  for (const o of obs.observations) {
    if (!o.pixel) continue;
    if (!byId.has(o.id)) byId.set(o.id, { id: o.id, X: o.X });
    byId.get(o.id)[o.kind] = o.pixel;
  }
  return [...byId.values()];
}

export function markerDistanceStats(scene) {
  const ids = scene.markers.map((m) => m.id);
  return ids;
}
