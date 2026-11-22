import { cameraFromLookAt, projectPoint } from './camera.js';
import {
  makeMirror,
  mirrorFrame,
  reflectPoint,
  reflectionValidity,
} from './mirror.js';
import { gaussian, mulberry32 } from './math.js';

const DEG = Math.PI / 180;

// Mirror setups (poses) of the movable mirror. The panel rotates about its own
// centre, so the panel centre is a fixed world point shared by every pose:
// that point is the shared mirror-frame anchor (共享镜框锚).
export const TIMELINE_POSES = {
  A: { yaw: -5 * DEG, tilt: 0.8 * DEG },
  B: { yaw: 6.5 * DEG, tilt: 1.2 * DEG },
  B2: { yaw: 11 * DEG, tilt: 1.6 * DEG },
};

// Default truth timeline: pose A for frames < 12, pose B for 12..17, and a
// further mirror move (B -> B2) at frame 18, after the right-side markers
// have left the viewfinder.
export function defaultPoseAt(frame) {
  if (frame < 12) return 'A';
  if (frame < 18) return 'B';
  return 'B2';
}

// A movable-mirror, multi-segment shot. The camera is locked off; the mirror
// is re-posed between setups. Calibration markers sit on both sides of the
// mirror frame (左侧/右侧标板), a straight calibration rail crosses the frame
// (穿过镜框的直线), corner markers ride on the panel frame, an actor
// temporarily occludes the mirror frame, and chirality-reversed distractor
// markers appear at the virtual-image positions of real markers.
export function createTimelineScene({
  frameCount = 24,
  noisePx = 0.5,
  poseAt = defaultPoseAt,
  occlusionSpan = [14, 17],
  rightExitFrame = 18,
  distractorSpan = [8, 11],
  seedBase = 1000,
} = {}) {
  const width = 1280;
  const height = 720;
  const camera = cameraFromLookAt({
    position: [0, 1.5, 2.9],
    target: [0, 1.0, 0.15],
    upHint: [0, 1, 0],
    f: 900,
    width,
    height,
  });

  const sharedFrame = {
    anchor: [0, 1.02, 0],
    width: 2.3,
    height: 1.9,
    restYaw: 0,
    restTilt: 0.01,
  };

  const markers = [];
  const addBoard = (side, origin, du, dv, nu, nv) => {
    let k = markers.filter((m) => m.side === side).length;
    for (let iv = 0; iv < nv; iv++) {
      for (let iu = 0; iu < nu; iu++) {
        const t = nu > 1 ? iu / (nu - 1) : 0.5;
        const s = nv > 1 ? iv / (nv - 1) : 0.5;
        k++;
        markers.push({
          id: `${side === 'left' ? 'L' : 'R'}${k}`,
          side,
          group: `board-${side}`,
          X: [
            origin[0] + du[0] * t + dv[0] * s,
            origin[1] + du[1] * t + dv[1] * s,
            origin[2] + du[2] * t + dv[2] * s,
          ],
        });
      }
    }
  };
  // 镜框两侧标板：左侧组与右侧组分别录入观测。
  addBoard('left', [-1.35, 0.75, 0.6], [0.3, 0, 0], [0, 0.6, 0], 3, 3);
  addBoard('right', [1.05, 0.75, 0.6], [0.3, 0, 0], [0, 0.6, 0], 3, 3);

  // Straight calibration rail touching the glass: 穿过镜框的直线。One end
  // rests on the mirror plane inside the panel; the rail is surveyed, so its
  // direct and reflected images must meet at the glass contact point.
  const rail = {
    a: [1.3, 1.5, 1.0],
    b: [0.35, 1.15, 0.0],
    samples: [],
  };
  const railTs = [0.18, 0.3, 0.42, 0.54, 0.66, 0.78];
  railTs.forEach((t, i) => {
    rail.samples.push({
      id: `T${i + 1}`,
      t,
      side: 'right',
      group: 'rail',
      X: [
        rail.a[0] + (rail.b[0] - rail.a[0]) * t,
        rail.a[1] + (rail.b[1] - rail.a[1]) * t,
        rail.a[2] + (rail.b[2] - rail.a[2]) * t,
      ],
    });
  });

  // Corner markers glued to the panel frame (local offsets in half sizes).
  // Their 3D positions are NOT surveyed — they only anchor the shared frame
  // through 2D observations (ray ∩ plane), so a moving panel stays honest.
  const corners = [
    { id: 'C0', su: -1, sv: 1 },
    { id: 'C1', su: 1, sv: 1 },
    { id: 'C2', su: 1, sv: -1 },
    { id: 'C3', su: -1, sv: -1 },
  ];

  const restMirror = makeMirror({
    center: sharedFrame.anchor,
    yaw: sharedFrame.restYaw,
    tilt: sharedFrame.restTilt,
    width: sharedFrame.width,
    height: sharedFrame.height,
  });

  const mirrorAt = (frame) => {
    const pose = TIMELINE_POSES[poseAt(frame)];
    return makeMirror({
      center: sharedFrame.anchor.slice(),
      yaw: pose.yaw,
      tilt: pose.tilt,
      width: sharedFrame.width,
      height: sharedFrame.height,
    });
  };

  // Chirality-reversed distractors: physical markers placed at the virtual
  // image positions of real left-board markers (behind the glass). In the
  // footage they sit at the same image positions as true reflections, but
  // they are direct (det = +1) content and must never enter a reflection solve.
  const mirrorA = makeMirror({
    center: sharedFrame.anchor,
    yaw: TIMELINE_POSES.A.yaw,
    tilt: TIMELINE_POSES.A.tilt,
    width: sharedFrame.width,
    height: sharedFrame.height,
  });
  const distractorSources = markers.filter((m) => m.side === 'left' && ['L1', 'L5', 'L9'].includes(m.id));
  const distractors = distractorSources.map((m, i) => ({
    id: `X${i + 1}`,
    side: 'left',
    group: 'distractor',
    mirrorsId: m.id,
    X: reflectPoint(mirrorA, m.X),
  }));

  const events = [
    { frame: 12, type: 'mirror-move', note: '镜面换位：姿态 A → B（镜框共享锚不变）' },
    { frame: occlusionSpan[0], type: 'occlusion-start', note: '演员遮住镜框：镜中观测与镜框角点缺失' },
    { frame: occlusionSpan[1] + 1, type: 'occlusion-end', note: '演员离开镜框' },
    { frame: distractorSpan[0], type: 'distractor-start', note: '手性相反干扰标记出现（同位置、直接内容）' },
    { frame: distractorSpan[1] + 1, type: 'distractor-end', note: '干扰标记移除' },
    { frame: rightExitFrame, type: 'side-exit', side: 'right', note: '右侧标记移出取景框' },
    { frame: rightExitFrame, type: 'mirror-move', note: '镜面角度继续改变：姿态 B → B2（右侧已缺观测）' },
  ];

  return {
    name: '可移动镜面多段镜头',
    frameCount,
    width,
    height,
    camera,
    sharedFrame,
    restMirror,
    markers,
    rail,
    corners,
    distractors,
    events,
    occlusionSpan,
    rightExitFrame,
    distractorSpan,
    noisePx,
    seedBase,
    poseAt,
    mirrorAt,
    asset: {
      name: '虚拟方盒',
      position: [0.42, 0.0, 0.95],
      yaw: -0.35,
      scale: [0.22, 0.34, 0.22],
      contact: [0.42, 0, 0.95],
    },
  };
}

// World position of a panel corner marker at a given mirror pose.
export function cornerWorld(mirror, corner) {
  const { u, v } = mirrorFrame(mirror);
  return [
    mirror.center[0] + u[0] * corner.su * (mirror.width / 2) + v[0] * corner.sv * (mirror.height / 2),
    mirror.center[1] + u[1] * corner.su * (mirror.width / 2) + v[1] * corner.sv * (mirror.height / 2),
    mirror.center[2] + u[2] * corner.su * (mirror.width / 2) + v[2] * corner.sv * (mirror.height / 2),
  ];
}

// Deterministic per-frame observation set. Recording the same frame twice
// (e.g. left side then right side) yields identical pixels.
export function observeTimelineFrame(scene, frame, { noisePx = scene.noisePx } = {}) {
  const rand = mulberry32(scene.seedBase + frame * 7919);
  const jitter = (p) => [p.x + noisePx * gaussian(rand), p.y + noisePx * gaussian(rand)];
  const mirror = scene.mirrorAt(frame);
  const cam = scene.camera;
  const occluded = frame >= scene.occlusionSpan[0] && frame <= scene.occlusionSpan[1];
  const rightGone = frame >= scene.rightExitFrame;
  const observations = [];
  const rejected = [];

  const pushMarker = (m) => {
    const gone = m.side === 'right' && m.group !== 'rail' && rightGone;
    const dp = projectPoint(cam, m.X);
    if (gone) {
      observations.push({ id: m.id, kind: 'direct', side: m.side, group: m.group, X: m.X, pixel: null, status: 'missing', reason: '右侧标记移出取景框' });
    } else if (dp) {
      observations.push({ id: m.id, kind: 'direct', side: m.side, group: m.group, X: m.X, pixel: jitter(dp), truePixel: [dp.x, dp.y], status: 'ok' });
    }
    const validity = reflectionValidity(cam, mirror, m.X);
    if (gone) {
      observations.push({ id: m.id, kind: 'mirror', side: m.side, group: m.group, X: m.X, pixel: null, status: 'missing', reason: '右侧标记移出取景框' });
    } else if (occluded) {
      observations.push({ id: m.id, kind: 'mirror', side: m.side, group: m.group, X: m.X, pixel: null, status: 'missing', reason: '演员遮挡镜框' });
    } else if (!validity.ok) {
      rejected.push({ id: m.id, kind: 'mirror', side: m.side, reason: validity.reasons.join('；') });
    } else {
      observations.push({ id: m.id, kind: 'mirror', side: m.side, group: m.group, X: m.X, pixel: jitter(validity.proj), truePixel: [validity.proj.x, validity.proj.y], H: validity.H, status: 'ok' });
    }
  };

  for (const m of scene.markers) pushMarker(m);
  for (const t of scene.rail.samples) pushMarker(t);

  for (const c of scene.corners) {
    const X = cornerWorld(mirror, c);
    const p = projectPoint(cam, X);
    if (occluded) {
      observations.push({ id: c.id, kind: 'corner', side: 'frame', pixel: null, status: 'missing', reason: '演员遮挡镜框' });
    } else if (p) {
      observations.push({ id: c.id, kind: 'corner', side: 'frame', pixel: jitter(p), truePixel: [p.x, p.y], status: 'ok' });
    }
  }

  if (frame >= scene.distractorSpan[0] && frame <= scene.distractorSpan[1]) {
    for (const d of scene.distractors) {
      const p = projectPoint(cam, d.X);
      if (!p) continue;
      observations.push({
        id: d.id,
        kind: 'direct',
        side: d.side,
        group: 'distractor',
        mirrorsId: d.mirrorsId,
        X: d.X,
        pixel: jitter(p),
        truePixel: [p.x, p.y],
        status: 'ok',
      });
    }
  }

  return { frame, observations, rejected, truth: { mirror, camera: cam } };
}
