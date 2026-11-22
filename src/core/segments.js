import { projectPoint, unprojectPixel } from './camera.js';
import {
  makeMirror,
  mirrorFrame,
  mirrorQuad,
  reflectPoint,
  reflectionValidity,
  virtualCamera,
  intersectRayPlane,
} from './mirror.js';
import { solveDirectCamera, solveMirrorPlane, rmsOf } from './solver.js';
import { levenbergMarquardt } from './optimize.js';
import { observeTimelineFrame, cornerWorld } from './timelineScene.js';
import {
  add,
  addScaled,
  clamp,
  dist,
  dot,
  median,
  scale,
  sub,
  DEG,
  RAD,
} from './math.js';

// Multi-segment calibration project for a movable mirror (可移动镜面的多段标定).
//
// Model: the camera is locked off; the mirror is re-posed between setups, so
// each timeline segment carries its own mirror plane (yaw/tilt/d). All
// segments share one physical mirror frame (共享镜框): panel size plus the
// in-plane anchor, which is observed through panel-corner markers and locked
// across segments. Observations are recorded separately from the two sides of
// the mirror frame (左侧/右侧标板).
//
// Two solvers are implemented and compared (对照后选定):
//   - solveProjectLinked: per-segment plane solves stitched by the shared
//     frame anchor (median of per-frame corner observations). CHOSEN: a bad
//     segment stays isolated and every segment can be re-estimated alone.
//   - solveProjectJoint: one LM over all segment planes + the shared anchor.
//     Statistically fine on clean data, but a poisoned segment bends the
//     shared anchor and thereby every other segment's panel placement.

// ------------------------- project model -------------------------

export function createSegmentProject(scene, { boundaries = [12] } = {}) {
  const segments = [];
  let start = 0;
  const cuts = [...boundaries, scene.frameCount];
  cuts.forEach((end, i) => {
    segments.push({
      id: `S${i}`,
      label: `区段 ${String.fromCharCode(65 + i)}`,
      range: [start, Math.min(end, scene.frameCount) - 1],
      calibration: null,
      attempted: false,
      invalid: null,
      lastGood: null,
      staleSince: null,
    });
    start = end;
  });
  return {
    frameCount: scene.frameCount,
    sharedFrame: {
      ...scene.sharedFrame,
      locked: true,
      anchorObserved: null,
      anchorSource: 'constraint',
    },
    segments,
    camera: null,
    cameraSource: null,
    cameraRmsPx: null,
    observations: new Map(),
    layers: buildLayers(segments),
    edits: [],
    seq: 0,
    kinkTolerancePx: 3,
    kinkAngleToleranceDeg: 1.5,
    asset: scene.asset,
    diagnostics: [],
    eventsSynced: false,
  };
}

function buildLayers(segments) {
  const layers = [];
  for (const seg of segments) {
    layers.push(
      { id: `L-${seg.id}-direct`, kind: 'direct', segmentId: seg.id, label: `${seg.label}·直接层`, isolated: false },
      { id: `L-${seg.id}-mirror`, kind: 'mirror', segmentId: seg.id, label: `${seg.label}·镜面层`, isolated: false },
      { id: `L-${seg.id}-asset`, kind: 'asset', segmentId: seg.id, label: `${seg.label}·合成物体层`, isolated: false },
    );
  }
  return layers;
}

export function logEdit(project, edit) {
  project.edits.push({ seq: ++project.seq, ...edit });
}

export function syncSceneEvents(project, scene) {
  if (project.eventsSynced) return;
  for (const e of scene.events) logEdit(project, e);
  project.eventsSynced = true;
}

export function segmentAtFrame(project, frame) {
  return project.segments.find((s) => frame >= s.range[0] && frame <= s.range[1]) ?? null;
}

export function boundariesOf(project) {
  return project.segments.slice(1).map((s) => s.range[0]);
}

// ------------------------- observation recording (两侧分别录入) -------------------------

function freshEntry(frame) {
  return {
    frame,
    sides: {
      left: { direct: [], mirror: [] },
      right: { direct: [], mirror: [] },
    },
    corners: [],
    distractors: [],
    missing: [],
    rejected: [],
    recordedSides: [],
  };
}

// Record one side's observations for one frame. Corner (frame) markers are
// shared evidence and recorded with either side. Chirality-reversed
// distractors are rejected geometrically: a "direct" point whose pixel falls
// inside the mirror frame but whose 3D position is behind the mirror plane.
export function recordSideObservations(project, scene, frame, side) {
  const obs = observeTimelineFrame(scene, frame);
  let entry = project.observations.get(frame);
  if (!entry) {
    entry = freshEntry(frame);
    project.observations.set(frame, entry);
  }
  if (!entry.recordedSides.includes(side)) entry.recordedSides.push(side);

  for (const o of obs.observations) {
    if (o.kind === 'corner') {
      if (o.pixel && !entry.corners.some((c) => c.id === o.id)) {
        entry.corners.push({ id: o.id, pixel: o.pixel });
      } else if (!o.pixel && !entry.missing.some((m) => m.id === o.id && m.kind === 'corner')) {
        entry.missing.push(o);
      }
      continue;
    }
    if (o.side !== side) continue;
    if (o.status === 'missing') {
      if (!entry.missing.some((m) => m.id === o.id && m.kind === o.kind)) entry.missing.push(o);
      continue;
    }
    if (!o.pixel) continue;
    if (o.kind === 'direct') {
      if (isChiralityDistractor(project, scene, o)) {
        entry.distractors.push({
          ...o,
          reason: '手性相反干扰标记：像素落在镜框内，但物点位于镜面背面（直接内容，非反射）',
        });
      } else {
        entry.sides[side].direct.push(o);
      }
    } else if (o.kind === 'mirror') {
      entry.sides[side].mirror.push(o);
    }
  }
  for (const r of obs.rejected) {
    if (r.side === side) entry.rejected.push(r);
  }
  return entry;
}

export function recordAllFrames(project, scene) {
  for (let f = 0; f < scene.frameCount; f++) {
    recordSideObservations(project, scene, f, 'left');
    recordSideObservations(project, scene, f, 'right');
  }
}

function restMirrorOf(project) {
  const sf = project.sharedFrame;
  return makeMirror({
    center: sf.anchorObserved ?? sf.anchor,
    yaw: sf.restYaw,
    tilt: sf.restTilt,
    width: sf.width,
    height: sf.height,
  });
}

export function isChiralityDistractor(project, scene, o) {
  const rest = restMirrorOf(project);
  const quad = mirrorQuad(rest)
    .map((p) => projectPoint(scene.camera, p))
    .filter(Boolean)
    .map((p) => [p.x, p.y]);
  if (quad.length !== 4) return false;
  if (!pointInPolygon2D(o.pixel, expandPolygon(quad, 1.12))) return false;
  const { n } = mirrorFrame(rest);
  return dot(n, sub(o.X, rest.center)) < 0;
}

function expandPolygon(poly, factor) {
  const c = poly.reduce((acc, p) => [acc[0] + p[0] / poly.length, acc[1] + p[1] / poly.length], [0, 0]);
  return poly.map((p) => [c[0] + (p[0] - c[0]) * factor, c[1] + (p[1] - c[1]) * factor]);
}

export function pointInPolygon2D(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// ------------------------- pooling -------------------------

export function pooledObservations(project, frames, kind) {
  const out = [];
  for (const f of frames) {
    const entry = project.observations.get(f);
    if (!entry) continue;
    for (const side of ['left', 'right']) out.push(...entry.sides[side][kind]);
  }
  return out;
}

function evidenceBySide(project, frames) {
  const ev = { left: null, right: null };
  for (const f of frames) {
    const entry = project.observations.get(f);
    if (!entry) continue;
    for (const side of ['left', 'right']) {
      const n = entry.sides[side].direct.length + entry.sides[side].mirror.length;
      if (n > 0) ev[side] = ev[side] ? [Math.min(ev[side][0], f), Math.max(ev[side][1], f)] : [f, f];
    }
  }
  return ev;
}

// Frames a segment should be solved from: recorded frames inside its range,
// cut at the first mirror-move event strictly inside the range (a new setup
// begins there and belongs to another segment).
export function defaultSolveFrames(project, segId) {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg) return [];
  const moves = project.edits.filter((e) => e.type === 'mirror-move').map((e) => e.frame);
  const innerMove = moves.find((m) => m > seg.range[0] && m <= seg.range[1]);
  const hi = innerMove != null ? innerMove - 1 : seg.range[1];
  const frames = [];
  for (let f = seg.range[0]; f <= hi; f++) {
    if (project.observations.has(f)) frames.push(f);
  }
  return frames;
}

// ------------------------- solving -------------------------

export function solveSharedCamera(project, scene) {
  const frames = [...project.observations.keys()].sort((a, b) => a - b);
  const pool = pooledObservations(project, frames, 'direct');
  const res = solveDirectCamera(pool, { width: scene.width, height: scene.height });
  if (!res.ok) {
    project.diagnostics.push({ level: 'error', stage: 'camera', message: res.error });
    return res;
  }
  project.camera = res.camera;
  project.cameraSource = 'direct-pooled';
  project.cameraRmsPx = res.rmsPx;
  logEdit(project, { type: 'solve-camera', note: `共享相机：${pool.length} 个直接观测，RMS ${res.rmsPx.toFixed(3)} px` });
  return res;
}

function anchorOf(project) {
  return project.sharedFrame.anchorObserved ?? project.sharedFrame.anchor;
}

// Re-estimate a single segment (单独重估). Only this segment's calibration is
// touched; every other segment keeps its solved state byte-for-byte.
export function solveSegment(project, scene, segId, { frames } = {}) {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg) return { ok: false, reason: 'no-such-segment' };
  seg.attempted = true;
  if (!project.camera) {
    const camRes = solveSharedCamera(project, scene);
    if (!camRes.ok) return { ok: false, reason: 'camera-failed' };
  }
  const pool = frames ?? defaultSolveFrames(project, segId);
  const mirrorObs = pooledObservations(project, pool, 'mirror');
  if (mirrorObs.length < 6) {
    seg.calibration = null;
    project.diagnostics.push({
      level: 'warn',
      stage: 'segments',
      message: `${seg.label} 镜面观测仅 ${mirrorObs.length} 个（求解不足），可补充观测后单独重估。`,
    });
    logEdit(project, { type: 'solve', segmentId: segId, note: `${seg.label} 求解不足：镜面观测 ${mirrorObs.length} 个` });
    return { ok: false, reason: 'underdetermined', mirrorObs: mirrorObs.length };
  }
  const constraint = project.sharedFrame.locked
    ? { center: anchorOf(project), width: project.sharedFrame.width, height: project.sharedFrame.height }
    : null;
  const res = solveMirrorPlane(project.camera, mirrorObs, {
    constraint,
    width: scene.width,
    height: scene.height,
  });
  if (!res.ok) {
    seg.calibration = null;
    project.diagnostics.push({ level: 'warn', stage: 'segments', message: `${seg.label} 镜面求解失败：${res.error}` });
    return { ok: false, reason: 'solve-failed', error: res.error };
  }
  seg.calibration = {
    yaw: res.yaw,
    tilt: res.tilt,
    d: res.d,
    mirror: res.mirror,
    rmsPx: res.rmsPx,
    solveFrames: pool,
    evidence: evidenceBySide(project, pool),
    seq: ++project.seq,
  };
  seg.lastGood = seg.calibration;
  seg.invalid = null;
  logEdit(project, {
    type: 'solve',
    segmentId: segId,
    note: `${seg.label} 标定：${pool.length} 帧 / ${mirrorObs.length} 个镜中观测，RMS ${res.rmsPx.toFixed(3)} px`,
  });
  return { ok: true, calibration: seg.calibration };
}

// CHOSEN IMPLEMENTATION (选定实现): per-segment solves stitched by the shared
// mirror-frame anchor. The anchor is estimated independently of the segment
// planes (corner ray ∩ plane per frame, then a component-wise median), so a
// failed or poisoned segment cannot bend the frame that other segments hang
// on, and every segment stays individually re-estimable.
export function solveProjectLinked(project, scene) {
  syncSceneEvents(project, scene);
  project.diagnostics = [];
  const camRes = project.camera ? { ok: true } : solveSharedCamera(project, scene);
  if (!camRes.ok) return { ok: false };
  const results = {};
  for (const seg of project.segments) {
    results[seg.id] = solveSegment(project, scene, seg.id);
  }
  observeSharedAnchor(project, scene);
  if (project.sharedFrame.locked && project.sharedFrame.anchorObserved) {
    for (const seg of project.segments) {
      if (!seg.calibration) continue;
      const c = seg.calibration;
      seg.calibration = {
        ...c,
        mirror: mirrorFromAnchorPlane(anchorOf(project), c.yaw, c.tilt, c.d, project.sharedFrame.width, project.sharedFrame.height),
      };
    }
  }
  markStaleSegments(project);
  return { ok: true, results };
}

// Shared frame anchor from panel-corner observations: for each evidence frame
// with all four corners visible, intersect the corner rays with that frame's
// solved plane and average; the per-frame estimates are then median-combined
// so a few bad frames (occlusion edges, a poisoned segment) cannot move it.
export function observeSharedAnchor(project, scene) {
  const perFrame = [];
  for (const seg of project.segments) {
    if (!seg.calibration) continue;
    const mirror = seg.calibration.mirror;
    for (const f of seg.calibration.solveFrames) {
      const entry = project.observations.get(f);
      if (!entry || entry.corners.length < 4) continue;
      const pts = [];
      for (const c of entry.corners) {
        const ray = unprojectPixel(project.camera, c.pixel[0], c.pixel[1], 1);
        const hit = intersectRayPlane(ray.origin, ray.dir, mirror);
        if (hit) pts.push(hit.point);
      }
      if (pts.length === 4) {
        perFrame.push([
          pts.reduce((s, p) => s + p[0], 0) / 4,
          pts.reduce((s, p) => s + p[1], 0) / 4,
          pts.reduce((s, p) => s + p[2], 0) / 4,
        ]);
      }
    }
  }
  if (!perFrame.length) {
    project.sharedFrame.anchorSource = 'constraint';
    project.diagnostics.push({ level: 'info', stage: 'frame', message: '镜框角点观测不足，共享锚使用场景约束值。' });
    return null;
  }
  const anchor = [
    median(perFrame.map((p) => p[0])),
    median(perFrame.map((p) => p[1])),
    median(perFrame.map((p) => p[2])),
  ];
  project.sharedFrame.anchorObserved = anchor;
  project.sharedFrame.anchorSource = 'observed';
  logEdit(project, { type: 'anchor', note: `共享镜框锚由 ${perFrame.length} 帧角点观测估计（逐分量中位数）` });
  return anchor;
}

export function mirrorFromAnchorPlane(anchor, yaw, tilt, d, width, height) {
  const probe = makeMirror({ center: anchor, yaw, tilt, width, height });
  const { n } = mirrorFrame(probe);
  const center = add(anchor, scale(n, d - dot(n, anchor)));
  return makeMirror({ center, yaw, tilt, width, height, centerSource: 'shared-anchor' });
}

// COMPARISON IMPLEMENTATION (对照): one joint LM over all segment planes and
// the shared anchor. Pure: reads recorded observations, returns a result
// object without mutating the project (except ensuring the shared camera).
export function solveProjectJoint(project, scene) {
  syncSceneEvents(project, scene);
  if (!project.camera) solveSharedCamera(project, scene);
  const cam = project.camera;
  const segs = project.segments;
  const pools = segs.map((s) => defaultSolveFrames(project, s.id));
  const mirrorObs = pools.map((pool) => pooledObservations(project, pool, 'mirror'));
  const sf = project.sharedFrame;
  const anchor0 = anchorOf(project);

  const inits = mirrorObs.map((obs) => {
    if (obs.length < 6) return null;
    const r = solveMirrorPlane(cam, obs, {
      constraint: { center: anchor0, width: sf.width, height: sf.height },
      width: scene.width,
      height: scene.height,
    });
    return r.ok ? r : null;
  });

  const cornerFrames = [];
  for (const [fi, entry] of project.observations) {
    if (entry.corners.length < 4) continue;
    const segIdx = segs.indexOf(segmentAtFrame(project, fi));
    if (segIdx < 0) continue;
    if (!pools[segIdx].includes(fi)) continue;
    cornerFrames.push({ frame: fi, segIdx, corners: entry.corners });
  }

  const p0 = [anchor0[0], anchor0[1], anchor0[2]];
  for (let i = 0; i < segs.length; i++) {
    if (inits[i]) p0.push(inits[i].yaw, inits[i].tilt, inits[i].d);
    else {
      const rest = restMirrorOf(project);
      const { n } = mirrorFrame(rest);
      p0.push(sf.restYaw, sf.restTilt, dot(n, anchor0));
    }
  }

  const build = (theta) => {
    const anchor = [theta[0], theta[1], theta[2]];
    const mirrors = segs.map((_, i) =>
      mirrorFromAnchorPlane(anchor, theta[3 + 3 * i], theta[4 + 3 * i], theta[5 + 3 * i], sf.width, sf.height),
    );
    return { anchor, mirrors };
  };

  const residuals = (theta) => {
    const { anchor, mirrors } = build(theta);
    const res = [];
    for (let i = 0; i < segs.length; i++) {
      const vc = virtualCamera(cam, mirrors[i]);
      for (const o of mirrorObs[i]) {
        const pr = projectPoint(vc, o.X);
        if (!pr) res.push(1e5, 1e5);
        else res.push(pr.x - o.pixel[0], pr.y - o.pixel[1]);
      }
    }
    for (const cf of cornerFrames) {
      const mirror = mirrors[cf.segIdx];
      for (const c of cf.corners) {
        const def = scene.corners.find((k) => k.id === c.id);
        const world = cornerWorld(mirror, def);
        const pr = projectPoint(cam, world);
        if (!pr) res.push(1e5, 1e5);
        else res.push(pr.x - c.pixel[0], pr.y - c.pixel[1]);
      }
    }
    return res;
  };

  const fit = levenbergMarquardt({ params: p0, residuals, maxIterations: 200, lambda0: 1e-3 });
  const { anchor, mirrors } = build(fit.params);
  return {
    anchor,
    mirrors,
    rmsPx: Math.sqrt(fit.cost / Math.max(1, mirrorObs.flat().length + cornerFrames.length * 4)),
    iterations: fit.iterations,
    converged: fit.converged,
    pools,
  };
}

// Run both solvers on identical recorded observations and grade them against
// the truth timeline. Returns the metrics table and the chosen implementation.
export function compareSegmentSolvers(project, scene) {
  const cloneA = deserializeProject(serializeProject(project));
  const linkedRes = solveProjectLinked(cloneA, scene);
  const cloneB = deserializeProject(serializeProject(project));
  const joint = solveProjectJoint(cloneB, scene);

  const grade = (mirrorsById, anchor) => {
    const perSegment = cloneA.segments.map((seg, i) => {
      const mirror = mirrorsById(seg.id, i);
      if (!mirror) return { id: seg.id, label: seg.label, ok: false };
      const cal = cloneA.segments[i].calibration;
      const frames = cal?.solveFrames?.length ? cal.solveFrames : defaultSolveFrames(cloneA, seg.id);
      const mid = frames.length ? frames[Math.floor(frames.length / 2)] : seg.range[0];
      const truth = scene.mirrorAt(mid);
      return { id: seg.id, label: seg.label, ok: true, ...segmentMetrics(mirror, truth) };
    });
    return {
      perSegment,
      anchorErrM: anchor ? dist(anchor, scene.sharedFrame.anchor) : null,
    };
  };

  const linked = grade(
    (id) => cloneA.segments.find((s) => s.id === id)?.calibration?.mirror ?? null,
    cloneA.sharedFrame.anchorObserved ?? cloneA.sharedFrame.anchor,
  );
  const jointG = grade((_, i) => joint.mirrors[i] ?? null, joint.anchor);

  return {
    linked: { ...linked, ok: linkedRes.ok !== false },
    joint: { ...jointG, rmsPx: joint.rmsPx, converged: joint.converged },
    choice: 'linked',
    rationale:
      '选定共享镜框锚衔接的分段解：干净数据上两者精度相当；但联合拟合把全部区段耦合进同一个目标，' +
      '一段观测被污染或求解不足时会把误差摊进共享锚，进而扭曲其它区段的面板位置。衔接解的锚由角点观测' +
      '独立估计（逐帧取中位数），失效只隔离在出事的区段内，且每个区段都可单独重估。',
  };
}

function segmentMetrics(mirror, truthMirror) {
  const n1 = mirrorFrame(mirror).n;
  const n2 = mirrorFrame(truthMirror).n;
  const normalErrDeg = Math.acos(clamp(dot(n1, n2), -1, 1)) * RAD;
  const planeDistErrM = Math.abs(dot(n1, truthMirror.center) - dot(n1, mirror.center));
  const centreErrM = dist(mirror.center, truthMirror.center);
  return { normalErrDeg, planeDistErrM, centreErrM };
}

// ------------------------- validation / isolation -------------------------

// Reflection-direction audit used when a segment is dragged back (拖回后核对
// 反射方向): the virtual camera must stay improper (det = -1), the camera and
// the observed markers must be on the reflective side, and the calibration
// must still explain its own mirrored observations.
export function validateSegmentMirror(mirror, camera, mirrorObs = []) {
  const checks = {};
  const nums = [...mirror.center, mirror.yaw, mirror.tilt, mirror.width, mirror.height];
  checks.finite = nums.every(Number.isFinite);
  const vc = virtualCamera(camera, mirror);
  const R = vc.R;
  const det =
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  checks.detVirtual = det;
  checks.chiralityOk = Math.abs(det + 1) < 1e-6;
  const { n } = mirrorFrame(mirror);
  checks.camSide = dot(n, sub(camera.C, mirror.center));
  checks.cameraInFront = checks.camSide > 1e-6;
  let front = 0;
  const errs = [];
  for (const o of mirrorObs) {
    if (dot(n, sub(o.X, mirror.center)) > 1e-6) front++;
    const pr = projectPoint(vc, o.X);
    if (pr) errs.push(Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]));
  }
  checks.objectsInFront = mirrorObs.length ? front / mirrorObs.length : 1;
  checks.rmsPx = errs.length ? rmsOf(errs) : null;
  const reasons = [];
  if (!checks.finite) reasons.push('参数含非有限值');
  if (!checks.chiralityOk) reasons.push(`虚拟相机手性错误（det=${det.toFixed(3)}，应为 -1）`);
  if (!checks.cameraInFront) reasons.push('相机位于镜面背面');
  if (checks.objectsInFront < 0.9) reasons.push('多数观测标记位于镜面背面');
  if (checks.rmsPx != null && checks.rmsPx > 3) reasons.push(`反射 RMS ${checks.rmsPx.toFixed(2)} px 超限`);
  return { ok: reasons.length === 0, checks, reasons };
}

// A segment's mirror parameters failed: isolate ONLY the composite layers
// that depend on that segment (仅隔离依赖该面的合成层).
export function invalidateSegment(project, segId, reason = '镜面参数失效') {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg) return null;
  if (seg.calibration) {
    seg.lastGood = seg.calibration;
    seg.calibration = {
      ...seg.calibration,
      yaw: seg.calibration.yaw + 25 * DEG,
      mirror: { ...seg.calibration.mirror, yaw: seg.calibration.mirror.yaw + 25 * DEG },
    };
  }
  seg.invalid = reason;
  for (const layer of project.layers) {
    if (layer.segmentId === segId) layer.isolated = true;
  }
  logEdit(project, { type: 'isolate', segmentId: segId, note: `${seg.label} ${reason}；仅隔离依赖该区段的合成层` });
  return seg;
}

// Drag the segment back (拖回): restore its last good calibration, but only
// after the reflection direction checks pass.
export function restoreSegment(project, scene, segId) {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg || !seg.lastGood) return { ok: false, reason: '无可拖回的标定' };
  const mirrorObs = pooledObservations(project, seg.lastGood.solveFrames ?? [], 'mirror');
  const check = validateSegmentMirror(seg.lastGood.mirror, project.camera, mirrorObs);
  if (!check.ok) {
    return { ok: false, reasons: check.reasons, check };
  }
  seg.calibration = seg.lastGood;
  seg.invalid = null;
  for (const layer of project.layers) {
    if (layer.segmentId === segId) layer.isolated = false;
  }
  logEdit(project, {
    type: 'restore',
    segmentId: segId,
    note: `${seg.label} 拖回后核对反射方向通过：det=-1、相机与标记均在正面、RMS ${check.rmsPx?.toFixed(3) ?? '—'} px`,
  });
  markStaleSegments(project);
  return { ok: true, check };
}

export function layersForSegment(project, segId) {
  return project.layers.filter((l) => l.segmentId === segId);
}

// ------------------------- staleness -------------------------

// A segment goes stale when the mirror moved inside its range after the
// segment's calibration evidence ends, or when a boundary drag pushed its
// evidence frames outside its range.
export function markStaleSegments(project) {
  for (const seg of project.segments) {
    seg.staleSince = null;
    if (!seg.calibration) continue;
    const evMax = Math.max(...seg.calibration.solveFrames);
    const move = project.edits
      .filter((e) => e.type === 'mirror-move' && e.frame > evMax && e.frame <= seg.range[1])
      .sort((a, b) => a.frame - b.frame)[0];
    const outOfRange = seg.calibration.solveFrames.some((f) => f < seg.range[0] || f > seg.range[1]);
    if (move) seg.staleSince = move.frame;
    else if (outOfRange) seg.staleSince = seg.range[0];
  }
}

export function segmentStatus(project, segId) {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg) return 'empty';
  if (seg.invalid) return 'invalid';
  if (!seg.calibration) return seg.attempted ? 'underdetermined' : 'empty';
  if (seg.staleSince != null) return 'stale';
  return 'ok';
}

// Everything a frame points back to (指回正确的相机和镜面状态): its segment,
// the shared camera, that segment's mirror, and the isolation state of the
// layers that would render it.
export function frameState(project, frame) {
  const seg = segmentAtFrame(project, frame);
  return {
    frame,
    segmentId: seg?.id ?? null,
    camera: project.camera,
    mirror: seg?.calibration?.mirror ?? null,
    status: seg ? segmentStatus(project, seg.id) : 'empty',
    isolatedLayers: seg ? project.layers.filter((l) => l.segmentId === seg.id && l.isolated).map((l) => l.id) : [],
  };
}

// Drag the mid-segment boundary (拖动中段边界). Frames are re-assigned to
// segments; calibrations stay put, and anything whose evidence no longer
// covers its range is flagged stale.
export function setSegmentBoundary(project, boundaryIndex, frame, { log = true } = {}) {
  const lo = project.segments[boundaryIndex].range[0] + 1;
  const hi =
    boundaryIndex + 2 < project.segments.length
      ? project.segments[boundaryIndex + 2].range[0] - 1
      : project.frameCount - 1;
  const f = clamp(Math.round(frame), lo, hi);
  project.segments[boundaryIndex].range[1] = f - 1;
  project.segments[boundaryIndex + 1].range[0] = f;
  if (log) logEdit(project, { type: 'boundary-drag', frame: f, note: `中段边界拖至第 ${f} 帧` });
  markStaleSegments(project);
  return f;
}

export function splitSegment(project, segId, frame) {
  const idx = project.segments.findIndex((s) => s.id === segId);
  if (idx < 0) return null;
  const seg = project.segments[idx];
  if (frame <= seg.range[0] || frame > seg.range[1]) return null;
  const newSeg = {
    id: `S${Math.max(...project.segments.map((s) => parseInt(s.id.slice(1), 10))) + 1}`,
    label: `区段 ${String.fromCharCode(65 + project.segments.length)}`,
    range: [frame, seg.range[1]],
    calibration: null,
    attempted: false,
    invalid: null,
    lastGood: null,
    staleSince: null,
  };
  seg.range = [seg.range[0], frame - 1];
  project.segments.splice(idx + 1, 0, newSeg);
  project.layers.push(...buildLayers([newSeg]));
  logEdit(project, { type: 'split', frame, segmentId: newSeg.id, note: `在第 ${frame} 帧分裂出 ${newSeg.label}` });
  markStaleSegments(project);
  return newSeg;
}

// ------------------------- crossing-line kink (穿过镜框的直线折角) -------------------------

export function fitLine2D(points) {
  const c = points.reduce((acc, p) => [acc[0] + p[0] / points.length, acc[1] + p[1] / points.length], [0, 0]);
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of points) {
    const dx = p[0] - c[0];
    const dy = p[1] - c[1];
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const tr = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const l1 = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  let dir;
  if (Math.abs(sxy) > 1e-12) {
    dir = [l1 - syy, sxy];
  } else {
    dir = sxx >= syy ? [1, 0] : [0, 1];
  }
  const len = Math.hypot(dir[0], dir[1]) || 1;
  return { point: c, dir: [dir[0] / len, dir[1] / len] };
}

export function intersectLines2D(L1, L2) {
  const [x1, y1] = L1.point;
  const [d1x, d1y] = L1.dir;
  const [x2, y2] = L2.point;
  const [d2x, d2y] = L2.dir;
  const denom = d1x * d2y - d1y * d2x;
  if (Math.abs(denom) < 1e-12) return null;
  const t = ((x2 - x1) * d2y - (y2 - y1) * d2x) / denom;
  return [x1 + t * d1x, y1 + t * d1y];
}

export function angleBetweenLines2D(L1, L2) {
  const d = Math.abs(L1.dir[0] * L2.dir[0] + L1.dir[1] * L2.dir[1]);
  return Math.acos(clamp(d, -1, 1)) * RAD;
}

export function linePlaneIntersection(a, b, mirror) {
  const { n } = mirrorFrame(mirror);
  const d = sub(b, a);
  const denom = dot(n, d);
  if (Math.abs(denom) < 1e-12) return null;
  const t = dot(n, sub(mirror.center, a)) / denom;
  return add(a, scale(d, t));
}

// The calibration rail is surveyed straight and touches the glass inside the
// panel, so its observed direct image and observed reflected image meet at
// the glass contact point. If a segment reuses stale calibration (沿用过期
// 标定), the junction predicted from its plane misses the observed junction:
// the straight line shows a kink (折角) at the mirror frame.
export function measureCrossingLineKink(project, scene, frame) {
  const seg = segmentAtFrame(project, frame);
  if (!seg?.calibration || !project.camera) return { ok: false, frame, reason: '区段无标定' };
  const entry = project.observations.get(frame);
  if (!entry) return { ok: false, frame, reason: '该帧未录入观测' };
  const railDirect = [];
  const railMirror = [];
  for (const side of ['left', 'right']) {
    railDirect.push(...entry.sides[side].direct.filter((o) => o.group === 'rail').map((o) => o.pixel));
    railMirror.push(...entry.sides[side].mirror.filter((o) => o.group === 'rail').map((o) => o.pixel));
  }
  if (railDirect.length < 2 || railMirror.length < 2) {
    return { ok: false, frame, reason: '直轨观测不足（遮挡或出画）' };
  }
  const L1 = fitLine2D(railDirect);
  const L2obs = fitLine2D(railMirror);
  const J_obs = intersectLines2D(L1, L2obs);
  const mirror = seg.calibration.mirror;
  const P = linePlaneIntersection(scene.rail.a, scene.rail.b, mirror);
  if (!P) return { ok: false, frame, reason: '直轨与镜面平面不相交' };
  const pr = projectPoint(project.camera, P);
  if (!pr) return { ok: false, frame, reason: '交点在相机后方' };
  const reflPx = scene.rail.samples
    .map((s) => projectPoint(project.camera, reflectPoint(mirror, s.X)))
    .filter(Boolean)
    .map((p) => [p.x, p.y]);
  const L2pred = fitLine2D(reflPx);
  // Detection signals (the raw junction intersection is ill-conditioned when
  // the direct and reflected images meet at a shallow angle, so the kink is
  // decided by well-conditioned line fits):
  //   angleDeg — observed vs predicted reflected-line direction;
  //   offsetPx — perpendicular offset between those two lines;
  //   gapPx    — perpendicular distance from the predicted junction to the
  //              observed reflected line (the visible 折角 at the frame edge).
  const angleDeg = angleBetweenLines2D(L2obs, L2pred);
  const n2 = [-L2obs.dir[1], L2obs.dir[0]];
  const offsetPx = Math.abs(
    (L2pred.point[0] - L2obs.point[0]) * n2[0] + (L2pred.point[1] - L2obs.point[1]) * n2[1],
  );
  const gapPx = Math.abs((pr.x - L2obs.point[0]) * n2[0] + (pr.y - L2obs.point[1]) * n2[1]);
  const kink = angleDeg > project.kinkAngleToleranceDeg || offsetPx > project.kinkTolerancePx;
  return {
    ok: true,
    frame,
    segmentId: seg.id,
    gapPx,
    offsetPx,
    angleDeg,
    J_obs,
    J_pred: [pr.x, pr.y],
    P,
    L1,
    L2obs,
    L2pred,
    kink,
  };
}

export function scanKinks(project, scene) {
  const out = new Map();
  for (let f = 0; f < project.frameCount; f++) {
    if (!project.observations.has(f)) continue;
    out.set(f, measureCrossingLineKink(project, scene, f));
  }
  return out;
}

// Attribute a detected kink to the edit that made the segment stale
// (定位到那次编辑): the mirror move the segment's calibration never covered.
export function localizeStaleEdit(project, scene) {
  const kinks = scanKinks(project, scene);
  const bad = [...kinks.values()].filter((k) => k.ok && k.kink);
  if (!bad.length) return { found: false, kinks };
  const first = Math.min(...bad.map((k) => k.frame));
  const seg = segmentAtFrame(project, first);
  const cal = seg?.calibration;
  const evMax = cal ? Math.max(...cal.solveFrames) : -1;
  const evMin = cal ? Math.min(...cal.solveFrames) : -1;
  const edit = project.edits
    .filter((e) => e.type === 'mirror-move' && e.frame > evMax && e.frame <= first)
    .sort((a, b) => b.frame - a.frame)[0] ?? null;
  const sideExit = project.edits
    .filter((e) => e.type === 'side-exit' && e.frame <= first)
    .sort((a, b) => b.frame - a.frame)[0] ?? null;
  const k = kinks.get(first);
  const message =
    `第 ${first} 帧起穿过镜框的直线出现折角（缺口 ${k.gapPx.toFixed(1)} px，夹角 ${k.angleDeg.toFixed(1)}°）：` +
    `${seg?.label ?? '?'} 沿用第 ${evMin}–${evMax} 帧标定` +
    (edit ? `，未覆盖第 ${edit.frame} 帧的镜面角度改变` : '') +
    (sideExit ? `；${sideExit.side === 'right' ? '右' : '左'}侧标记第 ${sideExit.frame} 帧起移出取景框缺观测` : '') +
    `。定位到编辑：${edit ? `#${edit.seq} ${edit.note}` : '（无对应编辑记录）'}`;
  return { found: true, frame: first, segmentId: seg?.id ?? null, edit, sideExit, gapPx: k.gapPx, angleDeg: k.angleDeg, kinks, message };
}

// ------------------------- virtual-image error (虚像误差) -------------------------

function virtualImageErrorWith(camera, mirror, entry) {
  if (!camera || !mirror || !entry) return null;
  const vc = virtualCamera(camera, mirror);
  const errs = [];
  for (const side of ['left', 'right']) {
    for (const o of entry.sides[side].mirror) {
      const pr = projectPoint(vc, o.X);
      if (pr) errs.push(Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]));
    }
  }
  return errs.length ? rmsOf(errs) : null;
}

// Virtual-image error of one frame under one segment's calibration.
export function virtualImageError(project, frame, segId) {
  const seg = project.segments.find((s) => s.id === segId);
  if (!seg?.calibration || !project.camera) return null;
  const entry = project.observations.get(frame);
  return virtualImageErrorWith(project.camera, seg.calibration.mirror, entry);
}

// Drag back to the same frame from both sides (从两侧拖回同一帧): the frames
// bracketing a boundary, each evaluated with BOTH segments' calibrations. The
// owning segment must explain its frame; the other segment's error is the
// measurable signature of the mirror move.
export function boundaryVirtualImageError(project, boundaryIndex) {
  const left = project.segments[boundaryIndex];
  const right = project.segments[boundaryIndex + 1];
  if (!left || !right) return null;
  const fL = left.range[1];
  const fR = right.range[0];
  return {
    boundary: fR,
    leftFrame: {
      frame: fL,
      ownSegment: left.id,
      own: virtualImageError(project, fL, left.id),
      other: virtualImageError(project, fL, right.id),
    },
    rightFrame: {
      frame: fR,
      ownSegment: right.id,
      own: virtualImageError(project, fR, right.id),
      other: virtualImageError(project, fR, left.id),
    },
  };
}

// ------------------------- persistence (保存分段工程) -------------------------

export function serializeProject(project) {
  return JSON.stringify({
    version: 1,
    frameCount: project.frameCount,
    sharedFrame: project.sharedFrame,
    segments: project.segments,
    camera: project.camera,
    cameraSource: project.cameraSource,
    cameraRmsPx: project.cameraRmsPx,
    observations: [...project.observations.entries()],
    layers: project.layers,
    edits: project.edits,
    seq: project.seq,
    kinkTolerancePx: project.kinkTolerancePx,
    kinkAngleToleranceDeg: project.kinkAngleToleranceDeg,
    asset: project.asset,
    diagnostics: project.diagnostics,
    eventsSynced: project.eventsSynced,
  });
}

export function deserializeProject(json) {
  const data = typeof json === 'string' ? JSON.parse(json) : json;
  const project = {
    frameCount: data.frameCount,
    sharedFrame: data.sharedFrame,
    segments: data.segments,
    camera: data.camera,
    cameraSource: data.cameraSource,
    cameraRmsPx: data.cameraRmsPx,
    observations: new Map(data.observations),
    layers: data.layers,
    edits: data.edits,
    seq: data.seq,
    kinkTolerancePx: data.kinkTolerancePx,
    kinkAngleToleranceDeg: data.kinkAngleToleranceDeg ?? 1.5,
    asset: data.asset,
    diagnostics: data.diagnostics ?? [],
    eventsSynced: Boolean(data.eventsSynced),
  };
  return project;
}
