import {
  createDefaultScene,
  observe,
  directObservations,
  mirrorObservations,
} from './scene.js';
import {
  solveDirectCamera,
  solveCameraFromMirror,
  solveMirrorPlane,
  refineJoint,
  augmentWithMirror,
  planarityOf,
} from './solver.js';
import { evaluateBranches } from './branches.js';
import {
  mirrorFrame,
  reflectionValidity,
  normalizeMirror,
} from './mirror.js';
import {
  placeAssetOnGround,
  assetReflectionCheck,
  flipVsReflection,
} from './asset.js';
import { buildCompositeFrame } from './composite.js';
import {
  dist,
  dot,
  rotationErrorDeg,
  sub,
} from './math.js';

// The whole tracking window state machine, kept UI-agnostic so the CLI and the
// browser app exercise exactly the same code.
export function runPipeline({
  scene = createDefaultScene(),
  camera = null,
  observationOptions = {},
  mirrorConstraint = null,
  fixF = false,
  fixMirror = false,
  assetPlacement = null,
  refit = true,
} = {}) {
  const obs = observe(scene, observationOptions);
  const direct = directObservations(obs);
  const mirrorObs = mirrorObservations(obs);
  const constraint = mirrorConstraint ? normalizeMirror(mirrorConstraint, scene.mirror) : null;
  const truth = {
    camera: scene.cameraTruth,
    mirror: scene.mirror,
  };
  const diagnostics = [];
  const cam = camera ?? scene.cameraTruth;

  // Surface observation-level exclusions: reflected rays that miss the finite
  // panel (or leave the frame) are NOT valid tie-points and never enter the
  // solve. Direct markers leaving frame are also reported.
  const rejectedMirror = obs.rejected.filter((r) => r.kind === 'mirror');
  const rejectedDirect = obs.rejected.filter((r) => r.kind === 'direct');
  if (rejectedMirror.length) {
    const byReason = {};
    for (const r of rejectedMirror) {
      const key = (r.reason || '未知原因').replace(/^已排除：/, '');
      byReason[key] = (byReason[key] ?? 0) + 1;
    }
    const parts = Object.entries(byReason).map(([k, v]) => `${v} 个${k}`);
    diagnostics.push({
      level: 'info',
      stage: 'mirror',
      message: `${rejectedMirror.length} 个镜中观测未作为有效观测参与求解（${parts.join('；')}）。`,
    });
  }
  if (rejectedDirect.length) {
    diagnostics.push({
      level: 'info',
      stage: 'camera',
      message: `${rejectedDirect.length} 个直接标记不在画面内，未参与直接求解。`,
    });
  }

  // 1. Direct camera solve. Coplanar marker sets cannot fix the camera on their
  //    own; if a mirror constraint is available, mirror observations supply the
  //    missing off-plane evidence (reflected object points are valid direct
  //    observations of the real camera).
  let camSolve = null;
  let solveObs = direct;
  let augmented = 0;
  const directPlanarity = planarityOf(direct.map((o) => o.X));
  const needAugment =
    constraint && mirrorObs.length >= 6 && (direct.length < 8 || directPlanarity.planar);
  if (needAugment) {
    solveObs = augmentWithMirror(direct, mirrorObs, constraint);
    augmented = solveObs.length - direct.length;
    diagnostics.push({
      level: 'info',
      stage: 'camera',
      message: `直接标记${direct.length < 8 ? '过少' : '共面'}（畸形标定）；用已知镜面约束把 ${augmented} 个镜中观测反射为直接证据参与相机求解。`,
    });
  }
  if (solveObs.length >= 6) {
    camSolve = solveDirectCamera(solveObs, {
      width: scene.width,
      height: scene.height,
      fixF,
      fPrior: fixF ? scene.cameraTruth.f : null,
    });
    if (!camSolve.ok) diagnostics.push({ level: 'error', message: camSolve.error, stage: 'camera' });
  } else {
    diagnostics.push({
      level: 'warn',
      stage: 'camera',
      message: `直接观测只有 ${direct.length} 个，无法单独求解相机。`,
    });
  }

  // 2. If direct solve unavailable but a mirror constraint exists, try mirror-only.
  let cameraSolved = camSolve?.ok ? camSolve.camera : null;
  let cameraSource = camSolve?.ok ? (augmented > 0 ? 'mirror-augmented' : 'direct') : null;
  if (!cameraSolved && constraint && mirrorObs.length >= 6) {
    const fromMirror = solveCameraFromMirror(mirrorObs, constraint, {
      width: scene.width,
      height: scene.height,
      fixF,
      fPrior: fixF ? scene.cameraTruth.f : null,
    });
    if (fromMirror.ok) {
      cameraSolved = fromMirror.camera;
      cameraSource = 'mirror-only';
      camSolve = { ...fromMirror, ok: true };
      diagnostics.push({
        level: 'info',
        stage: 'camera',
        message: '直接标记不足，改用镜面约束 + 镜中观测反解相机。',
      });
    } else {
      diagnostics.push({ level: 'error', stage: 'camera', message: fromMirror.error });
    }
  }

  if (!cameraSolved) {
    // Fall back to truth only for rendering a diagnosable failure; clearly flagged.
    cameraSolved = scene.cameraTruth;
    cameraSource = 'truth-fallback';
    diagnostics.push({
      level: 'error',
      stage: 'camera',
      message: '相机无法从观测求解，预览使用真值并标记为“未求解”。',
    });
  }

  // 3. Mirror plane solve.
  let mirrorSolve = null;
  if (mirrorObs.length >= 6) {
    mirrorSolve = solveMirrorPlane(cameraSolved, mirrorObs, {
      constraint,
      width: scene.width,
      height: scene.height,
    });
    if (!mirrorSolve.ok) {
      diagnostics.push({ level: 'error', stage: 'mirror', message: mirrorSolve.error });
    }
  } else {
    diagnostics.push({
      level: 'warn',
      stage: 'mirror',
      message: `镜面观测只有 ${mirrorObs.length} 个，无法求解镜面。`,
    });
  }

  let mirrorSolved = mirrorSolve?.ok ? mirrorSolve.mirror : (constraint ?? scene.mirror);
  if (!mirrorSolve?.ok) {
    diagnostics.push({ level: 'error', stage: 'mirror', message: '镜面无法从反射观测求解，使用约束值。' });
  }

  // 4. Joint refinement (camera + mirror plane) when both are available.
  let joint = null;
  if (refit && camSolve?.ok && mirrorSolve?.ok && direct.length >= 6 && mirrorObs.length >= 6) {
    const anchor = constraint?.center ?? mirrorSolve.mirror.center;
    joint = refineJoint({
      camera: cameraSolved,
      mirror: mirrorSolved,
      directObs: direct,
      mirrorObs,
      fixMirror,
      fixF,
      width: scene.width,
      height: scene.height,
      anchor,
    });
    cameraSolved = joint.camera;
    mirrorSolved = joint.mirror;
    cameraSource = `${cameraSource}+joint`;
  }

  // 5. Branch analysis on the solved configuration.
  const branches = evaluateBranches({
    camera: cameraSolved,
    mirror: mirrorSolved,
    mirrorObs,
    directObs: direct,
    width: scene.width,
    height: scene.height,
  });

  // 6. Mirror panel / validity audit.
  const audit = auditMirror({
    camera: cameraSolved,
    mirror: mirrorSolved,
    mirrorObs,
    scene,
    constraint,
  });
  diagnostics.push(...audit.diagnostics);

  // 7. Asset placement and composite.
  const assetBase = scene.virtualAsset;
  let placed = assetBase;
  if (assetPlacement?.groundPoint) {
    placed = placeAssetOnGround({ camera: cameraSolved, asset: assetBase, groundPoint: assetPlacement.groundPoint });
  } else if (assetPlacement?.screenPoint) {
    placed = placeAssetOnGround({ camera: cameraSolved, asset: assetBase, screenPoint: assetPlacement.screenPoint });
  }
  const composite = buildCompositeFrame({ camera: cameraSolved, mirror: mirrorSolved, asset: placed });
  const reflCheck = assetReflectionCheck(cameraSolved, mirrorSolved, placed);
  const flipCheck = flipVsReflection(cameraSolved, mirrorSolved, placed);
  if (!reflCheck.ok) {
    diagnostics.push({
      level: 'warn',
      stage: 'asset',
      message: `虚拟物体反射不可用：${reflCheck.reasons.join('；')}`,
    });
  }

  // 8. Truth grading (only available for the built-in synthetic shot).
  const grading = gradeAgainstTruth({
    camera: cameraSolved,
    mirror: mirrorSolved,
    truth,
    asset: placed,
  });

  return {
    scene,
    observations: obs,
    direct,
    mirrorObs,
    camera: cameraSolved,
    cameraSource,
    camSolve,
    mirror: mirrorSolved,
    mirrorSolve,
    joint,
    branches,
    audit,
    asset: placed,
    composite,
    reflectionCheck: reflCheck,
    flipCheck,
    fixF,
    augmented,
    constraint,
    constraints: describeConstraints({ constraint, fixF, scene, direct, mirrorObs, augmented }),
    rejected: obs.rejected,
    grading,
    diagnostics,
  };
}

// Human-readable list of the scene constraints actually adopted by this solve.
// This is what the UI shows as "采用的约束"; it never claims a constraint that
// was not supplied.
export function describeConstraints({ constraint, fixF, scene, direct, mirrorObs, augmented }) {
  const items = [];
  if (constraint) {
    items.push({
      key: 'mirror-panel',
      label: '镜面面板位置/尺寸（场景约束）',
      detail: `中心 (${constraint.center.map((v) => v.toFixed(2)).join(', ')}) m，` +
        `yaw ${(constraint.yaw * 180 / Math.PI).toFixed(1)}°，tilt ${(constraint.tilt * 180 / Math.PI).toFixed(1)}°，` +
        `${constraint.width.toFixed(2)}×${constraint.height.toFixed(2)} m`,
    });
  } else {
    items.push({
      key: 'mirror-plane',
      label: '镜面平面（仅由反射观测解算）',
      detail: '未提供面板位置约束；面内滑动不可观测，绝对距离按场景单位报告。',
    });
  }
  if (fixF) {
    items.push({ key: 'focal', label: '焦距固定（场景约束）', detail: `f = ${(scene.cameraTruth?.f ?? 0).toFixed(0)} px` });
  }
  if (augmented > 0) {
    items.push({
      key: 'augment',
      label: '镜面增广标定',
      detail: `直接标记不足/共面，用镜面约束把 ${augmented} 个镜中观测反射为直接证据。`,
    });
  }
  items.push({
    key: 'markers',
    label: '标点观测',
    detail: `直接 ${direct.length} 个，镜中有效 ${mirrorObs.length} 个。`,
  });
  return items;
}

export function auditMirror({ camera, mirror, mirrorObs, scene, constraint }) {
  const diagnostics = [];
  const { n } = mirrorFrame(mirror);
  const camSide = dot(n, sub(camera.C, mirror.center));
  if (camSide <= 0) {
    diagnostics.push({ level: 'error', stage: 'mirror', message: '相机位于镜面背面，反射几何不成立。' });
  }
  let outside = 0;
  let behindObject = 0;
  for (const o of mirrorObs) {
    const v = reflectionValidity(camera, mirror, o.X);
    if (!v.inBounds && v.H) outside++;
    if (v.objSide <= 0) behindObject++;
  }
  if (outside > 0) {
    diagnostics.push({
      level: 'warn',
      stage: 'mirror',
      message: `${outside} 个反射点落在镜面矩形外，已从有效观测中剔除。`,
    });
  }
  if (behindObject > 0) {
    diagnostics.push({
      level: 'warn',
      stage: 'mirror',
      message: `${behindObject} 个标记位于镜面背面，无有效反射路径。`,
    });
  }

  // Panel placement provenance.
  const source = mirror.centerSource;
  if (source === 'hit-centroid') {
    diagnostics.push({
      level: 'info',
      stage: 'mirror',
      message: '面板面内位置取自反射命中点质心（仅放置有限面板；面内滑动对反射不可观测）。',
    });
  } else if (source === 'constraint') {
    diagnostics.push({ level: 'info', stage: 'mirror', message: '面板位置由场景约束给定。' });
  }

  // Scale / metric evidence: refuse absolute claims if no metric constraint.
  const metricEvidence = Boolean(constraint?.center) || Boolean(constraint?.metric);
  if (!metricEvidence) {
    diagnostics.push({
      level: 'info',
      stage: 'scale',
      message: '未提供镜面位置/尺度约束：法线与面距可解，但绝对距离按场景单位报告，仅供参考。',
    });
  }
  return { outside, behindObject, camSide, metricEvidence, diagnostics };
}

export function gradeAgainstTruth({ camera, mirror, truth, asset }) {
  const camErr = dist(camera.C, truth.camera.C);
  const rotErr = rotationErrorDeg(camera.R, truth.camera.R);
  const fErr = Math.abs(camera.f - truth.camera.f);
  const n1 = mirrorFrame(mirror).n;
  const n2 = mirrorFrame(truth.mirror).n;
  const normalErrDeg = Math.acos(Math.max(-1, Math.min(1, dot(n1, n2)))) * (180 / Math.PI);
  const d1 = dot(n1, mirror.center);
  const d2 = dot(n2, truth.mirror.center);
  // Compare plane distance in a reflection-relevant way: signed distance of
  // the true mirror centre to the solved plane.
  const planeDist = Math.abs(dot(n1, truth.mirror.center) - d1);
  return {
    cameraCenterErrorM: camErr,
    cameraRotationErrorDeg: rotErr,
    focalErrorPx: fErr,
    mirrorNormalErrorDeg: normalErrDeg,
    mirrorPlaneDistanceErrorM: planeDist,
    note: '仅有合成片段才有真值；实拍不提供此评分。',
  };
}
