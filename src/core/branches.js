import {
  cross,
  dist,
  dot,
  mat3Mul,
  mat3T,
  normalize,
  rotationErrorDeg,
  sub,
} from './math.js';
import { projectPoint, makeCamera } from './camera.js';
import {
  virtualCamera,
  reflectionValidity,
  reflectionHit,
  mirrorFrame,
  insideMirrorBounds,
  reflectPoint,
} from './mirror.js';
import { dltCamera, decomposeCameraMatrix } from './dlt.js';
import { rmsOf } from './solver.js';

function det3(A) {
  return (
    A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
  );
}

// A branch is a complete interpretation of the mirrored pixels:
//  - 'reflection': proper virtual camera (det = -1) + finite panel clipping
//  - 'direct-dup': treat mirrored pixels as direct observations of the same
//    physical markers (the invalid "co-located second marker set" model)
//  - 'proper-flip': enforce det = +1 on the virtual camera (a screen-space
//    horizontal flip dressed up as geometry)
//  - 'far-mirror'/'near-mirror': depth ambiguity candidates for the plane
// Each is scored by its reprojection RMS plus geometric validity flags.
export function evaluateBranches({ camera, mirror, mirrorObs, directObs = [], width, height, panel }) {
  const results = [];
  const vc = virtualCamera(camera, mirror);

  // 1. Reflection branch (correct model).
  {
    const entries = [];
    for (const o of mirrorObs) {
      if (!o.pixel) continue;
      const pr = projectPoint(vc, o.X);
      const validity = reflectionValidity(camera, mirror, o.X);
      entries.push({
        id: o.id,
        err: pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity,
        predicted: pr ? [pr.x, pr.y] : null,
        pixel: o.pixel,
        H: validity.H,
        inBounds: validity.inBounds,
        ok: validity.ok,
        reasons: validity.reasons,
      });
    }
    results.push({
      key: 'reflection',
      label: '反射分支（虚拟相机 det=-1）',
      validModel: true,
      detVirtual: vc.det,
      handedness: det3(vc.R),
      rmsPx: rmsOf(entries.filter((e) => e.ok).map((e) => e.err)),
      rmsAllPx: rmsOf(entries.map((e) => e.err)),
      entries,
      insidePanel: entries.filter((e) => e.inBounds).length,
      outsidePanel: entries.filter((e) => e.H && !e.inBounds).length,
      accepted: entries.filter((e) => e.ok).length,
    });
  }

  // 2. Naive "mirrored pixels are another set of the same markers" branch.
  {
    const obs = mirrorObs.filter((o) => o.pixel);
    if (obs.length >= 6) {
      const { P } = dltCamera(obs.map((o) => o.X), obs.map((o) => o.pixel));
      const dec = decomposeCameraMatrix(P, { width, height });
      const cam = { ...dec, cx: width / 2, cy: height / 2, width, height };
      const errs = obs.map((o) => {
        const pr = projectPoint(cam, o.X);
        return pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity;
      });
      results.push({
        key: 'direct-dup',
        label: '错误模型：把镜中点当同位置直接标记',
        validModel: false,
        camera: cam,
        detR: det3(cam.R),
        rmsPx: rmsOf(errs),
        centerError: dist(cam.C, camera.C),
        rotationErrorDeg: rotationErrorDeg(cam.R, camera.R),
        why: '镜中像是反射虚拟相机的投影；当成直接观测会得到一个不存在的相机，残差无法收敛。',
      });
    }
  }

  // 3. Proper-flip branch: force the virtual camera to be a rotation.
  {
    const R = vc.R.map((r) => r.slice());
    const det = det3(R);
    const Rproper = R.map((row, i) => row.map((v, j) => (j === 0 ? v * Math.sign(det || 1) : v)));
    const camProp = { ...vc, R: Rproper };
    const errs = mirrorObs.filter((o) => o.pixel).map((o) => {
      const pr = projectPoint(camProp, o.X);
      return pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity;
    });
    results.push({
      key: 'proper-flip',
      label: '错误模型：仅水平翻转（强制 det=+1）',
      validModel: false,
      detOriginal: det,
      detForced: det3(Rproper),
      rmsPx: rmsOf(errs),
      why: '强制手性为正等价于镜像翻转贴图；只在一个方向上巧合对齐，无法同时满足全部反射观测。',
    });
  }

  // 4. Depth candidates: plane at the same normal but h scaled.
  //    Reflections alone fix h exactly, so the far/near candidates must show a
  //    much worse residual: this is the depth branch test.
  {
    const { n } = mirrorFrame(mirror);
    const h = dot(n, camera.C) - dot(n, mirror.center);
    for (const factor of [0.75, 1.25]) {
      const d2 = dot(n, camera.C) - h * factor;
      const center = sub(mirror.center, [n[0] * h * (1 - factor), n[1] * h * (1 - factor), n[2] * h * (1 - factor)]);
      const cand = { ...mirror, center };
      const vc2 = virtualCamera(camera, cand);
      const errs = mirrorObs.filter((o) => o.pixel).map((o) => {
        const pr = projectPoint(vc2, o.X);
        return pr ? Math.hypot(pr.x - o.pixel[0], pr.y - o.pixel[1]) : Infinity;
      });
      results.push({
        key: `depth-${factor}`,
        label: `深度候选：镜面距离 ×${factor}`,
        validModel: factor === 1,
        rmsPx: rmsOf(errs),
        hCandidate: h * factor,
        why: '反射观测随镜面深度变化敏感；偏离真实深度会使反射点整体错位。',
      });
    }
  }

  const correct = results.find((r) => r.key === 'reflection');
  const alt = results.find((r) => r.key === 'proper-flip');
  return {
    results,
    best: correct,
    chiralityOk: correct.detVirtual < 0,
    reflectionAtLeastAsGood:
      correct.rmsPx <= (alt?.rmsPx ?? Infinity),
    summary: {
      reflectionRms: correct.rmsPx,
      properFlipRms: alt?.rmsPx ?? null,
      accepted: correct.accepted,
      outsidePanel: correct.outsidePanel,
      detVirtual: correct.detVirtual,
    },
  };
}
