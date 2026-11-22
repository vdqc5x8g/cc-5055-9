#!/usr/bin/env node
// Headless assist for the multi-segment calibration editor (图形入口：
// npm start → http://localhost:4173/segments.html). Runs the whole segmented
// workflow on the synthetic movable-mirror shot and checks every guarantee:
// two-sided recording, shared-frame linked solve vs joint fit, isolation,
// per-segment re-estimation, save/load, boundary virtual-image error, and
// stale-calibration localisation via the crossing-line kink.

import { createTimelineScene } from '../src/core/timelineScene.js';
import {
  createSegmentProject,
  recordAllFrames,
  solveProjectLinked,
  solveSegment,
  compareSegmentSolvers,
  invalidateSegment,
  restoreSegment,
  segmentStatus,
  frameState,
  setSegmentBoundary,
  splitSegment,
  markStaleSegments,
  scanKinks,
  localizeStaleEdit,
  virtualImageError,
  boundaryVirtualImageError,
  serializeProject,
  deserializeProject,
  layersForSegment,
} from '../src/core/segments.js';
import { mirrorFrame } from '../src/core/mirror.js';
import { dist } from '../src/core/math.js';

const RAD = 180 / Math.PI;
let failures = 0;
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'OK  ' : 'FAIL'}  ${label}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
};
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const normalErr = (m1, m2) => {
  const a = mirrorFrame(m1).n;
  const b = mirrorFrame(m2).n;
  return Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * RAD;
};

const scene = createTimelineScene();
const project = createSegmentProject(scene, { boundaries: [12] });

console.log('多段标定编辑器 · 命令行自检（图形入口：npm start → /segments.html）');
console.log('='.repeat(78));

// ---- 1. two-sided recording + linked solve -------------------------------
console.log('\n[1] 从镜框两侧分别录入观测，共享镜框锚衔接分段解');
recordAllFrames(project, scene);
const sides = [0, 1].map(() => ({ left: 0, right: 0 }));
for (const [, entry] of project.observations) {
  sides[0].left += entry.sides.left.direct.length;
  sides[0].right += entry.sides.right.direct.length;
}
const res = solveProjectLinked(project, scene);
check(res.ok, '衔接分段解完成');
check(project.sharedFrame.anchorSource === 'observed', '共享镜框锚由角点观测估计', `来源 ${project.sharedFrame.anchorSource}`);
check(
  dist(project.sharedFrame.anchorObserved, scene.sharedFrame.anchor) < 0.03,
  '共享锚误差 < 3 cm',
  `${fmt(dist(project.sharedFrame.anchorObserved, scene.sharedFrame.anchor) * 1000, 1)} mm`,
);
for (const [i, seg] of project.segments.entries()) {
  const truth = scene.mirrorAt(i === 0 ? 5 : 14);
  const err = normalErr(seg.calibration.mirror, truth);
  check(err < 0.3, `${seg.label} 法线误差 < 0.3°`, `${fmt(err, 3)}°，RMS ${fmt(seg.calibration.rmsPx)} px`);
}
const nDistr = [...project.observations.values()].reduce((s, e) => s + e.distractors.length, 0);
check(nDistr === 3 * 4, '手性相反干扰标记全部被隔离（4 帧 × 3 个）', `${nDistr} 个`);
const occluded = [14, 15, 16, 17].every((f) => {
  const e = project.observations.get(f);
  return e.sides.left.mirror.length + e.sides.right.mirror.length === 0 && e.corners.length === 0;
});
check(occluded, '演员遮挡帧镜中/角点观测全部缺失（14–17）');

// ---- 2. joint fit comparison ---------------------------------------------
console.log('\n[2] 对照：联合拟合全部镜面段 vs 共享镜框锚衔接分段解');
const cmp = compareSegmentSolvers(project, scene);
console.log('           区段        法线误差°   面距误差m   面板中心m');
for (let i = 0; i < cmp.linked.perSegment.length; i++) {
  const L = cmp.linked.perSegment[i];
  const J = cmp.joint.perSegment[i];
  console.log(`  衔接解   ${L.label}  ${fmt(L.normalErrDeg, 4)}      ${fmt(L.planeDistErrM, 5)}     ${fmt(L.centreErrM, 5)}`);
  console.log(`  联合拟合 ${J.label}  ${fmt(J.normalErrDeg, 4)}      ${fmt(J.planeDistErrM, 5)}     ${fmt(J.centreErrM, 5)}`);
}
console.log(`  共享锚误差：衔接 ${fmt(cmp.linked.anchorErrM * 1000, 1)} mm，联合 ${fmt(cmp.joint.anchorErrM * 1000, 1)} mm`);
check(cmp.linked.perSegment.every((m) => m.normalErrDeg < 0.3), '衔接解全部区段法线误差 < 0.3°');
check(cmp.joint.perSegment.every((m) => m.normalErrDeg < 0.3), '联合拟合全部区段法线误差 < 0.3°（干净数据两者相当）');
console.log(`  选定实现：${cmp.choice === 'linked' ? '共享镜框锚衔接分段解' : '联合拟合'}`);
console.log(`  理由：${cmp.rationale}`);

// ---- 3. boundary drag rebinding -------------------------------------------
console.log('\n[3] 拖动中段边界：标记与合成物体指回正确的相机和镜面状态');
const mirrorA = frameState(project, 10).mirror;
const mirrorB = frameState(project, 20).mirror;
setSegmentBoundary(project, 0, 15);
check(frameState(project, 13).mirror === mirrorA, '边界拖到 15 后，第 13 帧指回区段 A 镜面');
check(frameState(project, 20).mirror === mirrorB, '第 20 帧仍指回区段 B 镜面');
check(frameState(project, 13).camera === frameState(project, 20).camera, '共享锁定机位相机');
setSegmentBoundary(project, 0, 12, { log: false });
markStaleSegments(project);

// ---- 4. isolation + drag-back ---------------------------------------------
console.log('\n[4] 区段参数失效：仅隔离依赖该面的合成层，拖回后核对反射方向');
invalidateSegment(project, 'S1', '镜面参数失效（模拟）');
check(layersForSegment(project, 'S1').every((l) => l.isolated), '区段 B 的全部合成层被隔离');
check(layersForSegment(project, 'S0').every((l) => !l.isolated), '区段 A 的合成层不受影响');
const back = restoreSegment(project, scene, 'S1');
check(back.ok, '拖回后反射方向核对通过', back.ok ? `det ${fmt(back.check.checks.detVirtual, 1)}，RMS ${fmt(back.check.checks.rmsPx)} px` : back.reasons?.join('；'));
check(layersForSegment(project, 'S1').every((l) => !l.isolated), '区段 B 合成层解除隔离');

// ---- 5. save + scrub back from both sides ---------------------------------
console.log('\n[5] 保存分段工程，从两侧拖回同一帧量取虚像误差');
const vie = boundaryVirtualImageError(project, 0);
console.log(`  边界帧 ${vie.boundary}：左帧 ${vie.leftFrame.frame} 本段 ${fmt(vie.leftFrame.own, 2)} px / 对侧 ${fmt(vie.leftFrame.other, 1)} px；` +
  `右帧 ${vie.rightFrame.frame} 本段 ${fmt(vie.rightFrame.own, 2)} px / 对侧 ${fmt(vie.rightFrame.other, 1)} px`);
check(vie.leftFrame.own < 1.5 && vie.rightFrame.own < 1.5, '本段标定解释边界帧（< 1.5 px）');
check(vie.leftFrame.other > 15 && vie.rightFrame.other > 15, '对侧标定虚像误差显著（> 15 px）');
const loaded = deserializeProject(serializeProject(project));
const vie2 = boundaryVirtualImageError(loaded, 0);
check(
  vie2.leftFrame.own === vie.leftFrame.own && vie2.rightFrame.other === vie.rightFrame.other,
  '保存/载入后虚像误差完全一致',
);
check(frameState(loaded, 11).segmentId === 'S0' && frameState(loaded, 12).segmentId === 'S1', '从两侧拖回边界帧状态一致');

// ---- 6. stale calibration -> crossing-line kink localises the edit --------
console.log('\n[6] 一侧出画后继续改镜面角度：过期标定由穿框直线折角定位');
console.log(`  区段 B 状态：${segmentStatus(project, 'S1')}（证据止于第 17 帧，第 18 帧镜面再次移动、右侧标记出画）`);
check(segmentStatus(project, 'S1') === 'stale' && project.segments[1].staleSince === 18, '区段 B 标记为过期（自第 18 帧）');
const kinks = scanKinks(project, scene);
const freshOk = [0, 5, 11, 12, 13].every((f) => kinks.get(f).ok && !kinks.get(f).kink);
const staleKink = [18, 19, 20, 21, 22, 23].every((f) => kinks.get(f).ok && kinks.get(f).kink);
check(freshOk, '第 18 帧前直线连续（夹角 < 1.5°，偏移 < 3 px）');
check(staleKink, '第 18–23 帧直线出现折角');
const loc = localizeStaleEdit(project, scene);
check(loc.found && loc.frame === 18 && loc.edit?.frame === 18, '折角定位到第 18 帧的镜面角度编辑');
console.log(`  ${loc.message}`);
const vieStale = virtualImageError(project, 20, 'S1');
check(vieStale > 8, '过期区段虚像误差显著', `${fmt(vieStale, 1)} px`);

// ---- 7. remedy: split + individual re-estimation --------------------------
console.log('\n[7] 求解不足/过期区段单独重估（分裂出新区段）');
const segC = splitSegment(project, 'S1', 18);
const re = solveSegment(project, scene, segC.id);
check(re.ok, '新区段单独重估成功', `${re.ok ? `RMS ${fmt(re.calibration.rmsPx)} px` : re.reason}`);
markStaleSegments(project);
const kinks2 = scanKinks(project, scene);
check([18, 20, 23].every((f) => kinks2.get(f).ok && !kinks2.get(f).kink), '重估后第 18–23 帧直线恢复连续');
check(segmentStatus(project, 'S1') === 'ok' && segmentStatus(project, segC.id) === 'ok', '全部区段状态正常');
const calA = JSON.stringify(project.segments[0].calibration);
solveSegment(project, scene, segC.id);
check(JSON.stringify(project.segments[0].calibration) === calA, '单独重估不影响其它区段标定');

console.log('\n' + '='.repeat(78));
if (failures) {
  console.error(`${failures} 项检查失败。`);
  process.exitCode = 1;
} else {
  console.log('全部检查通过。');
}
