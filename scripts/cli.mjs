#!/usr/bin/env node
// Command-line assist for mirror compositing. The graphical entry point is the
// lens-tracking window in index.html (npm start); this CLI runs the exact same
// pipeline headlessly to print reprojection/geometric errors, the adopted
// constraints and the chirality/depth branch comparison.

import { createDefaultScene } from '../src/core/scene.js';
import { runPipeline } from '../src/core/pipeline.js';
import { mirrorFrame } from '../src/core/mirror.js';

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  printHelp();
  process.exit(0);
}

const constraintMode = args.constraint ?? 'preset'; // preset | none
const scene = createDefaultScene();
const missingDirect = args.missing > 0
  ? scene.markers
      .slice(Math.max(0, Math.round((1 - args.missing) * scene.markers.length)))
      .map((m) => m.id)
  : [];
const constraint = constraintMode === 'none'
  ? null
  : {
      center: parseVec(args.center, [0, 1.02, 0]),
      yaw: (args.yaw ?? 0) * (Math.PI / 180),
      tilt: (args.tilt ?? 0.6) * (Math.PI / 180),
      width: args.width ?? 1.5,
      height: args.height ?? 2.0,
    };

const screen = args.screen ? args.screen.split(',').map(Number) : null;
const detection = screen ? { screenPoint: screen } : null;

const t0 = performance.now();
const result = runPipeline({
  scene,
  observationOptions: {
    noisePx: args.sigma ?? scene.noisePx,
    seed: args.seed ?? 11,
    missingDirect,
  },
  mirrorConstraint: constraint,
  fixF: Boolean(args.fixf),
  refit: !args.noRefit,
  assetPlacement: detection,
});
const ms = performance.now() - t0;

printReport(result, { ms, constraintMode, args });
if (args.json) process.stdout.write('\n' + JSON.stringify(serialize(result), null, 2) + '\n');

// Exit non-zero if the solve is not usable for compositing.
const usable =
  result.cameraSource !== 'truth-fallback' &&
  result.mirrorSolve?.ok &&
  result.branches?.best?.detVirtual === -1 &&
  result.audit?.camSide > 0;
if (!usable) process.exitCode = 1;

// ------------------------- report -------------------------

function printReport(r, { ms, constraintMode }) {
  const line = (s = '') => console.log(s);
  line('镜面合成 · 命令行辅助（图形入口：npm start → http://localhost:4173/）');
  line('='.repeat(74));
  line(`相机来源        ${labelCam(r.cameraSource)}`);
  line(`相机中心 C      ${vec(r.camera.C)} m`);
  line(`焦距 f          ${r.camera.f.toFixed(2)} px`);
  line(`相机 RMS        ${fmt(r.camSolve?.rmsPx)} px   （${r.direct.length} 直接观测${r.augmented ? ` + ${r.augmented} 镜面增广` : ''}）`);
  line('');
  const { n } = mirrorFrame(r.mirror);
  line(`镜面法线 n      ${vec(n)}   （yaw ${(r.mirrorSolve?.yaw ?? 0) * 180 / Math.PI}°）`);
  line(`镜面平面距      ${fmt(r.mirrorSolve?.d)}   （相机—镜面 ${fmt(r.mirrorSolve?.h)} m）`);
  line(`面板面内位置     ${labelPanel(r.mirrorSolve?.panelSource)}   ${vec(r.mirror.center)}`);
  line(`反射 RMS        ${fmt(r.branches?.best?.rmsPx)} px   （有效反射 ${r.branches?.best?.accepted ?? 0}，面板外 ${r.branches?.best?.outsidePanel ?? 0}）`);
  line(`虚拟相机 det(R)  ${r.branches?.best?.detVirtual}`);
  line('');

  line('采用约束：');
  for (const c of r.constraints ?? []) line(`  · ${c.label} — ${c.detail}`);
  line('');

  line('手性/深度分支（重投影误差）：');
  for (const b of r.branches?.results ?? []) {
    const tag = b.validModel ? '有效' : '错误';
    line(`  [${tag}] ${b.label.padEnd(26, ' ')} ${fmtPx(b.rmsPx)}`);
  }
  line('');

  line('反射位置（逐点，像素）：');
  const entries = (r.branches?.best?.entries ?? []).slice(0, argsShow());
  for (const e of entries) {
    const status = e.ok ? '有效' : (e.reasons?.[0] ?? '无效');
    line(`  ${e.id}′  err ${fmt(e.err)} px  H ${e.H ? vec(e.H, 2) : '—'}  ${status}`);
  }
  if ((r.branches?.best?.entries.length ?? 0) > entries.length) {
    line(`  … 共 ${r.branches.best.entries.length} 个（--all 显示全部）`);
  }
  line('');

  line('镜面完整性核对：');
  const rejMirror = r.rejected.filter((x) => x.kind === 'mirror');
  line(`  相机在镜面正面：${r.audit?.camSide > 0 ? '是' : '否'}`);
  line(`  有效镜中观测：${r.mirrorObs.length}   被剔除：${rejMirror.length}`);
  for (const x of rejMirror.slice(0, 6)) line(`    - ${x.id}：${x.reason}`);
  if (rejMirror.length > 6) line(`    … 另外 ${rejMirror.length - 6} 个`);
  line('');

  if (r.grading && Number.isFinite(r.grading.cameraCenterErrorM)) {
    line('真值评分（仅合成片段）：');
    line(`  中心误差 ${fmt(r.grading.cameraCenterErrorM, 4)} m   旋转 ${fmt(r.grading.cameraRotationErrorDeg, 3)}°   焦距 ${fmt(r.grading.focalErrorPx, 2)} px`);
    line(`  法线误差 ${fmt(r.grading.mirrorNormalErrorDeg, 3)}°   面距误差 ${fmt(r.grading.mirrorPlaneDistanceErrorM, 4)} m`);
    line('');
  }

  if (r.asset && r.reflectionCheck) {
    line('虚拟物体与接触：');
    line(`  接触点 ${vec(r.asset.contact)} m   反射可用：${r.reflectionCheck.ok ? '是' : `否（${r.reflectionCheck.reasons.join('；')}）`}`);
    if (Number.isFinite(r.flipCheck?.meanDiscrepancyPx)) {
      line(`  水平翻转 vs 真实反射偏差 ${fmt(r.flipCheck.meanDiscrepancyPx, 1)} px（翻转贴图不成立）`);
    }
    line('');
  }

  if (r.diagnostics.length) {
    line('诊断：');
    for (const d of r.diagnostics) line(`  [${d.level}] ${d.message}`);
    line('');
  }
  line(`耗时 ${ms.toFixed(1)} ms   约束模式 ${constraintMode === 'none' ? '无约束（仅解平面）' : '场景约束'}`);
}

function argsShow() {
  return args.all ? Infinity : 12;
}

// ------------------------- util -------------------------

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') out.help = true;
    else if (a === '--json') out.json = true;
    else if (a === '--all') out.all = true;
    else if (a === '--fixf') out.fixf = true;
    else if (a === '--no-refit') out.noRefit = true;
    else if (a === '--constraint') out.constraint = argv[++i];
    else if (a === '--center') out.center = argv[++i];
    else if (a === '--yaw') out.yaw = Number(argv[++i]);
    else if (a === '--tilt') out.tilt = Number(argv[++i]);
    else if (a === '--width') out.width = Number(argv[++i]);
    else if (a === '--height') out.height = Number(argv[++i]);
    else if (a === '--sigma') out.sigma = Number(argv[++i]);
    else if (a === '--seed') out.seed = Number(argv[++i]);
    else if (a === '--missing') out.missing = Number(argv[++i]);
    else if (a === '--screen') out.screen = argv[++i];
  }
  return out;
}

function printHelp() {
  console.log(`用法：node scripts/cli.mjs [选项]
  --constraint preset|none   镜面约束模式（默认 preset）
  --center x,y,z             面板中心（默认 0,1.02,0）
  --yaw deg / --tilt deg     面板偏航/俯仰
  --width m / --height m     面板尺寸
  --sigma px                 观测噪声（默认 0.5）
  --seed n                   噪声种子（默认 11）
  --missing 0..1             直接标记缺失比例
  --screen x,y               在此屏幕点按地面约束放置虚拟物体
  --fixf                     固定焦距
  --no-refit                 不做联合精化
  --all                      列出全部反射点
  --json                     额外输出机器可读 JSON
图形入口：npm start 后打开 http://localhost:4173/ （镜面合成镜头跟踪窗口）。`);
}

function labelCam(s) {
  return {
    direct: '直接观测',
    'direct+joint': '直接观测 + 联合精化',
    'mirror-augmented': '镜面增广（直接不足/共面）',
    'mirror-augmented+joint': '镜面增广 + 联合精化',
    'mirror-only': '镜面观测反解',
    'mirror-only+joint': '镜面观测反解 + 联合精化',
    'truth-fallback': '真值回退（未求解）',
  }[s] ?? s ?? '—';
}

function labelPanel(s) {
  return {
    constraint: '场景约束',
    'hit-centroid': '命中点质心（非约束）',
    'plane-origin': '平面原点',
  }[s] ?? s ?? '—';
}

function vec(v, d = 3) {
  if (!v) return '—';
  return `(${v.map((x) => (Number.isFinite(x) ? x.toFixed(d) : '—')).join(', ')})`;
}

function fmt(v, d = 3) {
  return Number.isFinite(v) ? v.toFixed(d) : '—';
}

function fmtPx(v) {
  if (v == null) return '—';
  if (!Number.isFinite(v)) return '∞';
  return `${v.toFixed(2)} px`;
}

function parseVec(s, def) {
  if (!s) return def;
  const v = s.split(',').map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? v : def;
}

function serialize(r) {
  return {
    cameraSource: r.cameraSource,
    camera: { C: r.camera.C, f: r.camera.f, R: r.camera.R },
    cameraRmsPx: r.camSolve?.rmsPx ?? null,
    mirror: {
      center: r.mirror.center,
      yaw: r.mirror.yaw,
      tilt: r.mirror.tilt,
      normal: mirrorFrame(r.mirror).n,
      panelSource: r.mirrorSolve?.panelSource ?? null,
      rmsPx: r.mirrorSolve?.rmsPx ?? null,
    },
    constraints: r.constraints,
    branches: (r.branches?.results ?? []).map((b) => ({
      key: b.key, label: b.label, validModel: b.validModel,
      rmsPx: Number.isFinite(b.rmsPx) ? b.rmsPx : null,
    })),
    grading: r.grading,
    flipDiscrepancyPx: r.flipCheck?.meanDiscrepancyPx ?? null,
    rejectedMirror: r.rejected.filter((x) => x.kind === 'mirror').map((x) => ({ id: x.id, reason: x.reason })),
    diagnostics: r.diagnostics,
  };
}
