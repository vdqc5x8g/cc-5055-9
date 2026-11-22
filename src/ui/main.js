import { createDefaultScene } from '../core/scene.js';
import { runPipeline } from '../core/pipeline.js';
import { drawScene, computeRays } from '../core/render.js';
import { mirrorFrame, mirrorQuad } from '../core/mirror.js';
import { projectPoint } from '../core/camera.js';
import { DEG } from '../core/math.js';

const $ = (id) => document.getElementById(id);

const state = {
  scene: createDefaultScene(),
  result: null,
  view: 'composite',
  selectedMarker: null,
  hoveredMarker: null,
  showRays: true,
  showDirect: true,
  showMirror: true,
  showTruth: false,
  lastGood: null,
};

// ------------------------- controls -------------------------

const num = (id, d = 0) => {
  const v = parseFloat($(id).value);
  return Number.isFinite(v) ? v : d;
};

function readConstraint() {
  return {
    center: [num('mx'), num('my'), num('mz')],
    yaw: num('myaw') * DEG,
    tilt: num('mtilt') * DEG,
    width: num('mw', 1.5),
    height: num('mh', 2.0),
  };
}

function readOptions() {
  const fixF = $('fixF').checked;
  return {
    fixF,
    fixFValue: num('fval', 1150),
    sigma: Math.max(0, num('sigma', 0.5)),
    missingFraction: parseFloat($('missing').value) || 0,
    refit: $('refit').checked,
    lockPos: $('lockPos').checked,
  };
}

function readAsset() {
  return {
    ...state.scene.virtualAsset,
    position: [num('ax', 0.42), 0, num('az', 0.95)],
    contact: [num('ax', 0.42), 0, num('az', 0.95)],
    yaw: num('ayaw', -20) * DEG,
  };
}

$('missing').addEventListener('input', () => {
  $('missingOut').textContent = `${Math.round(parseFloat($('missing').value) * 100)}%`;
});

// ------------------------- solve -------------------------


// ------------------------- UI update -------------------------

function updateUi() {
  const r = state.result;
  if (!r) return;
  const hasError = r.diagnostics.some((d) => d.level === 'error');
  const hasWarn = r.diagnostics.some((d) => d.level === 'warn');
  setState(hasError ? 'error' : hasWarn ? 'warn' : 'ok');
  renderSummary(r);
  renderErrorTable(r);
  renderBranches(r);
  const alert = pickAlert(r);
  if (alert) setAlert(alert.level, alert.text);
  else setAlert('ok', '求解完成：相机与镜面由实际观测解算，反射位置与误差已更新。');
  draw();
}

function pickAlert(r) {
  const err = r.diagnostics.find((d) => d.level === 'error');
  if (err) return { level: 'err', text: err.message };
  const warn = r.diagnostics.find((d) => d.level === 'warn');
  if (warn) return { level: 'warn', text: warn.message };
  return null;
}

function setState(kind) {
  const el = $('solveState');
  el.className = `pill ${kind === 'ok' ? 'ok' : kind === 'warn' ? 'warn' : kind === 'error' ? 'err' : 'idle'}`;
  el.textContent = kind === 'ok' ? '已求解' : kind === 'warn' ? '部分告警' : kind === 'error' ? '求解异常' : '未求解';
}

function setAlert(level, text) {
  const el = $('alertBar');
  el.className = `alert-bar ${level}`;
  el.textContent = text;
}

function fmt(v, digits = 3, unit = '') {
  if (v == null || !Number.isFinite(v)) return '—';
  return `${v.toFixed(digits)}${unit}`;
}

function renderSummary(r) {
  const boxes = [];
  const g = r.grading;
  const solvedCamera = r.cameraSource && !r.cameraSource.includes('fallback');
  boxes.push(`<dl class="kv">
    <dt>求解状态</dt><dd class="${solvedCamera ? 'good' : 'bad'}">${solvedCamera ? '已由观测求解' : '未求解（不可用于合成）'}</dd>
    <dt>相机来源</dt><dd>${labelCameraSource(r.cameraSource)}</dd>
    <dt>相机中心</dt><dd>${r.camera.C.map((v) => v.toFixed(3)).join(', ')}</dd>
    <dt>焦距 f</dt><dd>${fmt(r.camera.f, 1, ' px')}</dd>
    <dt>相机 RMS</dt><dd>${fmt(r.camSolve?.rmsPx, 3, ' px')}</dd>
    <dt>镜面法线</dt><dd>${fmtVec(mirrorFrame(r.mirror).n)}</dd>
    <dt>面内位置来源</dt><dd>${labelPanelSource(r.mirror.centerSource)}</dd>
    <dt>反射 RMS</dt><dd>${fmt(r.mirrorSolve?.rmsPx, 3, ' px')}</dd>
    <dt>耗时</dt><dd>${fmt(r.elapsedMs, 1, ' ms')}</dd>
  </dl>`);
  if (g && Number.isFinite(g.cameraCenterErrorM)) {
    boxes.push(`<dl class="kv" style="margin-top:8px">
      <dt>真值·中心误差</dt><dd>${fmt(g.cameraCenterErrorM, 4, ' m')}</dd>
      <dt>真值·旋转误差</dt><dd>${fmt(g.cameraRotationErrorDeg, 3, '°')}</dd>
      <dt>真值·焦距误差</dt><dd>${fmt(g.focalErrorPx, 2, ' px')}</dd>
      <dt>真值·镜面法线</dt><dd>${fmt(g.mirrorNormalErrorDeg, 3, '°')}</dd>
      <dt>真值·面距误差</dt><dd>${fmt(g.mirrorPlaneDistanceErrorM, 4, ' m')}</dd>
    </dl>`);
  }
  const flip = r.flipCheck?.meanDiscrepancyPx;
  boxes.push(`<p class="hint" style="margin-top:8px">水平翻转直射画面与真实反射的平均偏差：<b>${fmt(flip, 1, ' px')}</b>（越大说明“翻转贴图”越不成立）。</p>`);
  const cons = r.constraints ?? [];
  if (cons.length) {
    boxes.push(`<div class="constraints"><div class="constraints-title">采用的约束</div><ul>` +
      cons.map((c) => `<li><b>${c.label}</b><span>${c.detail}</span></li>`).join('') +
      `</ul></div>`);
  }
  $('resultSummary').className = 'result-body';
  $('resultSummary').innerHTML = boxes.join('');
}

function labelCameraSource(src) {
  if (!src) return '—';
  const map = {
    direct: '直接观测',
    'mirror-augmented': '镜面增广 + 联合精化',
    'mirror-augmented+joint': '镜面增广 + 联合精化',
    'direct+joint': '直接观测 + 联合精化',
    'mirror-only': '镜面观测反解',
    'mirror-only+joint': '镜面观测反解 + 联合精化',
    'truth-fallback': '真值回退（未求解）',
  };
  return map[src] ?? src;
}

function labelPanelSource(src) {
  if (src === 'constraint') return '场景约束';
  if (src === 'hit-centroid') return '命中点质心（非约束）';
  if (src === 'plane-origin') return '平面原点';
  return src ?? '—';
}

function fmtVec(v) {
  return `(${v.map((x) => x.toFixed(3)).join(', ')})`;
}

function renderErrorTable(r) {
  const tbody = $('errorTable').querySelector('tbody');
  const rows = [];
  const branch = r.branches?.best;
  if (branch) {
    for (const e of branch.entries) {
      const status = e.ok ? '有效' : (e.reasons?.[0] ?? '无效');
      const cls = e.ok ? 'ok' : 'bad';
      const H = e.H ? `(${e.H.map((x) => x.toFixed(2)).join(', ')})` : '—';
      rows.push(`<tr><td>${e.id}${'′'}</td><td>${fmt(e.err, 3)}</td><td>${H}</td><td class="${cls}">${status}</td></tr>`);
    }
  }
  tbody.innerHTML = rows.join('') || `<tr><td colspan="4" style="color:var(--muted)">无反射观测</td></tr>`;
}

function renderBranches(r) {
  const b = r.branches;
  const diags = r.diagnostics;
  const items = b.results.map((x) => {
    const cls = x.validModel ? 'valid' : 'invalid';
    const tag = x.validModel ? '有效模型' : '错误模型';
    return `<div class="branch-item ${cls}">
      <span class="name">${x.label}</span>
      <span><span class="rms">${x.rmsPx == null ? '—' : Number.isFinite(x.rmsPx) ? x.rmsPx.toFixed(2) + ' px' : '∞'}</span>
      <span class="tag">${tag}</span></span>
    </div>`;
  });
  const diagList = diags.length
    ? `<ul class="diag-list">${diags.map((d) => `<li class="${d.level}">${d.message}</li>`).join('')}</ul>`
    : '';
  $('branchBody').className = 'result-body';
  $('branchBody').innerHTML = items.join('') + diagList;
  const det = b.best?.detVirtual;
  const outsideRejected = (r.rejected ?? []).filter(
    (x) => x.kind === 'mirror' && /镜面矩形之外/.test(x.reason ?? ''),
  ).length;
  const inFrameRejected = (r.rejected ?? []).filter(
    (x) => x.kind === 'mirror' && /超出画面/.test(x.reason ?? ''),
  ).length;
  $('viewHint').textContent =
    `虚拟相机 det(R) = ${Number.isFinite(det) ? det.toFixed(2) : '—'}；` +
    `反射点镜内 ${b.best?.accepted ?? 0} / 镜外剔除 ${outsideRejected} / 出画剔除 ${inFrameRejected}`;
}

// ------------------------- drawing -------------------------

const canvas = $('view');

function draw() {
  const r = state.result;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width;
  const H = rect.height;
  if (!r) {
    ctx.fillStyle = '#0a0c10';
    ctx.fillRect(0, 0, W, H);
    return;
  }
  // Reshape camera native pixels to the viewport extent used by the renderer.
  const cam = adjustCameraForView(r.camera, state.view);
  const mirror = r.mirror;
  const rays = [];
  if (state.showRays && state.selectedMarker) {
    const marker = state.scene.markers.find((m) => m.id === state.selectedMarker);
    const obs = r.observations;
    const rs = computeRays({ camera: cam, mirror, marker, observation: obs });
    for (const ray of rs) {
      if (ray.color.includes('255,170') && !state.showDirect) continue;
      if (ray.color.includes('120,215') && !state.showMirror) continue;
      rays.push(ray);
    }
  }
  const assetForView = { ...r.asset, ...readAsset(), position: readAsset().position, contact: readAsset().contact };
  drawScene(ctx, { width: W, height: H }, {
    camera: cam,
    mirror,
    asset: assetForView,
    observations: r.observations,
    showDirectRays: state.showDirect,
    showMirrorRays: state.showMirror,
    showMirrorFrame: true,
    showTruth: false,
    rays,
    selectedMarker: state.selectedMarker,
    hoveredMarker: state.hoveredMarker,
  });
  drawTopViewOverlay(ctx, W, H, cam, mirror, assetForView);
  if (state.showTruth) drawTruthOverlay(ctx, W, H, cam);
}

// Native-resolution camera is mapped to the canvas; for the top view we still
// use the same projection then overlay a schematic.
function adjustCameraForView(camera, view) {
  return camera;
}

function drawTopViewOverlay(ctx, W, H, cam, mirror, asset) {
  if (state.view !== 'top') return;
  ctx.save();
  ctx.fillStyle = 'rgba(6,9,12,0.62)';
  ctx.fillRect(0, 0, W, H);
  // Schematic top view: X to the right, Z downward.
  const pad = 40;
  const sx = (W - pad * 2) / 5;
  const sz = (H - pad * 2) / 5;
  const ox = W / 2 - 0.3 * sx;
  const oz = H / 2;
  const P = (x, z) => ({ x: ox + x * sx, y: oz + z * sz });
  const camP = P(cam.C[0], cam.C[2]);
  const mP = P(mirror.center[0], mirror.center[2]);
  // Mirror plane segment.
  const { u } = mirrorFrame(mirror);
  const half = mirror.width / 2;
  const a = P(mirror.center[0] - u[0] * half, mirror.center[2] - u[2] * half);
  const b = P(mirror.center[0] + u[0] * half, mirror.center[2] + u[2] * half);
  ctx.strokeStyle = 'rgba(120,215,255,0.95)';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  // Camera marker + frustum.
  ctx.fillStyle = '#ffb26b';
  ctx.beginPath(); ctx.arc(camP.x, camP.y, 5, 0, Math.PI * 2); ctx.fill();
  const fwd = P(cam.C[0] + cam.R[2][0] * 1.2, cam.C[2] + cam.R[2][2] * 1.2);
  ctx.strokeStyle = 'rgba(255,178,107,0.7)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(camP.x, camP.y); ctx.lineTo(fwd.x, fwd.y); ctx.stroke();
  // Virtual camera (reflected).
  const vc = P(2 * mP.x - camP.x, 2 * mP.y - camP.y);
  ctx.strokeStyle = 'rgba(120,215,255,0.45)';
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(camP.x, camP.y); ctx.lineTo(vc.x, vc.y); ctx.stroke();
  ctx.setLineDash([]);
  // Asset.
  const ap = P(asset.contact[0], asset.contact[2]);
  ctx.fillStyle = '#ffd43b';
  ctx.beginPath(); ctx.arc(ap.x, ap.y, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(220,232,242,0.9)';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('相机', camP.x + 8, camP.y - 6);
  ctx.fillText('镜面', b.x + 6, b.y + 4);
  ctx.fillText('虚拟物体', ap.x + 8, ap.y + 12);
  ctx.fillText('俯视示意（X→, Z↓）', 12, 18);
  ctx.restore();
}

function drawTruthOverlay(ctx, W, H, cam) {
  // Truth outline is only meaningful in the synthetic shot.
  const r = state.result;
  if (!r?.scene?.mirror) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(47,158,68,0.9)';
  ctx.setLineDash([6, 5]);
  const truthCamQuad = mirrorQuad(r.scene.mirror).map((p) => {
    const pr = projectPoint(cam, p);
    return pr ? { x: pr.x * (W / cam.width), y: pr.y * (H / cam.height) } : null;
  });
  if (truthCamQuad.every(Boolean)) {
    ctx.beginPath();
    truthCamQuad.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

// ------------------------- interaction -------------------------

function pickMarker(ev) {
  const r = state.result;
  if (!r) return null;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  let best = null;
  for (const m of state.scene.markers) {
    const p = projectPoint(r.camera, m.X);
    if (!p) continue;
    const px = (p.x / r.camera.width) * rect.width;
    const py = (p.y / r.camera.height) * rect.height;
    const d = Math.hypot(px - x, py - y);
    if (d < 14 && (!best || d < best.d)) best = { id: m.id, d };
  }
  return best?.id ?? null;
}

canvas.addEventListener('mousemove', (ev) => {
  const id = pickMarker(ev);
  if (id !== state.hoveredMarker) {
    state.hoveredMarker = id;
    draw();
  }
  canvas.style.cursor = id ? 'pointer' : 'default';
});

canvas.addEventListener('click', (ev) => {
  state.selectedMarker = pickMarker(ev);
  draw();
});

document.querySelectorAll('.seg').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.seg').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.view = btn.dataset.view;
    draw();
  });
});

$('solveBtn').addEventListener('click', solve);
$('tglRays').addEventListener('change', (e) => { state.showRays = e.target.checked; draw(); });
$('tglDirect').addEventListener('change', (e) => { state.showDirect = e.target.checked; draw(); });
$('tglMirror').addEventListener('change', (e) => { state.showMirror = e.target.checked; draw(); });
$('tglTruth').addEventListener('change', (e) => { state.showTruth = e.target.checked; draw(); });

$('presetConstraint').addEventListener('click', () => {
  $('mx').value = 0;
  $('my').value = 1.02;
  $('mz').value = 0;
  $('myaw').value = 0;
  $('mtilt').value = 0.6;
  $('mw').value = 1.5;
  $('mh').value = 2.0;
  solve();
});

$('presetNoConstraint').addEventListener('click', () => {
  solve({ constraintMode: 'none' });
});

function solve({ constraintMode = 'constraint' } = {}) {
  const opts = readOptions();
  const scene = state.scene;
  scene.virtualAsset = readAsset();
  const t0 = performance.now();
  let result;
  try {
    result = runPipeline({
      scene,
      observationOptions: {
        noisePx: opts.sigma,
        seed: 11,
        extraDirectMissingFraction: opts.missingFraction,
      },
      mirrorConstraint: constraintMode === 'none' ? null : readConstraint(),
      fixF: opts.fixF,
      refit: opts.refit && !opts.lockPos,
    });
  } catch (e) {
    setAlert('err', `求解异常：${e.message}`);
    setState('error');
    return;
  }
  result.elapsedMs = performance.now() - t0;
  state.result = result;
  state.scene.virtualAsset = { ...scene.virtualAsset, ...result.asset };
  updateUi();
}

window.addEventListener('resize', () => {
  if (state.result) draw();
});

// ------------------------- boot -------------------------

// Start from a solved frame so the window is immediately informative.
solve();
requestAnimationFrame(() => draw());
