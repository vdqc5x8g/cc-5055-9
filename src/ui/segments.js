import { createTimelineScene } from '../core/timelineScene.js';
import {
  createSegmentProject,
  recordSideObservations,
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
  measureCrossingLineKink,
  scanKinks,
  localizeStaleEdit,
  boundaryVirtualImageError,
  serializeProject,
  deserializeProject,
  layersForSegment,
  logEdit,
} from '../core/segments.js';
import { drawScene } from '../core/render.js';
import { mirrorFrame } from '../core/mirror.js';
import { clamp, RAD } from '../core/math.js';

const $ = (id) => document.getElementById(id);

const state = {
  scene: createTimelineScene(),
  project: null,
  frame: 0,
  showDirect: true,
  showMirror: true,
  showRail: true,
  showDistractors: true,
  showTruth: false,
  kinks: new Map(),
  comparison: null,
  dragBoundary: null,
  dragPlayhead: false,
};

const STATUS_LABEL = {
  ok: '正常',
  stale: '过期',
  invalid: '失效',
  underdetermined: '求解不足',
  empty: '未标定',
};
const STATUS_CLASS = { ok: 'ok', stale: 'warn', invalid: 'err', underdetermined: 'warn', empty: 'idle' };

const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');

// ------------------------- boot -------------------------

function boot() {
  state.project = createSegmentProject(state.scene, { boundaries: [12] });
  recordAllFrames(state.project, state.scene);
  solveProjectLinked(state.project, state.scene);
  state.kinks = scanKinks(state.project, state.scene);
  updateAll();
  setAlert('ok', '已录入全部帧（两侧）并完成衔接分段解：共享镜框锚由角点观测估计，两个区段各自标定。');
}

// ------------------------- viewport -------------------------

const canvas = $('view');

function draw() {
  const { project, scene } = state;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width;
  const H = rect.height;

  const fs = frameState(project, state.frame);
  const camera = project.camera ?? scene.camera;
  const solved = Boolean(fs.mirror);
  const mirror = fs.mirror ?? scene.mirrorAt(state.frame);
  const layers = fs.segmentId ? layersForSegment(project, fs.segmentId) : [];
  const isolated = (kind) => layers.some((l) => l.kind === kind && l.isolated);

  drawScene(ctx, { width: W, height: H }, {
    camera,
    mirror,
    asset: isolated('asset') ? null : project.asset,
    observations: null,
    showMirrorFrame: true,
    showTruth: state.showTruth,
    truthMirror: scene.mirrorAt(state.frame),
  });

  const s = Math.max(W / camera.width, H / camera.height);
  const ox = (W - camera.width * s) / 2;
  const oy = (H - camera.height * s) / 2;
  const P = (px) => ({ x: ox + px[0] * s, y: oy + px[1] * s });

  const entry = project.observations.get(state.frame);
  if (entry) drawObservations(ctx, P, entry, isolated);

  let kinkInfo = null;
  if (state.showRail && solved) {
    const k = measureCrossingLineKink(project, scene, state.frame);
    if (k.ok) {
      kinkInfo = k;
      drawKinkOverlay(ctx, P, k);
    }
  }

  const seg = project.segments.find((x) => x.id === fs.segmentId);
  $('viewHint').textContent =
    `帧 ${state.frame} · ${seg?.label ?? '—'} · ${STATUS_LABEL[fs.status]}` +
    ` · 相机 ${project.cameraSource ? '共享锁定机位' : '真值（未求解）'}` +
    (solved ? '' : ' · 镜面未标定（显示真值姿态）') +
    (kinkInfo ? ` · 折角 ${kinkInfo.kink ? `是（夹角 ${kinkInfo.angleDeg.toFixed(1)}°）` : '否'}` : '') +
    (fs.isolatedLayers.length ? ` · 已隔离 ${fs.isolatedLayers.length} 层` : '');
}

function drawObservations(ctx, P, entry, isolated) {
  ctx.save();
  if (state.showDirect && !isolated('direct')) {
    for (const side of ['left', 'right']) {
      const color = side === 'left' ? 'rgba(255,178,107,0.98)' : 'rgba(140,233,154,0.98)';
      for (const o of entry.sides[side].direct) {
        if (!o.pixel) continue;
        const p = P(o.pixel);
        if (o.group === 'rail') {
          ctx.fillStyle = 'rgba(235,240,245,0.95)';
          ctx.fillRect(p.x - 2.5, p.y - 2.5, 5, 5);
        } else {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }
  if (state.showMirror && !isolated('mirror')) {
    for (const side of ['left', 'right']) {
      for (const o of entry.sides[side].mirror) {
        if (!o.pixel) continue;
        const p = P(o.pixel);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(120,215,255,0.95)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }
  // Frame-corner markers (shared anchor evidence).
  for (const c of entry.corners) {
    if (!c.pixel) continue;
    const p = P(c.pixel);
    ctx.fillStyle = 'rgba(255,212,59,0.95)';
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - 5);
    ctx.lineTo(p.x + 5, p.y);
    ctx.lineTo(p.x, p.y + 5);
    ctx.lineTo(p.x - 5, p.y);
    ctx.closePath();
    ctx.fill();
  }
  if (state.showDistractors) {
    for (const d of entry.distractors) {
      const p = P(d.pixel);
      ctx.strokeStyle = 'rgba(255,107,107,0.98)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p.x - 5, p.y - 5);
      ctx.lineTo(p.x + 5, p.y + 5);
      ctx.moveTo(p.x + 5, p.y - 5);
      ctx.lineTo(p.x - 5, p.y + 5);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,107,107,0.9)';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('手性干扰', p.x + 8, p.y - 6);
    }
  }
  ctx.restore();
}

function drawKinkOverlay(ctx, P, k) {
  const drawFit = (L, color, dashed = false) => {
    const a = P([L.point[0] - L.dir[0] * 2400, L.point[1] - L.dir[1] * 2400]);
    const b = P([L.point[0] + L.dir[0] * 2400, L.point[1] + L.dir[1] * 2400]);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    if (dashed) ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.restore();
  };
  drawFit(k.L1, 'rgba(235,240,245,0.85)');
  drawFit(k.L2obs, 'rgba(120,215,255,0.85)');
  drawFit(k.L2pred, k.kink ? 'rgba(255,107,107,0.95)' : 'rgba(105,219,124,0.9)', true);
  const jp = P(k.J_pred);
  ctx.save();
  ctx.beginPath();
  ctx.arc(jp.x, jp.y, 6, 0, Math.PI * 2);
  ctx.strokeStyle = k.kink ? 'rgba(255,107,107,0.98)' : 'rgba(105,219,124,0.95)';
  ctx.lineWidth = 2;
  ctx.stroke();
  if (k.J_obs) {
    const jo = P(k.J_obs);
    ctx.fillStyle = 'rgba(235,240,245,0.9)';
    ctx.beginPath();
    ctx.arc(jo.x, jo.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (k.kink) {
    ctx.fillStyle = 'rgba(255,107,107,0.98)';
    ctx.font = 'bold 12px system-ui, sans-serif';
    ctx.fillText(`折角 ${k.angleDeg.toFixed(1)}°`, jp.x + 10, jp.y - 10);
  }
  ctx.restore();
}

// ------------------------- timeline -------------------------

const tl = $('timeline');
const TL = { x0: 12, x1pad: 12, trackY: 20, trackH: 26, kinkY: 52, kinkH: 9, eventY: 70, h: 88 };

function tlGeom() {
  const rect = tl.getBoundingClientRect();
  const W = rect.width;
  const fx = (f) => TL.x0 + (f / state.scene.frameCount) * (W - TL.x0 - TL.x1pad);
  const xf = (x) => ((x - TL.x0) / (W - TL.x0 - TL.x1pad)) * state.scene.frameCount;
  return { W, H: rect.height, fx, xf };
}

function drawTimeline() {
  const { project, scene } = state;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = tl.getBoundingClientRect();
  tl.width = Math.max(1, Math.round(rect.width * dpr));
  tl.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = tl.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const { W, H, fx } = tlGeom();

  ctx.fillStyle = '#0d1219';
  ctx.fillRect(0, 0, W, H);

  // Event shading: occlusion, distractors, right-side exit.
  const span = (a, b, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(fx(a), TL.trackY, fx(b) - fx(a), TL.trackH);
  };
  span(scene.occlusionSpan[0], scene.occlusionSpan[1] + 1, 'rgba(255,107,107,0.13)');
  span(scene.distractorSpan[0], scene.distractorSpan[1] + 1, 'rgba(186,140,255,0.13)');
  span(scene.rightExitFrame, scene.frameCount, 'rgba(130,146,165,0.10)');

  // Segment blocks coloured by status.
  const fillFor = { ok: 'rgba(105,219,124,0.30)', stale: 'rgba(255,212,59,0.30)', invalid: 'rgba(255,107,107,0.34)', underdetermined: 'rgba(255,169,77,0.30)', empty: 'rgba(130,146,165,0.18)' };
  const edgeFor = { ok: 'rgba(105,219,124,0.8)', stale: 'rgba(255,212,59,0.8)', invalid: 'rgba(255,107,107,0.9)', underdetermined: 'rgba(255,169,77,0.8)', empty: 'rgba(130,146,165,0.5)' };
  for (const seg of project.segments) {
    const st = segmentStatus(project, seg.id);
    const a = fx(seg.range[0]);
    const b = fx(seg.range[1] + 1);
    ctx.fillStyle = fillFor[st];
    ctx.fillRect(a, TL.trackY, b - a, TL.trackH);
    ctx.strokeStyle = edgeFor[st];
    ctx.lineWidth = 1.2;
    ctx.strokeRect(a, TL.trackY, b - a, TL.trackH);
    ctx.fillStyle = '#dbe4ee';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(`${seg.label} · ${STATUS_LABEL[st]}`, a + 6, TL.trackY + 16);
  }

  // Boundary handles (draggable).
  for (let i = 1; i < project.segments.length; i++) {
    const x = fx(project.segments[i].range[0]);
    ctx.fillStyle = '#5cc8f2';
    ctx.fillRect(x - 2, TL.trackY - 4, 4, TL.trackH + 8);
    ctx.beginPath();
    ctx.moveTo(x - 5, TL.trackY - 6);
    ctx.lineTo(x + 5, TL.trackY - 6);
    ctx.lineTo(x, TL.trackY - 1);
    ctx.closePath();
    ctx.fill();
  }

  // Kink bars.
  for (const [f, k] of state.kinks) {
    if (!k.ok || !k.kink) continue;
    ctx.fillStyle = 'rgba(255,107,107,0.9)';
    ctx.fillRect(fx(f), TL.kinkY, Math.max(2, fx(f + 1) - fx(f) - 1), TL.kinkH);
  }
  ctx.fillStyle = 'rgba(130,146,165,0.9)';
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('折角', TL.x0, TL.kinkY + TL.kinkH + 10);

  // Event ticks.
  for (const e of scene.events) {
    const x = fx(e.frame);
    ctx.fillStyle = e.type === 'mirror-move' ? '#ffb26b' : 'rgba(130,146,165,0.8)';
    ctx.beginPath();
    ctx.moveTo(x, TL.eventY - 5);
    ctx.lineTo(x + 4, TL.eventY + 3);
    ctx.lineTo(x - 4, TL.eventY + 3);
    ctx.closePath();
    ctx.fill();
  }

  // Frame scale.
  ctx.fillStyle = 'rgba(130,146,165,0.75)';
  ctx.font = '10px ui-monospace, monospace';
  for (let f = 0; f < scene.frameCount; f += 6) {
    ctx.fillText(String(f), fx(f) - 3, H - 4);
  }

  // Playhead.
  const px = fx(state.frame + 0.5);
  ctx.strokeStyle = '#5cc8f2';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(px, 6);
  ctx.lineTo(px, H - 14);
  ctx.stroke();
  ctx.fillStyle = '#5cc8f2';
  ctx.beginPath();
  ctx.moveTo(px - 5, 2);
  ctx.lineTo(px + 5, 2);
  ctx.lineTo(px, 9);
  ctx.closePath();
  ctx.fill();
}

function seek(f) {
  state.frame = clamp(Math.round(f), 0, state.scene.frameCount - 1);
  $('frameLabel').textContent = `帧 ${state.frame} / ${state.scene.frameCount - 1}`;
  draw();
  drawTimeline();
}

function hitBoundary(x) {
  const { fx } = tlGeom();
  for (let i = 1; i < state.project.segments.length; i++) {
    if (Math.abs(x - fx(state.project.segments[i].range[0])) < 8) return i - 1;
  }
  return null;
}

tl.addEventListener('mousedown', (ev) => {
  const rect = tl.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const b = hitBoundary(x);
  if (b != null) {
    state.dragBoundary = b;
  } else {
    state.dragPlayhead = true;
    seek(tlGeom().xf(x) - 0.5);
  }
});

window.addEventListener('mousemove', (ev) => {
  if (state.dragBoundary == null && !state.dragPlayhead) return;
  const rect = tl.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  if (state.dragBoundary != null) {
    setSegmentBoundary(state.project, state.dragBoundary, tlGeom().xf(x), { log: false });
    state.kinks = scanKinks(state.project, state.scene);
    updateAll();
  } else if (state.dragPlayhead) {
    seek(tlGeom().xf(x) - 0.5);
  }
});

window.addEventListener('mouseup', () => {
  if (state.dragBoundary != null) {
    const b = state.project.segments[state.dragBoundary + 1].range[0];
    logEdit(state.project, { type: 'boundary-drag', frame: b, note: `中段边界拖至第 ${b} 帧` });
    markStaleSegments(state.project);
    state.kinks = scanKinks(state.project, state.scene);
    updateAll();
  }
  state.dragBoundary = null;
  state.dragPlayhead = false;
});

// ------------------------- panels -------------------------

function setAlert(level, text) {
  const el = $('alertBar');
  el.className = `alert-bar ${level}`;
  el.textContent = text;
}

function updateSolveState() {
  const statuses = state.project.segments.map((s) => segmentStatus(state.project, s.id));
  const kind = statuses.includes('invalid') ? 'err'
    : statuses.some((s) => s === 'stale' || s === 'underdetermined') ? 'warn'
    : statuses.every((s) => s === 'ok') ? 'ok' : 'idle';
  const el = $('solveState');
  el.className = `pill ${kind === 'idle' ? 'idle' : kind}`;
  el.textContent = kind === 'ok' ? '全部区段正常'
    : kind === 'warn' ? '存在过期/求解不足区段'
    : kind === 'err' ? '存在失效区段' : '未求解';
}

function renderSegmentCards() {
  const { project } = state;
  $('segmentCards').innerHTML = project.segments.map((seg) => {
    const st = segmentStatus(project, seg.id);
    const cal = seg.calibration;
    const ev = cal?.evidence;
    const evText = ev
      ? `左 ${ev.left ? `${ev.left[0]}–${ev.left[1]}` : '—'} · 右 ${ev.right ? `${ev.right[0]}–${ev.right[1]}` : '—'}`
      : '—';
    const stale = seg.staleSince != null ? `<span class="seg-stale-note">过期自第 ${seg.staleSince} 帧</span>` : '';
    const invalid = seg.invalid ? `<span class="seg-invalid-note">${seg.invalid}</span>` : '';
    const canSplit = state.frame > seg.range[0] && state.frame <= seg.range[1];
    return `<div class="seg-card">
      <div class="seg-card-head">
        <b>${seg.label}</b>
        <span class="pill ${STATUS_CLASS[st]}">${STATUS_LABEL[st]}</span>
      </div>
      <div class="seg-card-body mono">
        范围 ${seg.range[0]}–${seg.range[1]} · 证据帧 ${cal ? `${Math.min(...cal.solveFrames)}–${Math.max(...cal.solveFrames)}` : '—'}<br/>
        两侧证据 ${evText} · RMS ${cal ? fmt(cal.rmsPx, 3) : '—'} px
        ${stale}${invalid}
      </div>
      <div class="seg-card-actions">
        <button data-act="resolve" data-seg="${seg.id}" class="ghost">单独重估</button>
        <button data-act="invalidate" data-seg="${seg.id}" class="ghost" ${seg.invalid ? 'disabled' : ''}>失效模拟</button>
        <button data-act="restore" data-seg="${seg.id}" class="ghost" ${seg.lastGood && seg.invalid ? '' : 'disabled'}>拖回恢复</button>
        <button data-act="split" data-seg="${seg.id}" class="ghost" ${canSplit ? '' : 'disabled'}>分裂于播放头</button>
      </div>
    </div>`;
  }).join('');
  $('segmentCards').querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', () => onSegmentAction(btn.dataset.act, btn.dataset.seg));
  });
}

function onSegmentAction(act, segId) {
  const { project, scene } = state;
  if (act === 'resolve') {
    const res = solveSegment(project, scene, segId);
    setAlert(res.ok ? 'ok' : 'warn', res.ok
      ? `${segId} 单独重估完成：RMS ${fmt(res.calibration.rmsPx, 3)} px；其它区段标定保持不变。`
      : `${segId} 求解不足（镜中观测 ${res.mirrorObs ?? 0} 个），补充观测后可再次单独重估。`);
  } else if (act === 'invalidate') {
    invalidateSegment(project, segId, '镜面参数失效（模拟）');
    setAlert('warn', `${segId} 镜面参数失效：仅隔离依赖该区段的合成层，其它区段不受影响。`);
  } else if (act === 'restore') {
    const res = restoreSegment(project, scene, segId);
    setAlert(res.ok ? 'ok' : 'err', res.ok
      ? `${segId} 拖回后核对反射方向通过：det=-1、相机与标记均在正面、RMS ${fmt(res.check.checks.rmsPx, 3)} px。`
      : `${segId} 拖回未通过反射方向核对：${(res.reasons ?? [res.reason]).join('；')}`);
  } else if (act === 'split') {
    const seg = splitSegment(project, segId, state.frame);
    if (seg) setAlert('ok', `已在第 ${state.frame} 帧分裂出 ${seg.label}，可对其单独重估。`);
  }
  state.kinks = scanKinks(project, scene);
  updateAll();
}

function renderSegmentTable() {
  const { project, scene } = state;
  const tbody = $('segmentTable').querySelector('tbody');
  tbody.innerHTML = project.segments.map((seg) => {
    const st = segmentStatus(project, seg.id);
    const cal = seg.calibration;
    let nErr = '—';
    let cErr = '—';
    if (cal) {
      const mid = cal.solveFrames[Math.floor(cal.solveFrames.length / 2)];
      const truth = scene.mirrorAt(mid);
      const n1 = mirrorFrame(cal.mirror).n;
      const n2 = mirrorFrame(truth).n;
      nErr = (Math.acos(clamp(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2], -1, 1)) * RAD).toFixed(3);
      cErr = Math.hypot(
        cal.mirror.center[0] - truth.center[0],
        cal.mirror.center[1] - truth.center[1],
        cal.mirror.center[2] - truth.center[2],
      ).toFixed(4);
    }
    return `<tr>
      <td>${seg.label}</td><td>${seg.range[0]}–${seg.range[1]}</td>
      <td class="${st === 'ok' ? 'ok' : st === 'invalid' ? 'bad' : 'warn'}">${STATUS_LABEL[st]}</td>
      <td>${cal ? fmt(cal.rmsPx, 3) : '—'}</td><td>${nErr}</td><td>${cErr}</td>
    </tr>`;
  }).join('');
}

function renderFrameInfo() {
  const { project } = state;
  const sf = project.sharedFrame;
  const a = sf.anchorObserved ?? sf.anchor;
  $('frameInfo').innerHTML =
    `<span class="mono">锚 (${a.map((v) => v.toFixed(3)).join(', ')}) · 来源 ${sf.anchorSource === 'observed' ? '角点观测' : '场景约束'}<br/>` +
    `面板 ${sf.width.toFixed(2)} × ${sf.height.toFixed(2)} m · 全部区段共享同一镜框</span>`;
}

function renderRecordInfo() {
  const { project } = state;
  const entry = project.observations.get(state.frame);
  if (!entry) {
    $('recordInfo').textContent = `第 ${state.frame} 帧尚未录入观测。`;
    return;
  }
  const n = (side, kind) => entry.sides[side][kind].length;
  $('recordInfo').textContent =
    `第 ${state.frame} 帧：左侧 直接 ${n('left', 'direct')} / 镜中 ${n('left', 'mirror')}；` +
    `右侧 直接 ${n('right', 'direct')} / 镜中 ${n('right', 'mirror')}；` +
    `角点 ${entry.corners.length}；干扰 ${entry.distractors.length}；缺失 ${entry.missing.length}。`;
}

function renderEvents() {
  $('eventList').innerHTML = state.scene.events.map((e, i) =>
    `<div class="event-item" data-frame="${e.frame}"><span class="mono">第 ${e.frame} 帧</span>${e.note}</div>`,
  ).join('');
  $('eventList').querySelectorAll('.event-item').forEach((el) => {
    el.addEventListener('click', () => seek(parseInt(el.dataset.frame, 10)));
  });
}

function renderLayers() {
  $('layerChips').innerHTML = state.project.layers.map((l) =>
    `<span class="layer-chip ${l.isolated ? 'isolated' : ''}">${l.label}${l.isolated ? ' · 已隔离' : ''}</span>`,
  ).join('');
}

function renderDiagnostics() {
  const diags = state.project.diagnostics;
  $('diagPanel').innerHTML = diags.length
    ? `<ul class="diag-list">${diags.map((d) => `<li class="${d.level}">${d.message}</li>`).join('')}</ul>`
    : '';
}

function renderKinkPanel() {
  const { project, scene } = state;
  const loc = localizeStaleEdit(project, scene);
  if (!loc.found) {
    $('kinkPanel').className = 'result-body';
    $('kinkPanel').innerHTML = `<p class="kink-ok">未检测到折角：穿过镜框的直线在所有可测帧连续（夹角 &lt; ${project.kinkAngleToleranceDeg}°，偏移 &lt; ${project.kinkTolerancePx} px）。</p>`;
    return;
  }
  $('kinkPanel').className = 'result-body';
  $('kinkPanel').innerHTML =
    `<p class="kink-warn">${loc.message}</p>
     <div class="ctrl-actions"><button id="kinkRemedy" class="ghost">在折角起始帧分裂区段并重估</button></div>`;
  $('kinkRemedy').addEventListener('click', () => {
    const seg = splitSegment(project, loc.segmentId, loc.frame);
    if (!seg) return;
    const res = solveSegment(project, scene, seg.id);
    markStaleSegments(project);
    state.kinks = scanKinks(project, scene);
    setAlert(res.ok ? 'ok' : 'warn', res.ok
      ? `已在第 ${loc.frame} 帧分裂出 ${seg.label} 并单独重估（RMS ${fmt(res.calibration.rmsPx, 3)} px）：折角消除。`
      : `分裂后求解不足：${res.reason}`);
    updateAll();
  });
}

function renderViePanel() {
  const { project } = state;
  const rows = [];
  for (let i = 0; i < project.segments.length - 1; i++) {
    const vie = boundaryVirtualImageError(project, i);
    if (!vie) continue;
    const cell = (v) => (v == null ? '—' : v < 1.5 ? `<span class="vie-ok">${v.toFixed(2)}</span>` : `<span class="vie-bad">${v.toFixed(1)}</span>`);
    rows.push(`<dl class="kv">
      <dt>边界第 ${vie.boundary} 帧</dt><dd></dd>
      <dt>左帧 ${vie.leftFrame.frame}（${vie.leftFrame.ownSegment}）</dt><dd>本段 ${cell(vie.leftFrame.own)} px · 对侧 ${cell(vie.leftFrame.other)} px</dd>
      <dt>右帧 ${vie.rightFrame.frame}（${vie.rightFrame.ownSegment}）</dt><dd>本段 ${cell(vie.rightFrame.own)} px · 对侧 ${cell(vie.rightFrame.other)} px</dd>
    </dl>`);
  }
  $('viePanel').className = rows.length ? 'result-body' : 'result-body muted';
  $('viePanel').innerHTML = rows.length
    ? rows.join('') + '<p class="hint">从两侧拖回同一帧：本段标定应解释该帧（&lt; 1.5 px），对侧标定的虚像误差即镜面换位的可测特征。</p>'
    : '需要至少两个区段。';
}

function renderCompare() {
  const cmp = state.comparison;
  if (!cmp) return;
  const row = (m, tag) => (m.ok
    ? `<tr><td>${tag}</td><td>${m.label}</td><td>${m.normalErrDeg.toFixed(4)}</td><td>${m.planeDistErrM.toFixed(5)}</td><td>${m.centreErrM.toFixed(5)}</td></tr>`
    : `<tr><td>${tag}</td><td>${m.label}</td><td colspan="3" class="bad">求解失败</td></tr>`);
  let rows = '';
  for (let i = 0; i < cmp.linked.perSegment.length; i++) {
    rows += row(cmp.linked.perSegment[i], '衔接');
    rows += row(cmp.joint.perSegment[i], '联合');
  }
  $('comparePanel').className = 'result-body';
  $('comparePanel').innerHTML =
    `<div class="table-wrap"><table><thead><tr><th>实现</th><th>区段</th><th>法线误差°</th><th>面距误差 m</th><th>面板中心 m</th></tr></thead><tbody>${rows}</tbody></table></div>
     <dl class="kv" style="margin-top:8px">
       <dt>共享锚误差</dt><dd>衔接 ${(cmp.linked.anchorErrM * 1000).toFixed(1)} mm · 联合 ${(cmp.joint.anchorErrM * 1000).toFixed(1)} mm</dd>
       <dt>选定实现</dt><dd class="good">${cmp.choice === 'linked' ? '共享镜框锚衔接分段解' : '联合拟合'}</dd>
     </dl>
     <p class="hint">${cmp.rationale}</p>`;
}

function updateAll() {
  updateSolveState();
  drawTimeline();
  draw();
  renderSegmentCards();
  renderSegmentTable();
  renderFrameInfo();
  renderRecordInfo();
  renderLayers();
  renderDiagnostics();
  renderKinkPanel();
  renderViePanel();
  renderCompare();
  $('frameLabel').textContent = `帧 ${state.frame} / ${state.scene.frameCount - 1}`;
}

// ------------------------- actions -------------------------

$('solveBtn').addEventListener('click', () => {
  const res = solveProjectLinked(state.project, state.scene);
  state.kinks = scanKinks(state.project, state.scene);
  updateAll();
  setAlert(res.ok ? 'ok' : 'err', res.ok
    ? '衔接分段解完成：共享镜框锚由角点观测逐帧估计（中位数），各区段平面分别求解并挂到共享锚上。'
    : '求解失败。');
});

$('compareBtn').addEventListener('click', () => {
  state.comparison = compareSegmentSolvers(state.project, state.scene);
  renderCompare();
  setAlert('ok', '对照完成：联合拟合与共享锚衔接分段解已在相同观测上运行，结果见右下对照表。');
});

$('recLeft').addEventListener('click', () => {
  recordSideObservations(state.project, state.scene, state.frame, 'left');
  updateAll();
});
$('recRight').addEventListener('click', () => {
  recordSideObservations(state.project, state.scene, state.frame, 'right');
  updateAll();
});
$('recAll').addEventListener('click', () => {
  recordAllFrames(state.project, state.scene);
  updateAll();
  setAlert('ok', '已重新录入全部帧（两侧分别录入，像素确定可复现）。');
});

$('saveBtn').addEventListener('click', () => {
  const json = serializeProject(state.project);
  try {
    localStorage.setItem('segments-project', json);
  } catch { /* storage may be unavailable; the download below still works */ }
  const blob = new Blob([json], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'segments-project.json';
  a.click();
  URL.revokeObjectURL(a.href);
  setAlert('ok', '分段工程已保存（localStorage + 下载 segments-project.json）。');
});

$('loadBtn').addEventListener('click', () => {
  let json = null;
  try {
    json = localStorage.getItem('segments-project');
  } catch { /* ignore */ }
  if (json) {
    loadProject(json);
  } else {
    $('fileInput').click();
  }
});

$('fileInput').addEventListener('change', (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => loadProject(reader.result);
  reader.readAsText(file);
  ev.target.value = '';
});

function loadProject(json) {
  try {
    state.project = deserializeProject(json);
    state.kinks = scanKinks(state.project, state.scene);
    state.comparison = null;
    updateAll();
    setAlert('ok', '分段工程已载入：从两侧拖回同一帧的虚像误差与保存时一致。');
  } catch (e) {
    setAlert('err', `载入失败：${e.message}`);
  }
}

$('prevFrame').addEventListener('click', () => seek(state.frame - 1));
$('nextFrame').addEventListener('click', () => seek(state.frame + 1));

const toggles = [
  ['tglDirect', 'showDirect'],
  ['tglMirror', 'showMirror'],
  ['tglRail', 'showRail'],
  ['tglDistractor', 'showDistractors'],
  ['tglTruth', 'showTruth'],
];
for (const [id, key] of toggles) {
  $(id).addEventListener('change', (e) => {
    state[key] = e.target.checked;
    draw();
  });
}

window.addEventListener('resize', () => {
  draw();
  drawTimeline();
});

renderEvents();
boot();
