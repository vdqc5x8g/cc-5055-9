// Tiny software rasterizer for generating ground-truth clips without any native
// canvas dependency. Supports flat-shaded polygons, lines and discs with an
// optional clip rectangle. Coordinates are floating-point pixels, y down.

export function createRaster(width, height) {
  const rgb = new Uint8Array(width * height * 3);
  let clip = { x0: 0, y0: 0, x1: width, y1: height };
  let clipStack = [];

  function clear(c) {
    for (let i = 0; i < rgb.length; i += 3) {
      rgb[i] = c[0];
      rgb[i + 1] = c[1];
      rgb[i + 2] = c[2];
    }
  }

  function blendPixel(x, y, rgba) {
    if (x < clip.x0 || x >= clip.x1 || y < clip.y0 || y >= clip.y1) return;
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const a = rgba[3] ?? 255;
    const i = (y * width + x) * 3;
    if (a >= 255) {
      rgb[i] = rgba[0];
      rgb[i + 1] = rgba[1];
      rgb[i + 2] = rgba[2];
    } else {
      const t = a / 255;
      rgb[i] = rgb[i] * (1 - t) + rgba[0] * t;
      rgb[i + 1] = rgb[i + 1] * (1 - t) + rgba[1] * t;
      rgb[i + 2] = rgb[i + 2] * (1 - t) + rgba[2] * t;
    }
  }

  function fillPolygon(points, rgba) {
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const y0 = Math.max(clip.y0, Math.floor(minY));
    const y1 = Math.min(clip.y1 - 1, Math.ceil(maxY));
    const xs = [];
    for (let y = y0; y <= y1; y++) {
      const sy = y + 0.5;
      xs.length = 0;
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        if ((a.y <= sy && b.y > sy) || (b.y <= sy && a.y > sy)) {
          const t = (sy - a.y) / (b.y - a.y);
          xs.push(a.x + t * (b.x - a.x));
        }
      }
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        let xa = Math.ceil(xs[k] - 0.5);
        let xb = Math.floor(xs[k + 1] - 0.5);
        xa = Math.max(xa, clip.x0);
        xb = Math.min(xb, clip.x1 - 1);
        for (let x = xa; x <= xb; x++) blendPixel(x, y, rgba);
      }
    }
  }

  function strokePolygon(points, rgba, w = 1, closed = true) {
    const n = points.length;
    const last = closed ? n : n - 1;
    for (let i = 0; i < last; i++) line(points[i], points[(i + 1) % n], rgba, w);
  }

  function line(a, b, rgba, w = 1) {
    let x0 = Math.round(a.x);
    let y0 = Math.round(a.y);
    const x1 = Math.round(b.x);
    const y1 = Math.round(b.y);
    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    const r = Math.max(0, Math.floor((w - 1) / 2));
    for (;;) {
      for (let oy = -r; oy <= r; oy++) for (let ox = -r; ox <= r; ox++) blendPixel(x0 + ox, y0 + oy, rgba);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  function disc(cx, cy, radius, rgba) {
    const r = Math.ceil(radius);
    for (let y = -r; y <= r; y++) {
      for (let x = -r; x <= r; x++) {
        const d = Math.hypot(x, y);
        if (d > radius + 0.5) continue;
        // simple anti-aliased edge coverage
        const alpha = Math.max(0, Math.min(1, radius + 0.5 - d));
        blendPixel(
          Math.round(cx) + x,
          Math.round(cy) + y,
          [rgba[0], rgba[1], rgba[2], Math.round((rgba[3] ?? 255) * alpha)],
        );
      }
    }
  }

  function ring(cx, cy, radius, rgba, w = 1.5) {
    const r = Math.ceil(radius + w);
    for (let y = -r; y <= r; y++) {
      for (let x = -r; x <= r; x++) {
        const d = Math.hypot(x, y);
        const alpha = Math.max(0, Math.min(1, w / 2 + 0.5 - Math.abs(d - radius)));
        if (alpha <= 0) continue;
        blendPixel(
          Math.round(cx) + x,
          Math.round(cy) + y,
          [rgba[0], rgba[1], rgba[2], Math.round((rgba[3] ?? 255) * alpha)],
        );
      }
    }
  }

  function pushClipRect(x0, y0, x1, y1) {
    clipStack.push(clip);
    clip = {
      x0: Math.max(clip.x0, Math.floor(x0)),
      y0: Math.max(clip.y0, Math.floor(y0)),
      x1: Math.min(clip.x1, Math.ceil(x1)),
      y1: Math.min(clip.y1, Math.ceil(y1)),
    };
  }
  function popClip() {
    if (clipStack.length) clip = clipStack.pop();
  }

  return { width, height, rgb, clear, fillPolygon, strokePolygon, line, disc, ring, pushClipRect, popClip };
}
