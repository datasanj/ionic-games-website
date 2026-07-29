/**
 * Clouds backdrop + centrifuge tunnel (transparent darks) + centered logo.
 * Tunnel restored to upstream TypeGPU Centrifuge 2 intensity; darks keyed
 * softly so Holi clouds show through without killing bright streaks.
 * Pointer biases only far/incoming ring generation (depth-scaled); near
 * tunnel and camera stay fixed.
 *
 * Perf: half-res clouds (CSS upscale), alternate-frame cloud updates,
 * visibility pause — tunnel stays full-rate / full quality.
 */
import tgpu from 'typegpu';
import { createCloudsLayer } from './clouds/index.ts';
import { createTunnelLayer } from './tunnel.ts';
import './style.css';

const tunnelEl = document.querySelector<HTMLCanvasElement>('#tunnel');
const cloudsEl = document.querySelector<HTMLCanvasElement>('#clouds');
const fallback = document.querySelector<HTMLParagraphElement>('.fallback');

if (!tunnelEl || !cloudsEl) {
  throw new Error('Background canvases not found');
}

const tunnelCanvas = tunnelEl;
const cloudsCanvas = cloudsEl;

/** Soft volumes upscale cleanly; keep tunnel sharper than clouds. */
const TUNNEL_MAX_DPR = 1.25;
const CLOUDS_MAX_DPR = 1;
const CLOUDS_SCALE = 0.5;

function resizeCanvas(
  target: HTMLCanvasElement,
  opts: { maxDpr?: number; scale?: number } = {},
) {
  const maxDpr = opts.maxDpr ?? 1.25;
  const scale = opts.scale ?? 1;
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr) * scale;
  const width = Math.max(1, Math.floor(window.innerWidth * dpr));
  const height = Math.max(1, Math.floor(window.innerHeight * dpr));
  if (target.width !== width || target.height !== height) {
    target.width = width;
    target.height = height;
  }
}

async function start() {
  if (!navigator.gpu) {
    fallback?.removeAttribute('hidden');
    return;
  }

  resizeCanvas(tunnelCanvas, { maxDpr: TUNNEL_MAX_DPR });
  resizeCanvas(cloudsCanvas, { maxDpr: CLOUDS_MAX_DPR, scale: CLOUDS_SCALE });

  const root = await tgpu.init({
    adapter: { powerPreference: 'high-performance' },
  });

  const tunnel = createTunnelLayer(root, tunnelCanvas);
  tunnel.resize();

  const clouds = createCloudsLayer(root, cloudsCanvas);
  clouds.resize();

  let running = true;
  let frameIndex = 0;
  let lastTs = 0;
  /** EMA of frame time (ms). Target budget ~8.3ms for 120Hz. */
  let emaFrameMs = 8.3;
  /**
   * Cloud cadence: 1 = every frame, 2 = every other (default), 3 = every third.
   * Soft wind makes skipped frames nearly invisible; canvas retains prior pixels.
   */
  let cloudStride = 2;

  /** Pointer in NDC (−1‥1, y-up). Default center when no pointer / after leave. */
  let pointerTargetX = 0;
  let pointerTargetY = 0;
  let pointerSmoothX = 0;
  let pointerSmoothY = 0;
  const POINTER_SMOOTH = 0.1;

  const onResize = () => {
    resizeCanvas(tunnelCanvas, { maxDpr: TUNNEL_MAX_DPR });
    resizeCanvas(cloudsCanvas, { maxDpr: CLOUDS_MAX_DPR, scale: CLOUDS_SCALE });
    tunnel.resize();
    clouds.resize();
  };
  window.addEventListener('resize', onResize);

  /**
   * Restarting the loop on `visible` without tracking the outstanding handle
   * can leave two loops running, since a callback queued before the tab was
   * hidden may still be pending. Tracking the handle makes the loop
   * single-instance by construction.
   */
  let frameHandle = 0;
  const schedule = () => {
    if (frameHandle === 0) frameHandle = requestAnimationFrame(draw);
  };

  const onVisibility = () => {
    if (document.visibilityState === 'visible' && running) {
      lastTs = 0;
      schedule();
    }
  };
  document.addEventListener('visibilitychange', onVisibility);

  const setPointerFromClient = (clientX: number, clientY: number) => {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    pointerTargetX = (clientX / w) * 2 - 1;
    pointerTargetY = -((clientY / h) * 2 - 1);
  };

  const onPointerMove = (e: PointerEvent) => {
    setPointerFromClient(e.clientX, e.clientY);
  };
  const onPointerLeave = () => {
    pointerTargetX = 0;
    pointerTargetY = 0;
  };
  // pointermove covers mouse + touch/pen; leave resets to center
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  document.documentElement.addEventListener('pointerleave', onPointerLeave);

  function draw(timestamp: number) {
    frameHandle = 0;
    if (!running) return;
    if (document.visibilityState === 'hidden') return;

    if (lastTs > 0) {
      const dt = Math.min(33, timestamp - lastTs);
      emaFrameMs = emaFrameMs * 0.9 + dt * 0.1;
      // Adapt only cloud cadence — never touch tunnel quality
      if (emaFrameMs > 10.5) cloudStride = 3;
      else if (emaFrameMs < 7.2) cloudStride = 2;
    }
    lastTs = timestamp;

    pointerSmoothX += (pointerTargetX - pointerSmoothX) * POINTER_SMOOTH;
    pointerSmoothY += (pointerTargetY - pointerSmoothY) * POINTER_SMOOTH;

    try {
      // Half-res clouds + stride: ~4–8× less cloud GPU work vs every-frame 1×
      if (frameIndex % cloudStride === 0) {
        clouds.draw(timestamp);
      }
      tunnel.draw(timestamp, pointerSmoothX, pointerSmoothY);
    } catch (err) {
      console.error(err);
    }

    frameIndex += 1;
    schedule();
  }

  schedule();

  window.addEventListener('pagehide', () => {
    running = false;
    if (frameHandle !== 0) {
      cancelAnimationFrame(frameHandle);
      frameHandle = 0;
    }
    window.removeEventListener('resize', onResize);
    window.removeEventListener('pointermove', onPointerMove);
    document.documentElement.removeEventListener('pointerleave', onPointerLeave);
    document.removeEventListener('visibilitychange', onVisibility);
    root.destroy();
  });
}

start().catch((err) => {
  console.error(err);
  fallback?.removeAttribute('hidden');
});
