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
import tgpu, { d } from 'typegpu';
import {
  abs,
  add,
  atan2,
  cos,
  div,
  gt,
  length,
  mul,
  normalize,
  select,
  sign,
  smoothstep,
  saturate,
  sub,
  tanh,
} from 'typegpu/std';
import { createCloudsLayer } from './clouds/index.ts';
import './style.css';

const safeTanh = (v: d.v3f) => {
  'use gpu';
  return select(tanh(v), sign(v), gt(abs(v), d.vec3f(10)));
};

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

  const Params = d.struct({
    time: d.f32,
    aspectRatio: d.f32,
    cameraPos: d.vec2f,
    /** Max XY bend of the tunnel path at vanishing depth (pointer units). */
    steerOffset: d.vec2f,
    tunnelDepth: d.i32,
    bigStrips: d.f32,
    smallStrips: d.f32,
    dollyZoom: d.f32,
    color: d.vec3f,
  });

  // Upstream Centrifuge 2 defaults (depth 50, tone 0.005, color formula)
  const paramsUniform = root.createUniform(Params, {
    time: 0,
    aspectRatio: tunnelCanvas.clientWidth / tunnelCanvas.clientHeight,
    cameraPos: d.vec2f(0, -7),
    steerOffset: d.vec2f(0, 0),
    tunnelDepth: 55,
    bigStrips: 10,
    smallStrips: 5,
    dollyZoom: 0.2,
    // Upstream-style modulation with Holi-leaning magenta bias
    color: d.vec3f(0.32, 0.06, 0.28),
  });

  const tunnelRadius = 11;
  const moveSpeed = 5;
  /** Max far-end path bend in tunnel XY (pointer −1‥1 → world offset). */
  const FAR_STEER = 5.5;
  /** March distance where pointer influence begins (near/mid field stays put). */
  const STEER_NEAR_Z = 24;
  /** March distance where pointer influence is full (vanishing rings). */
  const STEER_FAR_Z = 52;
  const BASE_CAMERA_Y = -7;

  const fragmentMain = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    'use gpu';
    const params = paramsUniform.$;
    const ratio = d.vec2f(params.aspectRatio, 1);
    // Fixed look — camera / near tunnel never follow the pointer
    const dir = normalize(d.vec3f(mul(uv, ratio), -1));

    let z = d.f32(0);
    let acc = d.vec3f();
    for (let i = 0; i < params.tunnelDepth; i++) {
      const p = mul(dir, z);
      p.x += params.cameraPos.x;
      p.y += params.cameraPos.y;

      // Far-only steer: 0 through near/mid, ramps hard at vanishing distance.
      // Squared smoothstep keeps mid rings locked; only deep incoming rings bend.
      const farT = smoothstep(d.f32(STEER_NEAR_Z), d.f32(STEER_FAR_Z), z);
      const farW = mul(farT, farT);
      p.x += mul(params.steerOffset.x, farW);
      p.y += mul(params.steerOffset.y, farW);

      // Classic centrifuge coords (upstream TypeGPU / XorDev)
      const coords = d.vec3f(
        add(mul(atan2(p.y, p.x), params.bigStrips), params.time),
        sub(mul(p.z, params.dollyZoom), mul(moveSpeed, params.time)),
        sub(length(p.xy), tunnelRadius),
      );

      const coords2 = sub(
        cos(add(coords, cos(mul(coords, params.smallStrips)))),
        1,
      );
      // Upstream dd — no artificial clamp that softens strips
      const dd = sub(mul(length(d.vec4f(coords.z, coords2)), 0.5), 0.1);

      // Upstream color accumulation: (1.2 - cos(color * p.z)) / dd
      acc = add(acc, div(sub(1.2, cos(mul(params.color, p.z))), dd));
      z = add(z, dd);
    }

    // Upstream tone mapping (0.005) — restores strip brightness/contrast
    acc = safeTanh(mul(acc, 0.005));
    const luma = add(
      add(mul(acc.x, 0.2126), mul(acc.y, 0.7152)),
      mul(acc.z, 0.0722),
    );
    // Soft luminance key: true darks transparent for Holi clouds; mid/bright
    // streaks keep near-full intensity (no squared alpha floor)
    const alpha = saturate(smoothstep(0.003, 0.09, luma));
    return d.vec4f(mul(acc, alpha), alpha);
  });

  const vertexMain = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
  })((input) => {
    const pos = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
    return {
      pos: d.vec4f(pos[input.vertexIndex], 0, 1),
      uv: pos[input.vertexIndex],
    };
  });

  const tunnelContext = root.configureContext({
    canvas: tunnelCanvas,
    alphaMode: 'premultiplied',
  });

  const tunnelPipeline = root.createRenderPipeline({
    vertex: vertexMain,
    fragment: fragmentMain,
  });

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
    clouds.resize();
  };
  window.addEventListener('resize', onResize);

  const onVisibility = () => {
    if (document.visibilityState === 'visible' && running) {
      lastTs = 0;
      requestAnimationFrame(draw);
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

    // Positive steer: far rings bend toward cursor; camera / near field fixed
    paramsUniform.patch({
      aspectRatio: tunnelCanvas.clientWidth / tunnelCanvas.clientHeight,
      time: (timestamp * 0.001) % 1000,
      cameraPos: d.vec2f(0, BASE_CAMERA_Y),
      // Negate: SDF samples p+offset ⇒ cylinder axis at −offset; match cursor
      steerOffset: d.vec2f(
        -pointerSmoothX * FAR_STEER,
        -pointerSmoothY * FAR_STEER,
      ),
    });

    try {
      // Half-res clouds + stride: ~4–8× less cloud GPU work vs every-frame 1×
      if (frameIndex % cloudStride === 0) {
        clouds.draw(timestamp);
      }
      tunnelPipeline
        .withColorAttachment({
          view: tunnelContext,
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
        })
        .draw(3);
    } catch (err) {
      console.error(err);
    }

    frameIndex += 1;
    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);

  window.addEventListener('pagehide', () => {
    running = false;
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
