/**
 * Dark clouds (backdrop) + tunnel streaks over them + centered logo.
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
  max,
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

  // Match DPR on both layers — mismatched resolution can create edge seams when stacked
  resizeCanvas(tunnelCanvas, { maxDpr: 1 });
  resizeCanvas(cloudsCanvas, { maxDpr: 1 });

  const root = await tgpu.init({
    adapter: { powerPreference: 'high-performance' },
  });

  const Params = d.struct({
    time: d.f32,
    aspectRatio: d.f32,
    cameraPos: d.vec2f,
    tunnelDepth: d.i32,
    bigStrips: d.f32,
    smallStrips: d.f32,
    dollyZoom: d.f32,
  });

  const paramsUniform = root.createUniform(Params, {
    time: 0,
    aspectRatio: tunnelCanvas.clientWidth / tunnelCanvas.clientHeight,
    cameraPos: d.vec2f(0, -7),
    tunnelDepth: 28,
    // Integer fold count keeps cos(n*atan2) continuous across the branch cut
    bigStrips: 8,
    smallStrips: 4,
    dollyZoom: 0.2,
  });

  const tunnelRadius = 11;
  const moveSpeed = 5;

  const fragmentMain = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    'use gpu';
    const params = paramsUniform.$;
    const ratio = d.vec2f(params.aspectRatio, 1);
    const dir = normalize(d.vec3f(mul(uv, ratio), -1));

    let z = d.f32(0);
    let acc = d.vec3f();
    for (let i = 0; i < params.tunnelDepth; i++) {
      const p = mul(dir, z);
      p.x += params.cameraPos.x;
      p.y += params.cameraPos.y;

      // Angular coord — integer bigStrips keeps cos continuous across atan2 cut;
      // still clamp step so FP noise at the cut can't carve a dark radial seam
      const ang = mul(atan2(p.y, p.x), params.bigStrips);
      const coords = d.vec3f(
        add(ang, params.time),
        sub(mul(p.z, params.dollyZoom), mul(moveSpeed, params.time)),
        sub(length(p.xy), tunnelRadius),
      );

      const coords2 = sub(
        cos(add(coords, cos(mul(coords, params.smallStrips)))),
        1,
      );
      // Soft floor on dd — prevents 1/dd singularities / dark seam lines
      const ddRaw = sub(mul(length(d.vec4f(coords.z, coords2)), 0.5), 0.1);
      const dd = max(ddRaw, 0.035);

      // Holi multi-hue along depth (not a single magenta)
      const hueT = add(mul(p.z, 0.08), mul(params.time, 0.15));
      const streakCol = d.vec3f(
        add(0.55, mul(0.45, cos(hueT))),
        add(0.2, mul(0.55, cos(add(hueT, 2.1)))),
        add(0.35, mul(0.65, cos(add(hueT, 4.2)))),
      );
      acc = add(acc, div(mul(streakCol, 1.35), dd));
      z = add(z, dd);
    }

    acc = safeTanh(mul(acc, 0.0045));
    const luma = add(
      add(mul(acc.x, 0.2126), mul(acc.y, 0.7152)),
      mul(acc.z, 0.0722),
    );
    // High floor: only bright streaks composite — dark strip edges won't
    // punch thin dark lines through the cloud sun glow
    const alpha = saturate(smoothstep(0.1, 0.32, luma));
    // Soften edges further so strip boundaries don't read as hard seams
    const soft = mul(alpha, alpha);
    return d.vec4f(mul(acc, soft), soft);
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

  const onResize = () => {
    resizeCanvas(tunnelCanvas, { maxDpr: 1 });
    resizeCanvas(cloudsCanvas, { maxDpr: 1 });
    clouds.resize();
  };
  window.addEventListener('resize', onResize);

  function draw(timestamp: number) {
    if (!running) return;

    paramsUniform.patch({
      aspectRatio: tunnelCanvas.clientWidth / tunnelCanvas.clientHeight,
      time: (timestamp * 0.001) % 1000,
    });

    try {
      clouds.draw(timestamp);
      tunnelPipeline
        .withColorAttachment({
          view: tunnelContext,
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
        })
        .draw(3);
    } catch (err) {
      // TypeGPU resolve/compile can throw once on bad HMR; keep the loop alive
      // so a refresh/recover doesn't leave a permanent black frame.
      console.error(err);
    }

    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);

  window.addEventListener('pagehide', () => {
    running = false;
    window.removeEventListener('resize', onResize);
    root.destroy();
  });
}

start().catch((err) => {
  console.error(err);
  fallback?.removeAttribute('hidden');
});
