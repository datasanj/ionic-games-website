/**
 * Centrifuge 2 tunnel layer (upstream TypeGPU intensity, darks keyed to alpha).
 *
 * Steering works like flying, not like a screen-space warp. The pointer offset
 * is imprinted on a ring at the moment it spawns at the far end, and then
 * rides in with that ring: moving the cursor carves a path, and you watch the
 * path you carved flow toward you. See STEERING below.
 *
 * Perf: everything that never changes at runtime (march depth, strip counts,
 * dolly, camera, colour) is a module constant rather than a uniform field, so
 * the march bound is static and the strip maths const-folds. Only time,
 * aspect and the history cursor live in the per-frame uniform, and the trail
 * buffer is only re-uploaded when it actually advances.
 */
import type { TgpuRoot } from 'typegpu';
import tgpu, { d } from 'typegpu';
import {
  abs,
  add,
  atan2,
  clamp,
  cos,
  div,
  floor,
  gt,
  length,
  mix,
  mul,
  normalize,
  select,
  sign,
  smoothstep,
  saturate,
  sub,
  tanh,
} from 'typegpu/std';

const safeTanh = (v: d.v3f) => {
  'use gpu';
  return select(tanh(v), sign(v), gt(abs(v), d.vec3f(10)));
};

/** Upstream Centrifuge 2 march/geometry constants — fixed for the whole run. */
const TUNNEL_DEPTH = 55;
const TUNNEL_RADIUS = 11;
const MOVE_SPEED = 5;
const BIG_STRIPS = 10;
const SMALL_STRIPS = 5;
const DOLLY_ZOOM = 0.2;
/** Camera sits off-axis inside the tube and never follows the pointer. */
const CAMERA_Y = -7;
/** Upstream-style modulation with Holi-leaning magenta bias. */
const TUNNEL_COLOR = d.vec3f(0.32, 0.06, 0.28);

/* ---------------------------------- STEERING ---------------------------- */

/**
 * The strip phase is `p.z * DOLLY_ZOOM - MOVE_SPEED * time`, so a ring at a
 * fixed phase satisfies `p.z = 5·phase + 25·t`: rings travel toward the camera
 * at exactly MOVE_SPEED / DOLLY_ZOOM world units per second. That constant is
 * what lets a stored offset stay glued to its ring.
 */
const RING_SPEED = MOVE_SPEED / DOLLY_ZOOM;
const INV_RING_SPEED = 1 / RING_SPEED;

/**
 * Axial depth at which a ring counts as newly spawned. Sits just past the far
 * end of what the march actually lights up (measured: the vanishing point
 * draws its light from axial depth ~66), so a cursor move shows up almost
 * immediately at the back of the tube instead of waiting offscreen.
 */
const SPAWN_DEPTH = 85;
/** Seconds for a ring to travel from SPAWN_DEPTH to the camera. */
const TRAVEL_TIME = SPAWN_DEPTH / RING_SPEED;

/** Trail resolution. 64 entries over TRAVEL_TIME ≈ one sample per 1.3 world
 *  units, i.e. ~5 samples across the finest longitudinal strip feature. */
const HISTORY_LEN = 64;
const HISTORY_DT = TRAVEL_TIME / (HISTORY_LEN - 1);
const INV_HISTORY_DT = 1 / HISTORY_DT;
/** Converts an axial depth delta straight into a trail index delta. */
const AGE_TO_INDEX = INV_RING_SPEED * INV_HISTORY_DT;

/** Pointer NDC → world offset of the tube centre. */
const STEER_GAIN = 3.2;
/**
 * The offset now persists all the way in, so it also displaces the tube around
 * the camera. The camera sits 7 units off-axis in an 11-unit tube, so the
 * offset has to stay under ~4 or the camera ends up outside the wall and the
 * strip maths degenerates. 3.4 keeps ~0.6 units of margin at the worst-case
 * diagonal.
 */
const MAX_OFFSET = 3.4;

export type TunnelLayer = {
  /** steerX/steerY are smoothed pointer NDC (−1‥1, y-up). */
  draw: (timestamp: number, steerX: number, steerY: number) => void;
  resize: () => void;
};

export function createTunnelLayer(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
): TunnelLayer {
  const Params = d.struct({
    time: d.f32,
    aspectRatio: d.f32,
    /** Sub-step phase of the trail, 0‥1, so it slides instead of stepping. */
    histFrac: d.f32,
    pad: d.f32,
  });

  /**
   * Trail offsets, stored pre-paired: entry i holds (offset[i], offset[i+1]).
   * The march always interpolates between neighbours, so pairing them turns
   * the lookup into a single 16-byte load instead of two loads at a divergent
   * index — measured at ~0.27 ms/frame on a 2400x1500 target. Read-only
   * storage rather than uniform because the index varies per pixel.
   */
  const History = d.struct({
    pairs: d.arrayOf(d.vec4f, HISTORY_LEN),
  });

  const paramsUniform = root.createUniform(Params, {
    time: 0,
    aspectRatio: 1,
    histFrac: 0,
    pad: 0,
  });

  const histX = new Float32Array(HISTORY_LEN);
  const histY = new Float32Array(HISTORY_LEN);
  const pairs: d.v4f[] = [];
  for (let i = 0; i < HISTORY_LEN; i++) pairs.push(d.vec4f(0, 0, 0, 0));
  const historyValue = { pairs };
  const historyBuffer = root.createReadonly(History, historyValue);

  const rebuildPairs = () => {
    for (let i = 0; i < HISTORY_LEN; i++) {
      const j = i + 1 < HISTORY_LEN ? i + 1 : HISTORY_LEN - 1;
      const p = pairs[i];
      p.x = histX[i];
      p.y = histY[i];
      p.z = histX[j];
      p.w = histY[j];
    }
  };

  const fragmentMain = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })(({ uv }) => {
    'use gpu';
    const params = paramsUniform.$;
    const trail = historyBuffer.$;
    const ratio = d.vec2f(params.aspectRatio, 1);
    // Fixed look — the camera itself never follows the pointer
    const dir = normalize(d.vec3f(mul(uv, ratio), -1));
    const time = params.time;
    const timeSlide = mul(MOVE_SPEED, time);

    // Trail index is affine in the march distance:
    //   axialDepth = -dir.z * z, age = (SPAWN_DEPTH - axialDepth)/RING_SPEED
    //   hp = age/HISTORY_DT - histFrac  =  hpBase - hpSlope * z
    // so both coefficients hoist out of the march entirely.
    const hpBase = sub(SPAWN_DEPTH * AGE_TO_INDEX, params.histFrac);
    const hpSlope = mul(mul(dir.z, -1), AGE_TO_INDEX);

    let z = d.f32(0);
    let acc = d.vec3f();
    for (let i = 0; i < TUNNEL_DEPTH; i++) {
      const p = mul(dir, z);
      p.y += CAMERA_Y;

      // Offset this slice of tube by whatever the cursor was doing when this
      // ring spawned, interpolated between trail samples so it flows smoothly
      // instead of stepping.
      const hp = clamp(sub(hpBase, mul(hpSlope, z)), 0, HISTORY_LEN - 1);
      const hi = floor(hp);
      const pair = trail.pairs[d.i32(hi)];
      const off = mix(pair.xy, pair.zw, sub(hp, hi));
      p.x += off.x;
      p.y += off.y;

      // Classic centrifuge coords (upstream TypeGPU / XorDev)
      const coords = d.vec3f(
        add(mul(atan2(p.y, p.x), BIG_STRIPS), time),
        sub(mul(p.z, DOLLY_ZOOM), timeSlide),
        sub(length(p.xy), TUNNEL_RADIUS),
      );

      const coords2 = sub(cos(add(coords, cos(mul(coords, SMALL_STRIPS)))), 1);
      // Upstream dd — no artificial clamp that softens strips
      const dd = sub(mul(length(d.vec4f(coords.z, coords2)), 0.5), 0.1);

      // Upstream colour accumulation: (1.2 - cos(color * p.z)) / dd.
      // One reciprocal instead of three scalar divides.
      const invDd = div(1, dd);
      acc = add(acc, mul(sub(1.2, cos(mul(TUNNEL_COLOR, p.z))), invDd));
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

  const context = root.configureContext({
    canvas,
    alphaMode: 'premultiplied',
  });

  const pipeline = root.createRenderPipeline({
    vertex: vertexMain,
    fragment: fragmentMain,
  });

  // Hoisted so the draw loop allocates nothing per frame.
  const attachment = {
    view: context,
    clearValue: [0, 0, 0, 0],
    loadOp: 'clear' as const,
  };
  const patch = { time: 0, aspectRatio: 1, histFrac: 0, pad: 0 };

  /** Wall-clock (seconds) at which the newest trail sample was taken. */
  let lastSampleTime = -1;

  return {
    resize() {
      const h = canvas.clientHeight || canvas.height || 1;
      patch.aspectRatio = (canvas.clientWidth || canvas.width) / h;
    },
    draw(timestamp: number, steerX: number, steerY: number) {
      const now = timestamp * 0.001;

      // Negate: the march samples p + offset, so the tube centre lands at
      // −offset — this way the tunnel bends toward the cursor.
      let ox = -steerX * STEER_GAIN;
      let oy = -steerY * STEER_GAIN;
      const mag = Math.hypot(ox, oy);
      if (mag > MAX_OFFSET) {
        const s = MAX_OFFSET / mag;
        ox *= s;
        oy *= s;
      }

      let dirty = false;
      if (lastSampleTime < 0 || now - lastSampleTime > TRAVEL_TIME) {
        // First frame, or the tab was hidden long enough that the whole trail
        // is stale — refill rather than spinning the shift loop.
        histX.fill(ox);
        histY.fill(oy);
        lastSampleTime = now;
        dirty = true;
      } else {
        while (now - lastSampleTime >= HISTORY_DT) {
          lastSampleTime += HISTORY_DT;
          histX.copyWithin(1, 0, HISTORY_LEN - 1);
          histY.copyWithin(1, 0, HISTORY_LEN - 1);
          histX[0] = ox;
          histY[0] = oy;
          dirty = true;
        }
      }
      // ~19 uploads/sec instead of one per frame.
      if (dirty) {
        rebuildPairs();
        historyBuffer.write(historyValue);
      }

      patch.time = now % 1000;
      patch.histFrac = (now - lastSampleTime) * INV_HISTORY_DT;
      paramsUniform.write(patch);
      pipeline.withColorAttachment(attachment).draw(3);
    },
  };
}
