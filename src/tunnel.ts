/**
 * Centrifuge 2 tunnel layer (upstream TypeGPU intensity, darks keyed to alpha).
 *
 * Steering works like flying, not like a screen-space warp. A direction is
 * imprinted on a ring at the moment it spawns at the far end and then rides in
 * with that ring: moving the cursor carves a path, and you watch the path you
 * carved flow toward you. The cursor sets the *target* direction — unprojected
 * so that a fully settled tube points its vanishing point straight at the
 * pointer — and the spawn direction drifts toward it at a bounded rate, which
 * is what keeps the carved path a smooth continuous curve instead of a kink.
 * See STEERING, UNPROJECTION and DRIFT below.
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

/* ------------------------------ UNPROJECTION ----------------------------
 *
 * Where the cursor actually is, in world units, solved from the shader's own
 * ray construction rather than tuned by hand.
 *
 * The march samples `p = dir*z + (0, CAMERA_Y, 0) + off` with
 * `dir = normalize(vec3(uv * vec2(aspect, 1), -1))`, and the tube is
 * `length(p.xy) - TUNNEL_RADIUS`, so the tube axis sits at world `-off`.
 * Because `dir * z` collapses to `vec3(uv.x*aspect, uv.y, -1) * D` at axial
 * depth `D = -dir.z * z`, the camera ray through pointer NDC (mx, my) passes
 * through world `(mx*aspect*D, my*D + CAMERA_Y)`. Equating the two:
 *
 *   off(D) = (-mx*aspect*D, -my*D - CAMERA_Y)
 *
 * The lateral term is *linear in depth*, which is the whole game. A constant
 * offset — what this used to apply — projects to `off/D` and therefore decays
 * to nothing at the far end: it can never move the vanishing point no matter
 * how large it is, which is why the old effect read as invisible. It also
 * could not be made large: exact unprojection of a corner cursor as a constant
 * offset needs 157 world units, putting the camera 153 units from the axis of
 * an 11-unit tube (a radius of ~153 would be required to contain it).
 *
 * The depth-linear form has none of that trouble. `off(0) = 0`, so the axis
 * passes through the same place it always did and the camera keeps its 7-unit
 * off-axis perch inside the wall by construction — the wall-clipping clamp
 * that used to cap this at 3.4 units is simply not a constraint any more.
 * ------------------------------------------------------------------------ */

/**
 * Tilt at which the soft knee starts, and the hard ceiling it eases toward.
 * A 16:10 viewport reaches tilt 1.51 at the corner of the middle 80% of the
 * screen and 1.89 at the true corner, so the knee sits above the former: the
 * whole usable middle of the viewport tracks the cursor exactly, and only the
 * last sliver toward the corners is softened.
 */
const STEER_KNEE = 1.55;
const MAX_STEER = 2.1;

/* --------------------------------- DRIFT --------------------------------
 *
 * The unprojected cursor is a *target*, not a position to jump to.
 *
 * Because the lateral offset is depth-linear, any change in the imprinted
 * direction is multiplied by the depth it lands at, so a fast pointer move
 * shears the tube apart. Rings sit one `2π/(DOLLY_ZOOM*SMALL_STRIPS)` = 6.28
 * world units apart, which is `6.28/RING_SPEED` = 0.25 s of trail, so a
 * direction changing at rate r shifts the centres of adjacent rings by
 * `400 * r * 0.25` screen pixels against an apparent ring radius of only
 * ~52 px at the far end. Following the raw pointer (smoothed at τ≈0.17 s,
 * so r can hit ~9/s) means a 900 px stagger — seventeen ring radii, which
 * reads as a shattered tube rather than a tunnel.
 *
 * So the spawn direction eases toward the target instead: a slew cap bounds
 * the curvature during the big sweep, and an exponential tail lands it on the
 * target without the velocity step a bare slew limit would leave behind.
 * ------------------------------------------------------------------------ */

/**
 * Max change in tube tilt per second. At 0.28 the centres of adjacent rings
 * land 28 px apart, about half the 52 px apparent radius of a ring at the far
 * end and a third of it midway, so the centreline still reads as one
 * continuous curve — while the swing itself passes 7be5b9d's entire 16 px
 * range within the first half second and keeps going to ~440 px.
 */
const DRIFT_SLEW = 0.28;
/** Exponential time constant for the final approach, once inside the slew cap. */
const DRIFT_TAU = 1.4;
/**
 * A hidden tab or a stalled frame must not cash in as one huge drift step and
 * put a crease in the tube.
 */
const MAX_DRIFT_DT = 0.1;

/** Writes into `out` rather than returning, to keep the draw loop allocation-free. */
export const steerFor = (
  ndcX: number,
  ndcY: number,
  aspect: number,
  out: { x: number; y: number },
) => {
  let sx = -ndcX * aspect;
  let sy = -ndcY;
  // Tangent of the angle between the tube axis and the view axis. Full
  // unprojection (gain 1) holds until the tube is tilted far enough that the
  // march starts skating along the wall; past the knee it eases off.
  const tilt = Math.hypot(sx, sy);
  if (tilt > STEER_KNEE) {
    const room = MAX_STEER - STEER_KNEE;
    const eased = STEER_KNEE + room * Math.tanh((tilt - STEER_KNEE) / room);
    const scale = eased / tilt;
    sx *= scale;
    sy *= scale;
  }
  out.x = sx;
  out.y = sy;
};

/**
 * The `-CAMERA_Y` term of the unprojection is a constant, so applying it at
 * full strength would drag the axis through the camera and flatten the
 * off-axis Centrifuge perch that gives the near wall its rush. Ramping it in
 * over the first stretch of depth keeps the near geometry exactly as it was
 * while letting the far end sit precisely on the cursor ray. Smoothstep, not a
 * clamp, so the axis has no kink at a fixed depth for a stationary ring
 * artifact to latch onto.
 */
const AXIS_SETTLE_DEPTH = 26;
const INV_AXIS_SETTLE_DEPTH = 1 / AXIS_SETTLE_DEPTH;
const AXIS_LIFT = -CAMERA_Y;

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
    // Axial depth is affine in the march distance: depth = dz * z.
    const dz = mul(dir.z, -1);
    const hpBase = sub(SPAWN_DEPTH * AGE_TO_INDEX, params.histFrac);
    const hpSlope = mul(dz, AGE_TO_INDEX);
    const settleSlope = mul(dz, INV_AXIS_SETTLE_DEPTH);

    let z = d.f32(0);
    let acc = d.vec3f();
    for (let i = 0; i < TUNNEL_DEPTH; i++) {
      const p = mul(dir, z);
      p.y += CAMERA_Y;
      const depth = mul(dz, z);

      // Offset this slice of tube by whatever the cursor was doing when this
      // ring spawned, interpolated between trail samples so it flows smoothly
      // instead of stepping.
      const hp = clamp(sub(hpBase, mul(hpSlope, z)), 0, HISTORY_LEN - 1);
      const hi = floor(hp);
      const pair = trail.pairs[d.i32(hi)];
      const steer = mix(pair.xy, pair.zw, sub(hp, hi));

      // Unprojection: the axis rides the camera ray through the cursor, so the
      // lateral offset scales with depth and the ring this slice belongs to is
      // centred exactly where the pointer was when it spawned.
      const settle = saturate(mul(settleSlope, z));
      p.x += mul(steer.x, depth);
      p.y += add(
        mul(steer.y, depth),
        mul(AXIS_LIFT, mul(mul(settle, settle), sub(3, mul(2, settle)))),
      );

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
  const steerTarget = { x: 0, y: 0 };
  /** The direction actually imprinted on rings, easing toward the target. */
  let driftX = 0;
  let driftY = 0;
  let lastDrawTime = -1;

  /** Wall-clock (seconds) at which the newest trail sample was taken. */
  let lastSampleTime = -1;

  return {
    resize() {
      const h = canvas.clientHeight || canvas.height || 1;
      patch.aspectRatio = (canvas.clientWidth || canvas.width) / h;
    },
    draw(timestamp: number, steerX: number, steerY: number) {
      const now = timestamp * 0.001;

      // Negated inside steerFor: the march samples p + offset, so the tube
      // centre lands at −offset — this way the tunnel bends toward the cursor.
      steerFor(steerX, steerY, patch.aspectRatio, steerTarget);

      // Ease the spawn direction toward that target rather than assigning it,
      // so consecutive rings stay a smooth curve apart. Slew cap first (bounds
      // the bend), exponential tail second (lands without a velocity step).
      const dt =
        lastDrawTime < 0 ? 0 : Math.min(MAX_DRIFT_DT, now - lastDrawTime);
      lastDrawTime = now;
      if (dt > 0) {
        const ease = 1 - Math.exp(-dt / DRIFT_TAU);
        let stepX = (steerTarget.x - driftX) * ease;
        let stepY = (steerTarget.y - driftY) * ease;
        const step = Math.hypot(stepX, stepY);
        const maxStep = DRIFT_SLEW * dt;
        if (step > maxStep) {
          const k = maxStep / step;
          stepX *= k;
          stepY *= k;
        }
        driftX += stepX;
        driftY += stepY;
      }
      const ox = driftX;
      const oy = driftY;

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
