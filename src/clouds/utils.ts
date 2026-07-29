// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * TypeGPU Clouds raymarch — upstream noise/FBM structure.
 * Holi: density picks dark vs airy pigment families; soft world-space
 * sin/cos phases pick which Holi color within each family (exclusive-ish
 * weights so mixes don't mud into brown). lightVal is luminance only —
 * never hue — so the sun half-plane can't become a color diagonal.
 * https://docs.swmansion.com/TypeGPU/examples/#example=rendering--clouds
 */
import { tgpu, d, std } from 'typegpu';
import {
  CLOUD_AMPLITUDE,
  CLOUD_BRIGHT,
  CLOUD_COVERAGE,
  CLOUD_DARK,
  CLOUD_FREQUENCY,
  CLOUD_MAX_DISTANCE,
  CLOUD_MAX_STEPS,
  FBM_LACUNARITY,
  FBM_OCTAVES,
  FBM_PERSISTENCE,
  HOLI_CRIMSON,
  HOLI_LIME,
  HOLI_MAGENTA,
  HOLI_PLUM,
  HOLI_SAFFRON,
  HOLI_YELLOW,
  INV_NOISE_TEXTURE_SIZE,
  LIGHT_ABSORPTION,
  NOISE_Z_OFFSET,
  SUN_BRIGHTNESS,
} from './consts.ts';
import { cloudsLayout } from './types.ts';

const noise3d = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  'use gpu';
  // floor(p) + fract(p) reconstructs p exactly, so index the xy lattice from
  // pos directly, and only the z lerp weight is actually needed.
  const idxZ = std.floor(pos.z);
  const fracZ = pos.z - idxZ;
  const smoothZ = fracZ * fracZ * (3 - 2 * fracZ);

  // 1/256 is exact in binary, so this matches the original divide bit for bit.
  const texCoord0 = std.fract(
    (pos.xy + NOISE_Z_OFFSET * idxZ) * INV_NOISE_TEXTURE_SIZE,
  );
  const texCoord1 = std.fract(
    (pos.xy + NOISE_Z_OFFSET * (idxZ + 1)) * INV_NOISE_TEXTURE_SIZE,
  );

  const val0 = std.textureSampleLevel(
    cloudsLayout.$.noiseTexture,
    cloudsLayout.$.sampler,
    texCoord0,
    0,
  ).x;

  const val1 = std.textureSampleLevel(
    cloudsLayout.$.noiseTexture,
    cloudsLayout.$.sampler,
    texCoord1,
    0,
  ).x;

  return std.mix(val0, val1, smoothZ) * 2 - 1;
});

/**
 * How far every octave after the first can move the FBM sum. noise3d is in
 * [-1, 1], so this is just the sum of the remaining amplitudes, plus a slack
 * term so float rounding can never make the bound optimistic.
 */
const TAIL_BOUND = (() => {
  let sum = 1e-4;
  for (let i = 1; i < FBM_OCTAVES; i++) {
    sum += Math.abs(CLOUD_AMPLITUDE * FBM_PERSISTENCE ** i);
  }
  return sum;
})();

/** Adds octaves 1..N-1 onto an already-computed octave 0, in the original order. */
function fbmFrom(head: number, pos: d.v3f): number {
  'use gpu';
  let sum = head;

  for (const i of tgpu.unroll(std.range(FBM_OCTAVES - 1))) {
    sum +=
      noise3d(pos * (CLOUD_FREQUENCY * FBM_LACUNARITY ** (i + 1))) *
      (CLOUD_AMPLITUDE * FBM_PERSISTENCE ** (i + 1));
  }

  return sum;
}

/**
 * Density of the volume at `pos`; <= 0 means "nothing here".
 *
 * Octave 0 carries 2/3 of the FBM amplitude, and the tail octaves can only
 * move the sum by ±TAIL_BOUND. When octave 0 alone already puts the sum
 * outside the window where saturate() can still respond, the tail's four
 * texture fetches are skipped. This is exact rather than an approximation:
 * in the empty case the caller's `density > 0` test fails either way, and in
 * the saturated case the returned 0.5 is the value saturate() would produce.
 * The unskipped path sums octaves in the original order, so it is bit-exact.
 */
const sampleDensity = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  'use gpu';
  const coverage = CLOUD_COVERAGE - std.abs(pos.y) * 0.25;
  const head = noise3d(pos * CLOUD_FREQUENCY) * CLOUD_AMPLITUDE;
  const headSum = head + coverage;

  if (headSum + TAIL_BOUND <= 0.5) {
    return -0.5;
  }
  if (headSum - TAIL_BOUND >= 1) {
    return 0.5;
  }
  return std.saturate(fbmFrom(head, pos) + coverage) - 0.5;
});

const sampleDensityCheap = (pos: d.v3f): number => {
  'use gpu';
  const noise = noise3d(pos * CLOUD_FREQUENCY) * CLOUD_AMPLITUDE;
  return std.saturate(noise + CLOUD_COVERAGE - 0.5);
};

export const raymarch = tgpu.fn(
  [d.vec3f, d.vec3f, d.vec3f],
  d.vec4f,
)((rayOrigin, rayDir, sunDir) => {
  'use gpu';
  let accum = d.vec4f();

  const maxDepth = CLOUD_MAX_DISTANCE;
  const stepSize = 1 / CLOUD_MAX_STEPS;
  // Fixed offset (upstream uses randf) — avoids locked screen-door dither
  let dist = stepSize * 0.5;

  for (let i = 0; i < CLOUD_MAX_STEPS; i++) {
    const samplePos = rayOrigin + rayDir * dist * maxDepth;
    const cloudDensity = sampleDensity(samplePos);

    if (cloudDensity > 0.0) {
      const shadowPos = samplePos + sunDir;
      const shadowDensity = sampleDensityCheap(shadowPos);
      const shadow = std.saturate(cloudDensity - shadowDensity);
      // Luminance only — deep folds ↔ bright powder rims
      const lightVal = std.mix(0.05, 1.4, shadow);

      const fold = cloudDensity * cloudDensity;
      const airy = 1.0 - fold;

      // Soft world-space phases — z/y-weighted so they don't form a screen-X half-plane
      const phaseA =
        0.5 +
        0.5 *
          std.sin(samplePos.z * 0.48 + samplePos.y * 0.36 + samplePos.x * 0.18);
      const phaseB =
        0.5 +
        0.5 *
          std.cos(samplePos.z * 0.4 - samplePos.y * 0.44 + samplePos.x * 0.16);
      // Bright vs dark powder family (mostly along wind/depth, not screen X)
      const phaseC =
        0.5 + 0.5 * std.sin(samplePos.z * 0.52 + samplePos.y * 0.3);

      // Sharpen phases toward exclusive pigment picks (less mud)
      const darkPick = std.smoothstep(0.32, 0.68, phaseA);
      const brightPick = std.smoothstep(0.32, 0.68, phaseB);
      const brightFamily = std.smoothstep(0.4, 0.6, phaseC);
      const festPick = std.smoothstep(0.42, 0.58, phaseA * 0.5 + phaseB * 0.5);

      // Near-neutral luminance — Holi albedos keep their hue
      const lightAmt =
        std.mix(0.03, 1.35, lightVal) *
        std.mix(0.38, 1.28, airy) *
        SUN_BRIGHTNESS;

      // Never let-alias d.vec3f consts — TypeGPU black-screens on that
      let color = std.mix(CLOUD_BRIGHT, CLOUD_DARK, fold);

      // Dense folds always get crimson↔plum (dark pigment punch)
      color = std.mix(
        color,
        std.mix(HOLI_CRIMSON, HOLI_PLUM, darkPick),
        fold * 0.92,
      );

      // Airy body → plum↔magenta purple (kills beige saffron fills)
      color = std.mix(
        color,
        std.mix(HOLI_PLUM, HOLI_MAGENTA, brightPick),
        brightFamily * (0.5 + airy * 0.4),
      );

      // Yellow↔lime as Holi sparks only — not large beige slabs
      color = std.mix(
        color,
        std.mix(HOLI_YELLOW, HOLI_LIME, brightPick),
        brightFamily * airy * 0.22,
      );

      // Extra dark-family mid volumes so crimson/plum aren't only tiny folds
      color = std.mix(
        color,
        std.mix(HOLI_CRIMSON, HOLI_PLUM, darkPick),
        (1.0 - brightFamily) * airy * 0.4,
      );

      // Magenta / deep-violet accents (HOLI_SAFFRON is violet here)
      color = std.mix(
        color,
        HOLI_MAGENTA,
        festPick * (1.0 - brightFamily) * 0.42,
      );
      color = std.mix(
        color,
        HOLI_SAFFRON,
        (1.0 - festPick) * brightFamily * airy * 0.36,
      );

      const lit = color * lightAmt;

      const contrib =
        d.vec4f(lit, 1) * cloudDensity * (LIGHT_ABSORPTION - accum.a);
      accum += contrib;

      if (accum.a >= LIGHT_ABSORPTION - 0.001) {
        break;
      }
    }
    dist += stepSize;
  }
  return accum;
});
