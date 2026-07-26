// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * Clouds raymarch — Holi multi-pigment via soft world-space phase + density/light.
 * Density uses isotropic hash value-noise (no 2D-texture Z-shear).
 */
import { tgpu, d, std } from 'typegpu';
import {
  CLOUD_AMPLITUDE,
  CLOUD_COVERAGE,
  CLOUD_DARK,
  CLOUD_FREQUENCY,
  FBM_LACUNARITY,
  FBM_PERSISTENCE,
  HOLI_CYAN,
  HOLI_INDIGO,
  HOLI_LIME,
  HOLI_SAFFRON,
  HOLI_VIOLET,
  HOLI_YELLOW,
  CLOUD_BRIGHT,
  LIGHT_ABSORPTION,
  SKY_AMBIENT,
  SUN_BRIGHTNESS,
  SUN_COLOR,
} from './consts.ts';
import { cloudsLayout } from './types.ts';

/** Lattice hash → [-1, 1]. No texture UV shear, so no diagonal hatch. */
const hash31 = tgpu.fn(
  [d.vec3f],
  d.f32,
)((p) => {
  'use gpu';
  const p3 = std.fract(p * 0.1031);
  const p3b = p3 + std.dot(p3, p3.yzx + 33.33);
  return std.fract((p3b.x + p3b.y) * p3b.z) * 2 - 1;
});

/** Isotropic trilinear value noise (replaces sheared NOISE_Z_OFFSET texture pack). */
const noise3d = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  'use gpu';
  const i = std.floor(pos);
  const f = std.fract(pos);
  const u = f * f * (3 - 2 * f);

  const n000 = hash31(i);
  const n100 = hash31(i + d.vec3f(1, 0, 0));
  const n010 = hash31(i + d.vec3f(0, 1, 0));
  const n110 = hash31(i + d.vec3f(1, 1, 0));
  const n001 = hash31(i + d.vec3f(0, 0, 1));
  const n101 = hash31(i + d.vec3f(1, 0, 1));
  const n011 = hash31(i + d.vec3f(0, 1, 1));
  const n111 = hash31(i + d.vec3f(1, 1, 1));

  const x00 = std.mix(n000, n100, u.x);
  const x10 = std.mix(n010, n110, u.x);
  const x01 = std.mix(n001, n101, u.x);
  const x11 = std.mix(n011, n111, u.x);
  const y0 = std.mix(x00, x10, u.y);
  const y1 = std.mix(x01, x11, u.y);
  return std.mix(y0, y1, u.z);
});

function fbm(pos: d.v3f): number {
  'use gpu';
  let sum = noise3d(pos * CLOUD_FREQUENCY) * CLOUD_AMPLITUDE;
  sum +=
    noise3d(pos.yzx * (CLOUD_FREQUENCY * FBM_LACUNARITY)) *
    (CLOUD_AMPLITUDE * FBM_PERSISTENCE);
  sum +=
    noise3d(pos.zxy * (CLOUD_FREQUENCY * FBM_LACUNARITY * FBM_LACUNARITY)) *
    (CLOUD_AMPLITUDE * FBM_PERSISTENCE * FBM_PERSISTENCE);
  return sum;
}

const sampleDensity = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  'use gpu';
  const coverage = CLOUD_COVERAGE - std.abs(pos.y) * 0.22;
  return std.saturate(fbm(pos) + coverage) - 0.48;
});

const sampleDensityCheap = (pos: d.v3f): number => {
  'use gpu';
  const noise = noise3d(pos * CLOUD_FREQUENCY) * CLOUD_AMPLITUDE;
  return std.saturate(noise + CLOUD_COVERAGE - 0.5);
};

/**
 * Holi powder palette via soft world-space sin/cos phase.
 * No screen-space fract UV, no multi-noise3d pigment mix (those caused hatch).
 * Wide smoothstep bands → distinct magenta / saffron / yellow / lime / cyan / violet at once.
 */
const holiPigment = tgpu.fn(
  [d.vec3f, d.f32, d.f32],
  d.vec3f,
)((pos, densT, lightVal) => {
  'use gpu';
  // Smooth periodic fields only — no fract(), no mid-freq noise stacks
  const phase =
    std.sin(pos.x * 0.62 + pos.z * 0.41) * 0.42 +
    std.cos(pos.y * 0.55 + pos.x * 0.28) * 0.38 +
    std.sin(pos.z * 0.33 + pos.y * 0.47) * 0.28 +
    densT * 0.45 +
    lightVal * 0.18;
  const t = std.saturate(phase * 0.55 + 0.5);

  // Use pigment consts directly — aliasing d.vec3f refs into locals
  // can fail TypeGPU resolution ("references cannot be assigned to let").
  const c0 = std.mix(CLOUD_BRIGHT, HOLI_SAFFRON, std.smoothstep(0.05, 0.22, t));
  const c1 = std.mix(c0, HOLI_YELLOW, std.smoothstep(0.2, 0.38, t));
  const c2 = std.mix(c1, HOLI_LIME, std.smoothstep(0.36, 0.54, t));
  const c3 = std.mix(c2, HOLI_CYAN, std.smoothstep(0.52, 0.7, t));
  const c4 = std.mix(c3, HOLI_VIOLET, std.smoothstep(0.68, 0.88, t));
  // Wrap-ish return toward magenta for hot lit rims
  return std.mix(c4, CLOUD_BRIGHT, std.smoothstep(0.86, 1.0, t) * 0.55);
});

export const raymarch = tgpu.fn(
  [d.vec3f, d.vec3f, d.vec3f],
  d.vec4f,
)((rayOrigin, rayDir, sunDir) => {
  'use gpu';
  let accum = d.vec4f();

  const params = cloudsLayout.$.params;
  const maxSteps = params.maxSteps;
  const maxDepth = params.maxDistance;

  const stepSize = 1 / maxSteps;
  let dist = stepSize * 0.5;

  for (let i = 0; i < maxSteps; i++) {
    const samplePos = rayOrigin + rayDir * dist * maxDepth;
    const cloudDensity = sampleDensity(samplePos);

    if (cloudDensity > 0.0) {
      const shadowPos = samplePos + sunDir;
      const shadowDensity = sampleDensityCheap(shadowPos);
      const shadow = std.saturate(cloudDensity - shadowDensity);
      // Wide luminance range — deep indigo folds vs glowing powder rims
      const lightVal = std.mix(0.015, 1.75, shadow);

      const light = SKY_AMBIENT * 0.28 + SUN_COLOR * lightVal * SUN_BRIGHTNESS;
      const densT = std.smoothstep(0.0, 0.58, cloudDensity);
      const powder = holiPigment(samplePos, densT, lightVal);
      const darkAlbedo = std.mix(CLOUD_DARK, HOLI_INDIGO, densT);
      // Lit powder vs near-black folds
      const color = std.mix(
        powder,
        darkAlbedo,
        densT * (1.0 - std.smoothstep(0.15, 0.9, lightVal)),
      );
      const lit = color * light;

      const contrib = d.vec4f(lit, 1) * cloudDensity * (LIGHT_ABSORPTION - accum.a);
      accum += contrib;

      if (accum.a >= LIGHT_ABSORPTION - 0.001) {
        break;
      }
    }
    dist += stepSize;
  }
  return accum;
});
