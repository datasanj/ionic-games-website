// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * TypeGPU Clouds raymarch — upstream noise/FBM structure.
 * Holi dynamic range via albedo + lightVal only (no screen-space fract,
 * no multi-noise pigment stacks). Sky terminator stays soft in index.ts.
 * https://docs.swmansion.com/TypeGPU/examples/#example=rendering--clouds
 */
import { tgpu, d, std } from 'typegpu';
import {
  CLOUD_AMPLITUDE,
  CLOUD_BRIGHT,
  CLOUD_COVERAGE,
  CLOUD_DARK,
  CLOUD_FREQUENCY,
  FBM_LACUNARITY,
  FBM_OCTAVES,
  FBM_PERSISTENCE,
  HOLI_LIME,
  HOLI_MAGENTA,
  HOLI_SAFFRON,
  LIGHT_ABSORPTION,
  NOISE_TEXTURE_SIZE,
  NOISE_Z_OFFSET,
  SKY_AMBIENT,
  SUN_BRIGHTNESS,
  SUN_COLOR,
} from './consts.ts';
import { cloudsLayout } from './types.ts';

const noise3d = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  'use gpu';
  const idx = std.floor(pos);
  const frac = std.fract(pos);
  const smooth = frac * frac * (3 - 2 * frac);

  const texCoord0 = std.fract(
    (idx.xy + frac.xy + NOISE_Z_OFFSET * idx.z) / NOISE_TEXTURE_SIZE,
  );
  const texCoord1 = std.fract(
    (idx.xy + frac.xy + NOISE_Z_OFFSET * (idx.z + 1)) / NOISE_TEXTURE_SIZE,
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

  return std.mix(val0, val1, smooth.z) * 2 - 1;
});

function fbm(pos: d.v3f): number {
  'use gpu';
  let sum = d.f32();

  for (const i of tgpu.unroll(std.range(FBM_OCTAVES))) {
    sum +=
      noise3d(pos * (CLOUD_FREQUENCY * FBM_LACUNARITY ** i)) *
      (CLOUD_AMPLITUDE * FBM_PERSISTENCE ** i);
  }

  return sum;
}

const sampleDensity = tgpu.fn(
  [d.vec3f],
  d.f32,
)((pos) => {
  const coverage = CLOUD_COVERAGE - std.abs(pos.y) * 0.25;
  return std.saturate(fbm(pos) + coverage) - 0.5;
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

  const params = cloudsLayout.$.params;
  const maxSteps = params.maxSteps;
  const maxDepth = params.maxDistance;

  const stepSize = 1 / maxSteps;
  // Fixed offset (upstream uses randf) — avoids locked screen-door dither
  let dist = stepSize * 0.5;

  for (let i = 0; i < maxSteps; i++) {
    const samplePos = rayOrigin + rayDir * dist * maxDepth;
    const cloudDensity = sampleDensity(samplePos);

    if (cloudDensity > 0.0) {
      const shadowPos = samplePos + sunDir;
      const shadowDensity = sampleDensityCheap(shadowPos);
      const shadow = std.saturate(cloudDensity - shadowDensity);
      // Wide DR: deep shaded folds ↔ hot lit powder rims
      const lightVal = std.mix(0.06, 1.35, shadow);

      // Density^2 pushes pigment cores darker vs bright edges
      const fold = cloudDensity * cloudDensity;
      const light =
        SKY_AMBIENT * (0.55 + lightVal * 0.55) +
        SUN_COLOR * lightVal * SUN_BRIGHTNESS * 1.35;

      // Upstream albedo mix + Holi accents from lighting/density only
      let color = std.mix(CLOUD_BRIGHT, CLOUD_DARK, fold);
      color = std.mix(color, HOLI_SAFFRON, lightVal * (1.0 - fold) * 0.55);
      color = std.mix(color, HOLI_MAGENTA, lightVal * fold * 0.35);
      color = std.mix(color, HOLI_LIME, (1.0 - lightVal) * fold * 0.4);
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
