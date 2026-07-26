// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * Original TypeGPU clouds raymarch — Holi colors only via albedo/light constants.
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
  HOLI_CYAN,
  HOLI_INDIGO,
  HOLI_LIME,
  HOLI_RED,
  HOLI_SAFFRON,
  HOLI_VIOLET,
  HOLI_YELLOW,
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
  // Fixed offset — avoids locked dither patterns / diagonal screen-door
  let dist = stepSize * 0.5;

  for (let i = 0; i < maxSteps; i++) {
    const samplePos = rayOrigin + rayDir * dist * maxDepth;
    const cloudDensity = sampleDensity(samplePos);

    if (cloudDensity > 0.0) {
      const shadowPos = samplePos + sunDir;
      const shadowDensity = sampleDensityCheap(shadowPos);
      const shadow = std.saturate(cloudDensity - shadowDensity);
      // Deep folds vs glowing lit rims (luminance contrast)
      const lightVal = std.mix(0.06, 1.35, shadow);

      const light = SKY_AMBIENT * 0.55 + SUN_COLOR * lightVal * SUN_BRIGHTNESS;
      // Soft multi-pigment albedo from continuous world-space noise (no UV fract stripes)
      const n1 = noise3d(samplePos * 0.48) * 0.5 + 0.5;
      const n2 = noise3d(samplePos * 0.33 + d.vec3f(19.1, 7.3, 11.7)) * 0.5 + 0.5;
      const n3 = noise3d(samplePos * 0.21 + d.vec3f(3.7, 23.9, 5.1)) * 0.5 + 0.5;

      const warm = std.mix(HOLI_RED, std.mix(HOLI_SAFFRON, HOLI_YELLOW, n2), n1);
      const cool = std.mix(HOLI_LIME, HOLI_CYAN, n1);
      const magenta = std.mix(CLOUD_BRIGHT, HOLI_VIOLET, n2);
      const warmCool = std.mix(warm, cool, std.smoothstep(0.22, 0.78, n3));
      const brightAlbedo = std.mix(warmCool, magenta, std.smoothstep(0.32, 0.82, n2));
      const darkAlbedo = std.mix(CLOUD_DARK, HOLI_INDIGO, n1);
      // Dense cores → deep teal/indigo; thin edges keep saturated pigment
      const color = std.mix(brightAlbedo, darkAlbedo, std.smoothstep(0.04, 0.58, cloudDensity));
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
