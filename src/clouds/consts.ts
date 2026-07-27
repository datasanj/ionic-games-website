// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * Clouds constants — original TypeGPU structure + Holi-tinted albedos/sky.
 * Keep sky midtones soft (like upstream pastels) so sunDot terminator
 * never reads as a hard half-plane diagonal.
 */
import { d } from 'typegpu';

export const FOV_FACTOR = 1;

/** Match upstream: mostly +X so terminator is vertical-ish, not through logo */
export const SUN_DIRECTION = d.vec3f(1.0, 0.12, 0.18);
/** Upstream 0.9 — exponent ~1.37 keeps sunGlow a soft lobe, not a half-plane fill */
export const SUN_BRIGHTNESS = 0.9;
export const LIGHT_ABSORPTION = 0.88;

export const CLOUD_COVERAGE = 0.7;
export const CLOUD_AMPLITUDE = 1.0;
export const CLOUD_FREQUENCY = 1.4;
export const WIND_SPEED = 0.55;

export const FBM_OCTAVES = 3;
export const FBM_PERSISTENCE = 0.5;
export const FBM_LACUNARITY = 2.0;

/** Holi gulal pink — lit powder */
export const CLOUD_BRIGHT = d.vec3f(1.0, 0.45, 0.85);
/** Festival violet — shadow folds */
export const CLOUD_DARK = d.vec3f(0.28, 0.12, 0.72);
/** Soft saffron accent mixed by lightVal only (no screen UV) */
export const HOLI_SAFFRON = d.vec3f(1.0, 0.55, 0.12);
/** Mehndi lime accent */
export const HOLI_LIME = d.vec3f(0.45, 0.95, 0.35);

/** Upstream-style fill (soft, not near-black) */
export const SKY_AMBIENT = d.vec3f(0.55, 0.4, 0.7);
export const SUN_COLOR = d.vec3f(1.0, 0.72, 0.28);

/** Soft Holi sky — pastel midtones like upstream SKY_HORIZON / SUN_GLOW */
export const SKY_HORIZON = d.vec3f(0.55, 0.28, 0.62);
export const SKY_ZENITH_TINT = d.vec3f(0.85, 0.35, 0.55);
export const SUN_GLOW = d.vec3f(1.0, 0.5, 0.22);

export const NOISE_Z_OFFSET = d.vec2f(37.0, 239.0);
export const NOISE_TEXTURE_SIZE = 256;
