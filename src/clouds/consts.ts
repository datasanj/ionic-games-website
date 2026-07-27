// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * Clouds constants — upstream TypeGPU sky structure + Holi volume albedos.
 * Sky/sun stay soft pastels so sunDot never reads as a hard half-plane.
 * Punch lives in CLOUD_* / lighting, not sky terminator contrast.
 */
import { d } from 'typegpu';

export const FOV_FACTOR = 1;

/** Match upstream: mostly +X so terminator is vertical-ish, not through logo */
export const SUN_DIRECTION = d.vec3f(1.0, 0.1, 0.15);
/** Upstream 0.9 — exponent ~1.37 keeps sunGlow a soft lobe, not a half-plane fill */
export const SUN_BRIGHTNESS = 0.9;
export const LIGHT_ABSORPTION = 0.93;

/** Structure for bright rims vs pigment folds */
export const CLOUD_COVERAGE = 0.72;
export const CLOUD_AMPLITUDE = 1.35;
export const CLOUD_FREQUENCY = 1.5;
export const WIND_SPEED = 0.55;

export const FBM_OCTAVES = 3;
export const FBM_PERSISTENCE = 0.5;
export const FBM_LACUNARITY = 2.0;

/** Lit gulal rim base — bright powder */
export const CLOUD_BRIGHT = d.vec3f(1.0, 0.88, 0.98);
/** Near-black fold base before crimson/plum */
export const CLOUD_DARK = d.vec3f(0.03, 0.005, 0.04);

/** Dense shadow volumes — Holi crimson ink */
export const HOLI_CRIMSON = d.vec3f(0.95, 0.02, 0.1);
/** Dense shadow volumes — deep plum / violet ink */
export const HOLI_PLUM = d.vec3f(0.38, 0.01, 0.55);
/** Lit low-density rim — bright gulal yellow */
export const HOLI_YELLOW = d.vec3f(1.0, 0.95, 0.08);
/** Lit / mid powder — mehndi lime */
export const HOLI_LIME = d.vec3f(0.15, 1.0, 0.2);
/** Mid powder — festival magenta */
export const HOLI_MAGENTA = d.vec3f(1.0, 0.06, 0.58);
/** Warm gulal orange-saffron */
export const HOLI_SAFFRON = d.vec3f(1.0, 0.42, 0.02);

/**
 * Soft lavender sky close to upstream pastels (not near-black, not blown yellow).
 * Mild Holi bias only — leaves room for volume pigments to read.
 */
export const SKY_AMBIENT = d.vec3f(0.58, 0.42, 0.72);
export const SUN_COLOR = d.vec3f(1.0, 0.72, 0.32);
export const SKY_HORIZON = d.vec3f(0.68, 0.48, 0.78);
export const SKY_ZENITH_TINT = d.vec3f(0.9, 0.55, 0.5);
/** Soft peach lobe only — keep mild so it never reads as a half-plane fill */
export const SUN_GLOW = d.vec3f(0.78, 0.4, 0.3);

export const NOISE_Z_OFFSET = d.vec2f(37.0, 239.0);
export const NOISE_TEXTURE_SIZE = 256;
