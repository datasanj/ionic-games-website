// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
import { d } from 'typegpu';

export const FOV_FACTOR = 1;

export const SUN_DIRECTION = d.vec3f(1.0, 0.15, 0.25);
export const SUN_BRIGHTNESS = 0.9;
export const LIGHT_ABSORPTION = 0.88;

export const CLOUD_COVERAGE = 0.72;
export const CLOUD_AMPLITUDE = 1.0;
export const CLOUD_FREQUENCY = 1.15;
export const WIND_SPEED = 0.4;

export const FBM_OCTAVES = 3;
export const FBM_PERSISTENCE = 0.5;
export const FBM_LACUNARITY = 2.0;

/** Holi albedo — higher contrast so soft flats don't Bayer-dither */
export const CLOUD_BRIGHT = d.vec3f(1.0, 0.55, 0.95);
export const CLOUD_DARK = d.vec3f(0.25, 0.15, 0.9);

export const SKY_AMBIENT = d.vec3f(0.55, 0.75, 0.4);
export const SUN_COLOR = d.vec3f(1.0, 0.75, 0.15);
export const SKY_BASE = d.vec3f(0.04, 0.015, 0.07);
export const SKY_WARM = d.vec3f(1.0, 0.45, 0.2);

export const NOISE_Z_OFFSET = d.vec2f(37.0, 239.0);
export const NOISE_TEXTURE_SIZE = 256;
