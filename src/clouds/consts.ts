// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
import { d } from 'typegpu';

export const FOV_FACTOR = 1;

export const SUN_DIRECTION = d.vec3f(1.0, 0.15, 0.25);
export const SUN_BRIGHTNESS = 1.05;
export const LIGHT_ABSORPTION = 0.88;

export const CLOUD_COVERAGE = 0.72;
export const CLOUD_AMPLITUDE = 1.0;
export const CLOUD_FREQUENCY = 1.15;
export const WIND_SPEED = 0.4;

export const FBM_OCTAVES = 3;
export const FBM_PERSISTENCE = 0.5;
export const FBM_LACUNARITY = 2.0;

/** Holi gulal — vivid pink / magenta (lit powder) */
export const CLOUD_BRIGHT = d.vec3f(1.0, 0.18, 0.72);
/** Holi indigo — deep festival blue (dense / shadowed powder) */
export const CLOUD_DARK = d.vec3f(0.08, 0.04, 0.78);
/** Holi saffron / marigold — warm pigment swirl (world-space mix) */
export const HOLI_SAFFRON = d.vec3f(1.0, 0.52, 0.04);
/** Holi violet — rich purple powder accent */
export const HOLI_VIOLET = d.vec3f(0.62, 0.06, 0.92);

/** Mehndi green ambient fill */
export const SKY_AMBIENT = d.vec3f(0.28, 0.92, 0.38);
/** Saffron sun */
export const SUN_COLOR = d.vec3f(1.0, 0.68, 0.08);
/** Deep indigo night sky */
export const SKY_BASE = d.vec3f(0.035, 0.008, 0.09);
/** Marigold / saffron horizon wash */
export const SKY_WARM = d.vec3f(1.0, 0.38, 0.06);

export const NOISE_Z_OFFSET = d.vec2f(37.0, 239.0);
export const NOISE_TEXTURE_SIZE = 256;
