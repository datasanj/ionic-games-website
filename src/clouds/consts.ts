// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
import { d } from 'typegpu';

export const FOV_FACTOR = 1;

export const SUN_DIRECTION = d.vec3f(0.85, 0.35, 0.3);
/** Hotter rim light for glowing powder edges */
export const SUN_BRIGHTNESS = 1.55;
export const LIGHT_ABSORPTION = 0.86;

export const CLOUD_COVERAGE = 0.7;
export const CLOUD_AMPLITUDE = 1.05;
export const CLOUD_FREQUENCY = 1.05;
export const WIND_SPEED = 0.38;

export const FBM_OCTAVES = 3;
export const FBM_PERSISTENCE = 0.5;
export const FBM_LACUNARITY = 2.0;

/** Holi gulal — intense magenta / pink (lit powder rims) */
export const CLOUD_BRIGHT = d.vec3f(1.0, 0.06, 0.78);
/** Near-black teal folds — max luminance contrast in shadows */
export const CLOUD_DARK = d.vec3f(0.004, 0.02, 0.08);
/** Blood red → fire start of warm plume */
export const HOLI_RED = d.vec3f(1.0, 0.04, 0.05);
/** Holi saffron / marigold — fiery orange pigment */
export const HOLI_SAFFRON = d.vec3f(1.0, 0.42, 0.02);
/** Incandescent yellow highlight on warm billows */
export const HOLI_YELLOW = d.vec3f(1.0, 0.95, 0.1);
/** Mehndi lime — electric green pigment */
export const HOLI_LIME = d.vec3f(0.32, 1.0, 0.06);
/** Electric cyan / turquoise cool plume */
export const HOLI_CYAN = d.vec3f(0.02, 0.95, 0.92);
/** Rich festival violet / purple accent */
export const HOLI_VIOLET = d.vec3f(0.72, 0.04, 1.0);
/** Near-black indigo in deepest folds */
export const HOLI_INDIGO = d.vec3f(0.015, 0.008, 0.12);

/** Cool teal ambient fill (keeps folds dark, not washed out) */
export const SKY_AMBIENT = d.vec3f(0.05, 0.32, 0.4);
/** Hot saffron–yellow sun for glowing edges */
export const SUN_COLOR = d.vec3f(1.0, 0.9, 0.22);
/** Deep indigo night sky */
export const SKY_BASE = d.vec3f(0.005, 0.004, 0.028);
/** Marigold / saffron horizon wash */
export const SKY_WARM = d.vec3f(1.0, 0.38, 0.04);

/** Unused placeholder size (density is hash-based now). */
export const NOISE_TEXTURE_SIZE = 1;
