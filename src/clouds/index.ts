// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * TypeGPU Clouds backdrop — original composite + Holi color constants only.
 * https://docs.swmansion.com/TypeGPU/examples/#example=rendering--clouds
 */
import type { TgpuRoot } from 'typegpu';
import { common, d, std } from 'typegpu';
import {
  FOV_FACTOR,
  NOISE_TEXTURE_SIZE,
  SKY_BASE,
  SKY_WARM,
  SUN_BRIGHTNESS,
  SUN_COLOR,
  SUN_DIRECTION,
  WIND_SPEED,
} from './consts.ts';
import { cloudsLayout, CloudsParams } from './types.ts';
import { raymarch } from './utils.ts';

export type CloudsLayer = {
  draw: (timestamp: number) => void;
  resize: () => void;
};

export function createCloudsLayer(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
): CloudsLayer {
  const context = root.configureContext({
    canvas,
    alphaMode: 'premultiplied',
  });
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const paramsUniform = root.createUniform(CloudsParams, {
    time: 0,
    maxSteps: 48,
    maxDistance: 9.5,
  });
  const resolutionUniform = root.createUniform(
    d.vec2f,
    d.vec2f(canvas.width, canvas.height),
  );

  const noiseData = new Uint8Array(NOISE_TEXTURE_SIZE * NOISE_TEXTURE_SIZE);
  // Soft value-noise field (not pure white noise) — avoids diagonal texel crawl
  const raw = new Float32Array(NOISE_TEXTURE_SIZE * NOISE_TEXTURE_SIZE);
  for (let i = 0; i < raw.length; i += 1) {
    raw[i] = Math.random();
  }
  for (let y = 0; y < NOISE_TEXTURE_SIZE; y += 1) {
    for (let x = 0; x < NOISE_TEXTURE_SIZE; x += 1) {
      let sum = 0;
      let count = 0;
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          const ix = (x + ox + NOISE_TEXTURE_SIZE) % NOISE_TEXTURE_SIZE;
          const iy = (y + oy + NOISE_TEXTURE_SIZE) % NOISE_TEXTURE_SIZE;
          sum += raw[iy * NOISE_TEXTURE_SIZE + ix];
          count += 1;
        }
      }
      noiseData[y * NOISE_TEXTURE_SIZE + x] = Math.floor((sum / count) * 255);
    }
  }

  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

  const noiseTexture = root
    .createTexture({
      size: [NOISE_TEXTURE_SIZE, NOISE_TEXTURE_SIZE],
      format: 'r8unorm',
    })
    .$usage('sampled', 'render');
  noiseTexture.write(noiseData);

  const bindGroup = root.createBindGroup(cloudsLayout, {
    params: paramsUniform.buffer,
    noiseTexture,
    sampler,
  });

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: ({ uv }) => {
      'use gpu';
      const screenRes = resolutionUniform.$;
      const aspect = screenRes.x / screenRes.y;

      let screenPos = (uv - 0.5) * 2;
      screenPos = d.vec2f(
        screenPos.x * std.max(aspect, 1),
        screenPos.y * std.max(1 / aspect, 1),
      );

      const sunDir = std.normalize(SUN_DIRECTION);
      const time = cloudsLayout.$.params.time;
      const rayOrigin = d.vec3f(
        std.sin(time * 0.6) * 0.5,
        std.cos(time * 0.8) * 0.5 - 1,
        time * WIND_SPEED,
      );
      const rayDir = std.normalize(d.vec3f(screenPos.x, screenPos.y, FOV_FACTOR));

      const sunDot = std.saturate(std.dot(rayDir, sunDir));
      const sunGlow = std.pow(
        sunDot,
        1 / (SUN_BRIGHTNESS * SUN_BRIGHTNESS * SUN_BRIGHTNESS),
      );

      let skyCol = SKY_BASE + SKY_WARM * (1.0 - std.abs(rayDir.y)) * 0.55;
      skyCol += SUN_COLOR * sunGlow * 0.65;

      const cloudCol = raymarch(rayOrigin, rayDir, sunDir);
      let finalCol = skyCol * (1.1 - cloudCol.a) + cloudCol.rgb;
      // Break 8-bit ordered-dither / banding on soft gradients (not visible grain)
      const debands =
        (std.fract(std.sin(std.dot(uv, d.vec2f(12.9898, 78.233))) * 43758.5453) -
          0.5) *
        (1.5 / 255.0);
      finalCol += d.vec3f(debands);
      return d.vec4f(finalCol, 1.0);
    },
    targets: { format: presentationFormat },
  });

  return {
    resize() {
      resolutionUniform.write(d.vec2f(canvas.width, canvas.height));
    },
    draw(timestamp: number) {
      paramsUniform.patch({ time: (timestamp / 1000) % 500 });
      pipeline
        .with(bindGroup)
        .withColorAttachment({
          view: context,
          clearValue: [0.012, 0.01, 0.055, 1],
          loadOp: 'clear',
        })
        .draw(3);
    },
  };
}
