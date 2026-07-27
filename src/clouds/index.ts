// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * TypeGPU Clouds backdrop — restored upstream sky + texture noise.
 * Holi via soft pastel sky/sun constants (not near-black + blown yellow).
 * https://docs.swmansion.com/TypeGPU/examples/#example=rendering--clouds
 */
import type { TgpuRoot } from 'typegpu';
import { common, d, std } from 'typegpu';
import {
  FOV_FACTOR,
  NOISE_TEXTURE_SIZE,
  SKY_HORIZON,
  SKY_ZENITH_TINT,
  SUN_BRIGHTNESS,
  SUN_DIRECTION,
  SUN_GLOW,
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
  for (let i = 0; i < noiseData.length; i += 1) {
    noiseData[i] = Math.floor(Math.random() * 255);
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

      // Upstream sun lobe (same math) — safe because sky midtones are soft pastels
      const sunDot = std.saturate(std.dot(rayDir, sunDir));
      const sunGlow = std.pow(
        sunDot,
        1 / (SUN_BRIGHTNESS * SUN_BRIGHTNESS * SUN_BRIGHTNESS),
      );

      // Upstream sky: continuous horizon/zenith + sun glow (no near-black base)
      let skyCol = SKY_HORIZON - SKY_ZENITH_TINT * rayDir.y * 0.35;
      skyCol += SUN_GLOW * sunGlow;

      const cloudCol = raymarch(rayOrigin, rayDir, sunDir);
      const finalCol = skyCol * (1.1 - cloudCol.a) + cloudCol.rgb;
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
          clearValue: [0.02, 0.01, 0.04, 1],
          loadOp: 'clear',
        })
        .draw(3);
    },
  };
}
