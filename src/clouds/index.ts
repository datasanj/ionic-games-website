// @ts-nocheck — TypeGPU operator overloads are transformed by unplugin-typegpu
/**
 * TypeGPU Clouds backdrop — Holi color constants, isotropic hash density.
 * https://docs.swmansion.com/TypeGPU/examples/#example=rendering--clouds
 */
import type { TgpuRoot } from 'typegpu';
import { common, d, std } from 'typegpu';
import {
  FOV_FACTOR,
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

  // Placeholder 1×1 — density uses hash noise; layout still expects a texture binding
  const noiseTexture = root
    .createTexture({
      size: [1, 1],
      format: 'r8unorm',
    })
    .$usage('sampled', 'render');
  noiseTexture.write(new Uint8Array([128]));

  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

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

      let skyCol = SKY_BASE + SKY_WARM * (1.0 - std.abs(rayDir.y)) * 0.42;
      skyCol += SUN_COLOR * sunGlow * 0.7;
      // Cool cyan lift opposite the warm sun — more Holi sky contrast
      skyCol += d.vec3f(0.02, 0.12, 0.22) * std.saturate(-rayDir.x * 0.5 + 0.2);

      const cloudCol = raymarch(rayOrigin, rayDir, sunDir);
      // No screen-space UV hash deband — locked dither reads as hatch on soft gradients
      const finalCol = skyCol * (1.0 - cloudCol.a) + cloudCol.rgb;
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
          clearValue: [0.006, 0.005, 0.032, 1],
          loadOp: 'clear',
        })
        .draw(3);
    },
  };
}
