import { tgpu, d } from 'typegpu';

/** Step count / depth are compile-time constants; only time varies. */
export const CloudsParams = d.struct({
  time: d.f32,
});

export const cloudsLayout = tgpu.bindGroupLayout({
  params: { uniform: CloudsParams },
  noiseTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
});
