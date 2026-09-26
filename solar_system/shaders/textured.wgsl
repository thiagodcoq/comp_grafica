struct Matrix {
  vertex: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) pos: vec2<f32>,
            @location(1) texcoord: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.clip_position = global.projection * (matrix[instance_index].vertex * vec4<f32>(pos, 0.0, 1.0));
  out.texcoord = texcoord;
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  let color = textureSample(tex, samp, in.texcoord);
  if (color.a < 0.01) {
    discard;
  }
  return color;
}
