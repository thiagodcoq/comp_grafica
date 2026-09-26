struct Matrix {
  vertex: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  spin: f32,
}
@group(1) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

const PI: f32 = 3.14159265358979;

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) pos: vec2<f32>,
            @location(1) texcoord: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.clip_position = global.projection * (matrix[instance_index].vertex * vec4<f32>(pos, 0.0, 1.0));
  out.uv = texcoord;
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  let p = vec2<f32>(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0);
  let r2 = dot(p, p);
  let r = sqrt(r2);
  let a = clamp((1.0 - r) / 0.02, 0.0, 1.0);
  if (a <= 0.0) {
    discard;
  }

  let X = p.x;
  let Y = -p.y;
  let Z = sqrt(max(1.0 - r2, 0.0));

  let lat = asin(clamp(Y, -1.0, 1.0));
  let lon = atan2(X, Z) + global.spin;
  let su = lon / (2.0 * PI) + 0.5;
  let sv = 0.5 - lat / PI;
  var color = textureSample(tex, samp, vec2<f32>(su, sv)).rgb;

  let n = vec3<f32>(X, Y, Z);
  let ldir = normalize(vec3<f32>(-0.4, 0.4, 0.85));
  let d = clamp(dot(n, ldir), 0.0, 1.0);
  let shade = 0.4 + 0.7 * d;

  return vec4<f32>(color * shade, a);
}
