from __future__ import annotations

import os
import sys
import math
from typing import Any
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "python"))
sys.modules.pop("node", None)

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import *
from transform import *
from quad import *
from node import *
from shader import *
from pipeline import *
from scene import *
from renderer import *
from engine import *
from texture import *
from sampler import *
from textureset import *

from gen_textures import ensure_textures

SHADER_PATH = os.path.join(HERE, "shaders", "textured.wgsl")
EARTH_SHADER_PATH = os.path.join(HERE, "shaders", "earth.wgsl")

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera2D
scene: Scene
last_t: float = 0.0

WORLD = 10.0

SUN_SIZE = 4.6
MERCURY_ORBIT = 4.2
MERCURY_SIZE = 0.6
EARTH_ORBIT = 7.2
EARTH_SIZE = 1.4
MOON_ORBIT = 1.6
MOON_SIZE = 0.45

SUN_SPIN = 6.0
MERCURY_ORBIT_SPEED = 46.0
MERCURY_SPIN = 24.0
EARTH_ORBIT_SPEED = 20.0
EARTH_SPIN = 80.0
MOON_ORBIT_SPEED = 65.0


class Rotate (Engine):
  def __init__ (self, trf: Transform, deg_per_sec: float) -> None:
    self.trf = trf
    self.speed = deg_per_sec

  def update (self, dt: float) -> None:
    self.trf.rotate(self.speed * dt, 0, 0, 1)


class SpinGlobe (Engine):
  def __init__ (self, shader: Shader, deg_per_sec: float) -> None:
    self.shader = shader
    self.rad_per_sec = math.radians(deg_per_sec)
    self.angle = 0.0
    self.shader.set_value("spin", self.angle)

  def update (self, dt: float) -> None:
    self.angle += self.rad_per_sec * dt
    self.shader.set_value("spin", self.angle)


def _pos (x: float, y: float) -> Transform:
  t = Transform()
  t.translate(x, y, 0)
  return t


def _disk (size: float, texset: TextureSet, quad: Quad) -> Node:
  t = Transform()
  t.scale(size, size, 1)
  t.translate(-0.5, -0.5, 0)
  return Node(trf=t, apps=[texset], shps=[quad])


def build_scene (device: wgpu.GPUDevice, target_format: str) -> tuple[Scene, Camera2D]:
  cam = Camera2D(-WORLD, WORLD, -WORLD, WORLD)

  vertex_buffers = [
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ]
  blend = {
    "color": {"operation": "add", "src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one-minus-src-alpha"},
  }

  shader = Shader(device, SHADER_PATH)
  shader.set_vertex_buffers(vertex_buffers)
  pipeline = Pipeline(shader, target_format, depth_stencil=None, blend=blend)

  earth_shader = Shader(device, EARTH_SHADER_PATH)
  earth_shader.set_vertex_buffers(vertex_buffers)
  earth_pipeline = Pipeline(earth_shader, target_format, depth_stencil=None, blend=blend)

  paths = ensure_textures()
  sampler = Sampler(device, "samp", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
  earth_sampler = Sampler(device, "samp", address_mode_u="repeat", address_mode_v="clamp-to-edge")

  def texset (key: str, shd: Shader, smp: Sampler) -> TextureSet:
    ts = TextureSet([Texture(device, "tex", paths[key]), smp])
    shd.add_texture_set(ts)
    return ts

  space_ts = texset("space", shader, sampler)
  sun_ts = texset("sun", shader, sampler)
  mercury_ts = texset("mercury", shader, sampler)
  moon_ts = texset("moon", shader, sampler)
  earth_ts = texset("earth", earth_shader, earth_sampler)

  quad = Quad(device)

  bg_trf = Transform()
  bg_trf.scale(4 * WORLD, 4 * WORLD, 1)
  bg_trf.translate(-0.5, -0.5, 0)
  background = Node(trf=bg_trf, apps=[space_ts], shps=[quad])

  sun = Node()
  sun_spin_trf = Transform()
  sun_spin = Node(trf=sun_spin_trf, nodes=[_disk(SUN_SIZE, sun_ts, quad)])
  sun.add_node(sun_spin)

  mercury_orbit_trf = Transform()
  mercury_spin_trf = Transform()
  mercury_pos = Node(trf=_pos(MERCURY_ORBIT, 0), nodes=[
    Node(trf=mercury_spin_trf, nodes=[_disk(MERCURY_SIZE, mercury_ts, quad)])
  ])
  mercury_orbit = Node(trf=mercury_orbit_trf, nodes=[mercury_pos])
  sun.add_node(mercury_orbit)

  earth_orbit_trf = Transform()
  moon_orbit_trf = Transform()

  earth_disk = _disk(EARTH_SIZE, earth_ts, quad)
  earth_disk.set_pipeline(earth_pipeline)

  moon_pos = Node(trf=_pos(MOON_ORBIT, 0), nodes=[_disk(MOON_SIZE, moon_ts, quad)])
  moon_orbit = Node(trf=moon_orbit_trf, nodes=[moon_pos])

  earth_pos = Node(trf=_pos(EARTH_ORBIT, 0), nodes=[earth_disk, moon_orbit])
  earth_orbit = Node(trf=earth_orbit_trf, nodes=[earth_pos])
  sun.add_node(earth_orbit)

  root = Node(pipeline, nodes=[background, sun])
  sc = Scene(root)

  sc.add_engine(Rotate(sun_spin_trf, SUN_SPIN))
  sc.add_engine(Rotate(mercury_orbit_trf, MERCURY_ORBIT_SPEED))
  sc.add_engine(Rotate(mercury_spin_trf, MERCURY_SPIN))
  sc.add_engine(Rotate(earth_orbit_trf, EARTH_ORBIT_SPEED))
  sc.add_engine(SpinGlobe(earth_shader, EARTH_SPIN))
  sc.add_engine(Rotate(moon_orbit_trf, MOON_ORBIT_SPEED))

  return sc, cam


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene
  scene, camera = build_scene(device, target_format)


def update (dt: float) -> None:
  scene.update(dt)


def draw () -> None:
  global last_t
  t = time.perf_counter()
  update(t - last_t)
  last_t = t

  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)


def on_key (event: Any) -> None:
  if event["key"] in ("q", "Escape"):
    canvas.close()


def main () -> None:
  global canvas, device, context, renderer, last_t

  canvas = RenderCanvas(size=(900, 900), title="Mini-sistema solar 2D", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, clear_value=(0.0, 0.0, 0.0, 1.0))

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  last_t = time.perf_counter()
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()
