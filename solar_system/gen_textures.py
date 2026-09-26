from __future__ import annotations

import os

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "images", "solar")
EARTH_SRC = os.path.join(_HERE, "images", "earth.jpg")


def _disk_alpha (n: int, feather: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  ax = (np.arange(n) + 0.5) / n * 2.0 - 1.0
  dx, dy = np.meshgrid(ax, ax)
  r = np.sqrt(dx * dx + dy * dy)
  edge = 1.0 - feather / n
  alpha = np.clip((1.0 - r) / (1.0 - edge), 0.0, 1.0)
  return dx, dy, alpha


def _value_noise (shape: tuple[int, int], rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
  h, w = shape
  acc = np.zeros(shape, dtype=np.float32)
  amp = 1.0
  total = 0.0
  for o in range(octaves):
    cells = 2 ** (o + 2)
    low = rng.random((cells + 1, cells + 1)).astype(np.float32)
    img = np.array(Image.fromarray((low * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    acc += amp * img
    total += amp
    amp *= 0.5
  return acc / total


def _add_craters (rgb: np.ndarray, alpha: np.ndarray, rng: np.random.Generator, count: int) -> None:
  n = rgb.shape[0]
  ax = np.arange(n)
  gx, gy = np.meshgrid(ax, ax)
  for _ in range(count):
    cx, cy = rng.integers(0, n, size=2)
    rad = rng.integers(n // 40 + 2, n // 12 + 3)
    d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    inside = (d < rad) & (alpha > 0)
    rim = (d >= rad) & (d < rad + 2) & (alpha > 0)
    shade = 0.65 + 0.2 * (d[inside] / rad)
    rgb[inside] *= shade[:, None]
    rgb[rim] = np.clip(rgb[rim] * 1.25, 0, 1)


def _save (name: str, rgba: np.ndarray) -> str:
  os.makedirs(OUT_DIR, exist_ok=True)
  path = os.path.join(OUT_DIR, name)
  Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8)).save(path)
  return path


def make_space (w: int = 1600, h: int = 1000, seed: int = 7) -> None:
  rng = np.random.default_rng(seed)
  yy = np.linspace(0, 1, h)[:, None]
  base = np.zeros((h, w, 3), dtype=np.float32)
  base[..., 0] = 0.02 + 0.03 * yy
  base[..., 1] = 0.02 + 0.02 * yy
  base[..., 2] = 0.06 + 0.08 * yy
  neb = _value_noise((h, w), rng, octaves=5)
  neb = np.clip((neb - 0.55) * 2.0, 0, 1) ** 2
  base[..., 2] += 0.18 * neb
  base[..., 0] += 0.06 * neb
  nstars = 1400
  sx = rng.integers(0, w, nstars)
  sy = rng.integers(0, h, nstars)
  bright = rng.random(nstars) ** 2.2
  for x, y, b in zip(sx, sy, bright):
    base[y, x] = np.minimum(base[y, x] + b, 1.0)
    if b > 0.7:
      for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy2, xx2 = y + oy, x + ox
        if 0 <= yy2 < h and 0 <= xx2 < w:
          base[yy2, xx2] = np.minimum(base[yy2, xx2] + 0.4 * b, 1.0)
  rgba = np.dstack([base, np.ones((h, w), dtype=np.float32)])
  _save("space_bg.png", rgba)


def make_sun (n: int = 512, seed: int = 1) -> None:
  rng = np.random.default_rng(seed)
  dx, dy, _ = _disk_alpha(n)
  r = np.sqrt(dx * dx + dy * dy)
  rd = 0.66
  rg = 0.98
  gran = _value_noise((n, n), rng, octaves=5)
  core = np.array([1.0, 0.96, 0.75])
  mid = np.array([1.0, 0.72, 0.18])
  edge = np.array([1.0, 0.42, 0.05])
  t = np.clip(r / rd, 0, 1)
  rgb = (core[None, None] * (1 - t)[..., None] + mid[None, None] * t[..., None])
  rgb = rgb * (1 - t[..., None] ** 2) + edge[None, None] * (t[..., None] ** 2)
  rgb *= (0.82 + 0.35 * gran[..., None])
  rgb = np.clip(rgb, 0, 1)
  alpha = np.where(r <= rd, 1.0, np.clip((rg - r) / (rg - rd), 0, 1) * 0.5)
  rgba = np.dstack([rgb, alpha.astype(np.float32)])
  _save("sun.png", rgba)


def make_rocky (name: str, base_color: tuple[float, float, float], n: int, craters: int,
                seed: int) -> None:
  rng = np.random.default_rng(seed)
  dx, dy, alpha = _disk_alpha(n)
  base = np.array(base_color, dtype=np.float32)
  noise = _value_noise((n, n), rng, octaves=5)
  rgb = base[None, None] * (0.7 + 0.6 * noise[..., None])
  rgb = np.clip(rgb, 0, 1)
  _add_craters(rgb, alpha, rng, craters)
  z = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
  ldir = np.array([-0.5, -0.5, 0.7])
  ldir /= np.linalg.norm(ldir)
  ndotl = np.clip(dx * ldir[0] + dy * ldir[1] + z * ldir[2], 0, 1)
  shade = 0.35 + 0.75 * ndotl
  rgb = np.clip(rgb * shade[..., None], 0, 1)
  rgba = np.dstack([rgb, alpha.astype(np.float32)])
  _save(name, rgba)


def make_earth_map (w: int = 1024, h: int = 512, src: str | None = None, seed: int = 3) -> None:
  src = src or EARTH_SRC
  os.makedirs(OUT_DIR, exist_ok=True)
  out = os.path.join(OUT_DIR, "earth_map.png")
  if os.path.exists(src):
    img = Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)
    img.save(out)
  else:
    rng = np.random.default_rng(seed)
    land = _value_noise((h, w), rng, octaves=6) > 0.55
    rgb = np.where(land[..., None], np.array([0.20, 0.48, 0.20]), np.array([0.05, 0.20, 0.50]))
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out)


def ensure_textures (force: bool = False) -> dict[str, str]:
  paths = {
    "space": os.path.join(OUT_DIR, "space_bg.png"),
    "sun": os.path.join(OUT_DIR, "sun.png"),
    "earth": os.path.join(OUT_DIR, "earth_map.png"),
    "moon": os.path.join(OUT_DIR, "moon.png"),
    "mercury": os.path.join(OUT_DIR, "mercury.png"),
  }
  if force or not all(os.path.exists(p) for p in paths.values()):
    print("gerando texturas do sistema solar em", OUT_DIR)
    make_space()
    make_sun()
    make_earth_map()
    make_rocky("moon.png", (0.62, 0.62, 0.66), n=256, craters=26, seed=11)
    make_rocky("mercury.png", (0.58, 0.47, 0.38), n=256, craters=34, seed=23)
    print("texturas geradas.")
  return paths


if __name__ == "__main__":
  ensure_textures(force=True)
