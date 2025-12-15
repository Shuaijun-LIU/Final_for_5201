import os
import sys
from typing import Callable, Dict, Optional

from planner_config import PlannerConfig
from planner_types import Coord
from grid import round_to_step


def build_height_lookup(world: dict, cfg: PlannerConfig) -> Dict[Coord, float]:
    step = cfg.grid_step
    hmap: Dict[Coord, float] = {}
    for p in world.get("terrainMap", []):
        gx = round_to_step(float(p["x"]), step)
        gz = round_to_step(float(p["z"]), step)
        hmap[(gx, gz)] = float(p["y"])
    return hmap


def _find_terrain_generator_dir(terrain_json_path: str) -> Optional[str]:
    """
    Locate the `terrain_generator/` folder so we can import helper modules (height_map_utils.py, terrain_utils.py)
    regardless of where terrain_data.json is placed.

    Common layouts:
    - path2/data/terrain_data.json                 -> path2/terrain_generator/
    - path2/terrain_generator/data/terrain_data.json -> path2/terrain_generator/
    """
    data_dir = os.path.abspath(os.path.dirname(terrain_json_path))
    candidates = [
        os.path.abspath(os.path.join(data_dir, "..", "terrain_generator")),
        os.path.abspath(os.path.join(data_dir, "..")),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "height_map_utils.py")) and os.path.isfile(os.path.join(c, "terrain_utils.py")):
            return c
    for c in candidates:
        if os.path.isfile(os.path.join(c, "height_map_utils.py")) or os.path.isfile(os.path.join(c, "terrain_utils.py")):
            return c
    return None


def try_build_heightmap_height_fn(terrain_json_path: str, world: dict):
    hm = world.get("heightMap")
    if not isinstance(hm, dict) or "heightMap" not in hm:
        return None
    try:
        tg_dir = _find_terrain_generator_dir(terrain_json_path)
        if tg_dir is None:
            return None
        if tg_dir not in sys.path:
            sys.path.insert(0, tg_dir)
        from height_map_utils import HeightMap  # type: ignore

        height_map = HeightMap(hm)

        def height_fn(x: float, z: float) -> float:
            return float(height_map.get_height_interpolated(x, z))

        return height_fn
    except Exception:
        return None


def try_build_terrain_height_fn(terrain_json_path: str, world: dict):
    try:
        tg_dir = _find_terrain_generator_dir(terrain_json_path)
        if tg_dir is None:
            return None
        if tg_dir not in sys.path:
            sys.path.insert(0, tg_dir)

        from opensimplex import OpenSimplex  # type: ignore
        from terrain_utils import Mulberry32Random, TerrainGenerator  # type: ignore

        seed = int(float(world.get("seed", 0.0)))
        rng = Mulberry32Random(seed)
        simplex = OpenSimplex(seed)
        lakes = world.get("lakes", [])
        terrain_gen = TerrainGenerator(simplex, lakes, rng)

        def height_fn(x: float, z: float) -> float:
            return float(terrain_gen.get_terrain_height(x, z))

        return height_fn
    except Exception:
        return None


def select_height_fn(terrain_json_path: str, world: dict, cfg: PlannerConfig):
    """
    Prefer: heightMap -> deterministic regen -> sparse terrainMap
    Returns (height_fn, source_name)
    """
    height_fn = try_build_heightmap_height_fn(terrain_json_path, world)
    if height_fn is not None:
        return height_fn, "heightMap"

    height_fn = try_build_terrain_height_fn(terrain_json_path, world)
    if height_fn is not None:
        return height_fn, "seed_regen"

    height_lookup = build_height_lookup(world, cfg)

    def height_fn(x: float, z: float) -> float:
        return float(height_lookup.get((round_to_step(x, cfg.grid_step), round_to_step(z, cfg.grid_step)), 0.0))

    return height_fn, "terrainMap_sparse"


