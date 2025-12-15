import math
from typing import Sequence

from grid import clamp, grid_bounds, round_to_step
from grid import OccupancyGrid


def rasterize_buildings(grid: OccupancyGrid, buildings: Sequence[dict], buffer_m: float) -> None:
    cfg = grid.cfg
    step = cfg.grid_step
    lo, hi = grid_bounds(cfg)

    for b in buildings:
        x = float(b["x"])
        z = float(b["z"])
        hw = float(b["halfWidth"]) + buffer_m
        hd = float(b["halfDepth"]) + buffer_m

        min_x = clamp(round_to_step(x - hw, step), lo, hi)
        max_x = clamp(round_to_step(x + hw, step), lo, hi)
        min_z = clamp(round_to_step(z - hd, step), lo, hi)
        max_z = clamp(round_to_step(z + hd, step), lo, hi)

        gx = min_x
        while gx <= max_x:
            gz = min_z
            while gz <= max_z:
                if (abs(gx - x) <= hw) and (abs(gz - z) <= hd):
                    grid.mark_blocked(gx, gz)
                gz += step
            gx += step


def in_rotated_ellipse(x: float, z: float, lake: dict, buffer_m: float) -> bool:
    dx = x - float(lake["x"])
    dz = z - float(lake["z"])
    rot = float(lake.get("rot", 0.0))
    cos_r = math.cos(-rot)
    sin_r = math.sin(-rot)
    nx = dx * cos_r - dz * sin_r
    nz = dx * sin_r + dz * cos_r
    rx = float(lake["rx"]) + buffer_m
    rz = float(lake["rz"]) + buffer_m
    return (nx * nx) / (rx * rx) + (nz * nz) / (rz * rz) <= 1.0


def rasterize_lakes(grid: OccupancyGrid, lakes: Sequence[dict], buffer_m: float) -> None:
    cfg = grid.cfg
    step = cfg.grid_step
    lo, hi = grid_bounds(cfg)

    for lake in lakes:
        x0 = float(lake["x"])
        z0 = float(lake["z"])
        rx = float(lake["rx"]) + buffer_m
        rz = float(lake["rz"]) + buffer_m

        # conservative AABB for rotated ellipse
        rot = float(lake.get("rot", 0.0))
        c = abs(math.cos(rot))
        s = abs(math.sin(rot))
        ex = rx * c + rz * s
        ez = rx * s + rz * c

        min_x = clamp(round_to_step(x0 - ex, step), lo, hi)
        max_x = clamp(round_to_step(x0 + ex, step), lo, hi)
        min_z = clamp(round_to_step(z0 - ez, step), lo, hi)
        max_z = clamp(round_to_step(z0 + ez, step), lo, hi)

        gx = min_x
        while gx <= max_x:
            gz = min_z
            while gz <= max_z:
                if in_rotated_ellipse(gx, gz, lake, buffer_m=buffer_m):
                    grid.mark_blocked(gx, gz)
                gz += step
            gx += step


def rasterize_trees(grid: OccupancyGrid, trees: Sequence[dict], buffer_m: float) -> None:
    cfg = grid.cfg
    step = cfg.grid_step
    lo, hi = grid_bounds(cfg)
    r = max(0.0, float(buffer_m))

    for t in trees:
        x = float(t.get("x", 0.0))
        z = float(t.get("z", 0.0))
        scale = float(t.get("scale", 1.0))
        rr = r * max(0.5, min(2.0, scale))

        min_x = clamp(round_to_step(x - rr, step), lo, hi)
        max_x = clamp(round_to_step(x + rr, step), lo, hi)
        min_z = clamp(round_to_step(z - rr, step), lo, hi)
        max_z = clamp(round_to_step(z + rr, step), lo, hi)

        gx = min_x
        while gx <= max_x:
            gz = min_z
            while gz <= max_z:
                dx = float(gx) - x
                dz = float(gz) - z
                if dx * dx + dz * dz <= rr * rr:
                    grid.mark_blocked(gx, gz)
                gz += step
            gx += step


def rasterize_terrain_by_height(grid: OccupancyGrid, height_fn, threshold: float) -> None:
    for gx, gz in grid.iter_all_cells():
        if float(height_fn(gx, gz)) > threshold:
            grid.mark_blocked(gx, gz)


