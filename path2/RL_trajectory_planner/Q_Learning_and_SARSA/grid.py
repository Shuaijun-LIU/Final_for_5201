import math
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from planner_config import PlannerConfig
from planner_types import Coord


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def round_to_step(v: float, step: int) -> int:
    return int(round(v / step) * step)


def grid_bounds(cfg: PlannerConfig) -> Tuple[int, int]:
    lo = -cfg.map_half_size
    hi = cfg.map_half_size
    return lo, hi


def coord_in_bounds(c: Coord, cfg: PlannerConfig) -> bool:
    lo, hi = grid_bounds(cfg)
    return lo <= c[0] <= hi and lo <= c[1] <= hi


def neighbors_4(c: Coord, step: int) -> List[Coord]:
    x, z = c
    return [(x + step, z), (x - step, z), (x, z + step), (x, z - step)]


def neighbors_8(c: Coord, step: int) -> List[Coord]:
    x, z = c
    return [
        (x + step, z),
        (x - step, z),
        (x, z + step),
        (x, z - step),
        (x + step, z + step),
        (x + step, z - step),
        (x - step, z + step),
        (x - step, z - step),
    ]


class OccupancyGrid:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        lo, hi = grid_bounds(cfg)
        self.lo = lo
        self.hi = hi
        self.step = cfg.grid_step
        self.n = (hi - lo) // self.step + 1
        self.blocked: List[List[bool]] = [[False] * self.n for _ in range(self.n)]

    def _idx(self, gx: int) -> int:
        return (gx - self.lo) // self.step

    def mark_blocked(self, gx: int, gz: int) -> None:
        if not (self.lo <= gx <= self.hi and self.lo <= gz <= self.hi):
            return
        self.blocked[self._idx(gx)][self._idx(gz)] = True

    def is_blocked(self, c: Coord) -> bool:
        if not coord_in_bounds(c, self.cfg):
            return True
        return self.blocked[self._idx(c[0])][self._idx(c[1])]

    def iter_all_cells(self) -> Iterable[Coord]:
        for ix in range(self.n):
            gx = self.lo + ix * self.step
            for iz in range(self.n):
                gz = self.lo + iz * self.step
                yield (gx, gz)


def nearest_free(start: Coord, grid: OccupancyGrid) -> Optional[Coord]:
    if not grid.is_blocked(start):
        return start
    q = deque([start])
    seen = {start}
    step = grid.cfg.grid_step
    while q:
        c = q.popleft()
        for nb in neighbors_8(c, step):
            if nb in seen or not coord_in_bounds(nb, grid.cfg):
                continue
            if not grid.is_blocked(nb):
                return nb
            seen.add(nb)
            q.append(nb)
    return None


def bfs_shortest_path(start: Coord, goal: Coord, grid: OccupancyGrid) -> Optional[List[Coord]]:
    if grid.is_blocked(start) or grid.is_blocked(goal):
        return None
    q = deque([start])
    prev: Dict[Coord, Optional[Coord]] = {start: None}
    step = grid.cfg.grid_step
    while q:
        c = q.popleft()
        if c == goal:
            break
        for nb in neighbors_8(c, step):
            if not coord_in_bounds(nb, grid.cfg) or grid.is_blocked(nb) or nb in prev:
                continue
            prev[nb] = c
            q.append(nb)
    if goal not in prev:
        return None
    path: List[Coord] = []
    cur: Optional[Coord] = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def bfs_to_predicate(start: Coord, grid: OccupancyGrid, is_goal: Callable[[Coord], bool]) -> Optional[List[Coord]]:
    if grid.is_blocked(start):
        return None
    q = deque([start])
    prev: Dict[Coord, Optional[Coord]] = {start: None}
    step = grid.cfg.grid_step
    goal: Optional[Coord] = start if is_goal(start) else None
    while q and goal is None:
        c = q.popleft()
        for nb in neighbors_8(c, step):
            if nb in prev or (not coord_in_bounds(nb, grid.cfg)) or grid.is_blocked(nb):
                continue
            prev[nb] = c
            if is_goal(nb):
                goal = nb
                break
            q.append(nb)
    if goal is None:
        return None
    path: List[Coord] = []
    cur: Optional[Coord] = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def bfs_path_visit_users_in_order(
    start: Coord,
    user_points_all: List[Tuple[float, float]],
    visit_indices: List[int],
    grid: OccupancyGrid,
    cfg: PlannerConfig,
) -> Tuple[Optional[List[Coord]], Optional[int]]:
    if not visit_indices:
        return [start], None
    near_r2 = float(cfg.near_radius_m) * float(cfg.near_radius_m)

    def within_user(coord: Coord, idx: int) -> bool:
        gx, gz = coord
        ux, uz = user_points_all[idx]
        dx = float(gx) - float(ux)
        dz = float(gz) - float(uz)
        return dx * dx + dz * dz <= near_r2

    path_all: List[Coord] = [start]
    cur = start
    for idx in visit_indices:
        if within_user(cur, idx):
            continue
        sub = bfs_to_predicate(cur, grid, is_goal=lambda c, i=idx: within_user(c, i))
        if sub is None:
            return None, idx
        path_all.extend(sub[1:])
        cur = path_all[-1]
    return path_all, None


def world_users_to_grid(world: dict, cfg: PlannerConfig, idx: int) -> Coord:
    users = world.get("finalUsers", [])
    if idx < 0 or idx >= len(users):
        raise ValueError(f"finalUsers 索引越界: idx={idx}, len={len(users)}")
    u = users[idx]
    gx = clamp(round_to_step(float(u["x"]), cfg.grid_step), -cfg.map_half_size, cfg.map_half_size)
    gz = clamp(round_to_step(float(u["z"]), cfg.grid_step), -cfg.map_half_size, cfg.map_half_size)
    return (gx, gz)


