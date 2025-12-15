from typing import List, Optional, Tuple

from planner_config import PlannerConfig
from planner_types import Coord, State
from grid import coord_in_bounds
from grid import OccupancyGrid


class GridWorld:
    # 8-neighborhood movement
    ACTIONS: List[Coord] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, grid: OccupancyGrid, start: Coord, goal: Coord, cfg: PlannerConfig):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.cfg = cfg
        self.state = start

    def reset(self) -> Coord:
        self.state = self.start
        return self.state

    def step(self, action_idx: int) -> Tuple[Coord, float, bool]:
        dx, dz = self.ACTIONS[action_idx]
        step = self.cfg.grid_step
        ns: Coord = (self.state[0] + dx * step, self.state[1] + dz * step)
        if not coord_in_bounds(ns, self.cfg) or self.grid.is_blocked(ns):
            return self.state, self.cfg.reward_collision, True
        self.state = ns
        if ns == self.goal:
            return ns, self.cfg.reward_goal, True
        return ns, self.cfg.reward_step, False


class MultiUserGridWorld:
    """
    Multi-goal visit: entering any user's near-radius marks it visited.
    State = (Coord, visited_mask)
    """

    ACTIONS_8: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(
        self,
        grid: OccupancyGrid,
        start: Coord,
        user_points: List[Tuple[float, float]],  # (x,z) meters, not snapped
        cfg: PlannerConfig,
        height_fn,
        visit_order_fixed: Optional[List[int]] = None,
    ):
        self.grid = grid
        self.start = start
        self.user_points = user_points
        self.cfg = cfg
        self.height_fn = height_fn
        self.visit_order_fixed = visit_order_fixed

        self.all_mask = (1 << len(user_points)) - 1
        self.state: Tuple[Coord, int] = (start, 0)
        self.near_r2 = float(cfg.near_radius_m) * float(cfg.near_radius_m)

    @staticmethod
    def extract_coord(s: State) -> Coord:
        return s[0]  # type: ignore[index]

    def _mask_from_coord(self, c: Coord) -> int:
        gx, gz = c
        m = 0
        for i, (ux, uz) in enumerate(self.user_points):
            dx = float(gx) - float(ux)
            dz = float(gz) - float(uz)
            if dx * dx + dz * dz <= self.near_r2:
                m |= (1 << i)
        return m

    def reset(self) -> State:
        start_mask = self._mask_from_coord(self.start)
        self.state = (self.start, start_mask)
        return self.state

    def step(self, action_idx: int) -> Tuple[State, float, bool]:
        (gx, gz), mask = self.state
        dx, dz = self.ACTIONS_8[action_idx]
        step = self.cfg.grid_step
        ng = (gx + dx * step, gz + dz * step)
        if not coord_in_bounds(ng, self.cfg) or self.grid.is_blocked(ng):
            return self.state, self.cfg.reward_collision, True

        new_mask = mask | self._mask_from_coord(ng)

        h = float(self.height_fn(ng[0], ng[1]))
        r = float(self.cfg.reward_step) - float(self.cfg.height_penalty) * max(0.0, h)

        gained = new_mask ^ mask
        if gained:
            r += bin(gained).count("1") * float(self.cfg.reward_new_user)

        done = new_mask == self.all_mask
        if done:
            r += float(self.cfg.reward_all_users)

        self.state = (ng, new_mask)
        return self.state, r, done


