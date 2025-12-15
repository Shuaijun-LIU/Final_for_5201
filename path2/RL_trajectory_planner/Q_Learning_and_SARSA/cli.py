#!/usr/bin/env python3
"""
CLI entry for RL trajectory planner.

This module contains the command-line parsing and orchestration logic.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

from envs import GridWorld, MultiUserGridWorld
from exporter import to_path_json_with_height_fn
from grid import bfs_path_visit_users_in_order, bfs_shortest_path, nearest_free, world_users_to_grid
from grid import OccupancyGrid
from height_provider import select_height_fn
from obstacles import rasterize_buildings, rasterize_lakes, rasterize_terrain_by_height, rasterize_trees
from planner_config import PlannerConfig
from rl_algos import q_learning, rollout_greedy, sarsa, train_multi_user
from planner_types import Coord


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL trajectory planner (Q-learning / SARSA) for terrain_generator")
    p.add_argument("--terrain_json", required=True, help="path to terrain_data.json")
    p.add_argument("--out_json", required=True, help="output path json (for index.html)")
    p.add_argument("--algo", choices=["q_learning", "sarsa"], default="q_learning", help="单段 RL 规划时使用")
    p.add_argument("--method", choices=["multi_user_rl", "single_segment_rl"], default="multi_user_rl")
    p.add_argument("--start_user_idx", type=int, default=0, help="用户1（finalUsers[0]）作为起点")
    p.add_argument("--goal_user_idx", type=int, default=5, help="用户6（finalUsers[5]）作为终点（single_segment_rl）")
    p.add_argument("--grid_step", type=int, default=10)
    p.add_argument("--building_buffer", type=float, default=10.0)
    p.add_argument("--lake_buffer", type=float, default=5.0)
    p.add_argument("--terrain_block_threshold", type=float, default=30.0, help="地形高度>阈值视为障碍（默认开启 --block_terrain）")
    p.add_argument("--block_terrain", action="store_true", default=True, help="把高于阈值的地形当作硬障碍（默认开启）")
    p.add_argument("--no_block_terrain", action="store_false", dest="block_terrain", help="关闭地形硬避障")
    p.add_argument("--block_trees", action="store_true", default=True, help="把 trees 当作硬障碍（默认开启）")
    p.add_argument("--no_block_trees", action="store_false", dest="block_trees", help="关闭树木硬避障")
    p.add_argument("--tree_buffer", type=float, default=2.0, help="树木避障半径（米，默认 2.0）")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0, help="random seed for RL (0 means derive from terrain seed)")
    p.add_argument("--fallback_bfs", action="store_true", default=True)
    p.add_argument("--no_fallback_bfs", action="store_false", dest="fallback_bfs")
    p.add_argument("--path_name", default="RL_Path")
    p.add_argument("--color", type=int, default=0xFF0000)
    p.add_argument("--visit_user_indices", default="0,1,2,3,4,5", help="需要经过的用户索引列表（逗号分隔）")
    p.add_argument("--near_radius_m", type=float, default=30.0, help="认为“到用户附近”的半径（米）")
    p.add_argument("--reward_new_user", type=float, default=80.0, help="首次到达某用户附近的奖励")
    p.add_argument("--reward_all_users", type=float, default=400.0, help="访问完全部用户附近的终止奖励")
    p.add_argument("--height_penalty", type=float, default=0.02, help="地形高度惩罚系数（软避开山地）")
    p.add_argument("--fallback_greedy_bfs", action="store_true", default=True, help="multi_user_rl 失败时用 BFS 依序访问用户作为兜底（默认开启）")
    p.add_argument("--no_fallback_greedy_bfs", action="store_false", dest="fallback_greedy_bfs", help="关闭 multi_user_rl 的 BFS 兜底")
    return p


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    with open(args.terrain_json, "r") as f:
        world = json.load(f)

    cfg = PlannerConfig(
        grid_step=args.grid_step,
        building_buffer=args.building_buffer,
        lake_buffer=args.lake_buffer,
        episodes=args.episodes,
        terrain_block_threshold=args.terrain_block_threshold,
        near_radius_m=args.near_radius_m,
        reward_new_user=args.reward_new_user,
        reward_all_users=args.reward_all_users,
        height_penalty=args.height_penalty,
    )

    # seed RNG
    terrain_seed = int(float(world.get("seed", 0.0)))
    rl_seed = args.seed if args.seed != 0 else (terrain_seed & 0xFFFFFFFF)
    random.seed(rl_seed)

    # map size
    map_size = int(world.get("mapSize", cfg.map_half_size * 2))
    map_half = map_size // 2
    cfg = PlannerConfig(
        grid_step=cfg.grid_step,
        map_half_size=map_half,
        building_buffer=cfg.building_buffer,
        lake_buffer=cfg.lake_buffer,
        episodes=cfg.episodes,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
        reward_step=cfg.reward_step,
        reward_goal=cfg.reward_goal,
        reward_collision=cfg.reward_collision,
        max_steps_per_episode=cfg.max_steps_per_episode,
        max_rollout_steps=cfg.max_rollout_steps,
        y_clearance=cfg.y_clearance,
        terrain_block_threshold=cfg.terrain_block_threshold,
        near_radius_m=cfg.near_radius_m,
        reward_new_user=cfg.reward_new_user,
        reward_all_users=cfg.reward_all_users,
        height_penalty=cfg.height_penalty,
    )

    height_fn, height_source = select_height_fn(args.terrain_json, world, cfg)

    grid = OccupancyGrid(cfg)
    rasterize_buildings(grid, world.get("buildingColliders", []), buffer_m=cfg.building_buffer)
    rasterize_lakes(grid, world.get("lakes", []), buffer_m=cfg.lake_buffer)
    if args.block_trees:
        rasterize_trees(grid, world.get("trees", []), buffer_m=float(args.tree_buffer))
    if args.block_terrain:
        rasterize_terrain_by_height(grid, height_fn=height_fn, threshold=cfg.terrain_block_threshold)

    visit_indices = [int(s.strip()) for s in args.visit_user_indices.split(",") if s.strip() != ""]
    if not visit_indices:
        raise ValueError("visit_user_indices 为空")

    user_points_float: List[Tuple[float, float]] = []
    user_y_at_coord: Dict[Coord, float] = {}
    all_user_points_float: List[Tuple[float, float]] = [(float(u["x"]), float(u["z"])) for u in world.get("finalUsers", [])]
    for idx in visit_indices:
        u = world["finalUsers"][idx]
        user_points_float.append((float(u["x"]), float(u["z"])))
        c = world_users_to_grid(world, cfg, idx)
        user_y_at_coord[c] = max(user_y_at_coord.get(c, -1e18), float(u["y"]))

    if args.method == "single_segment_rl":
        start = nearest_free(world_users_to_grid(world, cfg, args.start_user_idx), grid) or world_users_to_grid(world, cfg, args.start_user_idx)
        goal = nearest_free(world_users_to_grid(world, cfg, args.goal_user_idx), grid) or world_users_to_grid(world, cfg, args.goal_user_idx)
        if grid.is_blocked(start) or grid.is_blocked(goal):
            raise RuntimeError("起点或终点在障碍物内且无法找到邻近可用网格点。请换 user_idx 或调小 buffer/阈值。")

        env = GridWorld(grid, start, goal, cfg)
        if args.algo == "q_learning":
            q = q_learning(env, cfg, action_n=len(GridWorld.ACTIONS))
        else:
            q = sarsa(env, cfg, action_n=len(GridWorld.ACTIONS))
        env.extract_coord = lambda s: s  # type: ignore[attr-defined]
        path = rollout_greedy(env, q, cfg, action_n=len(GridWorld.ACTIONS))
        if path is None and args.fallback_bfs:
            path = bfs_shortest_path(start, goal, grid)
        if path is None:
            raise RuntimeError("single_segment_rl 失败：RL rollout 与 BFS 都未找到可行解。")

        out_obj = to_path_json_with_height_fn(path, height_fn=height_fn, cfg=cfg, name=args.path_name, color=args.color, user_y_at_coord=user_y_at_coord)
        out_obj["_meta"] = {
            "method": args.method,
            "algo": args.algo,
            "episodes": cfg.episodes,
            "grid_step": cfg.grid_step,
            "building_buffer": cfg.building_buffer,
            "lake_buffer": cfg.lake_buffer,
            "block_trees": bool(args.block_trees),
            "tree_buffer": float(args.tree_buffer),
            "block_terrain": bool(args.block_terrain),
            "terrain_block_threshold": cfg.terrain_block_threshold,
            "height_source": height_source,
            "rl_seed": rl_seed,
            "start": {"gx": start[0], "gz": start[1]},
            "goal": {"gx": goal[0], "gz": goal[1]},
            "steps": len(path) - 1,
        }
    else:
        start = nearest_free(world_users_to_grid(world, cfg, args.start_user_idx), grid) or world_users_to_grid(world, cfg, args.start_user_idx)
        if grid.is_blocked(start):
            raise RuntimeError("起点在障碍物内且无法找到邻近可用网格点。请换 start_user_idx 或调小 buffer。")

        env = MultiUserGridWorld(grid=grid, start=start, user_points=user_points_float, cfg=cfg, height_fn=height_fn)
        q, best_path = train_multi_user(env, cfg, algo="q_learning" if args.algo == "q_learning" else "sarsa")
        path = best_path
        if path is None:
            path = rollout_greedy(env, q, cfg, action_n=len(MultiUserGridWorld.ACTIONS_8))
        if path is None and args.fallback_greedy_bfs:
            path, failed_idx = bfs_path_visit_users_in_order(start=start, user_points_all=all_user_points_float, visit_indices=visit_indices, grid=grid, cfg=cfg)
            if path is None and failed_idx is not None:
                raise RuntimeError(f"multi_user_rl 兜底失败：无法在避障约束下到达 user_idx={failed_idx} 的 near_radius 区域。")
        if path is None:
            raise RuntimeError("multi_user_rl 失败：RL 未出现成功轨迹，且 BFS 兜底也未找到可行解。")

        out_obj = to_path_json_with_height_fn(path, height_fn=height_fn, cfg=cfg, name=args.path_name or "MultiUser_RL_Path", color=args.color, user_y_at_coord=user_y_at_coord)
        out_obj["_meta"] = {
            "method": args.method,
            "algo": args.algo,
            "episodes": cfg.episodes,
            "grid_step": cfg.grid_step,
            "building_buffer": cfg.building_buffer,
            "lake_buffer": cfg.lake_buffer,
            "block_trees": bool(args.block_trees),
            "tree_buffer": float(args.tree_buffer),
            "block_terrain": bool(args.block_terrain),
            "terrain_block_threshold": cfg.terrain_block_threshold,
            "height_source": height_source,
            "near_radius_m": cfg.near_radius_m,
            "reward_new_user": cfg.reward_new_user,
            "reward_all_users": cfg.reward_all_users,
            "height_penalty": cfg.height_penalty,
            "fallback_greedy_bfs": bool(args.fallback_greedy_bfs),
            "visit_user_indices": visit_indices,
            "start_user_idx": args.start_user_idx,
            "steps": len(path) - 1,
        }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"✓ 输出轨迹到: {args.out_json}")
    print(f"  - method: {out_obj.get('_meta', {}).get('method')}, algo: {out_obj.get('_meta', {}).get('algo')}")
    print(f"  - steps: {out_obj.get('_meta', {}).get('steps')}")


if __name__ == "__main__":
    main()


