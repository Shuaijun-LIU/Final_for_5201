import random
from typing import Dict, Iterable, List, Optional, Tuple

from planner_config import PlannerConfig
from planner_types import Coord, State


def epsilon_schedule(cfg: PlannerConfig) -> Iterable[float]:
    eps = cfg.epsilon_start
    for _ in range(cfg.episodes):
        yield eps
        eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)


def choose_action(q: Dict[Tuple[State, int], float], s: State, action_n: int, eps: float) -> int:
    if random.random() < eps:
        return random.randrange(action_n)
    best_a = 0
    best_v = -1e18
    for a in range(action_n):
        v = q.get((s, a), 0.0)
        if v > best_v:
            best_v = v
            best_a = a
    return best_a


def q_learning(env, cfg: PlannerConfig, action_n: int) -> Dict[Tuple[State, int], float]:
    q: Dict[Tuple[State, int], float] = {}
    for eps in epsilon_schedule(cfg):
        s = env.reset()
        for _t in range(cfg.max_steps_per_episode):
            a = choose_action(q, s, action_n, eps)
            ns, r, done = env.step(a)
            max_next = max(q.get((ns, na), 0.0) for na in range(action_n))
            old = q.get((s, a), 0.0)
            q[(s, a)] = old + cfg.alpha * (r + cfg.gamma * max_next - old)
            s = ns
            if done:
                break
    return q


def sarsa(env, cfg: PlannerConfig, action_n: int) -> Dict[Tuple[State, int], float]:
    q: Dict[Tuple[State, int], float] = {}
    for eps in epsilon_schedule(cfg):
        s = env.reset()
        a = choose_action(q, s, action_n, eps)
        for _t in range(cfg.max_steps_per_episode):
            ns, r, done = env.step(a)
            na = choose_action(q, ns, action_n, eps) if not done else 0
            old = q.get((s, a), 0.0)
            nxt = q.get((ns, na), 0.0) if not done else 0.0
            q[(s, a)] = old + cfg.alpha * (r + cfg.gamma * nxt - old)
            s, a = ns, na
            if done:
                break
    return q


def rollout_greedy(env, q: Dict[Tuple[State, int], float], cfg: PlannerConfig, action_n: int) -> Optional[List[Coord]]:
    s = env.reset()
    path: List[Coord] = [env.extract_coord(s)]
    seen = {s}
    for _ in range(cfg.max_rollout_steps):
        best_a = 0
        best_v = -1e18
        for a in range(action_n):
            v = q.get((s, a), 0.0)
            if v > best_v:
                best_v = v
                best_a = a
        ns, _r, done = env.step(best_a)
        if ns in seen:
            return None
        seen.add(ns)
        path.append(env.extract_coord(ns))
        s = ns
        if done:
            return path
    return None


def train_multi_user(env, cfg: PlannerConfig, algo: str) -> Tuple[Dict[Tuple[State, int], float], Optional[List[Coord]]]:
    action_n = len(env.ACTIONS_8)
    q: Dict[Tuple[State, int], float] = {}
    best_path: Optional[List[Coord]] = None
    best_steps = 10**18

    for eps in epsilon_schedule(cfg):
        s = env.reset()
        ep_coords: List[Coord] = [env.extract_coord(s)]
        a = choose_action(q, s, action_n, eps) if algo == "sarsa" else -1

        for _t in range(cfg.max_steps_per_episode):
            if algo == "q_learning":
                a = choose_action(q, s, action_n, eps)
                ns, r, done = env.step(a)
                max_next = max(q.get((ns, na), 0.0) for na in range(action_n))
                old = q.get((s, a), 0.0)
                q[(s, a)] = old + cfg.alpha * (r + cfg.gamma * max_next - old)
                s = ns
            else:
                ns, r, done = env.step(a)
                na = choose_action(q, ns, action_n, eps) if not done else 0
                old = q.get((s, a), 0.0)
                nxt = q.get((ns, na), 0.0) if not done else 0.0
                q[(s, a)] = old + cfg.alpha * (r + cfg.gamma * nxt - old)
                s, a = ns, na

            ep_coords.append(env.extract_coord(s))
            if done:
                _coord, mask = s  # type: ignore[misc]
                if mask == env.all_mask:
                    steps = len(ep_coords) - 1
                    if steps < best_steps:
                        best_steps = steps
                        best_path = ep_coords
                break

    return q, best_path


