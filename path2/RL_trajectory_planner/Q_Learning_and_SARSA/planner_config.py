from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    grid_step: int = 10
    map_half_size: int = 500  # mapSize=1000 => [-500, 500]

    building_buffer: float = 10.0
    lake_buffer: float = 5.0

    max_steps_per_episode: int = 500
    max_rollout_steps: int = 5000
    episodes: int = 4000

    alpha: float = 0.25
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999

    reward_step: float = -1.0
    reward_goal: float = 200.0
    reward_collision: float = -200.0

    y_clearance: float = 8.0

    # terrain blocking (optional)
    terrain_block_threshold: float = 30.0

    # multi-user RL
    near_radius_m: float = 30.0
    reward_new_user: float = 80.0
    reward_all_users: float = 400.0
    height_penalty: float = 0.02


