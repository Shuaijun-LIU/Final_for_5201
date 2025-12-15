# RL Trajectory Planner (TD3 Skeleton)

This directory hosts SB3-based TD3 trajectory planning on `path2/data/terrain_data.json`, targeting a feasible obstacle-avoiding 3D path from User1 to User6.

## Directory Layout (current)
- `env/`
  - `config.py`: hyperparameters and paths.
  - `height_field.py`: height map loading and queries.
  - `collision.py`: terrain/building/tree/lake clearance and collision checks.
  - `terrain_env.py`: environment skeleton (state, action, reward, termination) to be refined.
- `scripts/`
  - `sb3_train_and_eval.py`: train & eval TD3 (Stable-Baselines3) and export `paths.json`.
  - `render_paths.py`: export trajectories to `data/paths.json` for Three.js (single-path format).
- `td3_trajectory_plan.md`: design draft.

## Dependencies
- Python 3.9+ (virtual env recommended).
- Install: `torch`, `numpy`, `stable-baselines3`, `gymnasium`.

## Quick Start (placeholder)
1) Create venv and install deps: `pip install torch numpy`.
2) Ensure `path2/data/terrain_data.json` is present (auto-loaded).
3) Run SB3 training & export path:
   ```bash
   cd path2/RL_trajectory_planner
   python3 scripts/sb3_train_and_eval.py
   ```

## Next Steps
- Refine `env/sb3_env.py` reward/termination/dynamics if needed.
- Adjust SB3 hyperparameters in `scripts/sb3_train_and_eval.py`.
- Visualize exported `data/paths.json` via `path2/index.html`.

