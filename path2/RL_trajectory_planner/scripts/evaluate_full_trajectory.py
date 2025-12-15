"""
Evaluate TD3 model and generate full trajectory for visualization.
Even if collisions occur, continues to generate path up to max_steps.
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from env.sb3_env import TerrainSB3Env  # noqa: E402
from env.config import MAX_STEPS  # noqa: E402
from scripts.render_paths import save_paths, DEFAULT_OUT  # noqa: E402

OUTPUT_DIR = BASE_DIR.parent / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "td3_model"


def obs_to_point(obs):
    """Convert observation to 3D point."""
    if isinstance(obs, (list, tuple, np.ndarray)):
        obs_array = np.array(obs).flatten()
        if len(obs_array) >= 3:
            return {"x": float(obs_array[0]), "y": float(obs_array[1]), "z": float(obs_array[2])}
    return {"x": 0.0, "y": 0.0, "z": 0.0}


class NonTerminatingWrapper:
    """Wrapper that prevents early termination, allowing full trajectory generation."""
    
    def __init__(self, env):
        self.env = env
        self._collision_occurred = False
        
    def reset(self, **kwargs):
        self._collision_occurred = False
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if collision occurred
        if "collisions" in info and any(info["collisions"].values()):
            self._collision_occurred = True
            # Mark as collision in info but don't terminate
            info["collision_occurred"] = True
        
        # Don't terminate early - always continue to max_steps
        # Only terminate if goal reached
        if terminated and not self._collision_occurred:
            # Goal reached - allow termination
            pass
        else:
            # Collision or other - continue anyway for visualization
            terminated = False
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped env."""
        return getattr(self.env, name)


def evaluate_full_trajectory(env: gym.Env, model: TD3, max_steps=800):
    """Evaluate model and generate full trajectory without early termination."""
    # Wrap environment to prevent early termination
    wrapped_env = NonTerminatingWrapper(env)
    
    obs, _ = wrapped_env.reset()
    path = [obs_to_point(obs)]
    rewards = []
    collisions = []
    
    for step in range(max_steps):
        # Predict action
        action, _ = model.predict(np.array(obs), deterministic=True)
        
        # Take step
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Record position
        path.append(obs_to_point(obs))
        rewards.append(reward)
        
        # Record collision info
        collision_info = {}
        if "collisions" in info:
            collision_info = info["collisions"].copy()
        if "collision_occurred" in info:
            collision_info["any_collision"] = info["collision_occurred"]
        collisions.append(collision_info)
        
        # Only stop if goal reached (distance < 5.0)
        if terminated:
            obs_array = np.array(obs).flatten()
            if len(obs_array) >= 7:
                distance = obs_array[6]
                if distance < 5.0:
                    print(f"Goal reached at step {step+1}!")
                    break
    
    total_reward = sum(rewards)
    collision_count = sum(1 for c in collisions if c.get("any_collision", False) or any(c.values()))
    
    print(f"Generated trajectory with {len(path)} points")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Collisions: {collision_count}/{len(path)-1} steps")
    
    return path


def main():
    print("Loading trained model...")
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please run train_with_logging.py first.")
        return
    
    # Load model
    model = TD3.load(str(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")
    
    # Create environment
    env = TerrainSB3Env()
    
    # Evaluate and generate full trajectory
    print(f"Generating full trajectory (max {MAX_STEPS} steps)...")
    path = evaluate_full_trajectory(env, model, max_steps=MAX_STEPS)
    
    # Save path
    save_paths([path], DEFAULT_OUT)
    print(f"Saved trajectory with {len(path)} points to {DEFAULT_OUT}")


if __name__ == "__main__":
    main()

