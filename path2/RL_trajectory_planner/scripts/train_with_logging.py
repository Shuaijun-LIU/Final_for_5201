"""
Train TD3 with logging for visualization.
Records episode rewards, lengths, success rates, and collision statistics.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from env.sb3_env import TerrainSB3Env  # noqa: E402
from env.config import ACTION_SCALE, MAX_STEPS  # noqa: E402
from scripts.render_paths import save_paths, DEFAULT_OUT  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_LOG_FILE = OUTPUT_DIR / "training_log.json"
MODEL_SAVE_DIR = OUTPUT_DIR / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


class TrainingCallback(BaseCallback):
    """Callback to log training statistics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.collision_stats = defaultdict(int)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_obs = None
        
    def _on_step(self) -> bool:
        # Get reward and done flags
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])
        
        if rewards:
            self.current_episode_reward += rewards[0]
        self.current_episode_length += 1
        
        # Get info from environment
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            # Check for collisions
            if "collisions" in info:
                collisions = info["collisions"]
                if isinstance(collisions, dict):
                    for collision_type, occurred in collisions.items():
                        if occurred:
                            self.collision_stats[collision_type] += 1
            
            # Store last observation to check distance at episode end
            if "obs" in self.locals:
                obs = self.locals["obs"]
                if hasattr(obs, 'shape') or isinstance(obs, (list, tuple, np.ndarray)):
                    if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                        self.last_obs = obs[0] if len(obs) > 0 else None
                    else:
                        self.last_obs = obs
        
        # Check if episode ended
        done = dones[0] if dones else False
        if done:
            # Check if success (no collision and reached goal)
            # Success is determined by terminated=True and distance < 5.0
            success = False
            if infos and "collisions" in infos[0]:
                collisions = infos[0]["collisions"]
                no_collision = not any(collisions.values())
                # Check distance (stored in observation index 6)
                if self.last_obs is not None:
                    try:
                        obs_array = np.array(self.last_obs).flatten()
                        if len(obs_array) >= 7:
                            distance = obs_array[6]
                            success = no_collision and distance < 5.0
                    except:
                        success = no_collision
            
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_success.append(success)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.last_obs = None
            
        return True
    
    def get_stats(self) -> Dict:
        """Return training statistics."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_success": self.episode_success,
            "collision_stats": dict(self.collision_stats),
            "total_episodes": len(self.episode_rewards),
            "success_rate": sum(self.episode_success) / len(self.episode_success) if self.episode_success else 0.0,
        }


def make_env():
    return TerrainSB3Env()


def evaluate_with_stats(env: gym.Env, model: TD3, max_steps=800, allow_full_trajectory=True) -> tuple:
    """Evaluate model and return path and statistics.
    
    If allow_full_trajectory=True, continues even after collisions to generate
    full trajectory for visualization purposes.
    """
    obs, _ = env.reset()
    path = []
    collisions = defaultdict(int)
    total_reward = 0
    steps = 0
    success = False
    collision_occurred = False
    
    # Helper to convert obs to point
    def obs_to_point(obs):
        if isinstance(obs, (list, tuple, np.ndarray)):
            obs_array = np.array(obs).flatten()
            if len(obs_array) >= 3:
                return {"x": float(obs_array[0]), "y": float(obs_array[1]), "z": float(obs_array[2])}
        # Fallback
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    # Always record the starting position
    path.append(obs_to_point(obs))
    
    for step in range(max_steps):
        try:
            # Predict action
            action, _ = model.predict(np.array(obs), deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record the new position AFTER the step
            path.append(obs_to_point(obs))
            total_reward += reward
            steps += 1
            
            # Record collisions
            if "collisions" in info:
                for collision_type, occurred in info["collisions"].items():
                    if occurred:
                        collisions[collision_type] += 1
                        collision_occurred = True
            
            # Check success (distance < 5.0) - only stop if goal reached
            if terminated:
                obs_array = np.array(obs).flatten()
                if len(obs_array) >= 7:
                    distance = obs_array[6]
                    if distance < 5.0:
                        success = True
                        # Goal reached, stop
                        break
                
                # If allow_full_trajectory, don't stop even after collision
                # Just continue to generate full path
                if not allow_full_trajectory:
                    break
            
            if truncated and not allow_full_trajectory:
                break
        except (IndexError, ValueError, KeyError) as e:
            # If we go out of bounds, clamp position and continue
            print(f"Warning: Position out of bounds at step {step}, clamping to bounds")
            # Try to get last valid position
            if len(path) > 1:
                # Use last valid position
                last_point = path[-1]
                # Create a simple obs from last point (approximate)
                obs = np.array([
                    last_point["x"],
                    last_point["y"],
                    last_point["z"],
                    0, 0, 0,  # dx, dy, dz
                    1000.0,  # large distance
                    0.0  # terrain_h
                ], dtype=np.float32)
            else:
                # Can't continue, stop
                break
    
    stats = {
        "path_length": len(path),
        "total_reward": float(total_reward),
        "steps": steps,
        "success": success,
        "collisions": dict(collisions),
    }
    
    return path, stats


def main():
    print("Starting TD3 training with logging...")
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Create TD3 model
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        batch_size=128,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
    )
    
    # Create callback
    callback = TrainingCallback()
    
    # Train - increase timesteps for better learning
    print(f"Training for 20000 timesteps...")
    model.learn(total_timesteps=20000, callback=callback, log_interval=500)
    
    # Save model
    model_path = MODEL_SAVE_DIR / "td3_model"
    model.save(str(model_path))
    print(f"Model saved to {model_path}.zip")
    
    # Get training stats
    train_stats = callback.get_stats()
    
    # Evaluate on fresh environment - allow full trajectory for visualization
    print("Evaluating model (generating full trajectory)...")
    eval_env = make_env()
    path, eval_stats = evaluate_with_stats(eval_env, model, max_steps=MAX_STEPS, allow_full_trajectory=True)
    
    # Save path
    save_paths([path], DEFAULT_OUT)
    print(f"Saved path with {len(path)} points to {DEFAULT_OUT}")
    
    # Combine all stats
    all_stats = {
        "training": train_stats,
        "evaluation": eval_stats,
    }
    
    # Save training log
    with open(TRAINING_LOG_FILE, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Training log saved to {TRAINING_LOG_FILE}")
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Total episodes: {train_stats['total_episodes']}")
    print(f"Success rate: {train_stats['success_rate']:.2%}")
    print(f"Average episode reward: {np.mean(train_stats['episode_rewards']):.2f}" if train_stats['episode_rewards'] else "N/A")
    print(f"Average episode length: {np.mean(train_stats['episode_lengths']):.2f}" if train_stats['episode_lengths'] else "N/A")
    print(f"\nCollision statistics:")
    for coll_type, count in train_stats['collision_stats'].items():
        print(f"  {coll_type}: {count}")
    
    print("\n=== Evaluation Summary ===")
    print(f"Path length: {eval_stats['path_length']} points")
    print(f"Total reward: {eval_stats['total_reward']:.2f}")
    print(f"Steps: {eval_stats['steps']}")
    print(f"Success: {eval_stats['success']}")
    print(f"Collisions: {eval_stats['collisions']}")


if __name__ == "__main__":
    main()

