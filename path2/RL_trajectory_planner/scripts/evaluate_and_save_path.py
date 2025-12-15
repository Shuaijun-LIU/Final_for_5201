"""
Evaluate a trained TD3 model and save the complete trajectory.
This script ensures we capture the full path even if episodes end early.
"""

import sys
from pathlib import Path
import json
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


def evaluate_model(env: gym.Env, model: TD3, max_steps=800, num_episodes=5):
    """Evaluate model multiple times and return the best path."""
    best_path = None
    best_reward = float('-inf')
    all_paths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        path = [obs_to_point(obs)]
        total_reward = 0
        collisions_count = 0
        
        for step in range(max_steps):
            # Predict action
            action, _ = model.predict(np.array(obs), deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record position
            path.append(obs_to_point(obs))
            total_reward += reward
            
            # Count collisions
            if "collisions" in info and any(info["collisions"].values()):
                collisions_count += 1
            
            # Stop if episode ended
            if terminated or truncated:
                break
        
        all_paths.append(path)
        
        # Keep the path with highest reward (or fewest collisions if same reward)
        if total_reward > best_reward or (total_reward == best_reward and len(path) > len(best_path) if best_path else True):
            best_reward = total_reward
            best_path = path
    
    return best_path, all_paths


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
    
    # Evaluate
    print("Evaluating model...")
    best_path, all_paths = evaluate_model(env, model, max_steps=MAX_STEPS, num_episodes=10)
    
    # Save best path
    save_paths([best_path], DEFAULT_OUT)
    print(f"Saved best path with {len(best_path)} points to {DEFAULT_OUT}")
    print(f"Path length: {len(best_path)} steps")
    
    # Also save all evaluated paths for comparison
    all_paths_file = OUTPUT_DIR / "evaluated_paths.json"
    with open(all_paths_file, "w") as f:
        json.dump([{"name": f"Path{i}", "points": path} for i, path in enumerate(all_paths)], f, indent=2)
    print(f"Saved {len(all_paths)} evaluated paths to {all_paths_file}")


if __name__ == "__main__":
    main()

