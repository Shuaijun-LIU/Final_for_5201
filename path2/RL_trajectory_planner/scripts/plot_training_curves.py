"""
Generate training curves visualization.
Reads training_log.json and generates:
- td3_training_curves.png: Episode rewards, lengths, success rate
- reward_convergence.png: Reward convergence over episodes
- success_rate_analysis.png: Success rate over training
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set matplotlib to use LaTeX-style fonts
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
TRAINING_LOG_FILE = OUTPUT_DIR / "training_log.json"
FIG_DIR = BASE_DIR / "report_latex" / "fig" / "part2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def moving_average(data: list, window: int = 10) -> np.ndarray:
    """Compute moving average."""
    if len(data) == 0:
        return np.array([])
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves():
    """Plot comprehensive training curves."""
    if not TRAINING_LOG_FILE.exists():
        print(f"Training log not found: {TRAINING_LOG_FILE}")
        print("Please run train_with_logging.py first.")
        return
    
    with open(TRAINING_LOG_FILE, "r") as f:
        data = json.load(f)
    
    train_stats = data.get("training", {})
    episode_rewards = train_stats.get("episode_rewards", [])
    episode_lengths = train_stats.get("episode_lengths", [])
    episode_success = train_stats.get("episode_success", [])
    
    if not episode_rewards:
        print("No training data found.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    episodes = np.arange(len(episode_rewards))
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    if len(episode_rewards) > 10:
        rewards_ma = moving_average(episode_rewards, window=10)
        ax.plot(episodes[:len(rewards_ma)], rewards_ma, 'b-', linewidth=1.5, label='Moving Average (window=10)')
        ax.plot(episodes, episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    else:
        ax.plot(episodes, episode_rewards, 'b-', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Rewards During Training')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    if len(episode_lengths) > 10:
        lengths_ma = moving_average(episode_lengths, window=10)
        ax.plot(episodes[:len(lengths_ma)], lengths_ma, 'g-', linewidth=1.5, label='Moving Average (window=10)')
        ax.plot(episodes, episode_lengths, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
    else:
        ax.plot(episodes, episode_lengths, 'g-', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths During Training')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Success Rate (cumulative)
    ax = axes[1, 0]
    if episode_success:
        cumulative_success = np.cumsum(episode_success) / (np.arange(len(episode_success)) + 1)
        ax.plot(episodes, cumulative_success, 'r-', linewidth=1.5)
        ax.axhline(y=train_stats.get("success_rate", 0), color='k', linestyle='--', 
                   label=f'Final Success Rate: {train_stats.get("success_rate", 0):.2%}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Success Rate')
        ax.set_title('Cumulative Success Rate')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 4. Reward Distribution
    ax = axes[1, 1]
    ax.hist(episode_rewards, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Episode Rewards')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_file = FIG_DIR / "td3_training_curves.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {out_file}")
    plt.close()


def plot_reward_convergence():
    """Plot reward convergence over episodes."""
    if not TRAINING_LOG_FILE.exists():
        return
    
    with open(TRAINING_LOG_FILE, "r") as f:
        data = json.load(f)
    
    train_stats = data.get("training", {})
    episode_rewards = train_stats.get("episode_rewards", [])
    
    if not episode_rewards:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(episode_rewards))
    
    # Raw rewards with low opacity
    ax.plot(episodes, episode_rewards, 'b-', alpha=0.2, linewidth=0.5, label='Raw Rewards')
    
    # Moving averages with different windows
    if len(episode_rewards) > 20:
        ma_10 = moving_average(episode_rewards, window=10)
        ma_50 = moving_average(episode_rewards, window=min(50, len(episode_rewards)//4))
        ax.plot(episodes[:len(ma_10)], ma_10, 'g-', linewidth=2, label='MA (window=10)')
        ax.plot(episodes[:len(ma_50)], ma_50, 'r-', linewidth=2.5, label='MA (window=50)')
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Episode Reward', fontsize=14)
    ax.set_title('TD3 Training: Reward Convergence', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    out_file = FIG_DIR / "reward_convergence.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved reward convergence plot to {out_file}")
    plt.close()


def plot_success_rate_analysis():
    """Plot success rate analysis."""
    if not TRAINING_LOG_FILE.exists():
        return
    
    with open(TRAINING_LOG_FILE, "r") as f:
        data = json.load(f)
    
    train_stats = data.get("training", {})
    episode_success = train_stats.get("episode_success", [])
    eval_stats = data.get("evaluation", {})
    
    if not episode_success:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    episodes = np.arange(len(episode_success))
    
    # 1. Cumulative success rate over training
    ax = axes[0]
    cumulative_success = np.cumsum(episode_success) / (episodes + 1)
    ax.plot(episodes, cumulative_success, 'b-', linewidth=2)
    final_rate = train_stats.get("success_rate", 0)
    ax.axhline(y=final_rate, color='r', linestyle='--', linewidth=2,
               label=f'Final Rate: {final_rate:.2%}')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Success Rate', fontsize=12)
    ax.set_title('Success Rate During Training', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # 2. Success rate in windows
    ax = axes[1]
    window_size = max(10, len(episode_success) // 10)
    if len(episode_success) >= window_size:
        windowed_success = []
        window_centers = []
        for i in range(0, len(episode_success) - window_size + 1, window_size):
            window = episode_success[i:i+window_size]
            windowed_success.append(sum(window) / len(window))
            window_centers.append(i + window_size // 2)
        ax.plot(window_centers, windowed_success, 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel('Episode (window center)', fontsize=12)
        ax.set_ylabel('Success Rate in Window', fontsize=12)
        ax.set_title(f'Success Rate (window={window_size})', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
    
    # Add evaluation result as text
    if eval_stats.get("success", False):
        fig.suptitle(f'Training Success Rate Analysis (Final Evaluation: SUCCESS)', 
                     fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle(f'Training Success Rate Analysis (Final Evaluation: FAILED)', 
                     fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    out_file = FIG_DIR / "success_rate_analysis.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved success rate analysis to {out_file}")
    plt.close()


def main():
    print("Generating training visualization plots...")
    plot_training_curves()
    plot_reward_convergence()
    plot_success_rate_analysis()
    print("Done!")


if __name__ == "__main__":
    main()

