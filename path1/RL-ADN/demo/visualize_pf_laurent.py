"""
Visualize training data from pf_laurent directory and generate publication-quality figures.

Reads metrics.json files from all algorithms (DDPG, TD3, SAC) in the pf_laurent directory,
and generates the following visualizations:
1. Episode Return training curve comparison
2. Reward breakdown (Power Reward vs Penalty Reward)
3. Violation Time (constraint violation counts)
4. Loss curves (Actor Loss vs Critic Loss)

Output:
- High-quality figures (300 DPI) saved to demo/outputs/paper_like/pf_laurent/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib parameters for publication-quality figures
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

# Algorithm color mapping (publication-style color scheme)
ALGO_COLORS = {
    'ddpg': '#1f77b4',  # Blue
    'td3': '#ff7f0e',   # Orange
    'sac': '#2ca02c',   # Green
}

ALGO_LABELS = {
    'ddpg': 'DDPG',
    'td3': 'TD3',
    'sac': 'SAC',
}


def load_metrics(data_dir: Path) -> Dict[str, Dict]:
    """
    Load metrics.json files for all algorithms.
    
    Args:
        data_dir: Path to pf_laurent directory
        
    Returns:
        Dictionary with algorithm names as keys and metrics data as values
    """
    metrics = {}
    algorithms = ['ddpg', 'td3', 'sac']
    
    for algo in algorithms:
        algo_dir = data_dir / algo
        # Find all seed directories
        seed_dirs = [d for d in algo_dir.iterdir() if d.is_dir()]
        if not seed_dirs:
            print(f"Warning: No seed directories found for {algo}")
            continue
            
        # Load metrics from the first found seed (usually there is only one)
        for seed_dir in seed_dirs:
            metrics_path = seed_dir / 'metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics[algo] = json.load(f)
                print(f"Loaded metrics for {algo} from {metrics_path}")
                break
    
    return metrics


def moving_average(data: List[float], window: int = 5) -> np.ndarray:
    """
    Compute moving average to smooth curves.
    
    Args:
        data: Raw data
        window: Moving average window size
        
    Returns:
        Smoothed data
    """
    if len(data) == 0:
        return np.array([])
    data = np.array(data)
    if len(data) < window:
        return data
    # Use padding to maintain original length
    padded = np.pad(data, (window-1, 0), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')


def plot_episode_return(metrics: Dict[str, Dict], output_dir: Path, smooth: bool = True):
    """
    Plot Episode Return training curve comparison.
    
    Args:
        metrics: Metrics data for all algorithms
        output_dir: Output directory
        smooth: Whether to smooth the curves
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        returns = np.array(data.get('episode_return', []))
        
        if len(episodes) == 0 or len(returns) == 0:
            continue
            
        # Optional: smoothing
        if smooth and len(returns) > 1:
            returns_smooth = moving_average(returns.tolist(), window=3)
        else:
            returns_smooth = returns
        
        # Ensure consistent length
        min_len = min(len(episodes), len(returns_smooth))
        episodes = episodes[:min_len]
        returns_smooth = returns_smooth[:min_len]
        
        ax.plot(episodes, returns_smooth, 
               color=ALGO_COLORS.get(algo, 'black'),
               label=ALGO_LABELS.get(algo, algo.upper()),
               linewidth=2.0,
               marker='o' if len(episodes) <= 30 else None,
               markersize=4)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Episode Return', fontweight='bold')
    ax.set_title('Training Curves: Episode Return Comparison', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'episode_return_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_breakdown(metrics: Dict[str, Dict], output_dir: Path, smooth: bool = True):
    """
    Plot reward breakdown (Power Reward vs Penalty Reward).
    
    Args:
        metrics: Metrics data for all algorithms
        output_dir: Output directory
        smooth: Whether to smooth the curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Power Reward
    ax1 = axes[0]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        power_rewards = np.array(data.get('reward_for_power', []))
        
        if len(episodes) == 0 or len(power_rewards) == 0:
            continue
            
        if smooth and len(power_rewards) > 1:
            power_rewards_smooth = moving_average(power_rewards.tolist(), window=3)
        else:
            power_rewards_smooth = power_rewards
        
        min_len = min(len(episodes), len(power_rewards_smooth))
        episodes = episodes[:min_len]
        power_rewards_smooth = power_rewards_smooth[:min_len]
        
        ax1.plot(episodes, power_rewards_smooth,
                color=ALGO_COLORS.get(algo, 'black'),
                label=ALGO_LABELS.get(algo, algo.upper()),
                linewidth=2.0,
                marker='o' if len(episodes) <= 30 else None,
                markersize=4)
    
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Power Reward', fontweight='bold')
    ax1.set_title('Power Reward', fontweight='bold')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Subplot 2: Penalty Reward
    ax2 = axes[1]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        penalty_rewards = np.array(data.get('reward_for_penalty', []))
        
        if len(episodes) == 0 or len(penalty_rewards) == 0:
            continue
            
        if smooth and len(penalty_rewards) > 1:
            penalty_rewards_smooth = moving_average(penalty_rewards.tolist(), window=3)
        else:
            penalty_rewards_smooth = penalty_rewards
        
        min_len = min(len(episodes), len(penalty_rewards_smooth))
        episodes = episodes[:min_len]
        penalty_rewards_smooth = penalty_rewards_smooth[:min_len]
        
        ax2.plot(episodes, penalty_rewards_smooth,
                color=ALGO_COLORS.get(algo, 'black'),
                label=ALGO_LABELS.get(algo, algo.upper()),
                linewidth=2.0,
                marker='o' if len(episodes) <= 30 else None,
                markersize=4)
    
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Penalty Reward', fontweight='bold')
    ax2.set_title('Penalty Reward', fontweight='bold')
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.suptitle('Reward Breakdown: Power vs Penalty', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'reward_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_violation_time(metrics: Dict[str, Dict], output_dir: Path, smooth: bool = True):
    """
    Plot Violation Time (constraint violation counts) comparison.
    
    Args:
        metrics: Metrics data for all algorithms
        output_dir: Output directory
        smooth: Whether to smooth the curves
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        violations = np.array(data.get('violation_time', []))
        
        if len(episodes) == 0 or len(violations) == 0:
            continue
            
        if smooth and len(violations) > 1:
            violations_smooth = moving_average(violations.tolist(), window=3)
        else:
            violations_smooth = violations
        
        min_len = min(len(episodes), len(violations_smooth))
        episodes = episodes[:min_len]
        violations_smooth = violations_smooth[:min_len]
        
        ax.plot(episodes, violations_smooth,
               color=ALGO_COLORS.get(algo, 'black'),
               label=ALGO_LABELS.get(algo, algo.upper()),
               linewidth=2.0,
               marker='o' if len(episodes) <= 30 else None,
               markersize=4)
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Violation Count', fontweight='bold')
    ax.set_title('Constraint Violations During Training', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / 'violation_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_curves(metrics: Dict[str, Dict], output_dir: Path, smooth: bool = True):
    """
    Plot loss curves (Actor Loss vs Critic Loss).
    
    Args:
        metrics: Metrics data for all algorithms
        output_dir: Output directory
        smooth: Whether to smooth the curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Actor Loss
    ax1 = axes[0]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        actor_losses = np.array(data.get('actor_loss', []))
        
        if len(episodes) == 0 or len(actor_losses) == 0:
            continue
            
        if smooth and len(actor_losses) > 1:
            actor_losses_smooth = moving_average(actor_losses.tolist(), window=3)
        else:
            actor_losses_smooth = actor_losses
        
        min_len = min(len(episodes), len(actor_losses_smooth))
        episodes = episodes[:min_len]
        actor_losses_smooth = actor_losses_smooth[:min_len]
        
        ax1.plot(episodes, actor_losses_smooth,
                color=ALGO_COLORS.get(algo, 'black'),
                label=ALGO_LABELS.get(algo, algo.upper()),
                linewidth=2.0,
                marker='o' if len(episodes) <= 30 else None,
                markersize=4)
    
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Actor Loss', fontweight='bold')
    ax1.set_title('Actor Loss', fontweight='bold')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Subplot 2: Critic Loss
    ax2 = axes[1]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
            
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        critic_losses = np.array(data.get('critic_loss', []))
        
        if len(episodes) == 0 or len(critic_losses) == 0:
            continue
            
        if smooth and len(critic_losses) > 1:
            critic_losses_smooth = moving_average(critic_losses.tolist(), window=3)
        else:
            critic_losses_smooth = critic_losses
        
        min_len = min(len(episodes), len(critic_losses_smooth))
        episodes = episodes[:min_len]
        critic_losses_smooth = critic_losses_smooth[:min_len]
        
        ax2.plot(episodes, critic_losses_smooth,
                color=ALGO_COLORS.get(algo, 'black'),
                label=ALGO_LABELS.get(algo, algo.upper()),
                linewidth=2.0,
                marker='o' if len(episodes) <= 30 else None,
                markersize=4)
    
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Critic Loss', fontweight='bold')
    ax2.set_title('Critic Loss', fontweight='bold')
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.suptitle('Training Loss Curves: Actor vs Critic', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comprehensive_comparison(metrics: Dict[str, Dict], output_dir: Path, smooth: bool = True):
    """
    Generate comprehensive comparison figure (all metrics in one figure using subplots).
    
    Args:
        metrics: Metrics data for all algorithms
        output_dir: Output directory
        smooth: Whether to smooth the curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Episode Return
    ax1 = axes[0, 0]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        returns = np.array(data.get('episode_return', []))
        if len(episodes) > 0 and len(returns) > 0:
            if smooth and len(returns) > 1:
                returns_smooth = moving_average(returns.tolist(), window=3)
            else:
                returns_smooth = returns
            min_len = min(len(episodes), len(returns_smooth))
            ax1.plot(episodes[:min_len], returns_smooth[:min_len],
                    color=ALGO_COLORS.get(algo, 'black'),
                    label=ALGO_LABELS.get(algo, algo.upper()),
                    linewidth=2.0)
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Episode Return', fontweight='bold')
    ax1.set_title('(a) Episode Return', fontweight='bold')
    ax1.legend(loc='best', frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Subplot 2: Violation Time
    ax2 = axes[0, 1]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        violations = np.array(data.get('violation_time', []))
        if len(episodes) > 0 and len(violations) > 0:
            if smooth and len(violations) > 1:
                violations_smooth = moving_average(violations.tolist(), window=3)
            else:
                violations_smooth = violations
            min_len = min(len(episodes), len(violations_smooth))
            ax2.plot(episodes[:min_len], violations_smooth[:min_len],
                    color=ALGO_COLORS.get(algo, 'black'),
                    label=ALGO_LABELS.get(algo, algo.upper()),
                    linewidth=2.0)
    ax2.set_xlabel('Episode', fontweight='bold')
    ax2.set_ylabel('Violation Count', fontweight='bold')
    ax2.set_title('(b) Constraint Violations', fontweight='bold')
    ax2.legend(loc='best', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Subplot 3: Actor Loss
    ax3 = axes[1, 0]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        actor_losses = np.array(data.get('actor_loss', []))
        if len(episodes) > 0 and len(actor_losses) > 0:
            if smooth and len(actor_losses) > 1:
                actor_losses_smooth = moving_average(actor_losses.tolist(), window=3)
            else:
                actor_losses_smooth = actor_losses
            min_len = min(len(episodes), len(actor_losses_smooth))
            ax3.plot(episodes[:min_len], actor_losses_smooth[:min_len],
                    color=ALGO_COLORS.get(algo, 'black'),
                    label=ALGO_LABELS.get(algo, algo.upper()),
                    linewidth=2.0)
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Actor Loss', fontweight='bold')
    ax3.set_title('(c) Actor Loss', fontweight='bold')
    ax3.legend(loc='best', frameon=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Subplot 4: Critic Loss
    ax4 = axes[1, 1]
    for algo in ['ddpg', 'td3', 'sac']:
        if algo not in metrics:
            continue
        data = metrics[algo]
        episodes = np.array(data.get('episode', []))
        critic_losses = np.array(data.get('critic_loss', []))
        if len(episodes) > 0 and len(critic_losses) > 0:
            if smooth and len(critic_losses) > 1:
                critic_losses_smooth = moving_average(critic_losses.tolist(), window=3)
            else:
                critic_losses_smooth = critic_losses
            min_len = min(len(episodes), len(critic_losses_smooth))
            ax4.plot(episodes[:min_len], critic_losses_smooth[:min_len],
                    color=ALGO_COLORS.get(algo, 'black'),
                    label=ALGO_LABELS.get(algo, algo.upper()),
                    linewidth=2.0)
    ax4.set_xlabel('Episode', fontweight='bold')
    ax4.set_ylabel('Critic Loss', fontweight='bold')
    ax4.set_title('(d) Critic Loss', fontweight='bold')
    ax4.legend(loc='best', frameon=True)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle('Comprehensive Training Comparison', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function: load data and generate all visualization figures."""
    # Determine paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'outputs' / 'paper_like' / 'pf_laurent'
    output_dir = data_dir / 'figures'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all metrics
    metrics = load_metrics(data_dir)
    
    if not metrics:
        print("Error: No metrics found. Please check the data directory.")
        return
    
    print(f"\nFound metrics for {len(metrics)} algorithm(s): {list(metrics.keys())}")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_episode_return(metrics, output_dir, smooth=True)
    plot_reward_breakdown(metrics, output_dir, smooth=True)
    plot_violation_time(metrics, output_dir, smooth=True)
    plot_loss_curves(metrics, output_dir, smooth=True)
    plot_comprehensive_comparison(metrics, output_dir, smooth=True)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

