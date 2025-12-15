"""
Generate collision statistics visualization.
Reads training_log.json to analyze collision patterns.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
TRAINING_LOG_FILE = OUTPUT_DIR / "training_log.json"
FIG_DIR = BASE_DIR / "report_latex" / "fig" / "part2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_collision_statistics():
    """Plot collision statistics from training."""
    if not TRAINING_LOG_FILE.exists():
        print(f"Training log not found: {TRAINING_LOG_FILE}")
        print("Please run train_with_logging.py first.")
        return
    
    with open(TRAINING_LOG_FILE, "r") as f:
        data = json.load(f)
    
    train_stats = data.get("training", {})
    collision_stats = train_stats.get("collision_stats", {})
    eval_stats = data.get("evaluation", {})
    eval_collisions = eval_stats.get("collisions", {})
    
    if not collision_stats:
        print("No collision statistics found.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Training collision statistics (bar chart)
    ax = axes[0]
    collision_types = list(collision_stats.keys())
    collision_counts = list(collision_stats.values())
    
    if collision_types:
        colors = plt.cm.Set3(np.linspace(0, 1, len(collision_types)))
        bars = ax.bar(collision_types, collision_counts, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Collision Type', fontsize=12)
        ax.set_ylabel('Number of Collisions', fontsize=12)
        ax.set_title('Collision Statistics During Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Evaluation collision breakdown (pie chart if collisions exist, else text)
    ax = axes[1]
    if eval_collisions and any(eval_collisions.values()):
        collision_types_eval = [k for k, v in eval_collisions.items() if v > 0]
        collision_counts_eval = [eval_collisions[k] for k in collision_types_eval]
        colors_eval = plt.cm.Pastel1(np.linspace(0, 1, len(collision_types_eval)))
        ax.pie(collision_counts_eval, labels=collision_types_eval, autopct='%1.1f%%',
              colors=colors_eval, startangle=90, textprops={'fontsize': 11})
        ax.set_title('Collision Distribution\n(Final Evaluation)', fontsize=14, fontweight='bold')
    else:
        # No collisions in evaluation
        ax.text(0.5, 0.5, 'No Collisions\nin Final Evaluation\n✓ SUCCESS', 
               ha='center', va='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.set_title('Final Evaluation Result', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    out_file = FIG_DIR / "collision_statistics.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved collision statistics to {out_file}")
    plt.close()


def main():
    print("Generating collision statistics visualization...")
    plot_collision_statistics()
    print("Done!")


if __name__ == "__main__":
    main()

