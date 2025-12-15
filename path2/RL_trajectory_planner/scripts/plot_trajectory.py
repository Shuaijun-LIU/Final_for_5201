"""
Generate 3D trajectory visualization.
Reads paths.json and terrain_data.json to visualize the planned trajectory.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12

BASE_DIR = Path(__file__).resolve().parents[2]
PATHS_FILE = BASE_DIR / "data" / "paths.json"
TERRAIN_FILE = BASE_DIR / "data" / "terrain_data.json"
FIG_DIR = BASE_DIR / "report_latex" / "fig" / "part2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_terrain_data():
    """Load terrain data for visualization."""
    if not TERRAIN_FILE.exists():
        return None, None, None
    
    with open(TERRAIN_FILE, "r") as f:
        data = json.load(f)
    
    users = data.get("finalUsers", [])
    buildings = data.get("buildingColliders", [])
    trees = data.get("trees", [])
    
    return users, buildings, trees


def plot_trajectory_3d():
    """Plot 3D trajectory with terrain and obstacles."""
    if not PATHS_FILE.exists():
        print(f"Paths file not found: {PATHS_FILE}")
        print("Please run training first to generate paths.json")
        return
    
    with open(PATHS_FILE, "r") as f:
        paths_data = json.load(f)
    
    if not paths_data:
        print("No paths found in paths.json")
        return
    
    # Get first path
    path_data = paths_data[0]
    path_points = path_data.get("points", [])
    
    if not path_points:
        print("No points in path")
        return
    
    # Extract coordinates
    x_coords = [p["x"] for p in path_points]
    y_coords = [p["y"] for p in path_points]
    z_coords = [p["z"] for p in path_points]
    
    # Load terrain data
    users, buildings, trees = load_terrain_data()
    
    # Create figure with two subplots: side view and top view
    fig = plt.figure(figsize=(16, 8))
    
    # Side view (X-Z plane, Y is height)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot trajectory
    ax1.plot(x_coords, z_coords, y_coords, 'b-', linewidth=2.5, label='TD3 Planned Path', zorder=10)
    ax1.scatter([x_coords[0]], [z_coords[0]], [y_coords[0]], 
                c='green', s=200, marker='o', label='Start (User1)', zorder=11)
    ax1.scatter([x_coords[-1]], [z_coords[-1]], [y_coords[-1]], 
                c='red', s=200, marker='*', label='Goal (User6)', zorder=11)
    
    # Plot users if available
    if users:
        user_x = [u["x"] for u in users]
        user_z = [u["z"] for u in users]
        user_y = [u["y"] for u in users]
        ax1.scatter(user_x, user_z, user_y, c='orange', s=100, marker='s', 
                   label='Users', alpha=0.6, zorder=9)
    
    # Plot buildings (simplified as boxes)
    if buildings:
        for b in buildings[:50]:  # Limit to first 50 for performance
            x, z = b["x"], b["z"]
            w, d = b.get("halfWidth", 5), b.get("halfDepth", 5)
            h = b.get("height", 20)
            # Draw wireframe box
            corners_x = [x-w, x+w, x+w, x-w, x-w]
            corners_z = [z-d, z-d, z+d, z+d, z-d]
            corners_y = [0, 0, 0, 0, 0]
            ax1.plot(corners_x, corners_z, corners_y, 'k-', alpha=0.3, linewidth=0.5)
            corners_y = [h, h, h, h, h]
            ax1.plot(corners_x, corners_z, corners_y, 'k-', alpha=0.3, linewidth=0.5)
            for i in range(4):
                ax1.plot([corners_x[i], corners_x[i]], [corners_z[i], corners_z[i]], 
                        [0, h], 'k-', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_zlabel('Y (Height, m)', fontsize=12)
    ax1.set_title('3D Trajectory: Side View', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.view_init(elev=20, azim=45)
    
    # Top view (X-Z plane, Y is color-coded)
    ax2 = fig.add_subplot(122)
    
    # Plot trajectory with color coding for altitude
    scatter = ax2.scatter(x_coords, z_coords, c=y_coords, cmap='viridis', 
                         s=30, edgecolors='black', linewidth=0.5, alpha=0.8, zorder=10)
    
    # Add trajectory line
    ax2.plot(x_coords, z_coords, 'b-', linewidth=1, alpha=0.5, zorder=9)
    
    # Mark start and goal
    ax2.scatter([x_coords[0]], [z_coords[0]], c='green', s=300, marker='o', 
               edgecolors='black', linewidth=2, label='Start (User1)', zorder=11)
    ax2.scatter([x_coords[-1]], [z_coords[-1]], c='red', s=300, marker='*', 
               edgecolors='black', linewidth=2, label='Goal (User6)', zorder=11)
    
    # Plot users
    if users:
        user_x = [u["x"] for u in users]
        user_z = [u["z"] for u in users]
        ax2.scatter(user_x, user_z, c='orange', s=150, marker='s', 
                   edgecolors='black', linewidth=1, label='Users', alpha=0.7, zorder=8)
    
    # Plot buildings (simplified)
    if buildings:
        for b in buildings[:100]:  # Limit for performance
            x, z = b["x"], b["z"]
            w, d = b.get("halfWidth", 5), b.get("halfDepth", 5)
            rect = plt.Rectangle((x-w, z-d), 2*w, 2*d, 
                               facecolor='gray', edgecolor='black', 
                               alpha=0.5, linewidth=0.5, zorder=7)
            ax2.add_patch(rect)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title('3D Trajectory: Top View (Color = Height)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Altitude (m)', fontsize=11)
    
    plt.tight_layout()
    out_file = FIG_DIR / "trajectory_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory visualization to {out_file}")
    plt.close()


def main():
    print("Generating trajectory visualization...")
    plot_trajectory_3d()
    print("Done!")


if __name__ == "__main__":
    main()

