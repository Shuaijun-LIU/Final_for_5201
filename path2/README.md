# UAV Trajectory Planning in Complex Terrain

This project applies reinforcement learning (RL) to 3D trajectory planning for Unmanned Aerial Vehicles (UAVs) in complex terrain environments. The work extends from successful RL application in UAV attitude control (DDQN-based FEAR-PID) to continuous action space trajectory planning using TD3, forming a comprehensive RL-driven UAV control framework.

## Current Status

### TD3-based Trajectory Planning

Unfortunately, the TD3-based trajectory planning approach did not achieve satisfactory results. Over 20,000 training timesteps, the agent achieved a **0% success rate**, with all episodes terminating immediately after the first action due to building collisions. The average episode length was exactly 1.0 steps, and the average episode reward was -501.17 (collision penalty -500 + step penalty -1.0).

**Root Causes Identified:**
1. **Starting Position Issues**: The starting position (User1) may be located inside or immediately adjacent to buildings, causing immediate collision
2. **Action Scale Too Large**: Action scale λ_a = 10.0 m is too large for the environment's obstacle density, making fine-grained navigation impossible
3. **Sparse Reward Problem**: The reward function lacks intermediate guidance signals (no distance reduction reward, no soft obstacle penalties), creating a classic sparse reward problem
4. **Environment Complexity**: 3D navigation requires simultaneous consideration of horizontal and vertical movement, with multiple obstacle types having different collision geometries
5. **Training Strategy Limitations**: Lack of curriculum learning and insufficient training duration

### Q-Learning and SARSA Alternative Approach

I also implemented Q-Learning and SARSA algorithms in the `Q_Learning_and_SARSA` directory as alternative approaches. These discrete RL methods successfully trained and converged, but the resulting trajectories still showed poor obstacle avoidance performance. The grid-based discretization (10m grid step) limits the precision of obstacle avoidance, and the learned policies tend to take suboptimal paths that pass too close to obstacles or fail to maintain adequate safety margins.

## Implementation Overview

### System Architecture

The project consists of three main components:

1. **Terrain Generator** (`terrain_generator/`): Generates complex 3D terrain data including buildings, trees, lakes, and user positions
2. **RL Trajectory Planner** (`RL_trajectory_planner/`): Implements TD3 and Q-learning/SARSA algorithms for trajectory planning
3. **Visualization**: Web-based 3D visualization using Three.js

### Terrain Generation

The terrain generation module creates a 1000m × 1000m map with:
- **Terrain**: Multi-octave SimplexNoise-based height field with city boundaries and mountain regions
- **Buildings**: ~1,500 buildings (1,300 city buildings + 200 mountain buildings) modeled as axis-aligned bounding boxes
- **Trees**: ~4,000 trees represented as cylindrical obstacles
- **Lakes**: Multiple elliptical lake regions
- **User Positions**: 6 user positions (User1 through User6) for start/goal locations

The terrain data is stored in `data/terrain_data.json` with a complete height map at 2m resolution (251,001 points) for fast O(1) height queries.

**Key Features:**
- Deterministic terrain generation (same seed produces identical terrain)
- Complete terrain coverage with 2m resolution height map
- Fast height lookup using `HeightMap` utility class
- Pre-analyzed mountain regions for quick identification

See `terrain_generator/README.md` for detailed documentation.

### RL Trajectory Planning

#### TD3 Implementation

The TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation uses Stable-Baselines3 for continuous action space trajectory planning:

- **State Space**: 8-dimensional vector `[x_t, y_t, z_t, dx_t, dy_t, dz_t, d_t, h_terrain]`
  - Position (x, y, z), relative vector to goal, Euclidean distance, terrain height
- **Action Space**: 3-dimensional continuous vector `[Δx, Δy, Δz] ∈ [-1, 1]³`
  - Scaled by λ_a = 10.0 m to obtain actual displacement
- **Reward Function**: 
  - +500 for reaching goal (within 5.0 m)
  - -500 for collision
  - -1.0 per step penalty
  - -0.1 × ||a_t||₂ smoothness penalty
- **Environment**: Gymnasium-compatible environment with comprehensive collision detection

**Training Configuration:**
- Learning rate: 10⁻³
- Batch size: 128
- Discount factor γ: 0.99
- Training timesteps: 20,000
- Max episode steps: 800

#### Q-Learning and SARSA Implementation

The discrete RL approaches use grid-based world representation:

- **Grid Resolution**: 10m grid step
- **Action Space**: 8-directional movement (4 cardinal + 4 diagonal)
- **State Space**: Grid coordinates + visited user mask (for multi-user scenarios)
- **Algorithms**: Q-Learning and SARSA with epsilon-greedy exploration
- **Obstacle Handling**: Rasterized buildings, trees, lakes, and terrain height blocks

**Features:**
- Multi-user trajectory planning (visiting multiple users in sequence)
- Single-segment planning (User1 to User6)
- BFS fallback for unreachable goals
- Configurable obstacle buffers and terrain blocking thresholds

See `RL_trajectory_planner/Q_Learning_and_SARSA/` for implementation details.

## Project Structure

```
path2/
├── README.md                          # This file
├── data/
│   ├── terrain_data.json              # Generated terrain data (buildings, trees, lakes, height map)
│   └── paths.json                     # Exported trajectory paths for visualization
├── outputs/
│   ├── models/                        # Trained RL models
│   │   └── td3_model.zip
│   ├── tensorboard/                   # TensorBoard logs
│   └── training_log.json              # Training metrics and statistics
├── terrain_generator/                  # Terrain generation module
│   ├── README.md                      # Detailed terrain generator documentation
│   ├── generate_terrain.py            # Main terrain generation script
│   ├── terrain_utils.py               # Terrain calculation utilities
│   ├── height_map_utils.py            # Height map loading and queries
│   ├── config.py                      # Configuration constants
│   ├── visualize.html                 # Web-based 3D visualization
│   ├── js/                            # JavaScript modules for visualization
│   └── data/                          # Generated terrain data
├── RL_trajectory_planner/             # RL trajectory planning module
│   ├── README.md                      # RL planner documentation
│   ├── env/                           # Environment implementation
│   │   ├── sb3_env.py                 # TD3 environment (Gymnasium-compatible)
│   │   ├── terrain_env.py             # Base terrain environment
│   │   ├── collision.py               # Collision detection system
│   │   ├── height_field.py            # Height map queries
│   │   └── config.py                  # Environment configuration
│   ├── scripts/                       # Training and evaluation scripts
│   │   ├── sb3_train_and_eval.py      # TD3 training and evaluation
│   │   ├── train_with_logging.py      # Training with comprehensive logging
│   │   ├── render_paths.py            # Export paths to JSON
│   │   └── plot_*.py                  # Visualization scripts
│   └── Q_Learning_and_SARSA/          # Discrete RL implementations
│       ├── cli.py                      # Command-line interface
│       ├── rl_algos.py                 # Q-learning and SARSA algorithms
│       ├── envs.py                     # Grid-based environments
│       ├── grid.py                     # Grid utilities and BFS
│       ├── obstacles.py                # Obstacle rasterization
│       ├── height_provider.py          # Height query utilities
│       └── planner_config.py          # Configuration
└── report_latex/                       # LaTeX report documentation
    └── fig/part2/                      # Figures and visualizations
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Step 1: Generate Terrain Data

First, generate the terrain data that will be used for trajectory planning:

```bash
cd terrain_generator
pip install -r requirements.txt
python3 generate_terrain.py
```

This generates `data/terrain_data.json` with terrain, buildings, trees, lakes, and user positions.

### Step 2: Train TD3 Agent

Train the TD3 agent for trajectory planning:

```bash
cd RL_trajectory_planner
pip install torch numpy stable-baselines3 gymnasium

# Train TD3 agent
python3 scripts/sb3_train_and_eval.py

# Or train with comprehensive logging
python3 scripts/train_with_logging.py
```

**Note**: As mentioned in the status section, TD3 training did not achieve satisfactory results. The training will complete but the agent will not learn effective obstacle avoidance strategies.

### Step 3: Train Q-Learning or SARSA (Alternative)

Train discrete RL algorithms as an alternative approach:

```bash
cd RL_trajectory_planner/Q_Learning_and_SARSA

# Train Q-Learning for multi-user trajectory
python3 cli.py \
    --terrain_json ../../data/terrain_data.json \
    --out_json ../../data/paths.json \
    --algo q_learning \
    --method multi_user_rl \
    --episodes 4000 \
    --grid_step 10

# Or train SARSA for single-segment planning
python3 cli.py \
    --terrain_json ../../data/terrain_data.json \
    --out_json ../../data/paths.json \
    --algo sarsa \
    --method single_segment_rl \
    --start_user_idx 0 \
    --goal_user_idx 5 \
    --episodes 4000
```

### Step 4: Visualize Results

Visualize the generated trajectories:

```bash
cd terrain_generator
./start_server.sh
# Or: python3 -m http.server 8000
```

Then open `http://localhost:8000/visualize.html` in your browser. The visualization will load `data/paths.json` and display the planned trajectories on the 3D terrain.

**Controls:**
- Left mouse drag: Rotate view
- Right mouse drag: Pan view
- Mouse wheel: Zoom

## Key Challenges and Future Work

### Identified Challenges

1. **Sparse Reward Problem**: The binary reward structure (goal reward vs. collision penalty) provides insufficient intermediate guidance for learning
2. **Action Space Configuration**: The 10m action scale is too large for precise obstacle avoidance in dense environments
3. **Starting Position Validation**: Need to ensure safe starting positions or implement automatic adjustment
4. **Environment Complexity**: 3D navigation with multiple obstacle types requires sophisticated training strategies
5. **Exploration-Exploitation Trade-off**: Exploration leads to immediate termination, preventing accumulation of positive learning experiences

### Proposed Improvements

1. **Reward Function Enhancement**: Add distance reduction rewards and safety bonuses for dense feedback
2. **Action Scale Reduction**: Reduce action scale from 10.0m to 2.0-5.0m for finer control
3. **Curriculum Learning**: Start with simpler scenarios and gradually increase complexity
4. **Starting Position Validation**: Implement automatic safe position detection and adjustment
5. **State Augmentation**: Add local obstacle information (nearest building distance, etc.) to enhance awareness
6. **Hybrid Approaches**: Combine RL with classical path planning methods (A*) for initialization

## Documentation

- **Terrain Generator**: See `terrain_generator/README.md` for detailed terrain generation documentation
- **RL Trajectory Planner**: See `RL_trajectory_planner/README.md` for RL implementation details
- **Terrain Storage**: See `terrain_generator/TERRAIN_STORAGE.md` for height map data structure details
- **Path Usage**: See `terrain_generator/PATH_USAGE.md` for trajectory visualization details

## Author

Shuaijun Liu  
Student ID: 50047151  
The Hong Kong University of Science and Technology (Guangzhou)

## License

This project is part of the IOTA5201 (L01) - RL for Intelligent Decision Making in Cyber-Physical Systems course assignment.

