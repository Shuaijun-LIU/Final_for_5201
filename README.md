# 最终作业
## 给老师和TA的留言

我最初选择了 path1 来完成这个作业，但我觉得似乎内容并不够充分。然后我又尝试完成path2，在这个过程中，一直没有解决问题，反而耽误了提交，希望老师和ta理解，再次抱歉，我将问题写在了report中，也希望能给我一点建议。我尝试的两条路径，包括：

**path1.** 复现一篇论文 (path1/paper.pdf) - [详细说明](path1/README.md)

这篇论文由代尔夫特理工大学电气可持续能源系于 2025 年发表在 Energy and AI 期刊上。

**path2.** 尝试将 TD3 算法应用于我的研究（相对复杂环境中的无人机轨迹规划）并提供演示 ([演示链接](https://shuaijun-liu.github.io/Final_for_5201/))，但是没有完成最终的版本 - [详细说明](path2/README.md)

---

# Final Assignment
## Message to Professor and TA

I initially chose path1 to complete this assignment, but I felt that the content was not sufficient enough. Then I tried to complete path2, but during this process, I kept encountering problems that I couldn't solve, which delayed my submission. I hope the professor and TA can understand, and I apologize again. I have documented the issues in my report and would appreciate any suggestions. The two paths I attempted include:

**path1.** Reproduce a paper (path1/paper.pdf) - [Detailed Documentation](path1/README.md)

This paper was published in Energy and AI journal in 2025 by the Department of Electrical Sustainable Energy, Delft University of Technology.

**path2.** Attempted to apply the TD3 algorithm to my research (drone trajectory planning in relatively complex environments) and provide a demo ([Demo Link](https://shuaijun-liu.github.io/Final_for_5201/)), but did not complete the final version - [Detailed Documentation](path2/README.md)

---

## Path 1: RL-ADN Reproduction and Evaluation

### Project Overview

This project implements two complementary approaches to validate the RL-ADN framework for energy storage systems (ESS) dispatch in active distribution networks:

1. **Independent Reproduction** (`path1/reproduce/`): A self-contained implementation built from scratch based on the paper's methodology, independent of the authors' library.
2. **Open-Source Code Validation** (`path1/RL-ADN/`): Validation using the authors' provided open-source implementation to obtain quantitative metrics.

### Key Results

**Independent Reproduction:**
- Zero voltage violations throughout training
- Penalty-driven learning patterns consistent with paper
- PandaPower averaging 13.1 ms per computation
- Complete MDP alignment with paper's state/action/reward formulation

**Open-Source Demo Validation:**
- Final episode return: 1.225 (positive, demonstrating successful learning)
- Voltage violations: Minimal (3 timesteps in final episode, magnitudes ~10⁻⁴ p.u.)
- Price responsiveness: Agent learns to charge during low prices and discharge during high prices
- Algorithm comparison: SAC achieves best performance (zero violations, stable returns)

### Quick Start

**Using Authors' Code:**
```bash
cd path1/RL-ADN
python demo/run_ddpg_quick_demo.py \
    --num-episode 100 \
    --target-step 2048 \
    --warm-up 4096 \
    --batch-size 256 \
    --gamma 0.99 \
    --lr 6e-4 \
    --seed 521
```

**Independent Reproduction:**
```bash
cd path1/reproduce
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python scripts/train_drl.py --algo ddpg --total_steps 10000 --seeds 521 --split train
python scripts/eval_drl.py --algo ddpg --seed 521 --split test --episodes 1
```

### Implementation Details

- **Environment**: PandaPower-based dispatch environment with Gymnasium-compatible interface
- **MDP Formulation**: State space (net loads, price, SOC), action space (ESS power setpoints), reward function (energy arbitrage revenue + voltage violation penalties)
- **DRL Algorithms**: Integration with Stable-Baselines3 (DDPG, TD3, SAC, PPO)
- **Data Augmentation**: GMM-BIC component selection + Copula-based correlation modeling

For more details, see [path1/README.md](path1/README.md).

---

## Path 2: UAV Trajectory Planning in Complex Terrain

### Project Overview

This project applies reinforcement learning (RL) to 3D trajectory planning for Unmanned Aerial Vehicles (UAVs) in complex terrain environments. The work extends from successful RL application in UAV attitude control to continuous action space trajectory planning using TD3.

### Current Status

Unfortunately, the TD3-based trajectory planning approach did not achieve satisfactory results. Over 20,000 training timesteps, the agent achieved a **0% success rate**, with all episodes terminating immediately after the first action due to building collisions.

**Root Causes Identified:**
1. Starting position issues (may be inside or immediately adjacent to buildings)
2. Action scale too large (λ_a = 10.0 m) for the environment's obstacle density
3. Sparse reward problem (lacks intermediate guidance signals)
4. Environment complexity (3D navigation with multiple obstacle types)
5. Training strategy limitations (lack of curriculum learning)

### Implementation Overview

**System Architecture:**
1. **Terrain Generator** (`path2/terrain_generator/`): Generates complex 3D terrain data including buildings, trees, lakes, and user positions
2. **RL Trajectory Planner** (`path2/RL_trajectory_planner/`): Implements TD3 and Q-learning/SARSA algorithms
3. **Visualization**: Web-based 3D visualization using Three.js

**TD3 Implementation:**
- **State Space**: 8-dimensional vector `[x_t, y_t, z_t, dx_t, dy_t, dz_t, d_t, h_terrain]`
- **Action Space**: 3-dimensional continuous vector `[Δx, Δy, Δz] ∈ [-1, 1]³` (scaled by 10.0 m)
- **Reward Function**: +500 for reaching goal, -500 for collision, -1.0 per step penalty
- **Training**: 20,000 timesteps with max 800 steps per episode

**Terrain Generation:**
- 1000m × 1000m map with ~1,500 buildings, ~4,000 trees, and multiple lakes
- Complete height map at 2m resolution (251,001 points) for fast O(1) height queries
- Deterministic terrain generation with pre-analyzed mountain regions

### Quick Start

**Generate Terrain:**
```bash
cd path2/terrain_generator
pip install -r requirements.txt
python3 generate_terrain.py
```

**Train TD3 Agent:**
```bash
cd path2/RL_trajectory_planner
python3 scripts/sb3_train_and_eval.py
```

**Train Q-Learning/SARSA (Alternative):**
```bash
cd path2/RL_trajectory_planner/Q_Learning_and_SARSA
python3 cli.py \
    --terrain_json ../../data/terrain_data.json \
    --algo q_learning \
    --method multi_user_rl \
    --episodes 4000
```

**Visualize Results:**
```bash
cd path2/terrain_generator
./start_server.sh
# Open http://localhost:8000/visualize.html
```

### Key Challenges

1. **Sparse Reward Problem**: Binary reward structure provides insufficient intermediate guidance
2. **Action Space Configuration**: 10m action scale too large for precise obstacle avoidance
3. **Starting Position Validation**: Need safe starting positions or automatic adjustment
4. **Environment Complexity**: 3D navigation with multiple obstacle types requires sophisticated training strategies
5. **Exploration-Exploitation Trade-off**: Exploration leads to immediate termination

### Proposed Improvements

1. Reward function enhancement (add distance reduction rewards)
2. Action scale reduction (from 10.0m to 2.0-5.0m)
3. Curriculum learning (start with simpler scenarios)
4. Starting position validation (automatic safe position detection)
5. State augmentation (add local obstacle information)
6. Hybrid approaches (combine RL with classical path planning)

For more details, see [path2/README.md](path2/README.md). 




