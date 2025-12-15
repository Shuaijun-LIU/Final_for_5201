# RL-ADN Reproduction

Paper-based reproduction using authors’ data; official code referenced only when paper is unclear.

## Components
- Environment: PandaPower dispatch env (`src/env_pandapower.py`) + Gym wrapper (`src/env_wrapper.py`).
- Baselines: Pyomo NLP (simplified + linearized grid) (`src/nlp_baseline.py`, `scripts/run_nlp_baseline.py`).
- DRL: DDPG/TD3/SAC/PPO via SB3 (`scripts/train_drl.py`, `scripts/eval_drl.py`).
- Augmentation: GMC (GMM + Copula) (`src/augment.py`, `scripts/augment_data.py`).
- Power flow compare: PandaPower vs Tensor PF (`scripts/compare_powerflow.py`).

## Current Outputs (已生成)
- Metrics:
  - DRL 训练示例：`outputs/metrics/train_ddpg_seed521.json`（2k steps）
  - 潮流对比：`outputs/metrics/pf_compare.json`（PandaPower vs Tensor PF）
  - NLP baseline：尚未生成（缺少可用 LP 求解器）
- Figures:
  - 训练曲线：`outputs/figures/train_reward_penalty.png`
  - 其他（performance bound、时序、越限热图）待有 eval/NLP 结果后生成
- 模型：`outputs/models/ddpg_seed521.zip`
- 增强数据：可通过脚本生成到 `outputs/augmented/`

## Usage examples
- Train: `PYTHONPATH=path1 python path1/reproduce/scripts/train_drl.py --algo ddpg --total_steps 10000 --seeds 521`
- Eval + performance bound（需可用求解器 glpk/HiGHS）：`PYTHONPATH=path1 python path1/reproduce/scripts/eval_drl.py --algo ddpg --seed 521 --split test --episodes 1`
- Plots: `PYTHONPATH=path1 python path1/reproduce/scripts/plot_results.py`；（eval 后可补）`plot_timeseries.py`、`plot_violation_heatmap.py`
- NLP baseline（需求解器）：`PYTHONPATH=path1 python path1/reproduce/scripts/run_nlp_baseline.py --mode simplified --day_idx 0`
- Data augmentation: `PYTHONPATH=path1 python path1/reproduce/scripts/augment_data.py --multiplier 5`
- Power flow compare: `PYTHONPATH=path1 python path1/reproduce/scripts/compare_powerflow.py`

