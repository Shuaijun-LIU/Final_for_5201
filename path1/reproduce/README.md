# RL-ADN Reproduction

Paper-based reproduction using authors’ data; official code referenced only when paper is unclear.

## Components
- Environment: PandaPower dispatch env (`src/env_pandapower.py`) + Gym wrapper (`src/env_wrapper.py`).
- Baselines: Pyomo NLP (simplified + linearized grid) (`src/nlp_baseline.py`, `scripts/run_nlp_baseline.py`).
- DRL: DDPG/TD3/SAC/PPO via SB3 (`scripts/train_drl.py`, `scripts/eval_drl.py`).
- Augmentation: GMC (GMM + Copula) (`src/augment.py`, `scripts/augment_data.py`).
- Power flow compare: PandaPower vs Tensor PF (`scripts/compare_powerflow.py`).

## Current Outputs
- Metrics:
  - DRL training sample: `outputs/metrics/train_ddpg_seed521.json` (2k steps)
  - Power-flow comparison: `outputs/metrics/pf_compare.json` (PandaPower vs Tensor PF)
  - NLP baseline: not produced yet (LP solver unavailable)
- Figures:
  - Training curves: `outputs/figures/train_reward_penalty.png`
  - Power-flow runtime comparison: `outputs/figures/pf_compare.png`
  - Others (performance bound, time series, violation heatmap) will appear after eval/NLP runs
- Models: `outputs/models/ddpg_seed521.zip`
- Augmented data: generated via scripts to `outputs/augmented/`

## Usage examples
- Train: `PYTHONPATH=path1 python path1/reproduce/scripts/train_drl.py --algo ddpg --total_steps 10000 --seeds 521`
- Eval + performance bound (requires a solver such as glpk/HiGHS): `PYTHONPATH=path1 python path1/reproduce/scripts/eval_drl.py --algo ddpg --seed 521 --split test --episodes 1`
- Plots: `PYTHONPATH=path1 python path1/reproduce/scripts/plot_results.py`; after eval, also run `plot_timeseries.py` and `plot_violation_heatmap.py`
- NLP baseline (needs a solver): `PYTHONPATH=path1 python path1/reproduce/scripts/run_nlp_baseline.py --mode simplified --day_idx 0`
- Data augmentation: `PYTHONPATH=path1 python path1/reproduce/scripts/augment_data.py --multiplier 5`
- Power flow compare: `PYTHONPATH=path1 python path1/reproduce/scripts/compare_powerflow.py`

