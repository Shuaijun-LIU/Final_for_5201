# 5201_RL Final Project Overview

## Repository structure
- `RL-ADN/` (upstream authors’ code): Original open-source library from the paper. We only reuse its environment/data/algorithm implementations to run demos; no core algorithm changes were made.
- `reproduce/` (our reproduction): Self-contained reproduction based on the paper, independent of the upstream library. See `reproduce/README.md` for components, scripts, and outputs.
- `demo/outputs/`: Generated demo results. Latest run: `run_ep100_step2048_seed521/` with figures (`fig_train_return.png`, `fig_train_violation.png`, `fig_train_reward_components.png`, `fig_eval_timeseries.png`, `fig_violation_heatmap.png`) plus `metrics.json` and `report.md`.

## How to run the quick demo (reuse upstream RL-ADN code)
```bash
cd RL-ADN
python demo/run_ddpg_quick_demo.py --num-episode 100 --target-step 2048 --warm-up 8192 --batch-size 256
```
Outputs will appear under `demo/outputs/run_ep100_step2048_seed521/`.

## Reproduction entry points (see `reproduce/README.md`)
- Train DRL (SB3): `python reproduce/scripts/train_drl.py --algo ddpg --total_steps 10000 --seeds 521`
- Eval + performance bound: `python reproduce/scripts/eval_drl.py --algo ddpg --seed 521 --split test --episodes 1`
- Plots (training/timeseries/heatmap): `python reproduce/scripts/plot_results.py` and related plotting scripts.
- NLP baseline: `python reproduce/scripts/run_nlp_baseline.py --mode simplified --day_idx 0`
- Data augmentation (GMM+Copula): `python reproduce/scripts/augment_data.py --multiplier 5`
- Power flow compare (PandaPower vs Tensor PF): `python reproduce/scripts/compare_powerflow.py`
