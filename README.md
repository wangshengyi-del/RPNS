# RPNS

Code and simulation data for **RPNS** (Rule-Prior Neural Scheduling): online multi-workflow scheduling on a shared cloud–edge pool. At each step, **EDF on LFT** picks the ready task; a hierarchical **GAT + PPO** policy places it on a processor.

Paper: *A Rule-Prior Neural Scheduling Agent for Deadline-Constrained Multi-Workflow Allocation in Cloud–Edge Systems* (Applied Soft Computing).

## Setup

Python 3.10+ with PyTorch. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy gym
```

Use `--device auto` to pick CUDA / Apple MPS / CPU. Environment simulation always runs on CPU; only the policy network uses the selected device.

Main layout:

| Path | Role |
|------|------|
| `env/` | Online simulator (Poisson arrivals, LFT, energy) |
| `model/` | Hierarchical GAT scheduler |
| `train_hierarchical_gat.py` | Training and in-loop val/test |
| `baseline_heft.py` | Random / HEFT / EDF / PCP-style |
| `scripts/revision_eval.py` | Unified 20-episode test collector |
| `eval_hybrid_diagnostics.py` | EDF/PPO × PPO/EFT 2×2 on one checkpoint |
| `dataset/daggen_factory/dags/` | Main val/test DAG pool (1000 + 1000) |
| `dataset/pegasus_wfcommons/` | Pegasus/WfCommons OOD graphs |
| `dataset/device/device_heterogeneous.json` | 3 cloud + 8 edge processors |
| `runs/hierarchical_ppo/` | Training logs and checkpoints |

## Simulation (main experiment)

- **Pool:** 3 cloud + 8 edge processors (`device_heterogeneous.json`).
- **Load:** Poisson job arrivals, \(\lambda = 0.55\).
- **SLA:** \(\rho \in \{1.5, 2.0, 3.0\}\) drawn per job; deadline \(= \) arrival \(+ \rho \times \widehat{M}\).
- **Train:** 201 episodes; online DAG window (not the frozen val/test files).
- **Val / test files:** `dataset/daggen_factory/dags` (`dag_dataset.json`). Each episode replays the same 1000 graphs in file order.
- **Seeds:** `--seed` sets PyTorch **and** the **training** arrival/DAG stream. Validation arrivals are pinned to `42 + episode_idx`. Test arrivals are pinned to `2024 + episode_idx`, so the hold-out process is independent of both training (including `--seed 42`) and checkpoint selection. All methods share that same test stream. Formal reporting uses `episode_idx = 0..19`.

`--seed` on `scripts/revision_eval.py` does not change test arrivals (leave it at 42).

## Train RPNS from scratch

CLI defaults are **not** RPNS (`joint_matching` defaults to 1, `task_policy` to `ppo`). Use this command (matches the seed-42 paper run):

```bash
nohup python3  -u train_hierarchical_gat.py \
  --seed 100 \
  --episodes 201 \
  --use_load_features 0 \
  --eft_aux_coef 0.0 \
  --task_aux_coef 0.0 \
  --task_reward_mode g2 \
  --dag_reward_timing complete \
  --model_type gat \
  --ablation none \
  --fusion_mode gate \
  --adj_direction down \
  --hidden_dim 256 \
  --lr 0.0004 \
  --lr_decay 1.0 \
  --entropy_coef_start 0.01 \
  --entropy_coef_end 0.0001 \
  --entropy_decay_episodes 100 \
  --ppo_epochs 1 \
  --eval_episodes 5 \
  --test_episodes 15 \
  --peak_guard 0 \
  --train_proc_only 0 \
  --device_config dataset/device/device_heterogeneous.json \
  --use_pair_features 1 \
  --task_policy edf \
  --joint_matching 0 \
  --run_tag edf_ppo > run1.log 2>&1 &
  
  
nohup python3 -u train_hierarchical_gat.py \
  --seed 100 \
  --episodes 201 \
  --use_load_features 0 \
  --eft_aux_coef 0.0 \
  --task_aux_coef 0.0 \
  --task_reward_mode g2 \
  --dag_reward_timing complete \
  --model_type gat \
  --ablation none \
  --fusion_mode gate \
  --adj_direction down \
  --hidden_dim 256 \
  --lr 0.0004 \
  --lr_decay 1.0 \
  --entropy_coef_start 0.01 \
  --entropy_coef_end 0.0001 \
  --entropy_decay_episodes 100 \
  --ppo_epochs 1 \
  --eval_episodes 5 \
  --test_episodes 15 \
  --peak_guard 0 \
  --train_proc_only 0 \
  --device_config dataset/device/device_heterogeneous.json \
  --use_pair_features 1 \
  --task_policy ppo \
  --joint_matching 1 \
  --run_tag ppo_joint > run2.log 2>&1 &
  
  

nohup python3 -u train_hierarchical_gat.py \
  --seed 100 \
  --episodes 201 \
  --use_load_features 0 \
  --eft_aux_coef 0.0 \
  --task_aux_coef 0.0 \
  --task_reward_mode g2 \
  --dag_reward_timing complete \
  --model_type gat \
  --ablation none \
  --fusion_mode gate \
  --adj_direction down \
  --hidden_dim 256 \
  --lr 0.0004 \
  --lr_decay 1.0 \
  --entropy_coef_start 0.01 \
  --entropy_coef_end 0.0001 \
  --entropy_decay_episodes 100 \
  --ppo_epochs 1 \
  --eval_episodes 5 \
  --test_episodes 15 \
  --peak_guard 0 \
  --train_proc_only 0 \
  --device_config dataset/device/device_heterogeneous.json \
  --use_pair_features 1 \
  --task_policy ppo \
  --joint_matching 0 \
  --run_tag ppo_twostage > run3.log 2>&1 &
    
    
nohup python3 -u train_hierarchical_gat.py \
  --seed 100 \
  --episodes 201 \
  --use_load_features 0 \
  --eft_aux_coef 0.0 \
  --task_aux_coef 0.0 \
  --task_reward_mode g2 \
  --dag_reward_timing complete \
  --model_type gat \
  --ablation none \
  --fusion_mode gate \
  --adj_direction down \
  --hidden_dim 256 \
  --lr 0.0004 \
  --lr_decay 1.0 \
  --entropy_coef_start 0.01 \
  --entropy_coef_end 0.0001 \
  --entropy_decay_episodes 100 \
  --ppo_epochs 1 \
  --eval_episodes 5 \
  --test_episodes 15 \
  --peak_guard 0 \
  --train_proc_only 0 \
  --device_config dataset/device/device_heterogeneous.json \
  --use_pair_features 1 \
  --task_policy est \
  --joint_matching 0 \
  --run_tag ppo_est > run4.log 2>&1 &
  
  
  nohup python3 -u scripts/revision_eval.py \
  --methods random,heft,edf,pcp \
  --episodes 15 --split test --seed 100 > run5.log 2>&1 &
```

Checkpoints land under `runs/hierarchical_ppo/<timestamp>_seed42_.../`:

- `best_dag_model.pt` — used for the main table (highest val job on-time rate after warmup)
- `peak_model.pt` / `final_model.pt` — extra snapshots

Validation still runs every 5 training episodes × 5 val episodes. `--test_episodes 20` only runs at the end (about two hours on MPS for RPNS). For a shorter train job, set `--test_episodes 1` and evaluate with `revision_eval.py`.

End-to-end PPO (joint) is the same command with `--task_policy ppo --joint_matching 1 --run_tag ppo_joint`. 

Two-stage PPO: `--task_policy ppo --joint_matching 0  --run_tag ppo_twostage`.

EST + PPO (earliest-ready task rule): `--task_policy est --joint_matching 0 --run_tag ppo_est`.

## Evaluate

Unified test (20 arrival seeds `2024 + episode_idx`, paired vs RPNS when both are run):

```bash
# Heuristics
python3 scripts/revision_eval.py \
  --methods random,heft,edf,pcp \
  --episodes 20 --split test --seed 42

# Learned methods (override paths after you train)
python3 scripts/revision_eval.py \
  --methods rpns,ppo_joint,ppo_twostage \
  --episodes 20 --split test --seed 42  \
  --ckpt_rpns runs/hierarchical_ppo/<rpns_run>/best_dag_model.pt \
  --ckpt_ppo_joint runs/hierarchical_ppo/<joint_run>/best_dag_model.pt \
  --ckpt_ppo_twostage runs/hierarchical_ppo/<twostage_run>/best_dag_model.pt
```

Pegasus OOD (use \(\lambda = 0.12\)):

```bash
python3 scripts/revision_eval.py \
  --methods heft,edf,rpns,ppo_joint \
  --dag_dataset_dir dataset/pegasus_wfcommons/dags \
  --arrival_rate 0.12 \
  --episodes 20 --split test --seed 42 --device auto \
  --ckpt_rpns runs/hierarchical_ppo/<rpns_run>/best_dag_model.pt \
  --ckpt_ppo_joint runs/hierarchical_ppo/<joint_run>/best_dag_model.pt
```

2×2 task/processor ablation on one checkpoint:

```bash
python3 eval_hybrid_diagnostics.py \
  --checkpoint runs/hierarchical_ppo/<run>/best_dag_model.pt \
  --split test --episodes 5 --device auto
```

Outputs: `revision/eval_runs/<tag>/revision_eval.json`, `hybrid_diagnostics/`.

## Regenerating Pegasus graphs

See `dataset/pegasus_wfcommons/README.md`. Default `--attribute_mode synthetic` keeps computation/data from this repo and only changes topology.

## License

Research code accompanying the paper. If you use it, please cite the RPNS article.
