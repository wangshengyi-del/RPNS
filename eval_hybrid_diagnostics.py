"""在同一 checkpoint 上隔离任务排序与处理器选择的贡献。

四种组合：
  PPO task + PPO proc
  EDF task + PPO proc
  PPO task + EFT proc
  EDF task + EFT proc
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from types import SimpleNamespace
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch

from baseline_heft import HEFTScheduler, run_baseline
from env.multi_dag_environment import MultiDAGSchedulerEnv, WAIT_TIME_MEAN, WAIT_TIME_STD
from model.ablation_scheduler import create_mixed_features_scheduler, create_no_inter_scheduler
from model.constants import (
    DAG_META_DIM,
    DYN_DIM,
    NUM_PROCS,
    STATIC_DIM,
)
from model.constants import ORIGINAL_TASK_DIM, PROC_DIM
from train_hierarchical_gat import (
    RULE_TASK_POLICIES,
    _uses_mixed_features,
    configure_cpu_threads,
    extract_env_data,
    extract_wait_time_from_ckpt,
    resolve_device,
    rule_ready_task_index,
    set_seed,
)
from model.mlp_scheduler import create_mlp_scheduler
from model.hierarchical_scheduler import HierarchicalGATScheduler

CHECKPOINT_ALIASES = {
    'dag': 'best_dag_model.pt',
    'reward': 'best_reward_model.pt',
    'final': 'final_model.pt',
}



def build_model(args, device: str) -> torch.nn.Module:
    common = dict(
        static_dim=STATIC_DIM,
        dyn_dim=DYN_DIM,
        dag_meta_dim=DAG_META_DIM,
        proc_dim=PROC_DIM,
        hidden_dim=args.hidden_dim,
        num_procs=NUM_PROCS,
        enable_hold=(args.enable_hold == 1),
        max_attn_dags=args.max_attn_dags,
        joint_matching=(args.joint_matching == 1),
        fusion_mode=args.fusion_mode,
        use_pair_features=(args.use_pair_features == 1),
    )
    if args.model_type == 'mlp':
        model = create_mlp_scheduler(**common)
    elif args.ablation == 'no_inter':
        model = create_no_inter_scheduler(**common, adj_direction=args.adj_direction)
    elif args.ablation == 'mixed_features':
        model = create_mixed_features_scheduler(**common, adj_direction=args.adj_direction)
    else:
        model = HierarchicalGATScheduler(**common, adj_direction=args.adj_direction)
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        meta = {k: v for k, v in ckpt.items() if k != 'model_state_dict' and k != 'optimizer_state_dict'}
    else:
        state = ckpt
        meta = {}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f'load_state_dict 异常: missing={missing}, unexpected={unexpected}')
    return meta


def apply_config_from_ckpt(args, meta: dict) -> None:
    """若 checkpoint 含 config，覆盖架构相关参数（避免手填不一致）。"""
    cfg = meta.get('config')
    if not isinstance(cfg, dict):
        return
    # 旧 checkpoint 没有该字段；多 checkpoint 连续评估时必须显式复位，
    # 避免前一个新 checkpoint 的架构设置泄漏到后一个旧 checkpoint。
    if 'use_pair_features' not in cfg:
        args.use_pair_features = 0
    if 'dag_early_miss_penalty' not in cfg:
        args.dag_early_miss_penalty = 0
    mapping = {
        'hidden_dim': 'hidden_dim',
        'enable_hold': 'enable_hold',
        'max_attn_dags': 'max_attn_dags',
        'joint_matching': 'joint_matching',
        'use_pair_features': 'use_pair_features',
        'fusion_mode': 'fusion_mode',
        'adj_direction': 'adj_direction',
        'model_type': 'model_type',
        'ablation': 'ablation',
        'dag_reward_timing': 'dag_reward_timing',
        'dag_early_miss_penalty': 'dag_early_miss_penalty',
        'task_reward_mode': 'task_reward_mode',
    }
    for ck, ak in mapping.items():
        if ck in cfg:
            setattr(args, ak, cfg[ck])
    print(f'  已从 checkpoint.config 覆盖架构参数: '
          f'type={args.model_type} ablation={args.ablation} '
          f'fusion={args.fusion_mode} joint={args.joint_matching} adj={args.adj_direction}')
    print(f'  processor pair features: {bool(args.use_pair_features)}')


def resolve_checkpoints(args) -> List[Tuple[str, str]]:
    """返回 [(显示名, 路径), ...]"""
    items: List[Tuple[str, str]] = []

    if args.checkpoint:
        for p in args.checkpoint:
            if not os.path.isfile(p):
                raise FileNotFoundError(f'checkpoint 不存在: {p}')
            items.append((os.path.basename(p), p))

    if args.run_dir:
        if not os.path.isdir(args.run_dir):
            raise FileNotFoundError(f'run_dir 不存在: {args.run_dir}')
        which = [w.strip().lower() for w in args.which.split(',') if w.strip()]
        for key in which:
            if key not in CHECKPOINT_ALIASES:
                raise ValueError(f'未知 --which={key}，可选: {list(CHECKPOINT_ALIASES)}')
            fname = CHECKPOINT_ALIASES[key]
            path = os.path.join(args.run_dir, fname)
            if not os.path.isfile(path):
                print(f'[跳过] 未找到 {path}')
                continue
            run_name = os.path.basename(os.path.normpath(args.run_dir))
            items.append((f'{run_name}/{key}', path))

    if not items:
        raise ValueError('请指定 --checkpoint 和/或 --run_dir（且至少有一个可加载的 .pt）')
    return items

class HybridScheduler:
    """按配置组合 PPO/EDF 任务选择和 PPO/EFT 处理器选择。"""

    def __init__(
        self,
        model,
        device: str,
        task_policy: str,
        proc_policy: str,
        device_env,
    ):
        if task_policy not in ("ppo",) + RULE_TASK_POLICIES:
            raise ValueError(f"未知 task_policy={task_policy}")
        if proc_policy not in ("ppo", "eft"):
            raise ValueError(f"未知 proc_policy={proc_policy}")
        self.model = model
        self.device = device
        self.task_policy = task_policy
        self.proc_policy = proc_policy
        self.eft = HEFTScheduler(device_env)

    def reset(self):
        self.eft.reset()
        self.model.eval()

    def _forward(self, env, env_data):
        with torch.no_grad():
            static_f = [
                f.to(self.device)
                for f in env_data["dag_task_static_features"]
            ]
            adjs = [a.to(self.device) for a in env_data["dag_adjs"]]
            task_logits, proc_logits, _ = self.model(
                dag_task_static_features=static_f,
                dag_adjs=adjs,
                dag_metadata=env_data["dag_metadata"].to(self.device),
                ready_task_indices=env_data["ready_task_indices"].to(self.device),
                ready_task_dyn_features=env_data[
                    "ready_task_dyn_features"
                ].to(self.device),
                proc_features=env_data["proc_features"].to(self.device),
                proc_states=env_data["proc_states"].to(self.device),
                task_dag_ids=env_data["task_dag_ids"].to(self.device),
                task_proc_pair_features=env_data[
                    "task_proc_pair_features"
                ].to(self.device),
            )
        return task_logits, proc_logits

    def select_action(self, env: MultiDAGSchedulerEnv) -> Optional[int]:
        env_data = extract_env_data(
            env,
            max_attn_dags=self.model.max_attn_dags,
            mixed_features=_uses_mixed_features(self.model),
        )
        if env_data is None:
            return None

        ready_candidate_indices = env_data["ready_candidate_indices"]
        candidate_types = env_data.get("candidate_types")

        # EDF+EFT 完全不依赖模型前向；其余组合需要相应 head。
        task_logits = proc_logits = None
        if self.task_policy == "ppo" or self.proc_policy == "ppo":
            task_logits, proc_logits = self._forward(env, env_data)

        if self.task_policy in RULE_TASK_POLICIES:
            task_id = rule_ready_task_index(
                env, ready_candidate_indices, self.task_policy
            )
        else:
            task_id = int(torch.argmax(task_logits).item())

        candidate_idx = ready_candidate_indices[task_id]
        task = env.candidate_tasks[candidate_idx]

        if self.proc_policy == "eft":
            proc_id, _ = self.eft.select_processor_eft(
                task,
                env.current_time,
                processors=env.processors,
            )
        else:
            selected = proc_logits[task_id].clone()
            if candidate_types is not None:
                selected = (
                    self.model.two_stage_scheduler.apply_candidate_type_mask(
                        selected,
                        int(candidate_types[task_id].item()),
                    )
                )
            if self.model.enable_hold:
                selected[-1] = -1e9
            proc_id = int(torch.argmax(selected).item())

        return candidate_idx * env.num_processors + proc_id


def _mean_std(results: Dict, key: str):
    values = [float(ep[key]) for ep in results["per_episode"]]
    return float(np.mean(values)), float(np.std(values))


def _summary(name: str, result: Dict) -> Dict:
    task_mean, task_std = _mean_std(result, "completion_rate")
    dag_mean, dag_std = _mean_std(result, "dag_completion_rate")
    delay_mean, delay_std = _mean_std(result, "avg_task_delay")
    energy_mean, energy_std = _mean_std(result, "avg_energy")
    row = {
        "name": name,
        "task_mean": task_mean,
        "task_std": task_std,
        "dag_mean": dag_mean,
        "dag_std": dag_std,
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }
    for priority, stats in sorted(result["priority_stats"].items()):
        prefix = f"p{priority:g}"
        total_dags = stats.get("total_dags", 0)
        total_tasks = stats.get("total_tasks", 0)
        row[f"{prefix}_dag_rate"] = (
            stats.get("on_time_dags", 0) / total_dags if total_dags else 0.0
        )
        row[f"{prefix}_task_rate"] = (
            stats.get("on_time_tasks", 0) / total_tasks if total_tasks else 0.0
        )
        row[f"{prefix}_avg_delay"] = (
            stats.get("total_delay", 0.0) / total_tasks
            if total_tasks
            else 0.0
        )
    return row


def _print_summary(row: Dict):
    print(f"\n{'=' * 68}")
    print(row["name"])
    print(f"{'=' * 68}")
    print(
        f"  TASK:   {row['task_mean']:.2%} ± {row['task_std']:.2%}\n"
        f"  DAG:    {row['dag_mean']:.2%} ± {row['dag_std']:.2%}\n"
        f"  DELAY:  {row['delay_mean']:.3f} ± {row['delay_std']:.3f}s\n"
        f"  ENERGY: {row['energy_mean']:.3f} ± {row['energy_std']:.3f}J"
    )
    for key in sorted(k for k in row if k.endswith("_dag_rate")):
        prefix = key[: -len("_dag_rate")]
        print(
            f"  {prefix}: DAG={row[key]:.2%}, "
            f"TASK={row[f'{prefix}_task_rate']:.2%}, "
            f"DELAY={row[f'{prefix}_avg_delay']:.3f}s"
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="PPO/EDF × PPO/EFT 四组合诊断")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max_dags",
        type=int,
        default=1000,
        help="每个 episode 的 DAG 数；默认与正式 Test 一致",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "mps", "cuda", "gpu"],
    )
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--arrival_rate", type=float, default=0.55)
    parser.add_argument(
        "--device_config",
        default="dataset/device/device_heterogeneous.json",
    )
    parser.add_argument("--out_dir", default="hybrid_diagnostics")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    device = resolve_device(args.device)
    configure_cpu_threads(device, args.num_threads)
    set_seed(args.seed, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # 提供旧 checkpoint 的架构回退值，随后优先读取 checkpoint.config。
    model_args = SimpleNamespace(
        hidden_dim=256,
        enable_hold=0,
        max_attn_dags=64,
        joint_matching=0,
        use_pair_features=0,
        fusion_mode="gate",
        adj_direction="down",
        model_type="gat",
        ablation="none",
        dag_reward_timing="complete",
        task_reward_mode="g2",
    )
    if isinstance(ckpt, dict):
        apply_config_from_ckpt(model_args, ckpt)

    model = build_model(model_args, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    wt_mean, wt_std = extract_wait_time_from_ckpt(
        ckpt if isinstance(ckpt, dict) else {},
        WAIT_TIME_MEAN,
        WAIT_TIME_STD,
    )
    env = MultiDAGSchedulerEnv(
        task_dim=ORIGINAL_TASK_DIM,
        proc_dim=PROC_DIM,
        dag_dataset_dir="dataset/daggen_factory/dags",
        max_candidates=15,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        dag_generator="notOur",
        device_config=args.device_config,
        dag_reward_timing=model_args.dag_reward_timing,
        task_reward_mode=model_args.task_reward_mode,
    )
    env.load_train_wait_time_stats(wt_mean, wt_std)
    env.set_eval_wait_time_stats(wt_mean, wt_std)
    env.max_dags[args.split] = args.max_dags

    print(
        f"checkpoint={args.checkpoint}\n"
        f"split={args.split}, episodes={args.episodes}, "
        f"max_dags={args.max_dags}, "
        f"wait_time=({wt_mean:.4f}, {wt_std:.4f})"
    )
    if model_args.joint_matching == 1:
        print(
            "提示：四组合中的 PPO task/PPO proc 使用两个 head 的两阶段贪心，"
            "不使用 joint argmax，以便独立诊断两个 head。"
        )

    combinations = [
        ("PPO-task + PPO-proc", "ppo", "ppo"),
        ("EDF-task + PPO-proc", "edf", "ppo"),
        ("PPO-task + EFT-proc", "ppo", "eft"),
        ("EDF-task + EFT-proc", "edf", "eft"),
    ]
    summaries = []
    details = {}

    for name, task_policy, proc_policy in combinations:
        scheduler = HybridScheduler(
            model=model,
            device=device,
            task_policy=task_policy,
            proc_policy=proc_policy,
            device_env=env.device_env,
        )
        result = run_baseline(
            env,
            scheduler,
            num_episodes=args.episodes,
            split=args.split,
            verbose=False,
            name=name,
        )
        row = _summary(name, result)
        summaries.append(row)
        details[name] = result
        _print_summary(row)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "hybrid_summary.csv")
    json_path = os.path.join(args.out_dir, "hybrid_detail.json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with open(json_path, "w") as f:
        json.dump(
            _json_safe({
                "checkpoint": args.checkpoint,
                "split": args.split,
                "episodes": args.episodes,
                "max_dags": args.max_dags,
                "wait_time_mean": wt_mean,
                "wait_time_std": wt_std,
                "summary": summaries,
                "details": details,
            }),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n汇总: {csv_path}")
    print(f"明细: {json_path}")


if __name__ == "__main__":
    main()
