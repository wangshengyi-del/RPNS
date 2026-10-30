#!/usr/bin/env python3
"""修订统一评测：同一测试池、episode_idx=0..N-1 到达种子。

阶段 A：启发式 + 已有 checkpoint。E7 新 run 也用同一入口。

示例:
  python3 scripts/revision_eval.py --methods random,heft,edf,pcp --episodes 20
  python3 scripts/revision_eval.py --methods rpns,ppo_joint --episodes 20 --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_heft import (  # noqa: E402
    EDFScheduler,
    HEFTScheduler,
    PCPStyleScheduler,
    RandomScheduler,
    run_baseline,
)
from env.multi_dag_environment import (  # noqa: E402
    EVAL_ARRIVAL_SEED,
    TEST_ARRIVAL_SEED,
    WAIT_TIME_MEAN,
    WAIT_TIME_STD,
    MultiDAGSchedulerEnv,
)
from eval_hybrid_diagnostics import (  # noqa: E402
    apply_config_from_ckpt,
    build_model,
    load_checkpoint,
)
from model.constants import ORIGINAL_TASK_DIM, PROC_DIM  # noqa: E402
from train_hierarchical_gat import (  # noqa: E402
    configure_cpu_threads,
    evaluate_policy,
    extract_wait_time_from_ckpt,
    resolve_device,
    set_seed,
)

DEFAULT_CKPTS = {
    "rpns": ROOT
    / "runs/hierarchical_ppo/20260729_140051_seed42_ppo(rpns)/best_dag_model.pt",
    "ppo_joint": ROOT
    / "runs/hierarchical_ppo/20260728_092351_seed42_ppo_joint/best_dag_model.pt",
    "ppo_twostage": ROOT
    / "runs/hierarchical_ppo/20260727_164430_seed42_cpu_dagcomplete_g2_peft0_peakguard_ppo/best_dag_model.pt",
    "rpns_final": ROOT
    / "runs/hierarchical_ppo/20260729_140051_seed42_ppo(rpns)/final_model.pt",
    "rpns_peak": ROOT
    / "runs/hierarchical_ppo/20260729_140051_seed42_ppo(rpns)/peak_model.pt",
}

LEARNED_TASK_POLICY = {
    "rpns": "edf",
    "rpns_final": "edf",
    "rpns_peak": "edf",
    "ppo_joint": "ppo",
    "ppo_twostage": "ppo",
    "est_ppo": "est",
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _priority_job_rate(priority_stats: dict, p: float) -> Optional[float]:
    bucket = None
    for key, val in (priority_stats or {}).items():
        try:
            if abs(float(key) - p) < 1e-6:
                bucket = val
                break
        except (TypeError, ValueError):
            continue
    if not bucket:
        return None
    total = bucket.get("total_dags") or 0
    if total <= 0:
        return None
    return float(bucket["on_time_dags"]) / float(total)


def _summarize(name: str, result: dict) -> dict:
    eps = result.get("per_episode") or []
    jobs = [float(r["dag_completion_rate"]) for r in eps]
    tasks = [float(r["completion_rate"]) for r in eps]
    energy = [float(r.get("avg_energy") or 0.0) for r in eps]
    tx = [float(r.get("tx_comm_time_per_task") or 0.0) for r in eps]
    tx_cd = [float(r.get("tx_comm_time_cross_domain") or 0.0) for r in eps]
    n_task = [max(int(r.get("total_tasks") or 0), 1) for r in eps]
    dec = [float(r.get("mean_decision_s") or 0.0) for r in eps]
    n_dags = sum(int(r.get("total_dags") or 0) for r in eps)
    n_late = sum(int(r.get("n_late_jobs") or 0) for r in eps)
    n_tol = sum(int(r.get("task_ontime_job_late") or 0) for r in eps)
    late_t = [
        r.get("late_job_mean_task_ontime")
        for r in eps
        if r.get("late_job_mean_task_ontime") is not None
        and int(r.get("n_late_jobs") or 0) > 0
    ]
    last_arr = [
        float(r["last_dag_arrival_time"])
        for r in eps
        if r.get("last_dag_arrival_time") is not None
    ]
    ps = result.get("priority_stats") or {}
    out = {
        "method": name,
        "n_episodes": len(eps),
        "job_mean": float(np.mean(jobs)) if jobs else None,
        "job_std": float(np.std(jobs)) if jobs else None,
        "task_mean": float(np.mean(tasks)) if tasks else None,
        "task_std": float(np.std(tasks)) if tasks else None,
        "energy_mean": float(np.mean(energy)) if energy else None,
        "tx_comm_per_task_mean": float(np.mean(tx)) if tx else None,
        "tx_comm_cross_domain_per_task_mean": (
            float(np.mean([c / n for c, n in zip(tx_cd, n_task)])) if tx_cd else None
        ),
        "cross_domain_task_ratio_mean": float(np.mean([
            r.get("cross_domain_task_ratio") or 0.0 for r in eps
        ])) if eps else None,
        "mean_decision_s": float(np.mean(dec)) if dec else None,
        "job_p1.5": _priority_job_rate(ps, 1.5),
        "job_p2.0": _priority_job_rate(ps, 2.0),
        "job_p3.0": _priority_job_rate(ps, 3.0),
        "task_ontime_job_late_rate": (n_tol / n_dags) if n_dags else None,
        "late_job_mean_task_ontime": float(np.mean(late_t)) if late_t else None,
        "n_late_jobs_total": n_late,
        "last_dag_arrival_mean": float(np.mean(last_arr)) if last_arr else None,
        "last_dag_arrival_std": float(np.std(last_arr)) if last_arr else None,
        "last_dag_arrival_min": float(np.min(last_arr)) if last_arr else None,
        "last_dag_arrival_max": float(np.max(last_arr)) if last_arr else None,
    }
    return out


def _paired(ref_eps: List[dict], other_eps: List[dict]) -> dict:
    n = min(len(ref_eps), len(other_eps))
    diffs = [
        float(ref_eps[i]["dag_completion_rate"])
        - float(other_eps[i]["dag_completion_rate"])
        for i in range(n)
    ]
    if not diffs:
        return {}
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
    return {
        "n": n,
        "diff_mean": float(np.mean(diffs)),
        "diff_std": float(np.std(diffs)),
        "n_positive": int(sum(s > 0 for s in signs)),
        "n_negative": int(sum(s < 0 for s in signs)),
        "n_zero": int(sum(s == 0 for s in signs)),
        "diffs": diffs,
    }


def make_env(args, seed: int) -> MultiDAGSchedulerEnv:
    # val 钉 EVAL_ARRIVAL_SEED；test 钉 TEST_ARRIVAL_SEED；此处 --seed 不影响配对到达
    env = MultiDAGSchedulerEnv(
        task_dim=ORIGINAL_TASK_DIM,
        proc_dim=PROC_DIM,
        dag_dataset_dir=args.dag_dataset_dir,
        max_candidates=15,
        arrival_rate=args.arrival_rate,
        seed=seed,
        dag_generator="notOur",
        device_config=args.device_config,
    )
    env.set_eval_wait_time_stats(WAIT_TIME_MEAN, WAIT_TIME_STD)
    if args.rho_mode == "all_1.5":
        env.priority_ratios = [1.5]
    elif args.rho_mode == "all_3.0":
        env.priority_ratios = [3.0]
    return env


def run_heuristic(env, method: str, episodes: int, split: str) -> dict:
    schedulers = {
        "random": RandomScheduler,
        "heft": HEFTScheduler,
        "edf": EDFScheduler,
        "pcp": PCPStyleScheduler,
    }
    scheduler = schedulers[method](env.device_env)
    return run_baseline(
        env,
        scheduler,
        num_episodes=episodes,
        split=split,
        verbose=True,
        name=method,
    )


def run_learned(
    env,
    method: str,
    ckpt_path: Path,
    episodes: int,
    split: str,
    device: str,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = SimpleNamespace(
        hidden_dim=256,
        enable_hold=0,
        max_attn_dags=64,
        joint_matching=0,
        use_pair_features=1,
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
    load_checkpoint(model, str(ckpt_path), device)
    model.eval()
    wt_mean, wt_std = extract_wait_time_from_ckpt(
        ckpt if isinstance(ckpt, dict) else {},
        WAIT_TIME_MEAN,
        WAIT_TIME_STD,
    )
    env.load_train_wait_time_stats(wt_mean, wt_std)
    env.set_eval_wait_time_stats(wt_mean, wt_std)
    task_policy = LEARNED_TASK_POLICY[method]
    print(
        f"  checkpoint={ckpt_path}\n"
        f"  task_policy={task_policy} joint={model_args.joint_matching} "
        f"wait=({wt_mean:.4f},{wt_std:.4f})"
    )
    return evaluate_policy(
        env,
        model,
        num_episodes=episodes,
        device=device,
        verbose=True,
        split=split,
        task_policy=task_policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="修订统一评测（20 到达种子）")
    parser.add_argument(
        "--methods",
        default="random,heft,edf,pcp",
        help="逗号分隔: random,heft,edf,pcp,rpns,ppo_joint,ppo_twostage,rpns_final,rpns_peak,est_ppo",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival_rate", type=float, default=0.55)
    parser.add_argument("--dag_dataset_dir", default="dataset/daggen_factory/dags")
    parser.add_argument(
        "--device_config",
        default="dataset/device/device_heterogeneous.json",
    )
    parser.add_argument(
        "--rho_mode",
        default="mixed",
        choices=["mixed", "all_1.5", "all_3.0"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--out_dir", default="revision/eval_runs")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--ckpt_rpns",
        default="",
        help="覆盖 rpns 的 best_dag_model.pt",
    )
    parser.add_argument("--ckpt_ppo_joint", default="")
    parser.add_argument("--ckpt_ppo_twostage", default="")
    parser.add_argument(
        "--ckpt_est_ppo",
        default="",
        help="EST+PPO 的 best_dag_model.pt（E6，须从头训练后指定）",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    configure_cpu_threads(device, args.num_threads)
    set_seed(args.seed, device)

    ckpt_map = dict(DEFAULT_CKPTS)
    if args.ckpt_rpns:
        ckpt_map["rpns"] = Path(args.ckpt_rpns)
    if args.ckpt_ppo_joint:
        ckpt_map["ppo_joint"] = Path(args.ckpt_ppo_joint)
    if args.ckpt_ppo_twostage:
        ckpt_map["ppo_twostage"] = Path(args.ckpt_ppo_twostage)
    if args.ckpt_est_ppo:
        ckpt_map["est_ppo"] = Path(args.ckpt_est_ppo)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    env = make_env(args, args.seed)
    results: Dict[str, dict] = {}
    summaries = []
    tag = args.tag or f"{args.split}_ep{args.episodes}_{args.rho_mode}"
    out_dir = ROOT / args.out_dir / tag
    os.makedirs(out_dir, exist_ok=True)
    out_json = out_dir / "revision_eval.json"
    print(
        f"split={args.split}  val_arrival_seed={EVAL_ARRIVAL_SEED}  "
        f"test_arrival_seed={TEST_ARRIVAL_SEED}  episodes={args.episodes}  "
        f"device={device}"
    )

    def _flush() -> None:
        paired = {}
        if "rpns" in results:
            ref = results["rpns"].get("per_episode") or []
            for other in ("ppo_joint", "edf", "heft", "pcp", "ppo_twostage", "est_ppo"):
                if other in results:
                    paired[f"rpns_minus_{other}"] = _paired(
                        ref, results[other].get("per_episode") or []
                    )
        payload = {
            "episodes": args.episodes,
            "split": args.split,
            "seed": args.seed,
            "val_arrival_seed": EVAL_ARRIVAL_SEED,
            "test_arrival_seed": TEST_ARRIVAL_SEED,
            "arrival_rate": args.arrival_rate,
            "dag_dataset_dir": args.dag_dataset_dir,
            "rho_mode": args.rho_mode,
            "methods": methods,
            "summary": summaries,
            "paired": paired,
            "details": results,
        }
        with open(out_json, "w") as f:
            json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)

    for method in methods:
        print("\n" + "=" * 70)
        print(f"method={method} episodes={args.episodes} rho={args.rho_mode}")
        print("=" * 70)
        if method in ("random", "heft", "edf", "pcp"):
            result = run_heuristic(env, method, args.episodes, args.split)
        elif method in LEARNED_TASK_POLICY:
            ckpt_path = ckpt_map.get(method)
            if ckpt_path is None or not Path(ckpt_path).is_file():
                print(f"跳过 {method}：找不到 checkpoint {ckpt_path}")
                continue
            result = run_learned(
                env, method, ckpt_path, args.episodes, args.split, device
            )
        else:
            raise ValueError(f"未知 method={method}")
        results[method] = result
        summaries.append(_summarize(method, result))
        row = summaries[-1]
        print(
            f"  Job={row['job_mean']:.2%} ± {row['job_std']:.2%}  "
            f"Task={row['task_mean']:.2%}  "
            f"E={row['energy_mean']:.2f}J  "
            f"tx_comm/task={row['tx_comm_per_task_mean']:.4f}s"
        )
        _flush()
        print(f"  已写入 {out_json}")

    print(f"\n写入 {out_json}")
    with open(out_json) as f:
        paired = json.load(f).get("paired") or {}
    if paired:
        print("配对差（Job 按时率）:")
        for k, v in paired.items():
            print(
                f"  {k}: mean={v['diff_mean']:+.4f}  "
                f"+{v['n_positive']}/-{v['n_negative']}/0={v['n_zero']}"
            )


if __name__ == "__main__":
    main()
