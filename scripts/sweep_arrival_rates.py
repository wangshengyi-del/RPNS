"""
Heuristic arrival-rate sweep on the paper test pool.

Purpose: show that λ=0.55 is a contested-but-discriminable operating point
jointly set by DAG scale (M̂) and SLA factor ρ — not a magic constant.
Does not retrain RPNS/PPO.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baseline_heft import EDFScheduler, HEFTScheduler
from env.multi_dag_environment import MultiDAGSchedulerEnv
from measure_load_pressure import OccupancyTracker, _summarize
from model.constants import ORIGINAL_TASK_DIM, PROC_DIM

LAMBDAS = (0.20, 0.35, 0.45, 0.55, 0.70)
NUM_EPISODES = 5
SPLIT = "test"
E_MHAT = 6.401262450011956
OFFERED_AT_055 = 5908.879690999999
TOTAL_MIPS = 10250.0
OUT_PATH = os.path.join(ROOT, "paper_results", "arrival_rate_sweep.json")


def _rho_rates(sojourns: List[Dict]) -> Dict[str, Dict[str, float]]:
    buckets = defaultdict(lambda: {"n": 0, "on": 0})
    for row in sojourns:
        key = f"{row['rho']:.1f}"
        buckets[key]["n"] += 1
        buckets[key]["on"] += int(row["on_time"])
    out = {}
    for key, val in sorted(buckets.items()):
        out[key] = {
            "n": val["n"],
            "job_on_time": (val["on"] / val["n"]) if val["n"] else float("nan"),
        }
    return out


def run_replay(env: MultiDAGSchedulerEnv, scheduler, num_episodes: int, tag: str) -> Dict:
    sojourns: List[Dict] = []
    step_ready: List[int] = []
    step_active: List[int] = []
    ep_rows: List[Dict] = []
    orig_agg = env._aggregate_dag_stats

    def hooked_agg(dag):
        sojourns.append(
            {
                "makespan_est": float(dag.makespan_est),
                "sojourn": float(dag.finish_time - dag.arrival_time),
                "rho": float(dag.priority),
                "on_time": bool(dag.finish_time <= dag.deadline),
            }
        )
        orig_agg(dag)

    env._aggregate_dag_stats = hooked_agg

    for ep in range(num_episodes):
        n_before = len(sojourns)
        occ = OccupancyTracker()
        env.reset(SPLIT, episode_idx=ep)
        scheduler.reset()
        occ.observe(env)
        steps = 0
        while True:
            occ.observe(env)
            action = scheduler.select_action(env)
            if action is None:
                event_type, next_time, min_proc_avail = env._get_next_event_time()
                if env._is_done():
                    break
                env.change_time(next_time)
                occ.observe(env)
                if event_type == "task_complete":
                    env._cleanup_completed_tasks()
                    env._update_candidate_tasks(min_proc_avail)
                    while event_type == "task_complete" and len(env.candidate_tasks) == 0:
                        if env._is_done():
                            break
                        event_type, next_time, min_proc_avail = env._get_next_event_time()
                        env.change_time(next_time)
                        occ.observe(env)
                        if event_type == "task_complete":
                            env._cleanup_completed_tasks()
                        elif event_type == "dag_arrival":
                            env._add_new_dag()
                        env._update_candidate_tasks(min_proc_avail)
                elif event_type == "dag_arrival":
                    env._add_new_dag()
                    env._update_candidate_tasks(min_proc_avail)
                elif event_type == "processor_idle":
                    env._update_candidate_tasks(min_proc_avail)
                env._cleanup_completed_dags()
                if env._is_done():
                    break
                continue

            step_active.append(len(env.active_dags))
            step_ready.append(sum(1 for t in env.candidate_tasks if not t.is_empty))
            _, _, done, _ = env.step(action)
            steps += 1
            occ.observe(env)
            if done:
                break

        occ.observe(env)
        ep_soj = sojourns[n_before:]
        stretch = [
            s["sojourn"] / s["makespan_est"]
            for s in ep_soj
            if s["makespan_est"] > 0
        ]
        ep_on = float(np.mean([s["on_time"] for s in ep_soj])) if ep_soj else float("nan")
        row = {
            "episode": ep,
            "steps": steps,
            "job_on_time": ep_on,
            **occ.summary(),
            "mean_stretch": float(np.mean(stretch)) if stretch else float("nan"),
        }
        ep_rows.append(row)
        print(
            f"  {tag} ep {ep + 1}/{num_episodes}: "
            f"Job={row['job_on_time']:.1%} "
            f"active={row['time_weighted_active_jobs']:.2f} "
            f"ready={row['time_weighted_ready_set']:.2f} "
            f"stretch={row['mean_stretch']:.2f}"
        )

    env._aggregate_dag_stats = orig_agg
    on_time = np.array([1.0 if s["on_time"] else 0.0 for s in sojourns])
    stretch_all = np.array(
        [s["sojourn"] / s["makespan_est"] for s in sojourns if s["makespan_est"] > 0]
    )
    return {
        "job_on_time_mean": float(np.mean(on_time)) if on_time.size else float("nan"),
        "job_on_time_std": float(np.std([r["job_on_time"] for r in ep_rows])),
        "by_rho": _rho_rates(sojourns),
        "time_weighted_active_jobs_mean": float(
            np.mean([r["time_weighted_active_jobs"] for r in ep_rows])
        ),
        "time_weighted_ready_set_mean": float(
            np.mean([r["time_weighted_ready_set"] for r in ep_rows])
        ),
        "decision_step_ready_set": _summarize(np.array(step_ready, dtype=float)),
        "decision_step_active_jobs": _summarize(np.array(step_active, dtype=float)),
        "stretch_mean": float(np.mean(stretch_all)) if stretch_all.size else float("nan"),
        "per_episode": ep_rows,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    env = MultiDAGSchedulerEnv(
        task_dim=ORIGINAL_TASK_DIM,
        proc_dim=PROC_DIM,
        dag_dataset_dir="dataset/daggen_factory/dags",
        max_candidates=15,
        arrival_rate=0.55,
        seed=42,
        dag_generator="notOur",
        device_config="dataset/device/device_heterogeneous.json",
    )
    env.set_eval_wait_time_stats(2.0, 3.5)

    payload = {
        "note": (
            "Heuristic-only sweep. Overlap index uses test-pool E[M̂]=6.40s. "
            "RPNS/PPO are not retrained."
        ),
        "lambdas": list(LAMBDAS),
        "results": {},
    }

    for lam in LAMBDAS:
        env.arrival_rate = lam
        overlap = E_MHAT * lam
        util_ub = (OFFERED_AT_055 * (lam / 0.55)) / TOTAL_MIPS
        print(f"\n== λ={lam:.2f}  1/λ={1/lam:.2f}s  E[M̂]λ={overlap:.2f}  ρ_comp≤{util_ub:.3f} ==")
        row = {
            "lambda": lam,
            "mean_interarrival_s": 1.0 / lam,
            "overlap_index_Mhat_times_lambda": overlap,
            "compute_utilization_upper": util_ub,
            "methods": {},
        }
        for name, ctor in (("EDF", EDFScheduler), ("HEFT", HEFTScheduler)):
            print(f"-- {name} --")
            row["methods"][name] = run_replay(
                env, ctor(env.device_env), NUM_EPISODES, f"λ={lam:.2f} {name}"
            )
        payload["results"][f"{lam:.2f}"] = row

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n== Summary (Job on-time %) ==")
    hdr = f"{'λ':>6} {'1/λ':>7} {'E[M]λ':>7} {'ρ̂':>6} {'EDF':>8} {'HEFT':>8} {'EDF ρ1.5':>9} {'EDF ρ3.0':>9} {'ready':>7}"
    print(hdr)
    for lam in LAMBDAS:
        r = payload["results"][f"{lam:.2f}"]
        edf = r["methods"]["EDF"]
        heft = r["methods"]["HEFT"]
        print(
            f"{lam:6.2f} {r['mean_interarrival_s']:7.2f} "
            f"{r['overlap_index_Mhat_times_lambda']:7.2f} "
            f"{r['compute_utilization_upper']:6.3f} "
            f"{100*edf['job_on_time_mean']:8.1f} "
            f"{100*heft['job_on_time_mean']:8.1f} "
            f"{100*edf['by_rho'].get('1.5', {}).get('job_on_time', float('nan')):9.1f} "
            f"{100*edf['by_rho'].get('3.0', {}).get('job_on_time', float('nan')):9.1f} "
            f"{edf['decision_step_ready_set']['mean']:7.1f}"
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
