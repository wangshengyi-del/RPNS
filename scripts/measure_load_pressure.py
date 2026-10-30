"""
Measure whether λ=0.55 is high load relative to platform capacity.

Reports:
  1. Uncontended HEFT makespan M̂ vs mean inter-arrival 1/λ
  2. Offered compute rate vs total MIPS
  3. Online EDF replay: active Jobs, ready-set size, processor occupancy,
     sojourn / M̂ (contention stretch)

Uses the paper's test pool, heterogeneous device file, and EDF+EFT.
Does not retrain.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baseline_heft import EDFScheduler
from env.dag_task import DAG
from env.multi_dag_environment import MultiDAGSchedulerEnv
from model.constants import NUM_PROCS, ORIGINAL_TASK_DIM, PROC_DIM


ARRIVAL_RATE = 0.55
NUM_EPISODES = 5
SPLIT = "test"
DEVICE_CONFIG = "dataset/device/device_heterogeneous.json"
DAG_DIR = "dataset/daggen_factory/dags"
OUT_PATH = os.path.join(ROOT, "paper_results", "load_pressure_lambda055.json")


def _pct(arr: np.ndarray, qs=(50, 90, 95, 99)) -> Dict[str, float]:
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def _summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), **_pct(arr)}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        **_pct(arr),
    }


def static_pool_stats(env: MultiDAGSchedulerEnv) -> Dict:
    files = env.dag_files[SPLIT]
    n_tasks, mi_all, mi_cloud, mi_edge, mi_flex, mhats = [], [], [], [], [], []

    avg_c = env.device_env.AVG_CLOUD_CPU
    avg_e = env.device_env.AVG_EDGE_CPU
    avg_a = env.device_env.AVG_CPU
    avg_b = env.device_env.AVG_BANDWIDTH

    for path in files:
        dag = DAG.from_dict(DAG.load_from_file(path))
        dag.compute_stats(2.0, avg_c, avg_e, avg_a, avg_b)
        n_tasks.append(len(dag.tasks))
        mhats.append(float(dag.makespan_est))
        mi = c = e = f = 0.0
        for t in dag.tasks:
            mi += float(t.computation)
            if t.candidate_type == "cloud_only":
                c += float(t.computation)
            elif t.candidate_type == "edge_only":
                e += float(t.computation)
            else:
                f += float(t.computation)
        mi_all.append(mi)
        mi_cloud.append(c)
        mi_edge.append(e)
        mi_flex.append(f)

    total_mips = float(sum(p.cpu for p in env.processors))
    cloud_mips = float(sum(p.cpu for p in env.processors if p.proc_type == 0))
    edge_mips = float(sum(p.cpu for p in env.processors if p.proc_type == 1))
    mean_mi = float(np.mean(mi_all))
    mean_mhat = float(np.mean(mhats))
    offered = ARRIVAL_RATE * mean_mi
    interarrival = 1.0 / ARRIVAL_RATE

    return {
        "n_dags": len(files),
        "n_tasks": _summarize(np.array(n_tasks)),
        "makespan_est_s": _summarize(np.array(mhats)),
        "mi_per_job": _summarize(np.array(mi_all)),
        "mean_mi_cloud_only": float(np.mean(mi_cloud)),
        "mean_mi_edge_only": float(np.mean(mi_edge)),
        "mean_mi_flexible": float(np.mean(mi_flex)),
        "lambda": ARRIVAL_RATE,
        "mean_interarrival_s": interarrival,
        "E_Mhat_over_interarrival": mean_mhat / interarrival,
        "total_mips": total_mips,
        "cloud_mips": cloud_mips,
        "edge_mips": edge_mips,
        "offered_mi_per_s": offered,
        "compute_utilization_upper": offered / total_mips if total_mips else float("nan"),
        "cloud_only_offered_over_cloud_mips": (
            ARRIVAL_RATE * float(np.mean(mi_cloud)) / cloud_mips
        ),
        "edge_only_offered_over_edge_mips": (
            ARRIVAL_RATE * float(np.mean(mi_edge)) / edge_mips
        ),
    }


class OccupancyTracker:
    def __init__(self):
        self.prev_t: Optional[float] = None
        self.snap: Optional[Dict] = None
        self.dt = 0.0
        self.active_dt = 0.0
        self.ready_dt = 0.0
        self.busy_dt = 0.0
        self.n_procs = 0

    def _snapshot(self, env: MultiDAGSchedulerEnv) -> Dict:
        return {
            "n_active": len(env.active_dags),
            "n_ready": sum(1 for task in env.candidate_tasks if not task.is_empty),
            "available_at": [float(p.available_at) for p in env.processors],
        }

    def observe(self, env: MultiDAGSchedulerEnv) -> None:
        t = float(env.current_time)
        self.n_procs = len(env.processors)
        if not np.isfinite(t):
            return
        if self.prev_t is not None and t > self.prev_t and self.snap is not None:
            dt = t - self.prev_t
            if np.isfinite(dt) and 0 < dt < 1e6:
                self.dt += dt
                self.active_dt += self.snap["n_active"] * dt
                self.ready_dt += self.snap["n_ready"] * dt
                for avail in self.snap["available_at"]:
                    if not np.isfinite(avail):
                        continue
                    overlap = min(t, max(avail, self.prev_t)) - self.prev_t
                    if overlap > 0:
                        self.busy_dt += overlap
        self.prev_t = t
        self.snap = self._snapshot(env)

    def summary(self) -> Dict[str, float]:
        if self.dt <= 0 or self.n_procs == 0:
            return {
                "horizon_s": 0.0,
                "time_weighted_active_jobs": float("nan"),
                "time_weighted_ready_set": float("nan"),
                "time_weighted_proc_util": float("nan"),
            }
        return {
            "horizon_s": float(self.dt),
            "time_weighted_active_jobs": float(self.active_dt / self.dt),
            "time_weighted_ready_set": float(self.ready_dt / self.dt),
            "time_weighted_proc_util": float(self.busy_dt / (self.dt * self.n_procs)),
        }


def run_edf_replay(env: MultiDAGSchedulerEnv, num_episodes: int) -> Dict:
    scheduler = EDFScheduler(env.device_env)
    sojourns: List[Dict] = []
    step_active: List[int] = []
    step_ready: List[int] = []
    step_busy: List[int] = []
    step_wait: List[float] = []
    ep_rows: List[Dict] = []

    orig_agg = env._aggregate_dag_stats

    def hooked_agg(dag):
        sojourns.append(
            {
                "makespan_est": float(dag.makespan_est),
                "sojourn": float(dag.finish_time - dag.arrival_time),
                "rho": float(dag.priority),
                "on_time": bool(dag.finish_time <= dag.deadline),
                "n_tasks": int(dag.total_tasks),
            }
        )
        orig_agg(dag)

    env._aggregate_dag_stats = hooked_agg

    for ep in range(num_episodes):
        n_soj_before = len(sojourns)
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

            n_ready = sum(1 for task in env.candidate_tasks if not task.is_empty)
            n_busy = sum(1 for p in env.processors if p.available_at > env.current_time)
            waits = [max(0.0, p.available_at - env.current_time) for p in env.processors]
            step_active.append(len(env.active_dags))
            step_ready.append(n_ready)
            step_busy.append(n_busy)
            step_wait.append(float(np.mean(waits)))

            _, _, done, _ = env.step(action)
            steps += 1
            occ.observe(env)
            if done:
                break

        occ.observe(env)
        busy_s = sum(
            (p.total_energy / p.power) if p.power > 0 else 0.0
            for p in env.processors
        )
        occ_sum = occ.summary()
        energy_util = (
            float(busy_s / (occ_sum["horizon_s"] * len(env.processors)))
            if occ_sum["horizon_s"] > 0
            else float("nan")
        )
        ep_soj = sojourns[n_soj_before:]
        stretch = [
            s["sojourn"] / s["makespan_est"]
            for s in ep_soj
            if s["makespan_est"] > 0
        ]
        ep_rows.append(
            {
                "episode": ep,
                "steps": steps,
                "job_on_time": float(env.on_time_completed_dags / env.total_dags_count)
                if env.total_dags_count
                else float("nan"),
                "n_jobs": int(env.total_dags_count),
                "mean_stretch": float(np.mean(stretch)) if stretch else float("nan"),
                "compute_busy_s": float(busy_s),
                "energy_proc_util": energy_util,
                **occ_sum,
            }
        )
        print(
            f"  EDF ep {ep + 1}/{num_episodes}: "
            f"Job={ep_rows[-1]['job_on_time']:.1%} "
            f"active={ep_rows[-1]['time_weighted_active_jobs']:.2f} "
            f"ready={ep_rows[-1]['time_weighted_ready_set']:.2f} "
            f"util={ep_rows[-1]['time_weighted_proc_util']:.1%} "
            f"e-util={ep_rows[-1]['energy_proc_util']:.1%} "
            f"stretch={ep_rows[-1]['mean_stretch']:.2f}"
        )

    env._aggregate_dag_stats = orig_agg

    stretch_all = np.array(
        [s["sojourn"] / s["makespan_est"] for s in sojourns if s["makespan_est"] > 0]
    )
    mhat_run = np.array([s["makespan_est"] for s in sojourns])
    sojourn_run = np.array([s["sojourn"] for s in sojourns])
    on_time = np.array([1.0 if s["on_time"] else 0.0 for s in sojourns])

    return {
        "per_episode": ep_rows,
        "decision_step_active_jobs": _summarize(np.array(step_active, dtype=float)),
        "decision_step_ready_set": _summarize(np.array(step_ready, dtype=float)),
        "decision_step_busy_procs": _summarize(np.array(step_busy, dtype=float)),
        "decision_step_mean_proc_wait_s": _summarize(np.array(step_wait, dtype=float)),
        "time_weighted_active_jobs_mean": float(
            np.mean([r["time_weighted_active_jobs"] for r in ep_rows])
        ),
        "time_weighted_ready_set_mean": float(
            np.mean([r["time_weighted_ready_set"] for r in ep_rows])
        ),
        "time_weighted_proc_util_mean": float(
            np.mean([r["time_weighted_proc_util"] for r in ep_rows])
        ),
        "energy_proc_util_mean": float(
            np.mean([r["energy_proc_util"] for r in ep_rows])
        ),
        "job_on_time_mean": float(np.mean(on_time)) if on_time.size else float("nan"),
        "makespan_est_online_s": _summarize(mhat_run),
        "sojourn_s": _summarize(sojourn_run),
        "stretch_sojourn_over_Mhat": _summarize(stretch_all),
        "frac_jobs_stretch_gt_1": float(np.mean(stretch_all > 1.0))
        if stretch_all.size
        else float("nan"),
        "frac_jobs_stretch_gt_rho": float(
            np.mean(
                [
                    s["sojourn"] > s["makespan_est"] * s["rho"]
                    for s in sojourns
                    if s["makespan_est"] > 0
                ]
            )
        )
        if sojourns
        else float("nan"),
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    env = MultiDAGSchedulerEnv(
        task_dim=ORIGINAL_TASK_DIM,
        proc_dim=PROC_DIM,
        dag_dataset_dir=DAG_DIR,
        max_candidates=15,
        arrival_rate=ARRIVAL_RATE,
        seed=42,
        dag_generator="notOur",
        device_config=DEVICE_CONFIG,
    )
    env.set_eval_wait_time_stats(2.0, 3.5)

    print("== Static test pool ==")
    static = static_pool_stats(env)
    m = static["makespan_est_s"]
    print(
        f"  n_dags={static['n_dags']}, "
        f"E[|V|]={static['n_tasks']['mean']:.1f}, "
        f"E[M̂]={m['mean']:.3f}s (p50={m['p50']:.3f}, p90={m['p90']:.3f})"
    )
    print(
        f"  1/λ={static['mean_interarrival_s']:.3f}s, "
        f"E[M̂]/(1/λ)={static['E_Mhat_over_interarrival']:.2f}"
    )
    print(
        f"  offered={static['offered_mi_per_s']:.1f} MI/s, "
        f"ΣMIPS={static['total_mips']:.0f}, "
        f"ρ_comp≤{static['compute_utilization_upper']:.3f}"
    )

    print("== Online EDF replay (λ=0.55, test, 5 episodes) ==")
    online = run_edf_replay(env, NUM_EPISODES)

    payload = {
        "setting": {
            "lambda": ARRIVAL_RATE,
            "episodes": NUM_EPISODES,
            "split": SPLIT,
            "device_config": DEVICE_CONFIG,
            "scheduler": "EDF+EFT",
        },
        "static_pool": static,
        "online_edf": online,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print("== Verdict inputs ==")
    print(
        f"  overlap index E[M̂]/(1/λ) = {static['E_Mhat_over_interarrival']:.3f} "
        f"(>1 ⇒ isolated jobs already longer than inter-arrival)"
    )
    print(
        f"  compute-only utilization upper bound = "
        f"{static['compute_utilization_upper']:.3f}"
    )
    print(
        f"  time-weighted active Jobs = {online['time_weighted_active_jobs_mean']:.2f}"
    )
    print(
        f"  decision-step ready-set mean = {online['decision_step_ready_set']['mean']:.2f} "
        f"(p90={online['decision_step_ready_set']['p90']:.1f})"
    )
    print(
        f"  time-weighted processor util = {online['time_weighted_proc_util_mean']:.1%}"
    )
    print(
        f"  mean stretch sojourn/M̂ = {online['stretch_sojourn_over_Mhat']['mean']:.2f} "
        f"(frac>1 = {online['frac_jobs_stretch_gt_1']:.1%})"
    )
    print(f"  EDF Job on-time = {online['job_on_time_mean']:.1%}")


if __name__ == "__main__":
    main()
