#!/usr/bin/env python3
"""汇总所有实验结果到 paper/result/all_result_details.csv。

覆盖：
  - 主实验 λ=0.55：Random / HEFT / EDF / PPO_* / EDF+PPO / Ablation_*
  - 归因 hybrid：Hybrid_EDF+EFT / Hybrid_EDF+PPO / Hybrid_PPO+EFT / Hybrid_PPO+PPO
  - OOD λ≈0.12：OOD_HEFT / OOD_EDF / OOD_EDF+PPO

用法:
  python3 scripts/build_all_result_details.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/result/all_result_details.csv"
COLS = [
    "method",
    "episode",
    "num_dags",
    "dag_completion_rate",
    "num_tasks",
    "task_completion_rate",
    "avg_reward",
    "steps",
    "avg_delay",
    "avg_energy",
    "dag_p1.5",
    "task_p1.5",
    "delay_p1.5",
    "energy_p1.5",
    "dag_p2.0",
    "task_p2.0",
    "delay_p2.0",
    "energy_p2.0",
    "dag_p3.0",
    "task_p3.0",
    "delay_p3.0",
    "energy_p3.0",
]
PRI = [1.5, 2.0, 3.0]
METHOD_ORDER = [
    "Random",
    "HEFT",
    "EDF",
    "PPO_twostage",
    "PPO_joint",
    "EDF+PPO",
    "Ablation_no_inter",
    "Ablation_no_pair",
    "Hybrid_EDF+EFT",
    "Hybrid_EDF+PPO",
    "Hybrid_PPO+EFT",
    "Hybrid_PPO+PPO",
    "OOD_HEFT",
    "OOD_EDF",
    "OOD_EDF+PPO",
]
MAIN_RUNS = {
    "EDF+PPO": "runs/hierarchical_ppo/20260729_140051_seed42_ppo(rpns)",
    "PPO_joint": "runs/hierarchical_ppo/20260728_092351_seed42_ppo_joint",
    "PPO_twostage": "runs/hierarchical_ppo/20260727_164430_seed42_cpu_dagcomplete_g2_peft0_peakguard_ppo",
    "Ablation_no_inter": "runs/hierarchical_ppo/20260728_173701_seed42_ppo(rpns)_no_inter",
    "Ablation_no_pair": "runs/hierarchical_ppo/20260729_140328_seed42_ppo(rpns)_no_pair",
}
HYBRID = {
    "PPO-task + PPO-proc": "Hybrid_PPO+PPO",
    "EDF-task + PPO-proc": "Hybrid_EDF+PPO",
    "PPO-task + EFT-proc": "Hybrid_PPO+EFT",
    "EDF-task + EFT-proc": "Hybrid_EDF+EFT",
}


def empty(**kw):
    row = {c: np.nan for c in COLS}
    row.update(kw)
    return row


def fill_pri_long(row, sub: pd.DataFrame):
    for _, pr in sub.iterrows():
        p = float(pr["priority"])
        row[f"dag_p{p:.1f}"] = float(pr["dag_rate"])
        row[f"task_p{p:.1f}"] = float(pr["task_rate"])
        row[f"delay_p{p:.1f}"] = float(pr["delay"])
        row[f"energy_p{p:.1f}"] = float(pr["energy"])


def fill_pri_ps(row, ps):
    for p in PRI:
        s = next(
            (v for k, v in (ps or {}).items() if abs(float(k) - p) < 1e-9),
            {},
        )
        td = s.get("total_dags", 0) or 0
        tt = s.get("total_tasks", 0) or 0
        row[f"dag_p{p:.1f}"] = s.get("on_time_dags", 0) / td if td else np.nan
        row[f"task_p{p:.1f}"] = s.get("on_time_tasks", 0) / tt if tt else np.nan
        row[f"delay_p{p:.1f}"] = s.get("total_delay", 0.0) / tt if tt else np.nan
        row[f"energy_p{p:.1f}"] = s.get("total_energy", 0.0) / tt if tt else np.nan


def parse_diag_cluster(cluster: str, method: str, ep: int):
    m_comp = re.search(r"valid_dag_completion=(\d+)/(\d+)", cluster)
    if not m_comp:
        return None
    m_tasks = re.search(r"total_tasks_completed=(\d+)", cluster)
    m_energy = re.search(r"avg_per_task=([\d.]+)J", cluster)
    m_rew = re.search(r"rewards:.*?mean=([-\d.]+)", cluster)
    num_dags = int(m_comp.group(2))
    num_tasks = int(m_tasks.group(1)) if m_tasks else np.nan
    row = empty(
        method=method,
        episode=ep,
        num_dags=num_dags,
        num_tasks=num_tasks,
        steps=num_tasks,
        dag_completion_rate=int(m_comp.group(1)) / num_dags,
        avg_reward=float(m_rew.group(1)) if m_rew else np.nan,
        avg_energy=float(m_energy.group(1)) if m_energy else np.nan,
    )
    weights_d, tasks, delays, energies = [], [], [], []
    for pm in re.finditer(
        r"priority=([\d.]+):\s*DAG=(\d+)/(\d+)=([\d.]+)%,\s*"
        r"TASK=([\d.]+)%,\s*avg_delay=([\d.]+)s(?:,\s*avg_energy=([\d.]+)J)?",
        cluster,
    ):
        p = float(pm.group(1))
        td = int(pm.group(3))
        row[f"dag_p{p:.1f}"] = float(pm.group(4)) / 100.0
        row[f"task_p{p:.1f}"] = float(pm.group(5)) / 100.0
        row[f"delay_p{p:.1f}"] = float(pm.group(6))
        row[f"energy_p{p:.1f}"] = float(pm.group(7)) if pm.group(7) else np.nan
        weights_d.append(td)
        tasks.append(row[f"task_p{p:.1f}"])
        delays.append(row[f"delay_p{p:.1f}"])
        energies.append(row[f"energy_p{p:.1f}"])
    if weights_d:
        w = np.array(weights_d, float)
        row["task_completion_rate"] = float(np.average(tasks, weights=w))
        row["avg_delay"] = float(np.average(delays, weights=w))
        if all(e == e for e in energies):
            row["avg_energy"] = float(np.average(energies, weights=w))
    return row


def parse_ood_body(body: str, method: str):
    chunks = re.split(r"-{5,}max_arrival_time=", body)
    eps = []
    for ch in chunks[1:]:
        row = parse_diag_cluster(ch, method, len(eps) + 1)
        if row:
            eps.append(row)
    return eps


def resolve_ood_method(raw: str):
    r = raw.strip().rstrip(".").lower()
    if r.startswith("learned") or "edf+ppo" in r or "edf_ppo" in r:
        return "OOD_EDF+PPO"
    if r == "heft" or r.startswith("heft"):
        return "OOD_HEFT"
    if r == "edf" or r.startswith("edf"):
        return "OOD_EDF"
    return None


def parse_ood_file(path: Path):
    text = path.read_text()
    parts = re.split(r">>> running\s+(\S+)", text)
    out = {}
    for i in range(1, len(parts), 2):
        raw, body = parts[i], parts[i + 1]
        method = resolve_ood_method(raw)
        if not method:
            continue
        eps = parse_ood_body(body, method)
        if eps:
            out[method] = eps
            print(f"  {path.name}: {raw} -> {method} x{len(eps)}")
    return out


def parse_baseline(base_text: str, pri_long: pd.DataFrame, method: str):
    starts = {
        "HEFT": "HEFT 在线调度",
        "EDF": "EDF 在线调度",
        "Random": "Random 在线调度",
    }
    ends = {"HEFT": "EDF 在线调度", "EDF": "Random 在线调度", "Random": None}
    a = base_text.find(starts[method])
    b = (
        base_text.find(ends[method], a + 1)
        if ends[method]
        else len(base_text)
    )
    sec = base_text[a:b]
    ep_re = re.compile(
        rf"\[{method} Ep (\d+)\]\s*Reward=([-\d.]+)\s*\|\s*"
        rf"TASK:\s*([\d.]+)%\s*\|\s*DAG:\s*([\d.]+)%\s*\|\s*"
        rf"DELAY:\s*([\d.]+)s\s*\|\s*ENERGY:\s*([\d.]+)J"
    )
    matches = list(ep_re.finditer(sec))
    out = []
    for i, em in enumerate(matches):
        chunk = sec[(matches[i - 1].end() if i else 0) : em.end()]
        m_tasks = re.search(
            r"total_tasks_completed=(\d+),\s*valid_dag_completion=(\d+)/(\d+)",
            chunk,
        )
        m_rew = re.search(r"rewards:.*?mean=([-\d.]+)", chunk)
        num_tasks = int(m_tasks.group(1)) if m_tasks else 31000
        num_dags = int(m_tasks.group(3)) if m_tasks else 1000
        ep = int(em.group(1))
        row = empty(
            method=method,
            episode=ep,
            num_dags=num_dags,
            num_tasks=num_tasks,
            steps=num_tasks,
            dag_completion_rate=float(em.group(4)) / 100.0,
            task_completion_rate=float(em.group(3)) / 100.0,
            avg_reward=(
                float(m_rew.group(1))
                if m_rew
                else float(em.group(2)) / num_tasks
            ),
            avg_delay=float(em.group(5)),
            avg_energy=float(em.group(6)),
        )
        sub = pri_long[
            (pri_long.setting == "main_lambda0.55")
            & (pri_long.method == method)
            & (pri_long.episode == ep)
        ]
        fill_pri_long(row, sub)
        out.append(row)
    return out


def main():
    rows = []

    # 1) Main learning (reevaled best_DAG rows)
    for method, run in MAIN_RUNS.items():
        df = pd.read_csv(ROOT / run / "training_log.csv")
        t = df[
            (df.type == "test")
            & (df.episode.astype(str).str.startswith("best_DAG_"))
        ].sort_values("episode")
        for _, r in t.iterrows():
            ep = int(str(r["episode"]).split("_")[-1])
            row = empty(
                method=method,
                episode=ep,
                num_dags=r["num_dags"],
                num_tasks=r["num_tasks"],
                dag_completion_rate=r["dag_completion_rate"],
                task_completion_rate=r["task_completion_rate"],
                avg_reward=r["avg_reward"],
                steps=r["steps"],
                avg_delay=r["avg_delay"],
                avg_energy=r["avg_energy"],
            )
            for p in PRI:
                for k in ("dag", "task", "delay", "energy"):
                    row[f"{k}_p{p:.1f}"] = r[f"{k}_p{p:.1f}"]
            rows.append(row)

    # 2) Main baselines
    pri_long = pd.read_csv(
        ROOT / "paper_results/all_methods_priority_per_episode.csv"
    )
    base_text = (ROOT / "runs/heft/20260727-test-0.55.txt").read_text()
    for m in ("Random", "HEFT", "EDF"):
        rows.extend(parse_baseline(base_text, pri_long, m))

    # 3) Hybrid attribution
    h = json.loads(
        (ROOT / "hybrid_diagnostics/hybrid_detail.json").read_text()
    )
    for raw, method in HYBRID.items():
        for i, ep in enumerate(h["details"][raw]["per_episode"], 1):
            steps = ep.get("steps") or 31000
            row = empty(
                method=method,
                episode=i,
                num_dags=ep.get("total_dags", 1000),
                num_tasks=ep.get("total_tasks", 31000),
                dag_completion_rate=ep["dag_completion_rate"],
                task_completion_rate=ep["completion_rate"],
                avg_reward=ep.get("total_reward", np.nan) / steps,
                steps=steps,
                avg_delay=ep["avg_task_delay"],
                avg_energy=ep["avg_energy"],
            )
            fill_pri_ps(row, ep.get("priority_stats", {}))
            rows.append(row)

    # 4) OOD
    print("Parsing OOD...")
    ood = {}
    ood.update(
        parse_ood_file(ROOT / "dataset/pegasus_wfcommons/ood_eval_run.log")
    )
    ood.update(
        parse_ood_file(
            ROOT / "dataset/pegasus_wfcommons/ood_eval_learned_1000.log"
        )
    )
    for method in ("OOD_HEFT", "OOD_EDF", "OOD_EDF+PPO"):
        eps = ood.get(method, [])
        if len(eps) != 5:
            raise RuntimeError(f"{method}: expected 5 eps, got {len(eps)}")
        rows.extend(eps)

    df = pd.DataFrame(rows, columns=COLS)
    df["method"] = pd.Categorical(
        df["method"], categories=METHOD_ORDER, ordered=True
    )
    df = df.sort_values(["method", "episode"]).reset_index(drop=True)
    for c in COLS:
        if c != "method":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6g")
    print(f"Wrote {OUT} rows={len(df)}")
    print(df.groupby("method", observed=True).size())
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("WARNING nulls:", nulls.to_dict())
    else:
        print("No missing values.")


if __name__ == "__main__":
    main()
