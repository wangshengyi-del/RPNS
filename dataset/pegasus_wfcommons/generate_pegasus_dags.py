#!/usr/bin/env python3
"""用 WfCommons 生成 Pegasus 风格科学工作流，并转为 daggen_factory 兼容 JSON。

依赖:
  pip install wfcommons

示例:
  # 生成默认 OOD 测试集（约 1000 个 test DAG）
  python dataset/pegasus_wfcommons/generate_pegasus_dags.py

  # 小规模冒烟
  python dataset/pegasus_wfcommons/generate_pegasus_dags.py \\
      --specs epigenomics:50:4,montage:100:4 --out_dir dataset/pegasus_wfcommons/dags_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 保证仓库根目录在 path 上
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.task_generator import (  # noqa: E402
    COMPUTATION_RANGE,
    DATA_SIZE_RANGE,
    LOGNORMAL_MU,
    LOGNORMAL_SIGMA,
    PARETO_ALPHA,
)
from model.constants import CLOUD_ONLY_RATIO, EDGE_ONLY_RATIO  # noqa: E402

# recipe 名 -> (最小建议任务数, 导入名)
# Seismology ≈ CyberShake 族；Epigenomics / Montage 为经典 Pegasus 应用
DEFAULT_SPECS = (
    "epigenomics:50:200,"
    "epigenomics:100:200,"
    "montage:100:200,"
    "montage:200:150,"
    "seismology:200:150,"
    "seismology:300:100"
)

REF_CPU = 1500.0  # computation(MI) ≈ runtime(s) * REF_CPU


def _load_recipes():
    try:
        from wfcommons import WorkflowGenerator
        from wfcommons.wfchef.recipes import (
            EpigenomicsRecipe,
            MontageRecipe,
            SeismologyRecipe,
        )
    except ImportError as e:
        raise SystemExit(
            "未安装 wfcommons。请先执行: pip install wfcommons\n"
            f"原始错误: {e}"
        ) from e
    return {
        "epigenomics": (EpigenomicsRecipe, WorkflowGenerator),
        "montage": (MontageRecipe, WorkflowGenerator),
        "seismology": (SeismologyRecipe, WorkflowGenerator),
    }


def parse_specs(spec_str: str) -> List[Tuple[str, int, int]]:
    """解析 'name:num_tasks:count,...' """
    specs = []
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(f"非法 --specs 项: {part!r}，期望 name:num_tasks:count")
        name, n_tasks, count = bits[0].lower(), int(bits[1]), int(bits[2])
        specs.append((name, n_tasks, count))
    return specs


def _pareto_computation(rng: np.random.Generator) -> float:
    x_min = COMPUTATION_RANGE[0]
    computation = x_min / (rng.random() ** (1.0 / PARETO_ALPHA))
    while computation > COMPUTATION_RANGE[1]:
        computation = x_min / (rng.random() ** (1.0 / PARETO_ALPHA))
    return float(computation)


def _lognormal_data_size(rng: np.random.Generator) -> float:
    data_size = float(rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA))
    return float(max(DATA_SIZE_RANGE[0], min(DATA_SIZE_RANGE[1], data_size)))


def _assign_candidate_type(rng: np.random.Generator) -> str:
    r = float(rng.random())
    if r < CLOUD_ONLY_RATIO:
        return "cloud_only"
    if r < CLOUD_ONLY_RATIO + EDGE_ONLY_RATIO:
        return "edge_only"
    return "flexible"


def workflow_to_dag_dict(
    wf,
    dag_id: str,
    split: str,
    attribute_mode: str,
    rng: np.random.Generator,
    ref_cpu: float = REF_CPU,
) -> Dict:
    """将 WfCommons Workflow 转为 env.DAG.from_dict 兼容结构。"""
    node_ids = list(wf.nodes)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    tasks = []
    for nid in node_ids:
        idx = id_to_idx[nid]
        task_obj = wf.nodes[nid]["task"]
        parents = sorted(id_to_idx[p] for p in wf.predecessors(nid))
        children = sorted(id_to_idx[c] for c in wf.successors(nid))

        if attribute_mode == "synthetic":
            computation = _pareto_computation(rng)
            data_size = _lognormal_data_size(rng)
        elif attribute_mode == "wfcommons":
            runtime = float(getattr(task_obj, "runtime", 0.0) or 0.0)
            computation = max(COMPUTATION_RANGE[0], min(COMPUTATION_RANGE[1], runtime * ref_cpu))
            out_bytes = sum(float(f.size) for f in (task_obj.output_files or []))
            data_size_mb = out_bytes / (1024.0 * 1024.0)
            if data_size_mb <= 0:
                data_size_mb = _lognormal_data_size(rng)
            data_size = max(DATA_SIZE_RANGE[0], min(DATA_SIZE_RANGE[1], data_size_mb))
        else:
            raise ValueError(f"未知 attribute_mode={attribute_mode}")

        tasks.append({
            "task_id": idx,
            "source_task_id": idx,
            "computation": round(computation, 2),
            "data_size": round(data_size, 2),
            "parents": parents,
            "children": children,
            "candidate_type": _assign_candidate_type(rng),
        })

    return {
        "dag_id": dag_id,
        "split": split,
        "num_tasks": len(tasks),
        "source": "wfcommons",
        "tasks": tasks,
    }


def generate_dataset(
    specs: List[Tuple[str, int, int]],
    out_dir: Path,
    attribute_mode: str,
    seed: int,
    split: str = "test",
) -> List[Dict]:
    recipes = _load_recipes()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    index: List[Dict] = []
    global_i = 0

    for recipe_name, num_tasks, count in specs:
        if recipe_name not in recipes:
            raise ValueError(
                f"未知 recipe={recipe_name}，可选: {sorted(recipes)}"
            )
        RecipeCls, WorkflowGenerator = recipes[recipe_name]
        print(f"[gen] {recipe_name} num_tasks≈{num_tasks} × {count} ...")
        try:
            recipe = RecipeCls.from_num_tasks(num_tasks)
        except Exception as e:
            raise RuntimeError(
                f"创建 {recipe_name}(num_tasks={num_tasks}) 失败（可能小于该 recipe 基图规模）: {e}"
            ) from e

        gen = WorkflowGenerator(recipe)
        for k in range(count):
            wf = gen.build_workflow()
            dag_id = f"{split}_{recipe_name}_{num_tasks}_{k:04d}"
            dag = workflow_to_dag_dict(
                wf,
                dag_id=dag_id,
                split=split,
                attribute_mode=attribute_mode,
                rng=rng,
            )
            # 记录真实生成规模与来源 meta
            dag["recipe"] = recipe_name
            dag["requested_num_tasks"] = num_tasks
            path = out_dir / f"dag_{dag_id}.json"
            with open(path, "w") as f:
                json.dump(dag, f, indent=2)
            index.append({
                "dag_id": dag_id,
                "split": split,
                "num_tasks": dag["num_tasks"],
                "recipe": recipe_name,
                "requested_num_tasks": num_tasks,
            })
            global_i += 1
            if (k + 1) % 50 == 0 or (k + 1) == count:
                print(f"  ... {k + 1}/{count} (actual_nodes={dag['num_tasks']})")

    index_path = out_dir / "dag_dataset.json"
    # 打乱索引，避免按 recipe/规模顺序取 DAG（否则 max_dags 截断或到达前缀会严重偏小图）
    rng_idx = np.random.default_rng(seed)
    order = rng_idx.permutation(len(index))
    index = [index[i] for i in order]
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    summary = {
        "seed": seed,
        "attribute_mode": attribute_mode,
        "split": split,
        "num_dags": len(index),
        "specs": [
            {"recipe": r, "requested_num_tasks": n, "count": c}
            for r, n, c in specs
        ],
        "node_stats": {
            "mean": float(np.mean([x["num_tasks"] for x in index])),
            "min": int(min(x["num_tasks"] for x in index)),
            "max": int(max(x["num_tasks"] for x in index)),
        },
        "out_dir": str(out_dir),
    }
    with open(out_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n完成: {len(index)} 个 DAG → {out_dir}")
    print(f"索引: {index_path}")
    print(
        f"节点数: mean={summary['node_stats']['mean']:.1f} "
        f"min={summary['node_stats']['min']} max={summary['node_stats']['max']}"
    )
    return index


def main():
    parser = argparse.ArgumentParser(
        description="WfCommons → daggen_factory 格式的 Pegasus 风格 DAG 生成器"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_REPO_ROOT / "dataset" / "pegasus_wfcommons" / "dags"),
        help="输出目录（将写入 dag_*.json 与 dag_dataset.json）",
    )
    parser.add_argument(
        "--specs",
        type=str,
        default=DEFAULT_SPECS,
        help="生成规格: name:num_tasks:count,... "
             "name∈{epigenomics,montage,seismology}",
    )
    parser.add_argument(
        "--attribute_mode",
        choices=["synthetic", "wfcommons"],
        default="synthetic",
        help="synthetic=沿用本仓库 Pareto/Lognormal（推荐，隔离结构 OOD）；"
             "wfcommons=用 runtime/文件大小映射",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="写入 split 字段；OOD 评测通常全部放 test",
    )
    args = parser.parse_args()

    specs = parse_specs(args.specs)
    generate_dataset(
        specs=specs,
        out_dir=Path(args.out_dir),
        attribute_mode=args.attribute_mode,
        seed=args.seed,
        split=args.split,
    )


if __name__ == "__main__":
    main()
