#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
混合 DAG 生成器
- 使用 DAGgen 生成 DAG 结构（分层随机拓扑）
- 使用你的 TASK 属性（Pareto 计算量, Lognormal 数据大小）
- 输出与现有 DAG 数据结构完全兼容
"""

import numpy as np
from typing import List, Dict, Optional
import random
from model.constants import CLOUD_ONLY_RATIO, EDGE_ONLY_RATIO

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入 DAGgen 模块
try:
    from rnddag import DAG as DAGgenDAG
    from generator import uunifast_discard, gen_period, apply_execution_times
except ImportError as e:
    print(f"Warning: Could not import DAGgen modules: {e}")
    DAGgenDAG = None
    uunifast_discard = None

# 导入环境 DAG
try:
    from env.dag_task import DAG as EnvDAG, DAGTask
except ImportError:
    # 绝对路径导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.dag_task import DAG as EnvDAG, DAGTask

# 从 task_generator 获取分布参数
try:
    from dataset.task_generator import (
        COMPUTATION_RANGE, LOGNORMAL_MU, LOGNORMAL_SIGMA, PARETO_ALPHA, DATA_SIZE_RANGE
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset.task_generator import (
        COMPUTATION_RANGE, LOGNORMAL_MU, LOGNORMAL_SIGMA, PARETO_ALPHA, DATA_SIZE_RANGE
    )


# 常量定义（从 task_generator.py）
COMPUTATION_MIN = 100


def generate_pareto_computation(rng: np.random.RandomState = None) -> float:
    """
    生成 Pareto 分布的计算量
    分布参数: x_min=100, alpha=1.25
    """
    if rng is None:
        rng = np.random.default_rng()

    x_min = COMPUTATION_RANGE[0]
    u = rng.uniform(0, 1)
    computation = x_min / (u ** (1.0 / PARETO_ALPHA))

    # 仅保留下界约束（帕累托天然满足 x >= x_min）
    # 上界建议用拒绝采样或业务层兜底，而非硬截断
    computation = max(x_min, computation)

    # 如果业务强制要求上界，改用拒绝采样避免边界堆积：
    while computation > COMPUTATION_RANGE[1]:
        u = rng.uniform(0, 1)
        computation = x_min / (u ** (1.0 / PARETO_ALPHA))

    return computation


def generate_lognormal_data_size(rng: np.random.RandomState = None) -> float:
    """
    生成 Lognormal 分布的数据大小
    分布参数: mu=3.5, sigma=0.5
    """
    if rng is None:
        rng = np.random.default_rng()

    # Lognormal: X = exp(mu + sigma * Z) 其中 Z ~ N(0,1)
    data_size = rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA)
    data_size = max(DATA_SIZE_RANGE[0], min(DATA_SIZE_RANGE[1], data_size))

    return data_size


def assign_candidate_type(rng: np.random.RandomState = None) -> str:
    """
    随机分配候选处理器类型
    分布: flexible=80%, cloud_only=10%, edge_only=10%
    """
    if rng is None:
        rng = np.random.default_rng()

    r = rng.uniform() if isinstance(rng, np.random.RandomState) else rng.random()

    if r < CLOUD_ONLY_RATIO:
        return 'cloud_only'
    elif r < CLOUD_ONLY_RATIO + EDGE_ONLY_RATIO:
        return 'edge_only'
    else:
        return 'flexible'


def convert_daggen_to_env_dag(
    daggen_dag: 'DAGgenDAG',
    dag_id: str,
    rng: np.random.RandomState = None,
    use_real_distributions: bool = True
) -> EnvDAG:
    """
    将 DAGgen 生成的图转换为环境兼容的 DAG

    Args:
        daggen_dag: DAGgen 的 DAG 对象
        dag_id: DAG ID
        rng: 随机数生成器
        use_real_distributions: True=使用你的 Pareto/Lognormal 分布，False=使用 DAGgen 的均匀分布

    Returns:
        EnvDAG: 与你的环境兼容的 DAG 对象
    """
    import networkx as nx

    G = daggen_dag.get_graph()
    all_nodes = list(G.nodes())
    n_nodes = len(all_nodes)

    if rng is None:
        rng = np.random.default_rng()

    # 获取节点属性（C 计算量）
    node_c = nx.get_node_attributes(G, 'C')

    # 拓扑排序获取正确的处理顺序
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        topo_order = all_nodes  # fallback

    env_tasks = []
    task_id_map = {}  # DAGgen node id -> env task_id (0-based)

    # 创建环境 TASK
    for idx, node in enumerate(topo_order):
        task_id_map[node] = idx

    # 再次遍历建立父子关系
    for idx, node in enumerate(topo_order):
        # 使用真实分布或 DAGgen 分布
        if use_real_distributions:
            computation = generate_pareto_computation(rng)
            data_size = generate_lognormal_data_size(rng)
            candidate_type = assign_candidate_type(rng)
        else:
            # DAGgen 的原始计算量（已归一化）
            computation = node_c.get(node, 1000)
            data_size = computation * 0.1  # 简化：数据大小约为计算量的 10%
            candidate_type = 'flexible'

        # 获取父节点和子节点（使用映射后的 ID）
        parents = []
        children = []
        for parent in G.predecessors(node):
            if parent in task_id_map:  # 排除源节点等
                parents.append(task_id_map[parent])
        for child in G.successors(node):
            if child in task_id_map:  # 排除汇节点等
                children.append(task_id_map[child])

        task = DAGTask(
            task_id=idx,
            source_task_id=idx,
            computation=computation,
            data_size=data_size,
            parents=parents,
            children=children,
            candidate_type=candidate_type
        )
        env_tasks.append(task)

    env_dag = EnvDAG(dag_id=dag_id, tasks=env_tasks)
    return env_dag


def generate_hybrid_dag(
    dag_id: str,
    dag_config: Dict,
    rng: Optional[np.random.RandomState] = None,
    use_real_distributions: bool = True
) -> EnvDAG:
    """
    生成单个混合 DAG

    Args:
        dag_id: DAG ID
        dag_config: DAG 结构配置（parallelism, layer_num_min, layer_num_max, connect_prob）
        rng: 随机数生成器
        use_real_distributions: 是否使用真实分布

    Returns:
        EnvDAG: 混合 DAG
    """
    # 创建 DAGgen DAG
    daggen_dag = DAGgenDAG(i=int(dag_id.split('_')[-1]) if '_' in dag_id else 0)

    # 生成随机结构
    daggen_dag.gen_rnd(
        parallelism=dag_config.get('parallelism', 8),
        layer_num_min=dag_config.get('layer_num_min', 5),
        layer_num_max=dag_config.get('layer_num_max', 12),
        connect_prob=dag_config.get('connect_prob', 0.5)
    )

    # 转换
    env_dag = convert_daggen_to_env_dag(
        daggen_dag,
        dag_id,
        rng,
        use_real_distributions
    )

    return env_dag


def generate_hybrid_multi_dag(
    n_dags: int,
    dag_config: Dict,
    utilization: float = 0.8,
    period_range: List[int] = [100, 500],
    rng: Optional[np.random.RandomState] = None,
    use_real_distributions: bool = True
) -> List[EnvDAG]:
    """
    生成多个混合 DAG（用于训练）

    Args:
        n_dags: DAG 数量
        dag_config: DAG 结构配置
        utilization: 总利用率（每个 DAG 均分）
        period_range: Period 范围 [min, max]
        rng: 随机数生成器
        use_real_distributions: 是否使用真实分布

    Returns:
        List[EnvDAG]: DAG 列表
    """
    if rng is None:
        rng = np.random.default_rng()

    dags = []
    utilization_per_dag = utilization / n_dags

    # 使用 uunifast 分配利用率
    try:
        u_sets = uunifast_discard(n_dags, utilization, nsets=1, ulimit=1.0)
        utilizations = u_sets[0]
    except RuntimeError:
        # fallback: 均分
        utilizations = [utilization / n_dags] * n_dags

    for i in range(n_dags):
        # 生成 DAG 结构
        daggen_dag = DAGgenDAG(i=i, U=utilizations[i], T=100)

        daggen_dag.gen_rnd(
            parallelism=dag_config.get('parallelism', 8),
            layer_num_min=dag_config.get('layer_num_min', 5),
            layer_num_max=dag_config.get('layer_num_max', 12),
            connect_prob=dag_config.get('connect_prob', 0.5)
        )

        # 计算工作量（基于利用率和周期）
        period = rng.integers(period_range[0], period_range[1] + 1)
        workload = utilizations[i] * period

        # 应用执行时间（DAGgen 的方式，用于获取结构）
        apply_execution_times(daggen_dag, daggen_dag.get_number_of_nodes(), workload)

        # 转换为环境 DAG
        env_dag = convert_daggen_to_env_dag(
            daggen_dag,
            f"hybrid_{i:04d}",
            rng,
            use_real_distributions
        )

        # 设置基本属性
        env_dag.arrival_time = 0.0  # 可以在外部设置

        dags.append(env_dag)

    return dags


class HybridDAGGenerator:
    """
    混合 DAG 生成器类
    提供与现有训练流程兼容的接口
    """

    def __init__(
        self,
        dag_config: Dict = None,
        use_real_distributions: bool = True,
        seed: int = 42
    ):
        """
        Args:
            dag_config: DAG 结构配置
            use_real_distributions: 是否使用真实分布
            seed: 随机种子
        """
        self.dag_config = dag_config or {
            'parallelism': 8,
            'layer_num_min': 5,
            'layer_num_max': 12,
            'connect_prob': 0.5
        }
        self.use_real_distributions = use_real_distributions
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def generate(self, dag_id: str = None) -> EnvDAG:
        """
        生成单个 DAG

        Args:
            dag_id: DAG ID（如果为 None，自动生成）

        Returns:
            EnvDAG: 混合 DAG
        """
        if dag_id is None:
            dag_id = f"hybrid_{self.rng.randint(0, 100000):06d}"

        return generate_hybrid_dag(
            dag_id=dag_id,
            dag_config=self.dag_config,
            rng=self.rng,
            use_real_distributions=self.use_real_distributions
        )

    def generate_batch(self, n: int) -> List[EnvDAG]:
        """
        批量生成 DAG

        Args:
            n: 数量

        Returns:
            List[EnvDAG]: DAG 列表
        """
        dags = []
        for i in range(n):
            dag_id = f"hybrid_{i:04d}"
            dag = generate_hybrid_dag(
                dag_id=dag_id,
                dag_config=self.dag_config,
                rng=self.rng,
                use_real_distributions=self.use_real_distributions
            )
            dags.append(dag)
        return dags

    def set_seed(self, seed: int):
        """设置随机种子"""
        self.rng = np.random.RandomState(seed)


def generate_dag_dataset(
        num_val: int = 200,
        num_test: int = 500,
        output_dir: str = "dataset"
):
    """生成 DAG 数据集

    Args:
        task_pool: Task 池
        num_val: 验证集数量（固定）
        num_test: 测试集数量（固定）
        output_dir: 输出目录
        complexity: 复杂度分布
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    dag_info = []

    # 训练集：动态生成（每次epoch新拓扑）
    val_count = num_val
    test_count = num_test

    total_num = val_count + test_count
    generator = HybridDAGGenerator(seed=42)
    dags = generator.generate_batch(total_num)
    for i in range(total_num):
        dag = dags[i]
        split = "val" if i < val_count else "test"
        dag_id = f"{split}_{i:04d}"
        data = {
            "dag_id": dag_id,
            "split": split,
            "num_tasks": dag.total_tasks,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "source_task_id": t.source_task_id,
                    "computation": round(t.computation, 2),
                    "data_size": round(t.data_size, 2),
                    "parents": t.parents,
                    "children": t.children,
                    "candidate_type": t.candidate_type
                }
                for t in dag.tasks
            ]
        }

        filepath = os.path.join(output_dir, f"dag_{dag_id}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        dag_info.append({
            "dag_id": dag_id,
            "split": split,
            "num_tasks": dag.total_tasks
        })

    # 保存索引
    index_file = os.path.join(output_dir, "dag_dataset.json")
    with open(index_file, 'w') as f:
        json.dump(dag_info, f, indent=2)

    # 统计
    print("DAG 数据集生成完成!")
    print(f"  总数: {len(dag_info)}")
    print(f"  验证集: {sum(1 for d in dag_info if d['split'] == 'val')}")
    print(f"  测试集: {sum(1 for d in dag_info if d['split'] == 'test')}")

    return dag_info


if __name__ == "__main__":
    # 测试
    generate_dag_dataset(1000, 1000, "dags")