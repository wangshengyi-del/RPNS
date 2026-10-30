"""
Task 生成器
基于 Alibaba Trace 统计特征校准
"""

import random
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List
from model.constants import CLOUD_ONLY_RATIO, EDGE_ONLY_RATIO

# Lognormal 参数 (μ=3.5, σ=0.5) 对应均值约 50MB
LOGNORMAL_MU = 3.5
LOGNORMAL_SIGMA = 0.5
# Pareto 分布（x_min=100，alpha=1.25）
PARETO_ALPHA = 1.25

# 基于真实数据统计的参数
DATA_SIZE_RANGE = (10, 100)
COMPUTATION_RANGE = (100, 10000)


@dataclass
class Task:
    """任务节点原始属性"""
    task_id: int
    # 计算量 (Million Instructions)
    computation: float
    # 数据大小 (MB)
    data_size: float
    # 隐私敏感标记（用于 edge_only 约束）
    privacy_sensitive: bool = False
    # 候选处理器类型（显式标记）
    # cloud_only: 只能在云端
    # edge_only: 只能在边缘
    # flexible: 云/边都可以
    candidate_type: str = "flexible"
    
    def __post_init__(self):
        """确保参数合理"""
        self.computation = max(1, self.computation)
        self.data_size = max(0.1, self.data_size)
        # 校验 candidate_type
        if self.candidate_type not in ["cloud_only", "edge_only", "flexible"]:
            self.candidate_type = "flexible"


class TaskGenerator:
    """Task 数据生成器"""
    # 分布比例
    SHORT_TASK_RATIO = 0.8  # 80% 短任务

    # 使用 numpy 的 lognorm 分布
    # candidate_type 分布（目标比例）
    # 调整说明：原 cloud_only 35%，有效动作仅 1 个，加剧动作稀疏问题
    # 调整为 15% 后，70% 的任务有 8-9 个处理器可选
    CLOUD_ONLY_RATIO = CLOUD_ONLY_RATIO
    EDGE_ONLY_RATIO = EDGE_ONLY_RATIO
    FLEXIBLE_RATIO = 1 - CLOUD_ONLY_RATIO - EDGE_ONLY_RATIO
    
    # 触发条件阈值（用于标记现实意义，但不强制约束分配）
    CLOUD_ONLY_COMP_THRESHOLD = 4000   # MI
    CLOUD_ONLY_DATA_THRESHOLD = 150    # MB
    
    # 隐私敏感比例（仅用于特征标记，不影响 candidate_type）
    PRIVACY_SENSITIVE_RATIO = 0.15
    
    def __init__(self, num_tasks: int = 1000, seed: int = 42):
        self.num_tasks = num_tasks
        self.seed = seed
        random.seed(seed)
    
    def _determine_candidate_type(self) -> str:
        """
        根据目标比例分配 candidate_type
        阈值仅用于设置特征属性，不影响类型分配
        """
        # 直接按目标分布比例随机分配
        r = random.random()
        if r < self.CLOUD_ONLY_RATIO:
            return "cloud_only"
        elif r < self.CLOUD_ONLY_RATIO + self.EDGE_ONLY_RATIO:
            return "edge_only"
        else:
            return "flexible"

    def generate(self) -> List[Task]:
        """生成 Task 列表"""
        tasks = []

        for i in range(self.num_tasks):
            # ========== 1. 计算量 ==========
            # Pareto 分布 (x_min=100, alpha=1.25)，采样后裁剪到 [100, 10000]
            # Pareto(x_min, alpha): x = x_min * (1 + U), U ~ Exp(1)
            # x_min = COMPUTATION_RANGE[0]
            # computation = x_min * (1 + np.random.exponential(PARETO_ALPHA))
            # computation = max(x_min, min(COMPUTATION_RANGE[1], computation))

            # 修订：使用逆变换采样
            # Pareto 分布 (x_min=100, alpha=1.25)，使用标准逆变换采样
            # 正确公式: X = x_min / U^(1/α), 其中 U ~ Uniform(0,1)
            x_min = COMPUTATION_RANGE[0]
            u = np.random.uniform(0, 1)
            computation = x_min / (u ** (1.0 / PARETO_ALPHA))

            # 仅保留下界约束（帕累托天然满足 x >= x_min）
            # 上界建议用拒绝采样或业务层兜底，而非硬截断
            computation = max(x_min, computation)

            # 如果业务强制要求上界，改用拒绝采样避免边界堆积：
            while computation > COMPUTATION_RANGE[1]:
                u = np.random.uniform(0, 1)
                computation = x_min / (u ** (1.0 / PARETO_ALPHA))

            # ========== 2. 数据大小 ==========
            # Lognormal 分布 (μ=3.5, σ=0.5)，均值约 50MB
            data_size = np.random.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA)
            data_size = max(DATA_SIZE_RANGE[0], min(DATA_SIZE_RANGE[1], data_size))

            # ========== 3. 隐私敏感标记 ==========
            privacy_sensitive = random.random() < self.PRIVACY_SENSITIVE_RATIO

            # ========== 4. 候选类型 ==========
            candidate_type = self._determine_candidate_type()

            task = Task(
                task_id=i,
                computation=computation,
                data_size=data_size,
                privacy_sensitive=privacy_sensitive,
                candidate_type=candidate_type
            )
            tasks.append(task)

        return tasks
    
    def save_to_file(self, tasks: List[Task], filepath: str):
        """保存到文件"""
        data = {
            "num_tasks": len(tasks),
            "seed": self.seed,
            "config": {
                "cloud_only_ratio": self.CLOUD_ONLY_RATIO,
                "edge_only_ratio": self.EDGE_ONLY_RATIO,
                "flexible_ratio": self.FLEXIBLE_RATIO,
                "cloud_only_comp_threshold": self.CLOUD_ONLY_COMP_THRESHOLD,
                "cloud_only_data_threshold": self.CLOUD_ONLY_DATA_THRESHOLD,
                "privacy_sensitive_ratio": self.PRIVACY_SENSITIVE_RATIO
            },
            "tasks": [asdict(t) for t in tasks]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def generate_task_pool(num_tasks: int = 1000, output_dir: str = "dataset"):
    """生成 Task 池"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    gen = TaskGenerator(num_tasks=num_tasks)
    tasks = gen.generate()
    
    filepath = os.path.join(output_dir, "task_pool.json")
    gen.save_to_file(tasks, filepath)
    
    print(f"Task 池已生成: {filepath}")
    print(f"  数量: {len(tasks)}")
    
    # 统计
    comps = [t.computation for t in tasks]
    sizes = [t.data_size for t in tasks]
    print(f"  计算量范围: {min(comps):.1f} - {max(comps):.1f} MI")
    print(f"  数据大小范围: {min(sizes):.1f} - {max(sizes):.1f} MB")
    print(f"  计算量中位数: {sorted(comps)[len(comps)//2]:.1f} MI")
    print(f"  数据大小中位数: {sorted(sizes)[len(sizes)//2]:.1f} MB")
    
    # candidate_type 统计
    type_counts = {"cloud_only": 0, "edge_only": 0, "flexible": 0}
    for t in tasks:
        type_counts[t.candidate_type] += 1
    print(f"  candidate_type 分布:")
    print(f"    cloud_only: {type_counts['cloud_only']} ({100*type_counts['cloud_only']/len(tasks):.1f}%)")
    print(f"    edge_only: {type_counts['edge_only']} ({100*type_counts['edge_only']/len(tasks):.1f}%)")
    print(f"    flexible: {type_counts['flexible']} ({100*type_counts['flexible']/len(tasks):.1f}%)")
    
    # privacy_sensitive 统计
    privacy_count = sum(1 for t in tasks if t.privacy_sensitive)
    print(f"  privacy_sensitive: {privacy_count} ({100*privacy_count/len(tasks):.1f}%)")
    
    return tasks


if __name__ == "__main__":
    generate_task_pool(1000, "dags")