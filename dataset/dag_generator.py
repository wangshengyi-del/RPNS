"""
DAG 生成器
基于 Task 池构建 DAG 结构
"""

import random
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Set


@dataclass
class DAGTaskInfo:
    """DAG 中的任务（引用原始 Task 属性）"""
    task_id: int          # 在 DAG 中的 ID
    source_task_id: int   # 原始 Task 池中的 ID
    computation: float    # 计算量 (MI)
    data_size: float      # 数据大小 (MB)
    parents: List[int]    # 父任务 ID 列表
    children: List[int]   # 子任务 ID 列表
    candidate_type: str   # 处理器类型


class DAGGenerator:
    """DAG 结构生成器"""
    
    # 基于真实数据统计
    NODE_COUNT_DIST = {
        'simple': (6, 15),      # 60%
        'complex': (16, 40),    # 30%
        'large': (41, 60)       # 10%
    }
    
    STRUCTURE_DIST = {
        'chain': 0.2,    # 链状
        'tree': 0.4,     # 树状
        'mesh': 0.4      # 网状
    }
    
    DENSITY = 0.95  # 边密度（提高以增加并行性）
    
    def __init__(
        self,
        task_pool: List[Dict],  # Task 池
        complexity: str = "mixed",
        structure: str = "mixed",
        seed: int = 42
    ):
        self.task_pool = task_pool
        self.complexity = complexity
        self.structure = structure
        self.seed = seed
        random.seed(seed)
    
    def _select_node_count(self) -> int:
        """根据复杂度选择节点数"""
        if self.complexity == "simple":
            return random.randint(*self.NODE_COUNT_DIST['simple'])
        elif self.complexity == "complex":
            return random.randint(*self.NODE_COUNT_DIST['complex'])
        elif self.complexity == "large":
            return random.randint(*self.NODE_COUNT_DIST['large'])
        else:  # mixed
            r = random.random()
            if r < 0.5:
                return random.randint(*self.NODE_COUNT_DIST['simple'])
            elif r < 0.8:
                return random.randint(*self.NODE_COUNT_DIST['complex'])
            else:
                return random.randint(*self.NODE_COUNT_DIST['large'])
    
    def _build_structure(self, num_nodes: int) -> Dict[int, List[int]]:
        """构建 DAG 依赖结构"""
        if self.structure == "mixed":
            struct_type = random.choices(
                list(self.STRUCTURE_DIST.keys()),
                weights=list(self.STRUCTURE_DIST.values())
            )[0]
        else:
            struct_type = self.structure
        
        # 按层分配节点
        num_layers = max(3, num_nodes // 4)
        nodes_per_layer = num_nodes // num_layers
        layer_assignment = []
        
        for layer in range(num_layers):
            count = nodes_per_layer
            layer_assignment.extend([layer] * count)
        
        # 剩余节点放到最后一层
        while len(layer_assignment) < num_nodes:
            layer_assignment.append(num_layers - 1)
        
        # 随机打乱但保持层数
        indices = list(range(num_nodes))
        random.shuffle(indices)
        
        # 构建依赖边
        edges = {i: [] for i in range(num_nodes)}
        
        if struct_type == "chain":
            # 链状：每个节点最多一个父节点
            for i in range(1, num_nodes):
                if random.random() < 0.7:
                    parent = random.randint(0, i-1)
                    edges[parent].append(i)
        
        elif struct_type == "tree":
            # 树状：多分支，但无双向汇合
            for i in range(1, num_nodes):
                num_parents = random.randint(0, min(2, i))
                if num_parents > 0:
                    parents = random.sample(range(i), num_parents)
                    for p in parents:
                        edges[p].append(i)
        
        else:  # mesh
            # 网状：可能出现多节点汇合
            for i in range(1, num_nodes):
                for j in range(i):
                    if layer_assignment[i] > layer_assignment[j]:
                        if random.random() < self.DENSITY * 0.3:
                            edges[j].append(i)
        
        return edges
    
    def generate(self) -> List[DAGTaskInfo]:
        """生成 DAG"""
        num_nodes = self._select_node_count()
        
        # 从 Task 池中随机选取任务
        selected_tasks = random.sample(self.task_pool, num_nodes)
        
        # 构建依赖结构
        edges = self._build_structure(num_nodes)
        
        # 创建 DAG 任务
        dag_tasks = []
        task_map = {}  # source_task_id -> dag_task_id
        
        for i, source_task in enumerate(selected_tasks):
            # 收集父节点
            parents = edges.get(i, [])
            
            task = DAGTaskInfo(
                task_id=i,
                source_task_id=source_task['task_id'],
                computation=source_task['computation'],
                data_size=source_task['data_size'],
                candidate_type=source_task['candidate_type'],
                parents=parents,
                children=[]
            )
            dag_tasks.append(task)
            task_map[source_task['task_id']] = i
        
        # 更新 children
        for task in dag_tasks:
            for parent_id in task.parents:
                dag_tasks[parent_id].children.append(task.task_id)
        
        return dag_tasks


def generate_dag_dataset(
    task_pool: List[Dict],
    num_val: int = 200,
    num_test: int = 500,
    output_dir: str = "dataset",
    complexity: str = "mixed"
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
    
    dags = []
    
    # 训练集：动态生成（每次epoch新拓扑）
    val_count = num_val
    test_count = num_test
    
    splits = {
        'val': val_count,
        'test': test_count
    }
    
    for split, count in splits.items():
        for i in range(count):
            seed = hash(f"{split}_{i}") % 10000
            gen = DAGGenerator(
                task_pool=task_pool,
                complexity=complexity,
                seed=seed
            )
            dag_tasks = gen.generate()
            
            dag_id = f"{split}_{i:04d}"
            
            data = {
                "dag_id": dag_id,
                "split": split,
                "num_tasks": len(dag_tasks),
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
                    for t in dag_tasks
                ]
            }
            
            filepath = os.path.join(output_dir, f"dag_{dag_id}.json")
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            dags.append({
                "dag_id": dag_id,
                "split": split,
                "num_tasks": len(dag_tasks)
            })
    
    # 保存索引
    index_file = os.path.join(output_dir, "dag_dataset.json")
    with open(index_file, 'w') as f:
        json.dump(dags, f, indent=2)
    
    # 统计
    print("DAG 数据集生成完成!")
    print(f"  总数: {len(dags)}")
    print(f"  验证集: {sum(1 for d in dags if d['split'] == 'val')}")
    print(f"  测试集: {sum(1 for d in dags if d['split'] == 'test')}")
    
    return dags


if __name__ == "__main__":
    # 1. 生成 Task 池
    from task_generator import generate_task_pool
    task_pool = generate_task_pool(1000, "dags")
    task_pool_data = [{"task_id": t.task_id, "computation": t.computation, "data_size": t.data_size, "candidate_type": t.candidate_type} for t in task_pool]
    
    # 2. 生成 DAG 数据集
    generate_dag_dataset(task_pool_data, 2000, 1000, "dataset/dags")