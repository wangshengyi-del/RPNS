"""
简化版调度仿真器（测试用）
"""

import random
import json
import math
import os


def run_simple_simulation():
    """简化版仿真测试"""
    
    # 加载 DAG 池
    dag_pool_file = "/Users/pingtaiyanjiusuo/.openclaw/workspace-sci-writer/dataset/dataset/dags/dag_dataset.json"
    with open(dag_pool_file, 'r') as f:
        dag_pool_idx = json.load(f)
    
    dags_dir = os.path.dirname(dag_pool_file)
    dag_pool = []
    for dag_info in dag_pool_idx:
        dag_file = os.path.join(dags_dir, f"dag_{dag_info['dag_id']}.json")
        with open(dag_file, 'r') as f:
            dag_pool.append(json.load(f))
    
    print(f"DAG 池大小: {len(dag_pool)}")
    
    # 参数
    arrival_rate = 0.1  # jobs/秒
    seed = 42
    max_time = 60.0  # 1分钟
    
    random.seed(seed)
    
    # 模拟到达
    arrived_dags = []
    t = 0.0
    arrival_count = 0
    
    print(f"\n模拟 DAG 到达 (seed={seed}, rate={arrival_rate}/s):")
    
    while t < max_time:
        # 泊松过程间隔
        interval = -math.log(random.random()) / arrival_rate
        t += interval
        
        if t > max_time:
            break
        
        # 有放回抽取
        dag = random.choice(dag_pool)
        arrival_count += 1
        arrived_dags.append((t, dag['dag_id'], dag['num_tasks']))
        
        if arrival_count <= 5:
            print(f"  时间 {t:.1f}s: DAG {dag['dag_id']} ({dag['num_tasks']} 任务)")
    
    print(f"\n1分钟内共到达 {arrival_count} 个 DAG")
    print(f"平均间隔: {max_time / arrival_count:.1f} 秒")
    
    # 简单统计
    total_tasks = sum(d[2] for d in arrived_dags)
    print(f"总任务数: {total_tasks}")
    
    return {
        "arrival_count": arrival_count,
        "total_tasks": total_tasks,
        "avg_interval": max_time / arrival_count
    }


if __name__ == "__main__":
    result = run_simple_simulation()
    print(f"\n结果: {result}")