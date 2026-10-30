"""
DAG Job 生成程序
生成 job 数据集（job_id, dag_id, 到达时间）
保证可重复性
"""

import random
import json
import math
import os
from datetime import datetime


def generate_job_dataset(
    num_jobs: int,
    dag_pool_file: str,
    arrival_rate: float = 0.1,
    seed: int = 42,
    output_file: str = None
) -> list:
    """
    生成 job 数据集
    
    Args:
        num_jobs: 需要的 job 数量
        dag_pool_file: DAG 池索引文件路径
        arrival_rate: 到达率 (jobs/秒)
        seed: 随机种子（保证可重复性）
        output_file: 输出文件路径
    
    Returns:
        job_list: [{"job_id": int, "dag_id": str, "arrival_time": float}, ...]
    """
    
    # 固定随机种子
    random.seed(seed)
    
    # 加载 DAG 池
    with open(dag_pool_file, 'r') as f:
        dag_pool = json.load(f)
    
    print(f"DAG 池大小: {len(dag_pool)}")
    
    # 生成 jobs
    jobs = []
    current_time = 0.0
    job_id = 0
    
    print(f"生成 {num_jobs} 个 jobs...")
    
    while len(jobs) < num_jobs:
        # 泊松过程：下一个到达间隔
        interval = -math.log(random.random()) / arrival_rate
        current_time += interval
        
        # 有放回抽取 DAG
        dag_info = random.choice(dag_pool)
        
        # 创建 job
        job = {
            "job_id": job_id,
            "dag_id": dag_info["dag_id"],
            "arrival_time": round(current_time, 2),
            "num_tasks": dag_info["num_tasks"]
        }
        jobs.append(job)
        
        job_id += 1
        
        # 打印进度
        if (len(jobs) % 10 == 0) or (len(jobs) == num_jobs):
            print(f"  已生成 {len(jobs)}/{num_jobs} jobs, 当前时间 {current_time:.1f}s")
    
    # 统计
    print(f"\n生成完成!")
    print(f"  总 jobs: {len(jobs)}")
    print(f"  总时间跨度: {jobs[-1]['arrival_time']:.1f} 秒")
    print(f"  平均到达间隔: {jobs[-1]['arrival_time'] / len(jobs):.2f} 秒")
    
    # 统计 DAG 分布
    dag_counts = {}
    for job in jobs:
        dag_id = job["dag_id"]
        split = dag_id.rsplit("_", 1)[0]  # train/val/test
        dag_counts[split] = dag_counts.get(split, 0) + 1
    
    print(f"  DAG 分布: {dag_counts}")
    
    # 保存
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(jobs, f, indent=2)
        print(f"\n已保存到: {output_file}")
    
    return jobs


def verify_reproducibility(
    dag_pool_file: str,
    num_jobs: int = 10,
    arrival_rate: float = 0.1
):
    """验证可重复性：两次运行应产生相同结果"""
    
    print("=" * 50)
    print("验证可重复性")
    print("=" * 50)
    
    jobs1 = generate_job_dataset(
        num_jobs=num_jobs,
        dag_pool_file=dag_pool_file,
        arrival_rate=arrival_rate,
        seed=42,
        output_file=None
    )
    
    print("\n--- 第二次运行 (相同 seed) ---")
    
    jobs2 = generate_job_dataset(
        num_jobs=num_jobs,
        dag_pool_file=dag_pool_file,
        arrival_rate=arrival_rate,
        seed=42,
        output_file=None
    )
    
    # 对比
    identical = all(
        j1["job_id"] == j2["job_id"] and
        j1["dag_id"] == j2["dag_id"] and
        abs(j1["arrival_time"] - j2["arrival_time"]) < 0.01
        for j1, j2 in zip(jobs1, jobs2)
    )
    
    print(f"\n结果: {'✓ 两次运行完全一致' if identical else '✗ 结果不一致'}")
    
    return identical


if __name__ == "__main__":
    import sys
    
    # 参数
    dag_pool_file = "/Users/pingtaiyanjiusuo/.openclaw/workspace-sci-writer/dataset/dataset/dags/dag_dataset.json"
    num_jobs = 100          # 生成 100 个 jobs
    arrival_rate = 0.1       # 到达率 0.1 jobs/秒
    seed = 42                # 固定种子
    output_file = "job_dataset.json"
    
    # 生成
    jobs = generate_job_dataset(
        num_jobs=num_jobs,
        dag_pool_file=dag_pool_file,
        arrival_rate=arrival_rate,
        seed=seed,
        output_file=output_file
    )
    
    # 验证可重复性
    print("\n")
    verify_reproducibility(dag_pool_file, num_jobs=10, arrival_rate=arrival_rate)
    
    # 显示前 5 个 job
    print("\n前 5 个 jobs:")
    for job in jobs[:5]:
        print(f"  {job}")