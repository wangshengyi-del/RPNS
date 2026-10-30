"""
从 Alibaba batch_task.csv 提取统计特征
用于校准 DAG 生成器参数
"""

import csv
import os
from collections import defaultdict
import statistics


def parse_batch_task(csv_file: str, limit: int = None):
    """解析 batch_task.csv"""
    
    durations = []      # 执行时间
    cpu_requests = []   # CPU 需求
    task_types = []     # 任务类型
    instance_nums = [] # 并行度
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if len(row) < 8:
                continue
            
            try:
                task_name, instance_num, job_name, task_type, status = row[:5]
                start_time, end_time = int(row[5]), int(row[6])
                plan_cpu, plan_mem = float(row[7]), float(row[8]) if len(row) > 8 else 0
                
                # 只统计已终止的任务
                if status == 'Terminated':
                    duration = end_time - start_time
                    if duration > 0:
                        durations.append(duration)
                        cpu_requests.append(plan_cpu)
                        task_types.append(task_type)
                        instance_nums.append(instance_num)
                
                count += 1
                if limit and count >= limit:
                    break
                    
            except (ValueError, IndexError):
                continue
    
    return {
        'durations': durations,
        'cpu_requests': cpu_requests,
        'task_types': task_types,
        'instance_nums': [int(x) for x in instance_nums]
    }


def analyze_stats(data: dict):
    """统计分析"""
    
    durations = data['durations']
    cpu_requests = data['cpu_requests']
    task_types = data['task_types']
    instance_nums = data['instance_nums']
    
    print("=" * 60)
    print("Alibaba Batch Task 统计特征")
    print("=" * 60)
    
    # 执行时间统计
    print(f"\n【执行时间】样本数: {len(durations)}")
    print(f"  平均值: {statistics.mean(durations):.1f} 秒")
    print(f"  中位数: {statistics.median(durations):.1f} 秒")
    print(f"  P90: {sorted(durations)[int(len(durations)*0.9)]:.1f} 秒")
    print(f"  P99: {sorted(durations)[int(len(durations)*0.99)]:.1f} 秒")
    print(f"  最大值: {max(durations):.1f} 秒")
    print(f"  范围: {min(durations):.1f} - {max(durations):.1f} 秒")
    
    # CPU 需求统计
    print(f"\n【CPU 需求】样本数: {len(cpu_requests)}")
    print(f"  平均值: {statistics.mean(cpu_requests):.1f}")
    print(f"  中位数: {statistics.median(cpu_requests):.1f}")
    print(f"  范围: {min(cpu_requests)} - {max(cpu_requests)}")
    
    # CPU 需求分布
    low = sum(1 for x in cpu_requests if x <= 50)
    medium = sum(1 for x in cpu_requests if 50 < x <= 100)
    high = sum(1 for x in cpu_requests if x > 100)
    total = len(cpu_requests)
    print(f"  分布:")
    print(f"    低 (≤50): {low} ({low/total*100:.1f}%)")
    print(f"    中 (50-100): {medium} ({medium/total*100:.1f}%)")
    print(f"    高 (>100): {high} ({high/total*100:.1f}%)")
    
    # 任务类型分布
    print(f"\n【任务类型分布】")
    type_counts = defaultdict(int)
    for t in task_types:
        type_counts[t] += 1
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {count} ({count/len(task_types)*100:.1f}%)")
    
    # 并行度分布
    print(f"\n【并行度 (instance_num)】")
    inst_stats = defaultdict(int)
    for inst in instance_nums:
        if inst == 1:
            inst_stats['1'] += 1
        elif inst <= 5:
            inst_stats['2-5'] += 1
        elif inst <= 10:
            inst_stats['6-10'] += 1
        elif inst <= 50:
            inst_stats['11-50'] += 1
        else:
            inst_stats['50+'] += 1
    for k in ['1', '2-5', '6-10', '11-50', '50+']:
        count = inst_stats.get(k, 0)
        print(f"  {k}: {count} ({count/len(instance_nums)*100:.1f}%)")
    
    # 建议生成器参数
    print("\n" + "=" * 60)
    print("生成器参数建议")
    print("=" * 60)
    print(f"  - 计算时间范围: {min(durations):.0f} - {max(durations):.0f} 秒")
    print(f"  - 建议范围: {min(durations):.0f} - {sorted(durations)[int(len(durations)*0.95)]:.0f} 秒 (P95)")
    print(f"  - CPU 需求中位数: {statistics.median(cpu_requests):.0f}")
    print(f"  - 单实例任务占比: {inst_stats.get('1',0)/len(instance_nums)*100:.1f}%")
    
    return {
        'duration_stats': {
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'p90': sorted(durations)[int(len(durations)*0.9)],
            'p99': sorted(durations)[int(len(durations)*0.99)],
            'min': min(durations),
            'max': max(durations)
        },
        'cpu_stats': {
            'mean': statistics.mean(cpu_requests),
            'median': statistics.median(cpu_requests),
            'low_pct': low/total*100,
            'medium_pct': medium/total*100,
            'high_pct': high/total*100
        }
    }


if __name__ == "__main__":
    csv_file = "/Users/pingtaiyanjiusuo/调度/data/batch_task.csv"
    
    print("正在解析 batch_task.csv (限制前 100000 行)...")
    data = parse_batch_task(csv_file, limit=100000)
    print(f"解析完成，有效任务数: {len(data['durations'])}")
    
    stats = analyze_stats(data)