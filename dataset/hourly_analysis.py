"""
分析 job 到达的小时分布
确定繁忙时间段和空闲时间段
"""

import csv
from collections import defaultdict
import json


def analyze_hourly_distribution(csv_file: str, limit: int = None):
    """按小时统计 job 到达分布"""
    
    # 统计每个小时的 job 数量
    hourly_counts = defaultdict(int)
    timestamps = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if len(row) < 6:
                continue
            try:
                job_name = row[2]
                start_time = int(row[5])
                hour = (start_time // 3600) % 24  # 0-23 小时
                hourly_counts[hour] += 1
                timestamps.append(start_time)
                count += 1
                if limit and count >= limit:
                    break
            except (ValueError, IndexError):
                continue
    
    # 统计
    total = sum(hourly_counts.values())
    hours = sorted(hourly_counts.items(), key=lambda x: x[0])
    
    print("=" * 60)
    print("每小时 Job 到达分布")
    print("=" * 60)
    print(f"总 Job 数: {total}")
    print()
    print(f"{'小时':>6} | {'Job数':>8} | {'占比':>8} | {'繁忙程度':>10}")
    print("-" * 40)
    
    avg_per_hour = total / 24
    
    busy_hours = []
    idle_hours = []
    
    for hour, count in hours:
        pct = count / total * 100
        ratio = count / avg_per_hour
        
        if ratio > 1.2:
            level = "繁忙▲"
            busy_hours.append(hour)
        elif ratio < 0.8:
            level = "空闲▼"
            idle_hours.append(hour)
        else:
            level = "正常"
        
        print(f"{hour:>6}时 | {count:>8} | {pct:>7.1f}% | {level:>10}")
    
    print()
    print(f"平均每小时: {avg_per_hour:.0f} 个 Job")
    print(f"繁忙时间段 (负载>120%): {busy_hours}")
    print(f"空闲时间段 (负载<80%): {idle_hours}")
    
    return hourly_counts, busy_hours, idle_hours


def calculate_interval_by_period(timestamps, busy_hours, idle_hours):
    """按繁忙/空闲时间段计算到达间隔"""
    
    # 按时间排序
    timestamps.sort()
    
    # 分类
    busy_timestamps = []
    idle_timestamps = []
    normal_timestamps = []
    
    for t in timestamps:
        hour = (t // 3600) % 24
        if hour in busy_hours:
            busy_timestamps.append(t)
        elif hour in idle_hours:
            idle_timestamps.append(t)
        else:
            normal_timestamps.append(t)
    
    # 计算间隔
    def calc_intervals(ts):
        if len(ts) < 2:
            return []
        intervals = []
        for i in range(1, len(ts)):
            iv = ts[i] - ts[i-1]
            if 0 < iv < 3600 * 24:  # 排除异常
                intervals.append(iv)
        return intervals
    
    busy_intervals = calc_intervals(busy_timestamps)
    idle_intervals = calc_intervals(idle_timestamps)
    normal_intervals = calc_intervals(normal_timestamps)
    
    print()
    print("=" * 60)
    print("各时间段到达间隔")
    print("=" * 60)
    
    import statistics
    
    if busy_intervals:
        print(f"\n【繁忙时段】 ({len(busy_timestamps)} jobs)")
        print(f"  平均间隔: {statistics.mean(busy_intervals):.1f} 秒")
        print(f"  中位数: {statistics.median(busy_intervals):.1f} 秒")
    
    if normal_intervals:
        print(f"\n【正常时段】 ({len(normal_timestamps)} jobs)")
        print(f"  平均间隔: {statistics.mean(normal_intervals):.1f} 秒")
        print(f"  中位数: {statistics.median(normal_intervals):.1f} 秒")
    
    if idle_intervals:
        print(f"\n【空闲时段】 ({len(idle_timestamps)} jobs)")
        print(f"  平均间隔: {statistics.mean(idle_intervals):.1f} 秒")
        print(f"  中位数: {statistics.median(idle_intervals):.1f} 秒")
    
    # 保存
    result = {
        "busy_hours": busy_hours,
        "idle_hours": idle_hours,
        "busy_intervals": {
            "count": len(busy_intervals),
            "mean": statistics.mean(busy_intervals) if busy_intervals else 0,
            "median": statistics.median(busy_intervals) if busy_intervals else 0
        },
        "idle_intervals": {
            "count": len(idle_intervals),
            "mean": statistics.mean(idle_intervals) if idle_intervals else 0,
            "median": statistics.median(idle_intervals) if idle_intervals else 0
        }
    }
    
    with open("hourly_analysis.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n分析结果已保存: hourly_analysis.json")
    
    return result


if __name__ == "__main__":
    csv_file = "/Users/pingtaiyanjiusuo/调度/data/batch_task.csv"
    
    hourly_counts, busy_hours, idle_hours = analyze_hourly_distribution(csv_file, limit=500000)
    
    # 重新读取 timestamps 计算间隔
    timestamps = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:
                try:
                    timestamps.append(int(row[5]))
                except:
                    pass
    
    calculate_interval_by_period(timestamps, busy_hours, idle_hours)