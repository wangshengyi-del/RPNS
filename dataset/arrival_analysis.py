"""
从 Alibaba batch_task.csv 提取真实到达间隔分布
按昼夜分段统计
"""

import csv
from collections import defaultdict
from datetime import datetime
import statistics


def get_hour(timestamp: int) -> int:
    """将 timestamp 转换为小时 (0-23)"""
    # Alibaba 数据的时间戳可能是秒或微秒
    # 假设是秒（需要根据实际数据调整）
    return (timestamp // 3600) % 24


def extract_job_arrivals(csv_file: str, limit: int = None):
    """提取 job 级别的到达时间"""
    
    # 按 job 统计：每个 job 的最早 start_time
    job_start_times = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if len(row) < 6:
                continue
            try:
                job_name = row[2]
                start_time = int(row[5])
                job_start_times[job_name].append(start_time)
                count += 1
                if limit and count >= limit:
                    break
            except (ValueError, IndexError):
                continue
    
    # 取每个 job 的最早开始时间
    earliest_times = []
    for job, times in job_start_times.items():
        earliest_times.append(min(times))
    
    earliest_times.sort()
    
    return earliest_times


def analyze_arrival_patterns(times: list):
    """分析到达模式"""
    
    # 全天统计
    intervals = []
    for i in range(1, len(times)):
        interval = times[i] - times[i-1]
        if interval > 0 and interval < 3600 * 24:  # 排除异常
            intervals.append(interval)
    
    intervals.sort()
    
    print("=" * 60)
    print("Job 到达间隔统计（整体）")
    print("=" * 60)
    print(f"总 job 数: {len(times)}")
    print(f"有效间隔数: {len(intervals)}")
    print()
    print(f"平均间隔: {statistics.mean(intervals):.1f} 秒 ({statistics.mean(intervals)/60:.1f} 分钟)")
    print(f"中位数间隔: {statistics.median(intervals):.1f} 秒 ({statistics.mean(intervals)/60:.1f} 分钟)")
    print(f"P90: {intervals[int(len(intervals)*0.9)]:.1f} 秒")
    print(f"P95: {intervals[int(len(intervals)*0.95)]:.1f} 秒")
    print(f"范围: {min(intervals):.1f} - {max(intervals):.1f} 秒")
    
    return intervals


def analyze_by_period(times: list):
    """按时段分析（昼夜规律）"""
    
    # 分类：白天 9-18点，晚间其他
    day_times = []   # 9:00-18:00
    night_times = [] # 18:00-9:00
    
    for t in times:
        hour = get_hour(t)
        if 9 <= hour < 18:
            day_times.append(t)
        else:
            night_times.append(t)
    
    day_times.sort()
    night_times.sort()
    
    # 计算各时段间隔
    def calc_intervals(ts):
        intervals = []
        for i in range(1, len(ts)):
            iv = ts[i] - ts[i-1]
            if 0 < iv < 3600 * 24:
                intervals.append(iv)
        return intervals
    
    day_intervals = calc_intervals(day_times)
    night_intervals = calc_intervals(night_times)
    
    print()
    print("=" * 60)
    print("昼夜到达间隔对比")
    print("=" * 60)
    
    print(f"\n【白天】(9:00-18:00)  job数: {len(day_times)}")
    if day_intervals:
        print(f"  平均间隔: {statistics.mean(day_intervals):.1f} 秒 ({statistics.mean(day_intervals)/60:.1f} 分钟)")
        print(f"  中位数: {statistics.median(day_intervals):.1f} 秒")
    
    print(f"\n【晚间】(18:00-9:00)  job数: {len(night_times)}")
    if night_intervals:
        print(f"  平均间隔: {statistics.mean(night_intervals):.1f} 秒 ({statistics.mean(night_intervals)/60:.1f} 分钟)")
        print(f"  中位数: {statistics.median(night_intervals):.1f} 秒")
    
    # 计算比例
    if day_intervals and night_intervals:
        day_mean = statistics.mean(day_intervals)
        night_mean = statistics.mean(night_intervals)
        print(f"\n对比: 白天到达率是晚间的 {night_mean/day_mean:.1f}x")
    
    return day_intervals, night_intervals


def save_distribution(intervals: list, filename: str):
    """保存间隔分布供生成器使用"""
    import json
    
    intervals_sorted = sorted(intervals)
    n = len(intervals_sorted)
    
    data = {
        "count": n,
        "min": min(intervals),
        "max": max(intervals),
        "mean": statistics.mean(intervals),
        "median": statistics.median(intervals),
        "p90": intervals_sorted[int(n*0.9)],
        "p95": intervals_sorted[int(n*0.95)],
        "percentiles": {
            "10": intervals_sorted[int(n*0.1)],
            "25": intervals_sorted[int(n*0.25)],
            "50": intervals_sorted[int(n*0.5)],
            "75": intervals_sorted[int(n*0.75)],
            "90": intervals_sorted[int(n*0.9)],
            "95": intervals_sorted[int(n*0.95)],
            "99": intervals_sorted[int(n*0.99)]
        },
        "sample_intervals": intervals_sorted[:100]  # 前100个样本
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n分布数据已保存: {filename}")


if __name__ == "__main__":
    csv_file = "/Users/pingtaiyanjiusuo/调度/data/batch_task.csv"
    
    print("正在提取 job 到达时间...")
    times = extract_job_arrivals(csv_file, limit=500000)
    print(f"提取到 {len(times)} 个 job")
    
    # 整体统计
    intervals = analyze_arrival_patterns(times)
    
    # 昼夜分析
    day_intervals, night_intervals = analyze_by_period(times)
    
    # 保存
    save_distribution(intervals, "arrival_intervals.json")
    
    if day_intervals:
        save_distribution(day_intervals, "arrival_intervals_day.json")
    if night_intervals:
        save_distribution(night_intervals, "arrival_intervals_night.json")