"""
DAG 结构统计分析脚本
基于预处理后的 DAG dict 数据
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set


def load_dags_from_csv(csv_file: str) -> List[Dict]:
    """从 CSV 文件加载 DAG 数据"""
    import ast
    import re
    
    dags = []
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 提取第一列：可能有双引号包裹
            # 格式1: {'1': []},['1']  (无引号)
            # 格式2: "{'2': [], '1': ['2']}",['1']  (有引号)
            
            # 使用正则提取 dict 部分
            # 匹配 {...} 部分
            match = re.match(r'^(["\']?)({.*})\1(,.*)?$', line)
            if match:
                dag_str = match.group(2)
            else:
                # 备用：直接取逗号前部分
                parts = line.split(',', 1)
                dag_str = parts[0].strip()
            
            # 去除首尾空格
            dag_str = dag_str.strip()
            
            try:
                dag = ast.literal_eval(dag_str)
                # 清理：确保所有键和值都是字符串
                cleaned_dag = {}
                for k, v in dag.items():
                    cleaned_dag[str(k)] = [str(p) for p in v]
                dags.append(cleaned_dag)
            except Exception as e:
                # 尝试替换单引号为双引号
                try:
                    dag_str_fixed = dag_str.replace("'", '"')
                    dag = json.loads(dag_str_fixed)
                    cleaned_dag = {str(k): [str(p) for p in v] for k, v in dag.items()}
                    dags.append(cleaned_dag)
                except:
                    continue
    
    return dags


def analyze_dag(dag: Dict) -> Dict:
    """分析单个 DAG 的特征"""
    # dag 格式: {node_id: [parent_ids]}
    nodes = set(dag.keys())
    all_nodes = set()
    for parents in dag.values():
        all_nodes.update(parents)
    nodes.update(all_nodes)
    
    num_nodes = len(nodes)
    
    # 统计入度
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)
    for node, parents in dag.items():
        out_degrees[node] = len(parents)
        for p in parents:
            in_degrees[node] += 1
    
    avg_in_degree = sum(in_degrees.values()) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    
    # 统计源节点和汇节点
    source_nodes = [n for n in nodes if in_degrees.get(n, 0) == 0]
    sink_nodes = [n for n in nodes if out_degrees.get(n, 0) == 0]
    
    # 关键路径长度（简化估算）
    critical_path_len = len(source_nodes) + len(sink_nodes)
    
    # 汇合节点（入度 > 1）
    bottleneck_nodes = sum(1 for d in in_degrees.values() if d > 1)
    
    return {
        'num_nodes': num_nodes,
        'num_edges': sum(out_degrees.values()),
        'avg_in_degree': avg_in_degree,
        'avg_out_degree': avg_out_degree,
        'num_sources': len(source_nodes),
        'num_sinks': len(sink_nodes),
        'bottleneck_nodes': bottleneck_nodes,
        'density': sum(out_degrees.values()) / num_nodes if num_nodes > 0 else 0
    }


def compute_critical_path(dag: Dict) -> int:
    """计算关键路径（最长路径）"""
    nodes = set(dag.keys())
    for parents in dag.values():
        nodes.update(parents)
    
    # 拓扑排序
    in_degree = defaultdict(int)
    for node in nodes:
        for parent in dag.get(node, []):
            in_degree[node] += 1
    
    queue = [n for n in nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for child, parents in dag.items():
            if node in parents:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    
    # DP 计算最长路径
    max_dist = {n: 1 for n in nodes}
    for node in topo_order:
        for parent in dag.get(node, []):
            if parent in max_dist:
                max_dist[node] = max(max_dist[node], max_dist[parent] + 1)
    
    return max(max_dist.values()) if max_dist else 0


def generate_report(dags: List[Dict], output_file: str = None):
    """生成统计报告"""
    if not dags:
        print("没有找到 DAG 数据")
        return
    
    # 分析每个 DAG
    results = []
    for dag in dags:
        stats = analyze_dag(dag)
        stats['critical_path'] = compute_critical_path(dag)
        results.append(stats)
    
    # 汇总统计
    num_dags = len(results)
    
    # 节点数分布
    node_counts = [r['num_nodes'] for r in results]
    node_dist = defaultdict(int)
    for n in node_counts:
        if n == 1:
            node_dist['1'] += 1
        elif n <= 5:
            node_dist['2-5'] += 1
        elif n <= 10:
            node_dist['6-10'] += 1
        elif n <= 20:
            node_dist['11-20'] += 1
        elif n <= 50:
            node_dist['21-50'] += 1
        else:
            node_dist['50+'] += 1
    
    # 关键路径分布
    cp_lengths = [r['critical_path'] for r in results]
    cp_dist = defaultdict(int)
    for cp in cp_lengths:
        if cp <= 3:
            cp_dist['1-3'] += 1
        elif cp <= 5:
            cp_dist['4-5'] += 1
        elif cp <= 10:
            cp_dist['6-10'] += 1
        else:
            cp_dist['10+'] += 1
    
    # 计算平均值
    avg_nodes = sum(node_counts) / num_dags
    avg_edges = sum(r['num_edges'] for r in results) / num_dags
    avg_in_deg = sum(r['avg_in_degree'] for r in results) / num_dags
    avg_out_deg = sum(r['avg_out_degree'] for r in results) / num_dags
    avg_density = sum(r['density'] for r in results) / num_dags
    avg_cp = sum(cp_lengths) / num_dags
    avg_bottleneck = sum(r['bottleneck_nodes'] for r in results) / num_dags
    
    # 打印报告
    print("=" * 60)
    print("DAG 结构统计分析报告")
    print("=" * 60)
    print(f"\n总 DAG 数量: {num_dags}")
    
    print(f"\n【节点数分布】")
    for key in ['1', '2-5', '6-10', '11-20', '21-50', '50+']:
        count = node_dist[key]
        pct = count / num_dags * 100
        print(f"  {key:>6} 节点: {count:>5} 个 ({pct:>5.1f}%)")
    
    print(f"\n【关键路径长度分布】")
    for key in ['1-3', '4-5', '6-10', '10+']:
        count = cp_dist[key]
        pct = count / num_dags * 100
        print(f"  {key:>6}: {count:>5} 个 ({pct:>5.1f}%)")
    
    print(f"\n【统计特征】")
    print(f"  平均节点数: {avg_nodes:.1f}")
    print(f"  平均边数: {avg_edges:.1f}")
    print(f"  平均入度: {avg_in_deg:.2f}")
    print(f"  平均出度: {avg_out_deg:.2f}")
    print(f"  平均密度: {avg_density:.2f}")
    print(f"  平均关键路径: {avg_cp:.1f}")
    print(f"  平均汇合节点数: {avg_bottleneck:.1f}")
    
    print("\n【用于生成器的建议参数】")
    print(f"  - 节点数: 建议 {int(avg_nodes)} 左右，分布参考上面")
    print(f"  - 边密度: {avg_density:.2f}")
    print(f"  - 平均入度: {avg_in_deg:.2f}")
    print(f"  - 平均出度: {avg_out_deg:.2f}")
    print(f"  - 关键路径长度: {avg_cp:.1f}")
    
    # 保存详细结果
    if output_file:
        report = {
            'num_dags': num_dags,
            'node_distribution': dict(node_dist),
            'critical_path_distribution': dict(cp_dist),
            'averages': {
                'num_nodes': avg_nodes,
                'num_edges': avg_edges,
                'avg_in_degree': avg_in_deg,
                'avg_out_degree': avg_out_deg,
                'density': avg_density,
                'critical_path': avg_cp,
                'bottleneck_nodes': avg_bottleneck
            },
            'details': results
        }
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n详细报告已保存到: {output_file}")
    
    return report


if __name__ == "__main__":
    # 数据文件路径
    csv_file = "/Users/pingtaiyanjiusuo/调度/data/workflow_task_data.csv"
    output_file = "/Users/pingtaiyanjiusuo/调度/data/dag_statistics.json"
    
    print("加载 DAG 数据...")
    dags = load_dags_from_csv(csv_file)
    print(f"共加载 {len(dags)} 个 DAG")
    
    print("\n生成统计报告...")
    generate_report(dags, output_file)