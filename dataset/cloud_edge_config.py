"""
云边环境配置文件
一云多边：1 云服务器 + 8 边缘服务器
与研究方案 5.1.5 对齐
"""

from dataclasses import dataclass
from typing import List
import random


@dataclass
class CloudServer:
    """云服务器配置（方案 5.1.5）"""
    proc_id: int = 0
    name: str = "cloud-0"
    mips: float = 5000.0          # CPU 计算能力 (MIPS)
    memory_gb: float = 32.0       # 内存 (GB)
    bandwidth_mbps: float = 100.0  # 带宽 (Mbps) (云↔边)
    power_w: float = 300.0         # 能耗 (W)
    power_per_mips: float = 0.06   # 每 MIPS 能耗
    power_per_mbps: float = 0.01   # 每 Mbps 通信能耗
    comm_latency_ms: float = 20.0  # 到边缘的延迟
    proc_type: str = "cloud"


@dataclass
class EdgeServer:
    """边缘服务器配置（方案 5.1.5）"""
    proc_id: int
    name: str
    mips: float                    # CPU 计算能力 (MIPS)，带微小差异 ±20%
    memory_gb: float = 8.0         # 内存 (GB)
    bandwidth_mbps: float = 200.0  # 带宽 (Mbps) (边↔边)
    power_w: float = 20.0          # 能耗 (W) (±20%)
    power_per_mips: float = 0.04   # 每 MIPS 能耗（边缘更高能效比）
    power_per_mbps: float = 0.02   # 每 Mbps 通信能耗
    comm_latency_ms: float = 2.0   # 边-边延迟
    proc_type: str = "edge"


class CloudEdgeEnvironment:
    """云边环境配置管理器
    
    与方案 5.1.5 对齐：
    - 云服务器: 5000 MIPS, 100 Mbps, 300 W
    - 边缘服务器: 400-600 MIPS (±20%), 200 Mbps, 16-24 W (±20%)
    """
    
    # 默认配置：1 云 + 8 边
    NUM_CLOUD = 1
    NUM_EDGE = 8
    
    # 方案 5.1.5 参数
    CLOUD_MIPS = 5000
    CLOUD_BANDWIDTH = 100  # Mbps (云↔边)
    CLOUD_POWER = 300  # W
    
    EDGE_MIPS_BASE = 500  # MIPS
    EDGE_MIPS_RANGE = (400, 600)  # ±20%
    EDGE_BANDWIDTH = 200  # Mbps (边↔边)
    EDGE_POWER_BASE = 20  # W
    EDGE_POWER_RANGE = (16, 24)  # ±20%
    
    def __init__(self, edge_variance: float = 0.2):
        self.edge_variance = edge_variance
        self.processors = self._generate_processors()
    
    def _generate_processors(self) -> List:
        """生成处理器列表（与方案 5.1.5 对齐）"""
        processors = []
        
        # 云服务器
        cloud = CloudServer()
        processors.append(cloud)
        
        # 边缘服务器（带性能差异 ±20%）
        for i in range(self.NUM_EDGE):
            mips_variance = random.uniform(*self.EDGE_MIPS_RANGE)
            power_variance = random.uniform(*self.EDGE_POWER_RANGE)
            
            edge = EdgeServer(
                proc_id=i + 1,
                name=f"edge-{i}",
                mips=mips_variance,
                power_w=power_variance
            )
            processors.append(edge)
        
        return processors
    
    def get_cloud(self):
        return self.processors[0]
    
    def get_edges(self):
        return self.processors[1:]
    
    def get_all(self):
        return self.processors
    
    def get_processor_by_name(self, name: str):
        """根据名称获取处理器"""
        for p in self.processors:
            if p.name == name:
                return p
        return None
    
    def summary(self):
        """打印环境摘要"""
        cloud = self.get_cloud()
        edges = self.get_edges()
        
        print("=" * 60)
        print("云边环境配置（方案 5.1.5 对齐）")
        print("=" * 60)
        print(f"\n【云服务器】")
        print(f"  数量: {self.NUM_CLOUD}")
        print(f"  CPU: {cloud.mips} MIPS")
        print(f"  内存: {cloud.memory_gb} GB")
        print(f"  带宽: {cloud.bandwidth_mbps} Mbps (云↔边)")
        print(f"  能耗: {cloud.power_w} W")
        
        print(f"\n【边缘服务器】")
        print(f"  数量: {self.NUM_EDGE}")
        mips_list = [e.mips for e in edges]
        power_list = [e.power_w for e in edges]
        avg_mips = sum(mips_list) / len(mips_list)
        avg_power = sum(power_list) / len(power_list)
        print(f"  CPU范围: {min(mips_list):.0f} - {max(mips_list):.0f} MIPS")
        print(f"  平均 CPU: {avg_mips:.1f} MIPS")
        print(f"  内存: {edges[0].memory_gb} GB")
        print(f"  带宽: {edges[0].bandwidth_mbps} Mbps (边↔边)")
        print(f"  功率范围: {min(power_list):.1f} - {max(power_list):.1f} W")
        print(f"  平均能耗: {avg_power:.1f} W")
        
        print(f"\n【性能差异】")
        print(f"  CPU: 云是边的 {cloud.mips/avg_mips:.1f}x")
        print(f"  带宽: 云是边的 {cloud.bandwidth_mbps/edges[0].bandwidth_mbps:.1f}x")
        print(f"  能耗: 云是边的 {cloud.power_w/avg_power:.1f}x")
        print("=" * 60)


# 通信延迟矩阵
COMM_LATENCY = {
    ("cloud", "cloud"): 5,      # ms
    ("edge", "edge"): 2,        # ms
    ("edge", "cloud"): 20,      # ms
    ("cloud", "edge"): 20,     # ms
}


def get_comm_latency(proc_type1: str, proc_type2: str) -> float:
    """获取两个处理器之间的通信延迟"""
    return COMM_LATENCY.get((proc_type1, proc_type2), 5)


if __name__ == "__main__":
    env = CloudEdgeEnvironment()
    env.summary()