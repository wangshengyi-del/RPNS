import os
import json
from model.constants import NUM_CLOUD, NUM_EDGE, EDGE_POWER, EDGE_MIPS, BANDWIDTH_MATRIX

def create_simplified_devices(seed: int = 42):
    """
    创建简化设备配置（1云+8边，研究方案5.1.5标准配置）

    云服务器: 5000 MIPS, 100 Mbps, 300 W
    边缘服务器: 500 MIPS (±20%), 200 Mbps, 20 W (±20%)
    """
    import random

    random.seed(seed)
    # 云端处理器 (index 0)
    cloud = [
        {
            "name": "cloud_0",
            "position": 0,
            "cloud": "cloud",
            "cpu": 1500,
            "bandwidth": BANDWIDTH_MATRIX.get((0, 1)),
            "power": 100
        },
        {
            "name": "cloud_1",
            "position": 1,
            "cloud": "cloud",
            "cpu": 2000,
            "bandwidth": BANDWIDTH_MATRIX.get((0, 1)),
            "power": 120
        },
        {
            "name": "cloud_2",
            "position": 2,
            "cloud": "cloud",
            "cpu": 2500,
            "bandwidth": BANDWIDTH_MATRIX.get((0, 1)),
            "power": 140
        }
    ]

    # 边缘处理器 (index 1-8)
    # CPU: 500 MIPS (±20%) = 400-600
    # Power: 20 W (±20%) = 16-24 W
    edge_cpus = [1000, 850, 700, 550, 400, 300, 250, 200]

    edges = []
    for i in range(NUM_EDGE):
        edge_cpu = random.uniform(EDGE_MIPS[0], EDGE_MIPS[1])  # ±20%
        edge_power = random.uniform(EDGE_POWER[0], EDGE_POWER[1])  # ±20%
        # edge_cpu = edge_cpus[i]
        # edge_power = random.uniform(EDGE_POWER[0], EDGE_POWER[1]) * edge_cpu / 500
        edge = {
            "name": f"edge_{i}",
            "position": i + NUM_CLOUD,
            "cloud": "edge",
            "cpu": edge_cpu,
            "bandwidth": BANDWIDTH_MATRIX.get((0, 1)),
            "power": edge_power
        }
        edges.append(edge)

    cloud.extend(edges)
    filepath = os.path.join("device", "device.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(cloud, f, indent=2)


if __name__ == "__main__":
    create_simplified_devices(seed=42)