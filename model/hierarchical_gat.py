"""
分层图注意力调度网络
与 ARCHITECTURE_DESIGN.md 对齐

Layer 1: Intra-DAG GAT - 任务结构编码
Layer 2: Inter-DAG Attention - DAG间协调
Layer 3: Two-Stage Decision - 两阶段决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

from .constants import (
    STATIC_DIM, DAG_META_DIM, DEFAULT_HIDDEN_DIM,
    DEFAULT_NUM_HEADS, DEFAULT_INTRA_LAYERS, DEFAULT_INTER_LAYERS, DEFAULT_DROPOUT,
    ADJ_DIRECTIONS, DEFAULT_ADJ_DIRECTION,
)


def apply_adj_direction(adj: torch.Tensor, mode: str = DEFAULT_ADJ_DIRECTION) -> torch.Tensor:
    """
    按消息传递方向变换 DAG 邻接矩阵。

    环境邻接 adj[i,j]=1 表示依赖边 i→j（父 i，子 j）。
    GAT 中节点 m 仅 attend 到 adj[m,n]=1 的 n。

    down: 父 attend 子（下游影响，默认，与 HEFT upward rank 直觉一致）
    up:   子 attend 父（上游依赖）
    both: 双向边（保留自环）
    """
    if mode not in ADJ_DIRECTIONS:
        raise ValueError(f"adj direction must be one of {ADJ_DIRECTIONS}, got {mode!r}")
    if mode == 'down':
        return adj
    if mode == 'up':
        return adj.transpose(-1, -2).contiguous()
    sym = adj + adj.transpose(-1, -2)
    return sym.clamp(max=1.0)


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（支持单样本和批量）
    实现残差连接 + LayerNorm
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Q, K, V 投影
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # 输出投影
        self.o_proj = nn.Linear(out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.orthogonal_(proj.weight, gain=1.0)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_dim] 或 [batch, num_nodes, in_dim]
            adj: [num_nodes, num_nodes] 或 None

        Returns:
            更新后的节点特征
        """
        is_batch = x.dim() == 3

        if not is_batch:
            x = x.unsqueeze(0)
            if adj is not None:
                adj = adj.unsqueeze(0)

        batch_size, num_nodes, _ = x.shape

        # Q, K, V: [batch, num_nodes, num_heads, head_dim]
        Q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # 注意力分数
        Q = Q.transpose(1, 2)  # [B, H, N, D]
        K = K.transpose(1, 2)  # [B, H, N, D]
        V = V.transpose(1, 2)  # [B, H, N, D]

        # 注意力分数
        scores = torch.einsum('bhmd,bhnd->bhmn', Q, K) / (self.head_dim ** 0.5)

        # 邻接矩阵 mask - 需要正确广播到 [B, H, N, N]
        if adj is not None:
            if adj.dim() == 2:
                # [N, N] -> [1, 1, N, N] -> broadcast to [B, H, N, N]
                adj = adj.unsqueeze(0).unsqueeze(0)
            elif adj.dim() == 3:
                # [B, N, N] -> need [B, H, N, N]
                adj = adj.unsqueeze(1)  # [B, 1, N, N]
            elif adj.dim() == 4:
                # Already [B, H, N, N]
                pass
            # Expand adj to match number of heads
            adj = adj.expand(-1, self.num_heads, -1, -1)  # [B, H, N, N]
            mask = (adj == 0).float() * -1e9
            scores = scores + mask

        # softmax + dropout + 聚合
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_output = torch.einsum('bhmn,bhnd->bhmd', attn, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.out_dim)

        # 输出投影 + 残差连接
        out = self.o_proj(attn_output)
        out = self.norm(x + out)

        if not is_batch:
            out = out.squeeze(0)

        return out


# ============================================================
# Layer 1: Intra-DAG Encoder (任务结构编码)
# ============================================================

class IntraDAGEncoder(nn.Module):
    """
    Layer 1: DAG内编码器

    输入: 每个DAG独立的12维静态结构特征 + 拓扑邻接矩阵
    机制: 两层GAT，提炼任务的拓扑关键度、HEFT静态预估与数据量属性
    输出: 任务结构嵌入 [num_tasks, hidden_dim]

    特点: 时间步内保持不变，无需重复计算
    """

    def __init__(
        self,
        static_dim: int = STATIC_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_INTRA_LAYERS,
        dropout: float = DEFAULT_DROPOUT
    ):
        super().__init__()

        self.static_dim = static_dim
        self.hidden_dim = hidden_dim

        # 任务特征投影（仅接收静态结构特征）
        self.task_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # DAG内GAT层
        self.dag_gnn = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 缓存：无梯度时避免重复编码相同静态DAG
        self._cache = {}

    @staticmethod
    def _fingerprint(feat: torch.Tensor, adj: torch.Tensor) -> tuple:
        return (tuple(feat.shape), tuple(adj.shape),
                feat.sum().item(), adj.sum().item())

    def clear_cache(self):
        self._cache.clear()

    def forward(
        self,
        task_static_features: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            task_static_features: [num_tasks, static_dim] 仅静态结构特征（12维）
            adj: [num_tasks, num_tasks] DAG内部邻接矩阵

        Returns:
            task_embs: [num_tasks, hidden_dim] 任务结构嵌入
        """
        # 1. 任务特征投影
        h = self.task_proj(task_static_features)

        # 2. DAG内GAT消息传递
        for gnn_layer in self.dag_gnn:
            h = gnn_layer(h, adj)

        return h

    def get_task_embeddings(
        self,
        dag_tasks_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        获取多个DAG的任务嵌入列表（逐个处理）

        Args:
            dag_tasks_features: List of [num_tasks_i, static_dim] 每个DAG的静态特征
            dag_adjs: List of [num_tasks_i, num_tasks_i] 每个DAG的邻接矩阵

        Returns:
            List of [num_tasks_i, hidden_dim] 每个DAG的任务嵌入
        """
        task_embs_list = []
        for features, adj in zip(dag_tasks_features, dag_adjs):
            task_emb = self.forward(features, adj)
            task_embs_list.append(task_emb)
        return task_embs_list

    def get_task_embeddings_batch(
        self,
        dag_tasks_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        获取多个DAG的任务嵌入列表（批量并行处理，支持缓存）

        当 torch.no_grad() 时自动缓存，避免采集阶段重复编码相同静态DAG
        """
        if len(dag_tasks_features) == 0:
            return []

        # 无梯度时走缓存（采集阶段）
        if not torch.is_grad_enabled():
            return self._get_task_embeddings_cached(dag_tasks_features, dag_adjs)

        return self._get_task_embeddings_impl(dag_tasks_features, dag_adjs)

    def _get_task_embeddings_cached(
        self,
        dag_tasks_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """带缓存的批量编码：只编码未缓存的DAG，其余直接复用"""
        fps = [self._fingerprint(f, a) for f, a in zip(dag_tasks_features, dag_adjs)]
        device = dag_tasks_features[0].device
        results = []

        for i, fp in enumerate(fps):
            if fp in self._cache:
                results.append((i, self._cache[fp].to(device)))
            else:
                results.append((i, None))

        # 批量编码未缓存的DAG
        uncached_idx = [i for i, (_, emb) in enumerate(results) if emb is None]
        uncached_f = [dag_tasks_features[i] for i in uncached_idx]
        uncached_a = [dag_adjs[i] for i in uncached_idx]

        if uncached_f:
            new_embs = self._get_task_embeddings_impl(uncached_f, uncached_a)
            for idx, emb, fp in zip(uncached_idx, new_embs, [fps[i] for i in uncached_idx]):
                self._cache[fp] = emb.cpu()
                results[idx] = (idx, emb)

        return [r[1] for r in results]

    def _get_task_embeddings_impl(
        self,
        dag_tasks_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """实际批量编码逻辑（padding + GAT forward + 切分）"""
        num_dags = len(dag_tasks_features)
        num_tasks_list = [f.size(0) for f in dag_tasks_features]
        max_num_tasks = max(num_tasks_list)

        # Padding后的批量特征: [num_dags, max_num_tasks, static_dim]
        padded_features = []
        for features in dag_tasks_features:
            num_tasks = features.size(0)
            if num_tasks < max_num_tasks:
                padding = torch.zeros(max_num_tasks - num_tasks, self.static_dim, dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            padded_features.append(features)
        batch_features = torch.stack(padded_features)  # [num_dags, max_num_tasks, static_dim]

        # Padding后的批量邻接矩阵: [num_dags, max_num_tasks, max_num_tasks]
        batch_adjs = []
        for adj, num_tasks in zip(dag_adjs, num_tasks_list):
            # 始终padding到max_num_tasks，确保维度一致
            if num_tasks < max_num_tasks:
                padding = torch.zeros(max_num_tasks, max_num_tasks, dtype=adj.dtype, device=adj.device)
                padding[:num_tasks, :num_tasks] = adj
                adj = padding
            elif num_tasks > max_num_tasks:
                # 这种情况不应该发生，但为了安全截断
                adj = adj[:max_num_tasks, :max_num_tasks]
            batch_adjs.append(adj)
        batch_adjs = torch.stack(batch_adjs)  # [num_dags, max_num_tasks, max_num_tasks]

        # 批量前向传播
        h = self.task_proj(batch_features)  # [num_dags, max_num_tasks, hidden_dim]

        for gnn_layer in self.dag_gnn:
            h = gnn_layer(h, batch_adjs)  # [num_dags, max_num_tasks, hidden_dim]

        # 切分回各个DAG（去除padding）
        task_embs_list = []
        for i, num_tasks in enumerate(num_tasks_list):
            task_embs_list.append(h[i, :num_tasks])
        return task_embs_list


# ============================================================
# Layer 2: Inter-DAG Attention (DAG间协调)
# ============================================================

class InterDAGAttention(nn.Module):
    """
    Layer 2: DAG间注意力层

    输入: 池化DAG嵌入 + 5维DAG元数据（归一化）
    机制: Self-Attention 隐式学习多DAG间的相对优先级与资源竞争关系
    输出: 更新后的高级DAG嵌入

    关键改进: DAG元数据注入避免"动态信息gap"
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dag_meta_dim: int = DAG_META_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_INTER_LAYERS,
        dropout: float = DEFAULT_DROPOUT
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dag_meta_dim = dag_meta_dim

        # 任务重要性打分器（用于注意力加权池化）
        self.task_importance = nn.Linear(hidden_dim, 1)

        # DAG嵌入投影（将hidden_dim + dag_meta_dim投影到hidden_dim）
        self.dag_proj = nn.Sequential(
            nn.Linear(hidden_dim + dag_meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # DAG间Self-Attention层
        self.dag_gnn = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def attention_pool(self, task_embs: torch.Tensor) -> torch.Tensor:
        """注意力加权池化: [num_tasks, hidden_dim] -> [hidden_dim]"""
        scores = self.task_importance(task_embs)  # [num_tasks, 1]
        weights = F.softmax(scores, dim=0)  # [num_tasks, 1]
        return (task_embs * weights).sum(dim=0)  # [hidden_dim]

    def forward(
        self,
        dag_task_embs_list: List[torch.Tensor],
        dag_metadata: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dag_task_embs_list: List of [num_tasks_i, hidden_dim] 每个DAG的任务嵌入
            dag_metadata: [num_active_dags, dag_meta_dim] 归一化后的DAG元数据

        Returns:
            updated_dag_embs: [num_active_dags, hidden_dim] 更新后的DAG嵌入
            dag_task_embs_list: 原始任务嵌入（用于后续融合）
        """
        num_active_dags = len(dag_task_embs_list)

        # Step 1: 注意力加权池化生成初始DAG嵌入
        dag_embs = []
        for task_embs in dag_task_embs_list:
            dag_embs.append(self.attention_pool(task_embs))
        dag_embs = torch.stack(dag_embs)  # [num_active_dags, hidden_dim]

        # Step 2: 拼接DAG元数据
        dag_embs_with_meta = torch.cat([dag_embs, dag_metadata], dim=-1)  # [num_active_dags, hidden_dim + dag_meta_dim]

        # Step 3: 投影到统一空间
        dag_embs = self.dag_proj(dag_embs_with_meta)  # [num_active_dags, hidden_dim]

        # Step 4: Self-Attention 进行DAG间交互（残差连接）
        h = dag_embs
        for gnn_layer in self.dag_gnn:
            h = gnn_layer(h, None)  # 全连接，自注意力
        dag_embs = h + dag_embs  # residual

        return dag_embs, dag_task_embs_list

    def forward_simple(
        self,
        dag_embeddings: torch.Tensor,
        dag_metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        简化版本：直接接收DAG嵌入（而非任务嵌入列表）

        Args:
            dag_embeddings: [num_active_dags, hidden_dim]
            dag_metadata: [num_active_dags, dag_meta_dim]

        Returns:
            updated_dag_embeddings: [num_active_dags, hidden_dim]
        """
        # 拼接DAG元数据
        dag_embs_with_meta = torch.cat([dag_embeddings, dag_metadata], dim=-1)

        # 投影
        dag_embs = self.dag_proj(dag_embs_with_meta)

        # Self-Attention（残差连接）
        h = dag_embs
        for gnn_layer in self.dag_gnn:
            h = gnn_layer(h, None)

        return h + dag_embs  # residual


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Layer 1 + Layer 2 测试")
    print("=" * 60)

    # 测试参数
    batch_size = 2
    num_dags = 3
    hidden_dim = 128
    static_dim = 13
    dag_meta_dim = 5

    # ========== DAG邻接矩阵构建 ==========
    def build_dag_adjacency(num_tasks: int, parents: List[List[int]]) -> torch.Tensor:
        """
        根据父子关系构建DAG邻接矩阵

        Args:
            num_tasks: 任务数量
            parents: 父节点列表，例如 [[], [0], [0,1]] 表示:
                - task 0: 无父节点
                - task 1: 父节点为 task 0
                - task 2: 父节点为 task 0 和 task 1

        Returns:
            adj: [num_tasks, num_tasks] 邻接矩阵，1表示有边
        """
        adj = torch.zeros(num_tasks, num_tasks, dtype=torch.float32)

        for child_id, parent_list in enumerate(parents):
            for parent_id in parent_list:
                # 父节点 -> 子节点 有边
                adj[parent_id, child_id] = 1.0

        # 添加自环
        for i in range(num_tasks):
            adj[i, i] = 1.0

        return adj

    # 模拟数据: 创建具有真实依赖关系的DAG
    # DAG 0: 4个任务 (线性链: 0 -> 1 -> 2 -> 3)
    # DAG 1: 3个任务 (并行: 0,1 -> 2)
    # DAG 2: 5个任务 (复杂结构)
    num_tasks_list = [4, 3, 5]

    dag_parents = [
        [[], [0], [1], [2]],           # DAG 0: 线性链 0->1->2->3
        [[], [], [0, 1]],              # DAG 1: 并行 0,1 -> 2
        [[], [0], [0], [1, 2], [3, 4]]  # DAG 2: 复杂DAG
    ]

    dag_task_features = []
    dag_adjs = []
    for i, num_tasks in enumerate(num_tasks_list):
        features = torch.randn(num_tasks, static_dim)
        dag_task_features.append(features)
        # 使用真实依赖关系构建邻接矩阵
        adj = build_dag_adjacency(num_tasks, dag_parents[i])
        dag_adjs.append(adj)

    dag_metadata = torch.randn(num_dags, dag_meta_dim)

    # ========== Layer 1 测试 ==========
    print("\n[Layer 1: IntraDAGEncoder]")
    intra_encoder = IntraDAGEncoder(
        static_dim=static_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2
    )

    task_embs_list = intra_encoder.get_task_embeddings(dag_task_features, dag_adjs)
    for i, emb in enumerate(task_embs_list):
        print(f"  DAG {i}: {emb.shape}")

    # ========== Layer 2 测试 ==========
    print("\n[Layer 2: InterDAGAttention]")
    inter_attention = InterDAGAttention(
        hidden_dim=hidden_dim,
        dag_meta_dim=dag_meta_dim,
        num_heads=4,
        num_layers=2
    )

    updated_dag_embs, _ = inter_attention(task_embs_list, dag_metadata)
    print(f"  Updated DAG embeddings: {updated_dag_embs.shape}")

    # ========== 简化版测试 ==========
    print("\n[Layer 2 简化版测试]")
    dag_embs_simple = torch.stack([emb.mean(dim=0) for emb in task_embs_list])
    print(f"  池化DAG嵌入: {dag_embs_simple.shape}")

    updated_dag_embs_simple = inter_attention.forward_simple(dag_embs_simple, dag_metadata)
    print(f"  更新后DAG嵌入: {updated_dag_embs_simple.shape}")

    print("\n" + "=" * 60)
    print("Phase 1 测试通过！")
    print("=" * 60)