"""
GAT + PPO 网络
与研究方案 4.3 融合设计对齐
- Actor: 当前任务嵌入 + 处理器嵌入 + 匹配分数
- Critic: 图嵌入 + 处理器嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（方案 4.3.2）
    支持异构图：任务-任务、处理器-处理器、任务-处理器
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
        
        # 边特征投影（如有）
        self.edge_proj = nn.Linear(1, self.head_dim) if False else None
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.orthogonal_(proj.weight, gain=1.0)  # 降低 gain 避免梯度不稳定
            nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 节点特征
               - 单样本: [num_nodes, in_dim]
               - Batch: [batch, num_nodes, in_dim]
            adj: 邻接矩阵
               - 单样本: [num_nodes, num_nodes]
               - Batch: [batch, num_nodes, num_nodes] 或 None

        Returns:
            更新后的节点特征
               - 单样本: [num_nodes, out_dim]
               - Batch: [batch, num_nodes, out_dim]
        """
        is_batch = x.dim() == 3

        if not is_batch:
            # 单样本模式：扩展为 batch_size=1
            x = x.unsqueeze(0)
            if adj is not None:
                adj = adj.unsqueeze(0)

        batch_size, num_nodes, _ = x.shape

        # Q, K, V: [batch, num_nodes, num_heads, head_dim]
        Q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # 注意力分数: [batch, num_heads, num_nodes, num_nodes]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        scores = torch.einsum('bhmd,bhnd->bhmn', Q, K) / (self.head_dim ** 0.5)

        # 邻接矩阵 mask
        if adj is not None:
            # adj shape: [num_nodes, num_nodes] or [batch, num_nodes, num_nodes]
            # scores shape: [batch, num_heads, num_nodes, num_nodes]
            # 需要转换为 [batch, 1, num_nodes, num_nodes] 以便 broadcast
            if adj.dim() == 2:
                # 无 batch 维度: [num_nodes, num_nodes] -> [1, 1, num_nodes, num_nodes]
                adj = adj.unsqueeze(0).unsqueeze(1)
            elif adj.dim() == 3:
                # 有 batch 维度: [batch, num_nodes, num_nodes] -> [batch, 1, num_nodes, num_nodes]
                adj = adj.unsqueeze(1)
            mask = (adj == 0).float() * -1e9
            scores = scores + mask

        # softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 聚合
        V = V.transpose(1, 2)
        out = torch.einsum('bhmd,bhmn->bhnd', V, attn)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.out_dim)

        # 输出投影 + 残差连接
        out = self.o_proj(out)
        out = self.norm(x + out)

        if not is_batch:
            out = out.squeeze(0)

        return out


class TaskEncoder(nn.Module):
    """
    任务编码器（方案 4.3.1）
    将任务特征编码为嵌入
    """
    
    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 中间层
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)  # 降低 gain 避免梯度不稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_features: [num_tasks, input_dim]
        
        Returns:
            任务嵌入 [num_tasks, hidden_dim]
        """
        x = self.input_proj(task_features)
        x = self.hidden_layers(x)
        x = self.output_proj(x)
        return x


class ProcessorEncoder(nn.Module):
    """
    处理器编码器（方案 4.3.1）
    将处理器特征编码为嵌入
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)  # 降低 gain 避免梯度不稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, proc_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proc_features: [num_procs, input_dim]
        
        Returns:
            处理器嵌入 [num_procs, hidden_dim]
        """
        return self.net(proc_features)


class HeterogeneousGraphEncoder(nn.Module):
    """
    异构图编码器（方案 4.3）
    包含：
    - 任务自注意力
    - 处理器自注意力  
    - 跨类型注意力（任务-处理器匹配）
    """
    
    def __init__(
        self,
        task_dim: int,
        proc_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 编码器
        self.task_encoder = TaskEncoder(task_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.proc_encoder = ProcessorEncoder(proc_dim, hidden_dim, dropout)
        
        # 任务自注意力层
        self.task_attention = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 处理器自注意力层
        self.proc_attention = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 跨类型注意力（任务→处理器）
        self.cross_attention = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 任务-处理器匹配层
        self.match_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            task_features:
                - 单样本: [num_tasks, task_dim]
                - Batch: [batch, num_tasks, task_dim]
            proc_features:
                - 单样本: [num_procs, proc_dim]
                - Batch: [batch, num_procs, proc_dim]
            task_adj: 任务邻接矩阵 [num_tasks, num_tasks] 或 None
            proc_adj: 处理器邻接矩阵 [num_procs, num_procs] 或 None
            cross_adj: 跨类型邻接矩阵 [num_tasks+num_procs, num_tasks+num_procs] 或 None

        Returns:
            task_embed: [num_tasks, hidden_dim] 或 [batch, num_tasks, hidden_dim]
            proc_embed: [num_procs, hidden_dim] 或 [batch, num_procs, hidden_dim]
            match_scores: [num_tasks, num_procs] 或 [batch, num_tasks, num_procs]
        """
        is_batch = task_features.dim() == 3

        if not is_batch:
            # 单样本模式：扩展为 batch_size=1
            task_features = task_features.unsqueeze(0)
            proc_features = proc_features.unsqueeze(0)

        batch_size, num_tasks, _ = task_features.size()
        num_procs = proc_features.size(1)

        # 编码
        task_embed = self.task_encoder(task_features)  # [batch, num_tasks, hidden_dim]
        proc_embed = self.proc_encoder(proc_features)  # [batch, num_procs, hidden_dim]

        # 任务自注意力（残差连接）
        for attn in self.task_attention:
            task_embed = attn(task_embed, task_adj)

        # 处理器自注意力（残差连接）
        for attn in self.proc_attention:
            proc_embed = attn(proc_embed, proc_adj)

        # 跨类型注意力：任务→处理器
        # 拼接后 [batch, num_tasks + num_procs, hidden_dim]
        concat_embed = torch.cat([task_embed, proc_embed], dim=1)

        # 跨类型注意力计算（使用交叉邻接矩阵）
        for cross_attn in self.cross_attention:
            concat_embed = cross_attn(concat_embed, cross_adj)

        # 分离回任务和处理器嵌入
        task_embed = concat_embed[:, :num_tasks]
        proc_embed = concat_embed[:, num_tasks:]

        # 最终匹配分数（结合双线性匹配）
        match_scores = self._compute_match(task_embed, proc_embed)

        if not is_batch:
            # 压缩回单样本
            task_embed = task_embed.squeeze(0)
            proc_embed = proc_embed.squeeze(0)
            match_scores = match_scores.squeeze(0)

        return task_embed, proc_embed, match_scores
    
    def _compute_match(
        self,
        task_embed: torch.Tensor,
        proc_embed: torch.Tensor
    ) -> torch.Tensor:
        """计算任务-处理器匹配分数"""
        is_batch = task_embed.dim() == 3

        if not is_batch:
            task_embed = task_embed.unsqueeze(0)
            proc_embed = proc_embed.unsqueeze(0)

        batch_size, num_tasks, hidden_dim = task_embed.size()
        num_procs = proc_embed.size(1)

        # 扩展维度
        # [batch, num_tasks, hidden_dim] -> [batch, num_tasks, 1, hidden_dim] -> [batch, num_tasks, num_procs, hidden_dim]
        task_expanded = task_embed.unsqueeze(2).expand(-1, -1, num_procs, -1)
        proc_expanded = proc_embed.unsqueeze(1).expand(-1, num_tasks, -1, -1)

        # 拼接后通过匹配层
        concat = torch.cat([task_expanded, proc_expanded], dim=-1)  # [batch, num_tasks, num_procs, hidden_dim*2]
        scores = self.match_layer(concat).squeeze(-1)  # [batch, num_tasks, num_procs]

        if not is_batch:
            scores = scores.squeeze(0)

        return scores  # [num_tasks, num_procs] 或 [batch, num_tasks, num_procs]


class Actor(nn.Module):
    """
    Actor 网络（方案 4.3.3）
    输入：当前任务嵌入 + 处理器嵌入 + 匹配分数
    输出：动作 logits
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_procs: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_procs = num_procs
        
        # 输入维度: task_embed + proc_embed + match_score = 256 + 256 + 1 = 513
        input_dim = hidden_dim * 2 + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 输出 1 维分数（不是 num_procs）
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 最后一层用较大的 gain，便于 logits 拉开差距
                if m is self.net[-1]:  # 最后一层
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)  # 降低 gain 避免梯度不稳定
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_joint_logits(
        self,
        task_embed: torch.Tensor,
        proc_embed: torch.Tensor,
        match_scores: torch.Tensor,
        num_tasks: int,
        num_procs: int,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取联合动作空间的 logits

        Args:
            task_embed:
                - 单样本: [num_tasks, hidden_dim]
                - Batch: [batch, num_tasks, hidden_dim]
            proc_embed:
                - 单样本: [num_procs, hidden_dim]
                - Batch: [batch, num_procs, hidden_dim]
            match_scores: [num_tasks, num_procs] 或 [batch, num_tasks, num_procs]
            num_tasks: 任务数
            num_procs: 处理器数
            action_mask: [num_tasks * num_procs] 或 [batch, num_tasks * num_procs]

        Returns:
            joint_logits: [num_tasks * num_procs] 或 [batch, num_tasks * num_procs]
        """
        is_batch = task_embed.dim() == 3

        if not is_batch:
            # 单样本模式：扩展为 batch_size=1
            task_embed = task_embed.unsqueeze(0)
            proc_embed = proc_embed.unsqueeze(0)
            match_scores = match_scores.unsqueeze(0)

        batch_size = task_embed.size(0)

        # 处理器嵌入（扩展到 num_tasks 维度）
        # [batch, num_procs, hidden_dim] -> [batch, num_tasks, num_procs, hidden_dim]
        proc_expanded = proc_embed.unsqueeze(1).expand(-1, num_tasks, -1, -1)

        # 任务嵌入（扩展到 num_procs 维度）
        # [batch, num_tasks, hidden_dim] -> [batch, num_tasks, num_procs, hidden_dim]
        task_expanded = task_embed.unsqueeze(2).expand(-1, -1, num_procs, -1)

        # 拼接：任务嵌入 + 处理器嵌入 + 匹配分数
        concat = torch.cat([
            task_expanded,
            proc_expanded,
            match_scores.unsqueeze(-1)
        ], dim=-1)  # [batch, num_tasks, num_procs, hidden_dim*2+1]

        # 通过网络
        logits = self.net(concat)  # [batch, num_tasks, num_procs, 1]
        logits = logits.squeeze(-1)  # [batch, num_tasks, num_procs]

        # 展平为联合动作空间
        joint_logits = logits.view(batch_size, -1)  # [batch, num_tasks * num_procs]

        # 应用 mask
        if action_mask is not None:
            if action_mask.dim() == 1:
                # 单样本 mask [num_tasks * num_procs]
                joint_logits = joint_logits + action_mask
            else:
                # Batch mask [batch, num_tasks * num_procs]
                joint_logits = joint_logits + action_mask

        if not is_batch:
            joint_logits = joint_logits.squeeze(0)

        return joint_logits


class Critic(nn.Module):
    """
    Critic 网络
    输入：图嵌入 → 平均池化 → 简单 MLP → value
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tasks: int = 15,
        num_procs: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()

        # 简单 MLP：1024 → 256 → 128 → 1
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 1024 → 256
            nn.Tanh(),  # Tanh 比 GELU 更稳定
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)  # 128 → 1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        task_embed: torch.Tensor,
        proc_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            task_embed: [num_tasks, hidden_dim] 或 [batch, num_tasks, hidden_dim]
            proc_embed: [num_procs, hidden_dim] 或 [batch, num_procs, hidden_dim]

        Returns:
            value: [1] 或 [batch]
        """
        is_batch = task_embed.dim() == 3

        if not is_batch:
            task_embed = task_embed.unsqueeze(0)
            proc_embed = proc_embed.unsqueeze(0)

        # 平均池化 + 最大池化（捕捉不同信息）
        task_pooled = task_embed.mean(dim=1)  # [batch, hidden_dim]
        task_max, _ = task_embed.max(dim=1)   # [batch, hidden_dim]
        proc_pooled = proc_embed.mean(dim=1)  # [batch, hidden_dim]
        proc_max, _ = proc_embed.max(dim=1)   # [batch, hidden_dim]

        # 拼接
        concat = torch.cat([task_pooled, task_max, proc_pooled, proc_max], dim=1)  # [batch, hidden_dim * 4]

        # 通过网络
        value = self.net(concat)  # [batch, 1]

        if not is_batch:
            value = value.squeeze(0)

        return value


class GATPPO(nn.Module):
    """
    GAT + PPO 完整网络（分离 Actor/Critic 编码器）
    """

    def __init__(
        self,
        task_dim: int = 25,
        proc_dim: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_procs: int = 9,
        max_candidates: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_procs = num_procs
        self.max_candidates = max_candidates

        # Actor 专用编码器
        self.actor_encoder = HeterogeneousGraphEncoder(
            task_dim=task_dim,
            proc_dim=proc_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Critic 专用编码器（独立学习）
        self.critic_encoder = HeterogeneousGraphEncoder(
            task_dim=task_dim,
            proc_dim=proc_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Actor
        self.actor = Actor(
            hidden_dim=hidden_dim,
            num_procs=num_procs,
            dropout=dropout
        )

        # Critic
        self.critic = Critic(
            hidden_dim=hidden_dim,
            num_tasks=max_candidates,
            num_procs=num_procs,
            dropout=dropout
        )

    def build_task_adj(
        self,
        candidate_tasks: List,
        max_candidates: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        从候选任务构建任务邻接矩阵

        Args:
            candidate_tasks: 候选任务列表
            max_candidates: 最大候选数
            device: 设备

        Returns:
            adj: [max_candidates, max_candidates] 邻接矩阵，1表示有边
        """
        adj = torch.zeros((max_candidates, max_candidates), dtype=torch.float32, device=device)

        for i, task in enumerate(candidate_tasks):
            if i >= max_candidates:
                break
            # 添加父节点连接
            for parent_id in task.parents:
                # 找到父节点在候选列表中的索引
                for j, t in enumerate(candidate_tasks):
                    if t.task_id == parent_id and j < max_candidates:
                        adj[j, i] = 1.0  # 父->子
                        break
            # 添加子节点连接
            for child_id in task.children:
                for j, t in enumerate(candidate_tasks):
                    if t.task_id == child_id and j < max_candidates:
                        adj[i, j] = 1.0  # 子->父
                        break

        # 添加自环
        for i in range(min(len(candidate_tasks), max_candidates)):
            adj[i, i] = 1.0

        return adj

    def build_proc_adj(
        self,
        num_procs: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        构建处理器邻接矩阵（全连接）

        Args:
            num_procs: 处理器数量
            device: 设备

        Returns:
            adj: [num_procs, num_procs] 全1矩阵
        """
        # 处理器之间全连接
        adj = torch.ones((num_procs, num_procs), dtype=torch.float32, device=device)
        return adj

    def build_cross_adj(
        self,
        candidate_tasks: List,
        num_procs: int,
        max_candidates: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        构建跨类型邻接矩阵（任务-处理器）

        Args:
            candidate_tasks: 候选任务列表
            num_procs: 处理器数量
            max_candidates: 最大候选数
            device: 设备

        Returns:
            adj: [max_candidates + num_procs, max_candidates + num_procs]
                联合邻接矩阵，任务和处理器之间全连接
        """
        total = max_candidates + num_procs
        adj = torch.zeros((total, total), dtype=torch.float32, device=device)
        # 任务之间已有邻接
        # 处理器之间全连接
        for i in range(num_procs):
            for j in range(num_procs):
                adj[max_candidates + i, max_candidates + j] = 1.0
        # 任务-处理器全连接
        for i in range(max_candidates):
            for j in range(num_procs):
                adj[i, max_candidates + j] = 1.0
                adj[max_candidates + j, i] = 1.0
        return adj

    def encode(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码状态（用于 Actor）

        支持两种输入格式：
        1. 单样本: task_features [max_candidates, task_dim], proc_features [num_procs, proc_dim]
        2. Batch: task_features [batch, max_candidates, task_dim], proc_features [batch, num_procs, proc_dim]

        Args:
            task_features: 任务特征
            proc_features: 处理器特征
            task_adj: 任务邻接矩阵 [num_tasks, num_tasks] 或 [batch, num_tasks, num_tasks]
            proc_adj: 处理器邻接矩阵 [num_procs, num_procs] 或 [batch, num_procs, num_procs]
            cross_adj: 跨类型邻接矩阵 [num_tasks+num_procs, num_tasks+num_procs]
        """
        # 使用 Actor 编码器
        task_embed, proc_embed, match_scores = self.actor_encoder(
            task_features, proc_features, task_adj, proc_adj, cross_adj
        )

        return task_embed, proc_embed, match_scores

    def encode_for_critic(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码状态（用于 Critic）

        Args:
            task_features: 任务特征
            proc_features: 处理器特征
            task_adj: 任务邻接矩阵
            proc_adj: 处理器邻接矩阵
            cross_adj: 跨类型邻接矩阵
        """
        task_embed, proc_embed, _ = self.critic_encoder(
            task_features, proc_features, task_adj, proc_adj, cross_adj
        )
        return task_embed, proc_embed

    def get_action(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作

        Args:
            task_features: [max_candidates, task_dim]
            proc_features: [num_procs, proc_dim]
            action_mask: [max_candidates * num_procs]
            deterministic: 是否确定性选择
            task_adj: 任务邻接矩阵
            proc_adj: 处理器邻接矩阵
            cross_adj: 跨类型邻接矩阵

        Returns:
            action: 动作 ID
            action_logprob: 对数概率
            value: 状态价值
        """
        # Actor 编码
        task_embed, proc_embed, match_scores = self.encode(task_features, proc_features, task_adj, proc_adj, cross_adj)

        # Actor: 获取联合动作 logits
        logits = self.actor.get_joint_logits(
            task_embed, proc_embed, match_scores,
            self.max_candidates, self.num_procs,
            action_mask
        )

        # 采样
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)  # [batch] 或 scalar
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)

        # Critic: 使用独立的 Critic 编码器
        task_embed_c, proc_embed_c = self.encode_for_critic(task_features, proc_features, task_adj, proc_adj, cross_adj)
        value = self.critic(task_embed_c, proc_embed_c)

        return action, action_logprob, value

    def evaluate_actions(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）

        Args:
            task_features: [batch, max_candidates, task_dim]
            proc_features: [batch, num_procs, proc_dim]
            actions: [batch]
            action_mask: [batch, max_candidates * num_procs] (可选)
            task_adj: 任务邻接矩阵
            proc_adj: 处理器邻接矩阵
            cross_adj: 跨类型邻接矩阵

        Returns:
            action_logprobs: [batch]
            values: [batch]
            entropy: [batch]
        """
        # Actor 编码
        task_embed, proc_embed, match_scores = self.encode(task_features, proc_features, task_adj, proc_adj, cross_adj)

        # Actor: 获取联合动作 logits
        logits = self.actor.get_joint_logits(
            task_embed, proc_embed, match_scores,
            self.max_candidates, self.num_procs,
            action_mask  # 支持 None
        )

        # 计算 logprob 和 entropy
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Critic: 使用独立的 Critic 编码器
        task_embed_c, proc_embed_c = self.encode_for_critic(task_features, proc_features, task_adj, proc_adj, cross_adj)
        values = self.critic(task_embed_c, proc_embed_c)

        return action_logprobs, values.squeeze(-1), entropy

    def get_values(
        self,
        task_features: torch.Tensor,
        proc_features: torch.Tensor,
        task_adj: Optional[torch.Tensor] = None,
        proc_adj: Optional[torch.Tensor] = None,
        cross_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取状态价值（用于 value target 计算）

        Args:
            task_features: [batch, max_candidates, task_dim]
            proc_features: [batch, num_procs, proc_dim]
            task_adj: 任务邻接矩阵
            proc_adj: 处理器邻接矩阵
            cross_adj: 跨类型邻接矩阵

        Returns:
            values: [batch]
        """
        # 使用 Critic 专用编码器
        task_embed, proc_embed = self.encode_for_critic(task_features, proc_features, task_adj, proc_adj, cross_adj)

        # 批量计算价值
        values = self.critic(task_embed, proc_embed)

        return values.squeeze(-1)


class PPOTrainer:
    """
    PPO 训练器（方案 4.4.6）
    """

    def __init__(
        self,
        model: GATPPO,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.9,  # 降低 GAE lambda 减少方差
        lr_decay: float = 0.999,  # 每 episode 衰减，500 episode 后仍保留 ~61%
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        # 分离 actor/critic 优化器
        self.optimizer = torch.optim.Adam([
            {'params': model.actor_encoder.parameters(), 'lr': lr_actor},
            {'params': model.actor.parameters(), 'lr': lr_actor},
            {'params': model.critic_encoder.parameters(), 'lr': lr_critic},
            {'params': model.critic.parameters(), 'lr': lr_critic}
        ])

        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.lr_decay = lr_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

    def decay_lr(self):
        """每 episode 结束时调用，学习率衰减"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)

        Args:
            rewards: [batch]
            values: [batch]
            next_values: [batch]
            masks: [batch] (1 = continue, 0 = terminal)

        Returns:
            advantages: [batch]
            returns: [batch]
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_val = next_values[-1] if len(next_values) > 0 else 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                v_next = 0  # 终止状态
            else:
                v_next = values[t + 1]

            delta = rewards[t] + self.gamma * v_next * masks[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        task_features_batch,
        proc_features_batch,
        next_task_features_batch,
        next_proc_features_batch,
        actions_batch,
        old_logprobs_batch,
        rewards_batch,
        masks_batch,
        action_masks_batch=None
    ) -> dict:
        """
        PPO 更新

        Args:
            task_features_batch: [batch, max_candidates, task_dim]
            proc_features_batch: [batch, num_procs, proc_dim]
            next_task_features_batch: [batch, max_candidates, task_dim]
            next_proc_features_batch: [batch, num_procs, proc_dim]
            actions_batch: [batch]
            old_logprobs_batch: [batch]
            rewards_batch: [batch]
            masks_batch: [batch] (episode continue flag)
            action_masks_batch: [batch, max_candidates * num_procs]

        Returns:
            loss info
        """
        # 转换为 tensor
        task_features = torch.FloatTensor(task_features_batch).to(self.device)
        proc_features = torch.FloatTensor(proc_features_batch).to(self.device)
        next_task_features = torch.FloatTensor(next_task_features_batch).to(self.device)
        next_proc_features = torch.FloatTensor(next_proc_features_batch).to(self.device)
        actions = torch.LongTensor(actions_batch).to(self.device)
        old_logprobs = torch.FloatTensor(old_logprobs_batch).to(self.device)
        rewards = torch.FloatTensor(rewards_batch).to(self.device)
        masks = torch.FloatTensor(masks_batch).to(self.device)

        if action_masks_batch is not None:
            action_masks = torch.FloatTensor(action_masks_batch).to(self.device)
        else:
            action_masks = None

        # 计算 next_values（用于 GAE）
        with torch.no_grad():
            next_values = self.model.get_values(next_task_features, next_proc_features).squeeze()

        # 评估当前策略（单次调用获取所有需要的信息）
        action_logprobs, values, entropy = self.model.evaluate_actions(
            task_features, proc_features, actions, action_masks
        )
        values = values.squeeze()
        values_eval = values  # 重用，避免重复计算

        # 计算 GAE 优势估计
        advantages, returns = self._compute_gae(rewards, values, next_values, masks)

        # === DEBUG: 记录标准化前的原始值 ===
        returns_raw = returns.clone()
        advantages_raw = advantages.clone()
        debug_info = {
            'rewards_mean': rewards.mean().item(),
            'rewards_std': rewards.std().item(),
            'rewards_min': rewards.min().item(),
            'rewards_max': rewards.max().item(),
            'masks_sum': masks.sum().item(),
            'masks_count': masks.numel(),
            'values_mean': values.mean().item(),
            'next_values_mean': next_values.mean().item(),
            'returns_raw_mean': returns_raw.mean().item(),
            'returns_raw_std': returns_raw.std().item(),
            'advantages_raw_mean': advantages_raw.mean().item(),
            'advantages_raw_std': advantages_raw.std().item(),
        }
        # 每 50 次 update 打印一次调试信息
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            print(f"\n[DEBUG GAE #{self._debug_counter}]")
            print(f"  rewards: mean={debug_info['rewards_mean']:.4f}, std={debug_info['rewards_std']:.4f}, min={debug_info['rewards_min']:.4f}, max={debug_info['rewards_max']:.4f}")
            print(f"  masks: sum={debug_info['masks_sum']:.0f}/{debug_info['masks_count']:.0f}")
            print(f"  values: mean={debug_info['values_mean']:.4f}, next_values mean={debug_info['next_values_mean']:.4f}")
            print(f"  returns_raw: mean={debug_info['returns_raw_mean']:.6f}, std={debug_info['returns_raw_std']:.4f}")
            print(f"  advantages_raw: mean={debug_info['advantages_raw_mean']:.6f}, std={debug_info['advantages_raw_std']:.4f}")

        # 标准化 advantages（稳定训练）
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 标准化 returns（与 values_eval 分布对齐，避免 scale mismatch）
        if returns.std() > 1e-8:
            returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns_norm = returns

        # values_eval 已通过上面的 evaluate_actions 获取，直接使用
        if self._debug_counter % 50 == 0:
            # 使用 Critic 专用编码器获取 embeddings（用于调试）
            with torch.no_grad():
                task_embed_debug, proc_embed_debug = self.model.encode_for_critic(task_features, proc_features)
            task_flat_debug = task_embed_debug.view(task_embed_debug.size(0), -1)
            proc_flat_debug = proc_embed_debug.view(proc_embed_debug.size(0), -1)
            print(f"  [Critic Debug] task_embed std={task_flat_debug.std().item():.4f}, proc_embed std={proc_flat_debug.std().item():.4f}")
            print(f"  [Critic Debug] values_eval (norm) mean={values_eval.mean().item():.4f}, std={values_eval.std().item():.4f}")
            print(f"  [Critic Debug] returns_norm mean={returns_norm.mean().item():.4f}, std={returns_norm.std().item():.4f}")
            print(f"  [Critic Debug] returns_raw mean={returns_raw.mean().item():.4f}, std={returns_raw.std().item():.4f}")

        # PPO 损失
        ratio = torch.exp(action_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value 损失（使用标准化后的 returns，与 values_eval 分布对齐）
        value_loss = F.mse_loss(values_eval, returns_norm)

        # 熵损失
        entropy_loss = -entropy.mean()

        # 各损失组件贡献（用于诊断）
        policy_contrib = policy_loss.item()
        entropy_contrib = self.entropy_coef * entropy_loss.item()
        value_contrib = self.value_coef * value_loss.item()

        # 总损失
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 学习率衰减在每 episode 统一调用，这里不再衰减
        # 使用前面已克隆的 returns_raw 和 advantages_raw（避免重复克隆）
        returns_raw = returns_raw if 'returns_raw' in dir() else returns.clone()
        advantages_raw = advantages_raw if 'advantages_raw' in dir() else advantages.clone()

        return {
            'policy_loss': policy_loss.item(),
            'policy_contrib': policy_contrib,
            'entropy_contrib': entropy_contrib,
            'value_contrib': value_contrib,
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': loss.item(),
            'grad_norm': grad_norm,
            'advantage_mean': advantages_raw.mean().item(),      # 标准化前
            'advantage_std': advantages_raw.std().item(),        # 标准化前
            'ratio_mean': ratio.mean().item(),
            'value_pred': values_eval.mean().item(),             # Critic 预测
            'value_target': returns_raw.mean().item(),      # 训练目标（原始 returns）
            'value_target_raw': returns_raw.mean().item(),      # 原始 returns 均值（参考用）
            'returns_std': returns_raw.std().item(),            # 原始 returns 标准差
            # 调试信息
            'debug_rewards_mean': debug_info['rewards_mean'],
            'debug_masks_sum': debug_info['masks_sum'],
            'debug_masks_count': debug_info['masks_count'],
        }


# 测试
if __name__ == "__main__":
    # 创建网络
    model = GATPPO(
        task_dim=25,
        proc_dim=4,
        hidden_dim=256,
        num_procs=9,
        max_candidates=15
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向
    batch_size = 4
    task_features = torch.randn(batch_size, 15, 25)
    proc_features = torch.randn(batch_size, 9, 4)
    action_mask = torch.zeros(15 * 9)
    action_mask[::3] = -1e9  # 模拟 mask
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        action, logprob, value = model.get_action(
            task_features[0],
            proc_features[0],
            action_mask,
            deterministic=False
        )
        print(f"\n动作: {action.item()}, logprob: {logprob.item():.3f}, value: {value.item():.3f}")
    
    # 测试训练
    print("\n测试训练...")
    trainer = PPOTrainer(model, lr_actor=3e-4, lr_critic=3e-4)
    
    # 构造 batch
    task_batch = torch.randn(8, 15, 25)
    proc_batch = torch.randn(8, 9, 4)
    actions = torch.randint(0, 135, (8,))
    old_logprobs = torch.randn(8)
    rewards = torch.randn(8)
    masks = torch.ones(8)
    action_masks = torch.zeros(8, 135)
    
    info = trainer.update(
        task_batch, proc_batch, actions, old_logprobs, rewards, masks, action_masks
    )
    print(f"训练更新: {info}")
    
    print("\n✓ GAT+PPO 网络测试通过")