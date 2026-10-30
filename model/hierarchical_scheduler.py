"""
分层图注意力调度网络 - Phase 3
完整的 HierarchicalGATScheduler

整合:
- Layer 1: IntraDAGEncoder (12维静态特征 → 任务结构嵌入)
- Layer 2: InterDAGAttention (DAG间协调)
- Layer 3: TwoStageScheduler (两阶段决策)

与 ARCHITECTURE_DESIGN.md Section 6.4 对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hierarchical_gat import IntraDAGEncoder, InterDAGAttention, GraphAttentionLayer, apply_adj_direction
from model.hierarchical_gat_layer3 import TwoStageScheduler
from .constants import (
    STATIC_DIM, DYN_DIM, DAG_META_DIM, PROC_DIM, NUM_PROCS, NUM_CLOUD,
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_HEADS,
    DEFAULT_INTRA_LAYERS, DEFAULT_INTER_LAYERS, DEFAULT_DROPOUT,
    MAX_DAGS, MAX_CANDIDATES, GLOBAL_FEATURES_DIM,
    CRITIC_NORM_MAX_CANDIDATES, CRITIC_NORM_MAX_DAGS,
    ADJ_DIRECTIONS, DEFAULT_ADJ_DIRECTION,
)


def compute_global_features(
    proc_features: torch.Tensor,
    num_ready: int,
    num_active_dags: int,
    max_candidates: int = CRITIC_NORM_MAX_CANDIDATES,
    max_dags: int = CRITIC_NORM_MAX_DAGS,
) -> torch.Tensor:
    """
    Critic 用全局排队特征（采集与 PPO 更新须使用同一函数）。

    ① queue_pressure: 处理器平均 z-scored log 等待时间 (proc_features[:, 1])
    ② ready_task_ratio: 就绪任务数 / max_candidates
    ③ active_dag_ratio: 原始活跃 DAG 数 / max_dags（非裁剪后 DAG 数）
    """
    queue_pressure = proc_features[:, 1].mean()
    ready_task_ratio = min(num_ready / max_candidates, 1.0)
    active_dag_ratio = min(num_active_dags / max_dags, 1.0)
    return torch.stack([
        queue_pressure,
        torch.tensor(ready_task_ratio, device=proc_features.device, dtype=torch.float32),
        torch.tensor(active_dag_ratio, device=proc_features.device, dtype=torch.float32),
    ])


class HierarchicalGATScheduler(nn.Module):
    """
    分层GAT多DAG调度器

    创新点：
    1. Intra-DAG GAT: 仅用12维静态结构特征学习DAG内部依赖结构
    2. DAG元数据注入: 池化后拼接DAG元数据再送入Inter-DAG Attention

    与 ARCHITECTURE_DESIGN.md Section 6.4 对齐
    """

    def __init__(
        self,
        static_dim: int = STATIC_DIM,
        dyn_dim: int = DYN_DIM,
        dag_meta_dim: int = DAG_META_DIM,
        proc_dim: int = PROC_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_intra_layers: int = DEFAULT_INTRA_LAYERS,
        num_inter_layers: int = DEFAULT_INTER_LAYERS,
        max_dags: int = MAX_DAGS,
        max_candidates: int = MAX_CANDIDATES,
        num_procs: int = NUM_PROCS,
        dropout: float = DEFAULT_DROPOUT,
        enable_hold: bool = True,
        max_attn_dags: int = 64,
        joint_matching: bool = False,
        fusion_mode: str = 'gate',
        adj_direction: str = DEFAULT_ADJ_DIRECTION,
        use_pair_features: bool = False,
        use_load_features: bool = False,
    ):
        super().__init__()

        if fusion_mode not in ('add', 'gate'):
            raise ValueError(f"fusion_mode must be 'add' or 'gate', got {fusion_mode!r}")
        if adj_direction not in ADJ_DIRECTIONS:
            raise ValueError(f"adj_direction must be one of {ADJ_DIRECTIONS}, got {adj_direction!r}")

        self.static_dim = static_dim
        self.dyn_dim = dyn_dim
        self.dag_meta_dim = dag_meta_dim
        self.proc_dim = proc_dim
        self.hidden_dim = hidden_dim
        self.max_dags = max_dags
        self.max_candidates = max_candidates
        self.num_procs = num_procs
        self.enable_hold = enable_hold
        self.max_attn_dags = max_attn_dags
        self.joint_matching = joint_matching
        self.fusion_mode = fusion_mode
        self.adj_direction = adj_direction
        self.use_pair_features = use_pair_features
        self.use_load_features = use_load_features

        # Layer 1: DAG内编码器（仅接收12维静态特征）
        self.intra_dag_encoder = IntraDAGEncoder(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_intra_layers,
            dropout=dropout
        )

        # Layer 2: DAG间注意力（接收DAG元数据）
        self.inter_dag_attention = InterDAGAttention(
            hidden_dim=hidden_dim,
            dag_meta_dim=dag_meta_dim,
            num_heads=num_heads,
            num_layers=num_inter_layers,
            dropout=dropout
        )

        # Layer 3: 两阶段决策器
        self.two_stage_scheduler = TwoStageScheduler(
            hidden_dim=hidden_dim,
            dyn_dim=dyn_dim,
            proc_dim=proc_dim,
            num_procs=num_procs,
            dropout=dropout,
            enable_hold=enable_hold,
            use_pair_features=use_pair_features,
            use_load_features=use_load_features,
        )

        # 全局特征投影（排队压力、候选积压、系统负载 → 16维）
        self.global_proj = nn.Linear(GLOBAL_FEATURES_DIM, 16)
        self.default_global = nn.Parameter(torch.zeros(16))

        # Critic用于估计状态价值（拼接全局排队特征）
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Critic 最后一层零初始化，初始值预测接近 0，避免 GAE 第一轮就爆炸
        nn.init.zeros_(self.critic_net[-1].weight)
        nn.init.zeros_(self.critic_net[-1].bias)

        # Layer 2 → Layer 3: 任务嵌入与 DAG 上下文融合（gate 为默认）
        if fusion_mode == 'gate':
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, 2.0)  # sigmoid≈0.88，初始接近 plain add
        else:
            self.fusion_gate = None

    def _fuse_task_dag_context(
        self,
        task_embs: torch.Tensor,
        dag_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        将 Inter-DAG 更新后的 DAG 嵌入注入任务表示。

        gate: task + sigmoid(W·[task; dag]) ⊙ dag
        add:  task + dag（消融基线）
        """
        if self.fusion_mode == 'add':
            return task_embs + dag_embs
        gate = torch.sigmoid(self.fusion_gate(torch.cat([task_embs, dag_embs], dim=-1)))
        return task_embs + gate * dag_embs

    def _select_attn_dags(
        self,
        num_dags: int,
        dag_metadata: torch.Tensor,
        ready_dag_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        选择参与 Inter-DAG Attention 的 DAG 子集。
        优先保留有就绪任务的 DAG，其余按紧迫度补足。

        Args:
            num_dags: 活跃 DAG 总数
            dag_metadata: [num_dags, dag_meta_dim]，列1为 completion_ratio
            ready_dag_mask: [num_dags] bool，True 表示该 DAG 有就绪任务

        Returns:
            selected_idx: [num_selected] 选中的 DAG 索引（有序）
        """
        device = dag_metadata.device
        if num_dags <= self.max_attn_dags:
            return torch.arange(num_dags, device=device)

        # urgency: deadline紧迫度
        urgency = -dag_metadata[:, 0]

        # 有就绪任务的 DAG 强制保留（加一个大偏移确保排在最前）
        if ready_dag_mask is not None:
            urgency = urgency + ready_dag_mask.float() * 100.0

        _, indices = torch.topk(urgency, self.max_attn_dags)
        return indices.sort()[0]

    def _prepare_adjs(self, adjs: List[torch.Tensor]) -> List[torch.Tensor]:
        """按 adj_direction 变换邻接矩阵（仅 Intra-DAG GAT 使用）"""
        if self.adj_direction == 'down':
            return adjs
        return [apply_adj_direction(a, self.adj_direction) for a in adjs]

    def _encode_tasks(self, features: List[torch.Tensor], adjs: List[torch.Tensor]) -> List[torch.Tensor]:
        """编码 DAG 任务（可被子类重载用于消融实验）"""
        return self.intra_dag_encoder.get_task_embeddings_batch(features, self._prepare_adjs(adjs))

    def _encode_dags(self, dag_embs: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """编码 DAG（可被子类重载用于消融实验）"""
        return self.inter_dag_attention.forward_simple(dag_embs, metadata)

    def _pool_dag(self, task_embs: torch.Tensor) -> torch.Tensor:
        """DAG 嵌入池化（可被子类重载用于消融实验）"""
        return self.inter_dag_attention.attention_pool(task_embs)

    def _encode_dags_batch(self, dag_embs, dag_meta, dag_adj):
        """批量 Inter-DAG Attention（可被子类重载用于消融实验）"""
        dag_embs_with_meta = torch.cat([dag_embs, dag_meta], dim=-1)
        h = self.inter_dag_attention.dag_proj(dag_embs_with_meta)
        updated = h
        for gnn_layer in self.inter_dag_attention.dag_gnn:
            updated = gnn_layer(updated, dag_adj)
        return updated + h

    def forward(
        self,
        dag_task_static_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor],
        dag_metadata: torch.Tensor,
        ready_task_indices: torch.Tensor,
        ready_task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        task_dag_ids: torch.Tensor,
        task_proc_pair_features: Optional[torch.Tensor] = None,
        num_active_dags: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播

        Args:
            dag_task_static_features: List of [num_tasks_i, static_dim] 每个DAG的静态特征
            dag_adjs: List of [num_tasks_i, num_tasks_i] 每个DAG的邻接矩阵
            dag_metadata: [num_active_dags, dag_meta_dim] DAG元数据
            ready_task_indices: [M] 就绪任务的全局索引
            ready_task_dyn_features: [M, dyn_dim] 就绪任务的动态特征
            proc_features: [N, proc_dim] 处理器特征
            proc_states: [N] 处理器剩余等待时间（原始秒数，供其他模块使用）
            task_dag_ids: [total_tasks] 每个任务对应的DAG ID
            num_active_dags: 原始活跃 DAG 数（裁剪前）；供负载特征 / Critic 用

        Returns:
            task_logits: [M] 任务选择logits
            proc_logits: [M, N+1] 处理器选择logits
            info: 调试信息
        """
        model_device = next(self.parameters()).device

        # 所有张量迁到模型设备
        static_features = [f.to(model_device) for f in dag_task_static_features]
        adjs = [a.to(model_device) for a in dag_adjs]
        metadata = dag_metadata.to(model_device)
        ready_idx = ready_task_indices.to(model_device)
        ready_dyn = ready_task_dyn_features.to(model_device)
        proc_feat = proc_features.to(model_device)
        proc_st = proc_states.to(model_device)
        dag_ids = task_dag_ids.to(model_device)

        # ===== Step 1: Intra-DAG编码（批量并行） =====
        dag_task_embs_list = self._encode_tasks(static_features, adjs)

        # ===== Step 2: 注意力加权DAG嵌入池化 =====
        dag_embeddings = []
        for task_embs in dag_task_embs_list:
            dag_embeddings.append(self._pool_dag(task_embs))
        dag_embeddings = torch.stack(dag_embeddings)  # [num_active_dags, hidden_dim]

        # ===== Step 3: Inter-DAG注意力（DAG过多时裁剪到 max_attn_dags） =====
        num_dags = len(dag_task_embs_list)
        if num_dags > self.max_attn_dags:
            ready_dag_ids = dag_ids[ready_idx].unique()
            ready_dag_mask = torch.zeros(num_dags, dtype=torch.bool, device=metadata.device)
            ready_dag_mask[ready_dag_ids] = True
            selected_idx = self._select_attn_dags(num_dags, metadata, ready_dag_mask)
            selected_embs = dag_embeddings[selected_idx]
            selected_meta = metadata[selected_idx]
            updated_selected = self._encode_dags(selected_embs, selected_meta)
            updated_dag_embeddings = dag_embeddings.clone()
            updated_dag_embeddings[selected_idx] = updated_selected
        else:
            updated_dag_embeddings = self._encode_dags(dag_embeddings, metadata)

        # ===== Step 4: 获取就绪任务嵌入 =====
        # 收集所有任务的嵌入
        all_task_embs = torch.cat(dag_task_embs_list, dim=0)  # [total_tasks, hidden_dim]

        # 根据就绪任务索引获取嵌入
        ready_task_embs = all_task_embs[ready_idx]  # [M, hidden_dim]

        # 获取就绪任务对应的DAG嵌入并融合
        ready_task_dag_ids = dag_ids[ready_idx]  # [M]
        ready_dag_embs = updated_dag_embeddings[ready_task_dag_ids]  # [M, hidden_dim]

        # 融合任务静态嵌入与 DAG 上下文（门控或加法）
        ready_task_embs = self._fuse_task_dag_context(ready_task_embs, ready_dag_embs)

        # ===== Step 5: 两阶段决策（可选：全局负载 → 任务头） =====
        m_ready = ready_task_embs.size(0)
        if num_active_dags is None:
            num_active_dags = num_dags
        global_load = None
        if self.use_load_features:
            global_load = compute_global_features(
                proc_feat, m_ready, int(num_active_dags)
            )

        task_logits, proc_logits = self.two_stage_scheduler(
            ready_task_embs,
            ready_dyn,
            proc_feat,
            proc_st,
            task_proc_pair_features,
            global_load_features=global_load,
        )

        info = {
            'dag_embeddings': dag_embeddings,
            'updated_dag_embeddings': updated_dag_embeddings,
            'ready_task_embs': ready_task_embs,
            'global_load_features': global_load,
        }

        return task_logits, proc_logits, info

    def get_action(
        self,
        dag_task_static_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor],
        dag_metadata: torch.Tensor,
        ready_task_indices: torch.Tensor,
        ready_task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        task_dag_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        candidate_types: Optional[torch.Tensor] = None,
        num_active_dags: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个动作

        Args:
            dag_task_static_features: List of [num_tasks_i, static_dim]
            dag_adjs: List of [num_tasks_i, num_tasks_i]
            dag_metadata: [num_active_dags, dag_meta_dim]
            ready_task_indices: [M]
            ready_task_dyn_features: [M, dyn_dim]
            proc_features: [N, proc_dim]
            proc_states: [N]
            task_dag_ids: [total_tasks]
            valid_mask: [M] 可选，任务有效掩码
            candidate_types: [M] 可选，任务类型 (0=cloud_only, 1=edge_only, 2=flexible)
            num_active_dags: 原始活跃 DAG 总数（Critic 用）；默认 len(dag_task_static_features)

        Returns:
            task_id: 选中的任务索引（0到M-1）
            proc_id: 选中的处理器索引（0到N-1为物理，N为Global Hold）
            value: 状态价值估计
        """
        task_logits, proc_logits, info = self.forward(
            dag_task_static_features, dag_adjs, dag_metadata,
            ready_task_indices, ready_task_dyn_features,
            proc_features, proc_states, task_dag_ids,
            num_active_dags=num_active_dags,
        )

        # 应用有效掩码
        if valid_mask is not None:
            task_logits = task_logits + (1 - valid_mask.float()) * -1e9

        # Stage 1: 选择任务
        task_probs = F.softmax(task_logits, dim=-1)
        task_dist = torch.distributions.Categorical(task_probs)
        task_id = task_dist.sample()

        # Stage 2: 选择处理器（应用candidate_type约束的mask）
        proc_logits_selected = proc_logits[task_id]

        if candidate_types is not None:
            task_type = candidate_types[task_id].item()
            proc_logits_selected = self.two_stage_scheduler.apply_candidate_type_mask(
                proc_logits_selected, task_type
            )

        proc_probs = F.softmax(proc_logits_selected, dim=-1)
        proc_dist = torch.distributions.Categorical(proc_probs)
        proc_id = proc_dist.sample()

        # Critic 估计（与采集 / PPO 更新使用相同全局特征）
        ready_task_embs = info['ready_task_embs']
        if num_active_dags is None:
            num_active_dags = len(dag_task_static_features)
        num_ready = ready_task_indices.numel()
        global_feat = compute_global_features(proc_features, num_ready, num_active_dags)
        value = self.estimate_value(ready_task_embs, global_feat)

        return task_id, proc_id, value

    def estimate_value(self, task_embs: torch.Tensor,
                       global_features: torch.Tensor = None) -> torch.Tensor:
        """
        估计状态价值

        Args:
            task_embs: [M, hidden_dim] 任务嵌入
            global_features: [GLOBAL_FEATURES_DIM] 全局排队特征，可选

        Returns:
            value: [1] 价值估计
        """
        pooled = task_embs.mean(dim=0)  # [hidden_dim]
        if global_features is None:
            global_feat = self.default_global
        else:
            global_feat = self.global_proj(global_features.to(pooled.device))
        pooled = torch.cat([pooled, global_feat])
        value = self.critic_net(pooled)  # [1]
        return value

    def evaluate_actions(
        self,
        dag_task_static_features: List[torch.Tensor],
        dag_adjs: List[torch.Tensor],
        dag_metadata: torch.Tensor,
        ready_task_indices: torch.Tensor,
        ready_task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        task_dag_ids: torch.Tensor,
        actions_task: torch.Tensor,
        actions_proc: torch.Tensor,
        num_active_dags: Optional[int] = None,
        task_proc_pair_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于训练）

        Args:
            dag_task_static_features: List of [num_tasks_i, static_dim]
            dag_adjs: List of [num_tasks_i, num_tasks_i]
            dag_metadata: [num_active_dags, dag_meta_dim]
            ready_task_indices: [M]
            ready_task_dyn_features: [M, dyn_dim]
            proc_features: [N, proc_dim]
            proc_states: [N]
            task_dag_ids: [total_tasks]
            actions_task: [batch] 任务选择动作
            actions_proc: [batch] 处理器选择动作

        Returns:
            action_logprobs: [batch] 动作对数概率
            values: [batch] 价值估计
            entropy: [batch] 熵
        """
        task_logits, proc_logits, info = self.forward(
            dag_task_static_features, dag_adjs, dag_metadata,
            ready_task_indices, ready_task_dyn_features,
            proc_features, proc_states, task_dag_ids,
            task_proc_pair_features,
            num_active_dags=num_active_dags,
        )

        ready_task_embs = info['ready_task_embs']

        # 计算任务选择的logprob
        task_probs = F.softmax(task_logits, dim=-1)
        task_dist = torch.distributions.Categorical(task_probs)
        task_logprobs = task_dist.log_prob(actions_task)

        # 计算处理器选择的logprob
        # proc_logits: [M, N+1], actions_proc: [batch]
        # 需要根据选择的task_id获取对应的proc_logits
        proc_logits_selected = proc_logits[actions_task]  # [batch, N+1]
        proc_probs = F.softmax(proc_logits_selected, dim=-1)
        proc_dist = torch.distributions.Categorical(proc_probs)
        proc_logprobs = proc_dist.log_prob(actions_proc)

        # 联合动作的logprob（相加）
        action_logprobs = task_logprobs + proc_logprobs

        # 熵
        task_entropy = task_dist.entropy()
        proc_entropy = proc_dist.entropy()
        entropy = task_entropy + proc_entropy

        if num_active_dags is None:
            num_active_dags = len(dag_task_static_features)
        num_ready = ready_task_indices.numel()
        global_feat = compute_global_features(proc_features, num_ready, num_active_dags)
        values = self.estimate_value(ready_task_embs, global_feat).expand_as(action_logprobs)

        return action_logprobs, values, entropy

    @staticmethod
    def _dag_fingerprint(feat: torch.Tensor, adj: torch.Tensor) -> tuple:
        """快速DAG指纹，用于跨样本去重（CPU tensor）"""
        return (tuple(feat.shape), tuple(adj.shape),
                feat.sum().item(), adj.sum().item())

    def evaluate_actions_batch(
        self,
        batch_env_data: dict,
        batch_actions: dict,
        device: str = 'cpu'
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ]:
        """
        批量评估动作 — 优化版：去重 Intra-DAG + 批量 Inter-DAG Attention

        Returns:
            action_logprobs: [batch]
            values: [batch]
            entropy: [batch]
            eft_aux_loss: processor EFT 交叉熵
            eft_top1_acc: processor EFT top-1 准确率
            task_aux_loss: 最小归一化 laxity 任务交叉熵
            task_top1_acc: SLA 任务 top-1 准确率
        """
        batch_size = len(batch_env_data['dag_task_static_features'])

        # ==================================================================
        # Phase 1: 去重 Intra-DAG 编码（同一 DAG 在 batch 中只编码一次）
        # ==================================================================
        unique_dag_idx = {}  # fingerprint -> unique_index
        unique_features = []
        unique_adjs = []
        # sample_dag_fps[i] = [fp0, fp1, ...]  每个样本每个DAG的指纹
        sample_dag_fps = []

        for i in range(batch_size):
            static_list = batch_env_data['dag_task_static_features'][i]
            adj_list = batch_env_data['dag_adjs'][i]
            fps = []
            for feat, adj in zip(static_list, adj_list):
                fp = self._dag_fingerprint(feat, adj)
                if fp not in unique_dag_idx:
                    unique_dag_idx[fp] = len(unique_features)
                    unique_features.append(feat.to(device))
                    unique_adjs.append(adj.to(device))
                fps.append(fp)
            sample_dag_fps.append(fps)

        # 只对唯一DAG做 Intra-DAG 编码
        if len(unique_features) > 0:
            unique_task_embs = self._encode_tasks(unique_features, unique_adjs)
            # fp -> task_embeddings [num_tasks, hidden_dim]
            fp_to_emb = {fp: unique_task_embs[idx] for fp, idx in unique_dag_idx.items()}
        else:
            fp_to_emb = {}

        # 重建每个样本的任务嵌入列表 + 预计算 dag_ids_for_tasks
        sample_task_embs_list = []  # List of List of [num_tasks_i, hidden_dim]
        sample_dag_ids_for_tasks = []  # List of [total_tasks] tensor (dag index per task)
        sample_task_counts = []
        sample_cum_task_offsets = []
        task_cum = 0

        for i in range(batch_size):
            fps = sample_dag_fps[i]
            embs = [fp_to_emb[fp] for fp in fps]
            sample_task_embs_list.append(embs)
            counts = [e.size(0) for e in embs]
            sample_task_counts.append(counts)
            sample_cum_task_offsets.append(task_cum)
            task_cum += sum(counts)

            # 预计算 dag_ids_for_tasks（每个任务属于哪个DAG）
            ids = []
            for di, c in enumerate(counts):
                ids.extend([di] * c)
            sample_dag_ids_for_tasks.append(
                torch.tensor(ids, device=device, dtype=torch.long)
            )

        # 拼接所有任务嵌入
        all_embs = []
        for embs in sample_task_embs_list:
            all_embs.extend(embs)
        all_task_embs_cat = torch.cat(all_embs, dim=0) if all_embs else \
            torch.zeros(0, self.intra_dag_encoder.hidden_dim, device=device)

        # ==================================================================
        # Phase 2: 批量 Inter-DAG Attention（DAG 过多时裁剪到 max_attn_dags）
        # ==================================================================
        original_max_dags = max(len(fps) for fps in sample_dag_fps)
        attn_max_dags = min(original_max_dags, self.max_attn_dags)
        need_prune = original_max_dags > self.max_attn_dags
        hd = self.intra_dag_encoder.hidden_dim

        # 全量池化 DAG 嵌入（被裁剪的 DAG 在 Phase 3 中用作 fallback）
        batch_dag_embs_full = torch.zeros(batch_size, original_max_dags, hd, device=device)
        for i in range(batch_size):
            embs = sample_task_embs_list[i]
            for di, emb in enumerate(embs):
                batch_dag_embs_full[i, di] = self._pool_dag(emb)

        # 构建裁剪后的批量 DAG 嵌入 + 元数据 + 邻接 mask
        batch_dag_embs = torch.zeros(batch_size, attn_max_dags, hd, device=device)
        batch_dag_meta = torch.zeros(batch_size, attn_max_dags, self.dag_meta_dim, device=device)
        batch_dag_adj = torch.zeros(batch_size, attn_max_dags, attn_max_dags, device=device)
        dag_attn_maps = []  # List of [n] tensor, mapping original_dag_idx -> attn_idx (-1=pruned)

        for i in range(batch_size):
            embs = sample_task_embs_list[i]
            n = len(embs)
            dag_meta_i = batch_env_data['dag_metadata'][i][:n].to(device)

            if n <= self.max_attn_dags:
                # 无需裁剪，直接映射
                mapping = torch.arange(n, device=device)
            else:
                # 找出有就绪任务的 DAG
                dag_ids = sample_dag_ids_for_tasks[i]
                ready_idx = batch_env_data['ready_task_indices'][i].to(device).long()
                ready_dag_set = set(dag_ids[ready_idx].tolist())
                ready_dag_mask = torch.zeros(n, dtype=torch.bool, device=device)
                for d in ready_dag_set:
                    ready_dag_mask[d] = True
                selected_idx = self._select_attn_dags(n, dag_meta_i, ready_dag_mask)
                mapping = torch.full((n,), -1, dtype=torch.long, device=device)
                for j, orig in enumerate(selected_idx.tolist()):
                    mapping[orig] = j

            dag_attn_maps.append(mapping)

            # 填充裁剪后的批量张量
            for orig_idx in range(n):
                attn_idx = mapping[orig_idx].item()
                if attn_idx >= 0:
                    batch_dag_embs[i, attn_idx] = batch_dag_embs_full[i, orig_idx]
                    batch_dag_meta[i, attn_idx] = dag_meta_i[orig_idx]

            # 邻接: 选中的 DAG 全连接，padding 仅自环
            n_sel = n if n <= self.max_attn_dags else self.max_attn_dags
            batch_dag_adj[i, :n_sel, :n_sel] = 1.0
            for di in range(n_sel, attn_max_dags):
                batch_dag_adj[i, di, di] = 1.0

        # 批量 Inter-DAG Attention（残差连接）— 通过 hook 方法调用
        dag_embs_with_meta = torch.cat([batch_dag_embs, batch_dag_meta], dim=-1)
        updated_dag_embs = self._encode_dags_batch(batch_dag_embs, batch_dag_meta, batch_dag_adj)

        # ==================================================================
        # Phase 3: 构建就绪任务嵌入（融合DAG嵌入）
        # ==================================================================
        all_ready_task_embs = []
        max_ready = 0

        for i in range(batch_size):
            task_counts = sample_task_counts[i]
            dag_ids_tensor = sample_dag_ids_for_tasks[i]

            ready_indices = batch_env_data['ready_task_indices'][i].to(device)
            task_offset = sample_cum_task_offsets[i]
            num_tasks = sum(task_counts)

            if num_tasks > 0 and all_task_embs_cat.size(0) > 0:
                task_embs_in_sample = all_task_embs_cat[task_offset:task_offset + num_tasks]
            else:
                task_embs_in_sample = torch.zeros(0, hd, device=device)

            if ready_indices.numel() > 0 and len(task_embs_in_sample) > 0:
                ready_idx = ready_indices.long()
                ready_task_embs = task_embs_in_sample[ready_idx]
                ready_dag_ids_orig = dag_ids_tensor[ready_idx]

                # 查找 DAG 嵌入（通过裁剪映射）
                mapping = dag_attn_maps[i]  # [n], -1 表示被裁剪
                attn_indices = mapping[ready_dag_ids_orig]  # [M]
                ready_dag_embs = torch.zeros(ready_idx.size(0), hd, device=device)
                sel = attn_indices >= 0
                if sel.any():
                    ready_dag_embs[sel] = updated_dag_embs[i, attn_indices[sel]]
                if not sel.all():
                    # 被裁剪的 DAG 使用原始池化嵌入
                    ready_dag_embs[~sel] = batch_dag_embs_full[i, ready_dag_ids_orig[~sel]]

                ready_task_embs = self._fuse_task_dag_context(ready_task_embs, ready_dag_embs)
            else:
                ready_task_embs = torch.zeros(1, hd, device=device)

            all_ready_task_embs.append(ready_task_embs)
            max_ready = max(max_ready, ready_task_embs.size(0))

        # ==================================================================
        # Phase 4: 两阶段决策（批量 task_scorer + per-sample proc_selector）
        # ==================================================================
        proc_features = batch_env_data['proc_features'].to(device)
        proc_states = batch_env_data['proc_states'].to(device)
        pair_features = batch_env_data.get('task_proc_pair_features')

        # Stage 1: 批量任务打分
        task_embs_batch = []
        task_dyn_batch = []
        for i in range(batch_size):
            ready_emb = all_ready_task_embs[i]
            ready_dyn = batch_env_data['ready_task_dyn_features'][i].to(device)
            if ready_emb.size(0) < max_ready:
                pad = torch.zeros(max_ready - ready_emb.size(0), ready_emb.size(1), device=device)
                ready_emb = torch.cat([ready_emb, pad], dim=0)
                dyn_pad = torch.zeros(max_ready - ready_dyn.size(0), ready_dyn.size(1), device=device)
                ready_dyn = torch.cat([ready_dyn, dyn_pad], dim=0)
            task_embs_batch.append(ready_emb)
            task_dyn_batch.append(ready_dyn)

        task_embs_packed = torch.stack(task_embs_batch)
        if self.two_stage_scheduler.dyn_dim > 0:
            task_dyn_packed = torch.stack(task_dyn_batch)
            task_emb_base = torch.cat([task_embs_packed, task_dyn_packed], dim=-1)
        else:
            task_emb_base = task_embs_packed

        if self.use_load_features:
            load_rows = []
            for i in range(batch_size):
                n_ready = int(batch_env_data['num_ready'][i])
                n_active = int(batch_env_data['num_active_dags'][i])
                gf = compute_global_features(
                    proc_features[i], n_ready, n_active
                ).to(device=device, dtype=task_emb_base.dtype)
                load_rows.append(gf.unsqueeze(0).expand(max_ready, -1))
            load_packed = torch.stack(load_rows, dim=0)
            task_emb_with_dyn = torch.cat([task_emb_base, load_packed], dim=-1)
        else:
            task_emb_with_dyn = task_emb_base

        task_logits_flat = self.two_stage_scheduler.task_scorer(task_emb_with_dyn).squeeze(-1)

        # Stage 2: 处理器选择（per-sample，因为每样本处理器特征不同）
        num_options = self.num_procs + (1 if self.two_stage_scheduler.enable_hold else 0)
        proc_logits_packed = torch.zeros(batch_size, max_ready, num_options, device=device)

        for i in range(batch_size):
            ready_embs = all_ready_task_embs[i]
            num_ready = ready_embs.size(0)  # Phase 4 用 padded 大小, 计算所有行（含 padding）
            if num_ready > 0:
                pair_i = None
                if pair_features is not None:
                    pair_i = pair_features[i][:num_ready].to(device)
                scores = self.two_stage_scheduler.score_processors(
                    ready_embs,
                    proc_features[i],
                    proc_states[i],
                    pair_i,
                )
                proc_logits_packed[i, :num_ready] = scores

        # ==================================================================
        # Phase 5: 计算 logprob / entropy / value（per-sample）
        # ==================================================================
        actions_task = batch_actions['task'].to(device)
        actions_proc = batch_actions['proc'].to(device)
        candidate_types_list = batch_env_data.get('candidate_types')
        if candidate_types_list is not None:
            candidate_types_list = [ct.to(device) for ct in candidate_types_list]

        logprobs_list = []
        values_list = []
        entropy_list = []
        eft_aux_sum = torch.tensor(0.0, device=device)
        eft_aux_count = 0
        eft_aux_correct = torch.tensor(0.0, device=device)
        eft_targets_list = batch_env_data.get('eft_targets')
        task_aux_sum = torch.tensor(0.0, device=device)
        task_aux_count = 0
        task_aux_correct = torch.tensor(0.0, device=device)
        sla_task_targets = batch_env_data.get('sla_task_targets')

        for i in range(batch_size):
            num_ready = batch_env_data['num_ready'][i]

            if num_ready == 0:
                logprobs_list.append(torch.tensor(0.0, device=device))
                values_list.append(torch.tensor(0.0, device=device))
                entropy_list.append(torch.tensor(0.0, device=device))
                continue

            task_log = task_logits_flat[i, :num_ready]
            proc_log = proc_logits_packed[i, :num_ready]

            # 每个状态监督一个最紧迫任务：argmin normalized laxity。
            if sla_task_targets is not None:
                task_target = sla_task_targets[i].to(device).long()
                if 0 <= task_target.item() < num_ready:
                    task_aux_sum = task_aux_sum + F.cross_entropy(
                        task_log.unsqueeze(0),
                        task_target.unsqueeze(0),
                        reduction='sum',
                    )
                    task_aux_correct = task_aux_correct + (
                        task_log.argmax(dim=-1) == task_target
                    ).float()
                    task_aux_count += 1

            # 对所有 ready task 提供密集的 EFT processor 排序监督。
            if eft_targets_list is not None:
                aux_log = proc_log.clone()
                if candidate_types_list is not None and i < len(candidate_types_list):
                    ct = candidate_types_list[i]
                    for k in range(num_ready):
                        aux_log[k] = self.two_stage_scheduler.apply_candidate_type_mask(
                            aux_log[k], ct[k].item()
                        )
                if self.two_stage_scheduler.enable_hold:
                    aux_log[:, -1] = -1e9
                targets = eft_targets_list[i][:num_ready].to(device).long()
                eft_aux_sum = eft_aux_sum + F.cross_entropy(
                    aux_log, targets, reduction='sum'
                )
                eft_aux_correct = eft_aux_correct + (
                    aux_log.argmax(dim=-1) == targets
                ).float().sum()
                eft_aux_count += num_ready

            if self.joint_matching:
                # 联合匹配：joint = task_scorer + proc_scores
                joint = task_log.unsqueeze(1) + proc_log  # [num_ready, num_opts]
                M_ready = joint.shape[0]
                n_opt = joint.shape[1]

                # 逐任务 candidate_type mask
                if candidate_types_list is not None and i < len(candidate_types_list):
                    ct = candidate_types_list[i]
                    if ct.numel() > 0:
                        for k in range(M_ready):
                            joint[k] = self.two_stage_scheduler.apply_candidate_type_mask(
                                joint[k], ct[k].item()
                            )

                # 与采集阶段一致：环境不支持 Hold 时屏蔽 Hold 槽位
                if self.two_stage_scheduler.enable_hold:
                    joint[:, -1] = -1e9

                joint_flat = joint.flatten()
                joint_dist = torch.distributions.Categorical(logits=joint_flat)
                action_id = actions_task[i] * n_opt + actions_proc[i]
                logprobs_list.append(joint_dist.log_prob(action_id))
                entropy_list.append(joint_dist.entropy())
            else:
                # 两阶段（原方案）
                task_probs = F.softmax(task_log, dim=-1)
                task_dist = torch.distributions.Categorical(task_probs)
                t_logprob = task_dist.log_prob(actions_task[i])

                proc_logits_sel = proc_log[actions_task[i]]
                if candidate_types_list is not None and i < len(candidate_types_list):
                    ct = candidate_types_list[i]
                    if ct.numel() > 0:
                        task_type = ct[actions_task[i]].item()
                        proc_logits_sel = self.two_stage_scheduler.apply_candidate_type_mask(
                            proc_logits_sel, task_type
                        )
                if self.two_stage_scheduler.enable_hold:
                    proc_logits_sel = proc_logits_sel.clone()
                    proc_logits_sel[-1] = -1e9
                proc_probs = F.softmax(proc_logits_sel, dim=-1)
                proc_dist = torch.distributions.Categorical(proc_probs)
                p_logprob = proc_dist.log_prob(actions_proc[i])

                # EDF 固定任务时仅用 proc logprob / entropy，与采集一致
                if getattr(self, "proc_only_logprob", False):
                    logprobs_list.append(p_logprob)
                    entropy_list.append(proc_dist.entropy().mean())
                else:
                    logprobs_list.append(t_logprob + p_logprob)
                    entropy_list.append(
                        task_dist.entropy().mean() + proc_dist.entropy().mean()
                    )

            # Value 估计（与采集阶段同一 compute_global_features）
            num_active = batch_env_data['num_active_dags'][i]
            num_ready = batch_env_data['num_ready'][i]
            global_feat = compute_global_features(proc_features[i], num_ready, num_active)
            value = self.estimate_value(all_ready_task_embs[i][:num_ready], global_feat)
            values_list.append(value.squeeze())

        action_logprobs = torch.stack(logprobs_list)
        values = torch.stack(values_list)
        entropy = torch.stack(entropy_list)
        eft_aux_loss = (
            eft_aux_sum / eft_aux_count
            if eft_aux_count > 0
            else torch.tensor(0.0, device=device)
        )
        eft_top1_acc = (
            eft_aux_correct / eft_aux_count
            if eft_aux_count > 0
            else torch.tensor(0.0, device=device)
        )
        task_aux_loss = (
            task_aux_sum / task_aux_count
            if task_aux_count > 0
            else torch.tensor(0.0, device=device)
        )
        task_top1_acc = (
            task_aux_correct / task_aux_count
            if task_aux_count > 0
            else torch.tensor(0.0, device=device)
        )

        return (
            action_logprobs,
            values,
            entropy,
            eft_aux_loss,
            eft_top1_acc,
            task_aux_loss,
            task_top1_acc,
        )


# ============================================================
# 工厂函数：创建完整调度器
# ============================================================

def create_hierarchical_scheduler(
    static_dim: int = STATIC_DIM,
    dyn_dim: int = DYN_DIM,
    dag_meta_dim: int = DAG_META_DIM,
    proc_dim: int = PROC_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_intra_layers: int = DEFAULT_INTRA_LAYERS,
    num_inter_layers: int = DEFAULT_INTER_LAYERS,
    max_dags: int = MAX_DAGS,
    max_candidates: int = MAX_CANDIDATES,
    num_procs: int = NUM_PROCS,
    dropout: float = DEFAULT_DROPOUT,
    enable_hold: bool = True,
    max_attn_dags: int = 64,
    joint_matching: bool = False,
    fusion_mode: str = 'gate',
    adj_direction: str = DEFAULT_ADJ_DIRECTION,
    use_pair_features: bool = False,
    use_load_features: bool = False,
) -> HierarchicalGATScheduler:
    """
    工厂函数：创建分层GAT调度器
    """
    return HierarchicalGATScheduler(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        dag_meta_dim=dag_meta_dim,
        proc_dim=proc_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_intra_layers=num_intra_layers,
        num_inter_layers=num_inter_layers,
        max_dags=max_dags,
        max_candidates=max_candidates,
        num_procs=num_procs,
        dropout=dropout,
        enable_hold=enable_hold,
        max_attn_dags=max_attn_dags,
        joint_matching=joint_matching,
        fusion_mode=fusion_mode,
        adj_direction=adj_direction,
        use_pair_features=use_pair_features,
        use_load_features=use_load_features,
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    from model.constants import STATIC_DIM, DYN_DIM, DAG_META_DIM, PROC_DIM, NUM_PROCS

    print("=" * 60)
    print("Phase 3: HierarchicalGATScheduler 测试")
    print("=" * 60)

    # 测试参数
    hidden_dim = 128
    num_active_dags = 3

    # ========== DAG邻接矩阵构建 ==========
    def build_dag_adjacency(num_tasks: int, parents: List[List[int]]) -> torch.Tensor:
        """
        根据父子关系构建DAG邻接矩阵
        """
        adj = torch.zeros(num_tasks, num_tasks, dtype=torch.float32)
        for child_id, parent_list in enumerate(parents):
            for parent_id in parent_list:
                adj[parent_id, child_id] = 1.0
        for i in range(num_tasks):
            adj[i, i] = 1.0
        return adj

    # 模拟数据: 创建具有真实依赖关系的DAG
    num_tasks_per_dag = [4, 3, 5]
    dag_parents = [
        [[], [0], [1], [2]],           # DAG 0: 线性链 0->1->2->3
        [[], [], [0, 1]],              # DAG 1: 并行 0,1 -> 2
        [[], [0], [0], [1, 2], [3, 4]]  # DAG 2: 复杂DAG
    ]

    dag_task_static_features = []
    dag_adjs = []
    task_dag_ids_list = []

    for dag_id, num_tasks in enumerate(num_tasks_per_dag):
        features = torch.randn(num_tasks, STATIC_DIM)
        dag_task_static_features.append(features)
        # 使用真实依赖关系构建邻接矩阵
        adj = build_dag_adjacency(num_tasks, dag_parents[dag_id])
        dag_adjs.append(adj)
        task_dag_ids_list.extend([dag_id] * num_tasks)

    task_dag_ids = torch.tensor(task_dag_ids_list, dtype=torch.long)

    dag_metadata = torch.randn(num_active_dags, DAG_META_DIM)

    # 就绪任务（假设所有任务都就绪）
    ready_task_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)
    ready_task_dyn_features = torch.randn(len(ready_task_indices), DYN_DIM)
    proc_features = torch.randn(NUM_PROCS, PROC_DIM)
    proc_states = torch.tensor([100.0, 150.0, 80.0, 120.0, 90.0, 110.0, 130.0, 95.0, 105.0])

    # ========== HierarchicalGATScheduler 测试 ==========
    print("\n[HierarchicalGATScheduler]")

    scheduler = HierarchicalGATScheduler(
        static_dim=STATIC_DIM,
        dyn_dim=DYN_DIM,
        dag_meta_dim=DAG_META_DIM,
        proc_dim=PROC_DIM,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_intra_layers=2,
        num_inter_layers=2,
        num_procs=NUM_PROCS
    )

    # forward 测试
    task_logits, proc_logits, info = scheduler(
        dag_task_static_features, dag_adjs, dag_metadata,
        ready_task_indices, ready_task_dyn_features,
        proc_features, proc_states, task_dag_ids
    )

    print(f"  task_logits shape: {task_logits.shape}")
    print(f"  proc_logits shape: {proc_logits.shape}")
    print(f"  dag_embeddings shape: {info['dag_embeddings'].shape}")
    print(f"  ready_task_embs shape: {info['ready_task_embs'].shape}")

    # get_action 测试
    print("\n[get_action 测试]")
    task_id, proc_id, value = scheduler.get_action(
        dag_task_static_features, dag_adjs, dag_metadata,
        ready_task_indices, ready_task_dyn_features,
        proc_features, proc_states, task_dag_ids
    )
    print(f"  task_id: {task_id.item()}, proc_id: {proc_id.item()}")
    print(f"  value: {value.item():.4f}")

    # evaluate_actions 测试
    print("\n[evaluate_actions 测试]")
    actions_task = torch.tensor([0, 1, 2])
    actions_proc = torch.tensor([0, 1, 2])
    logprobs, values, entropy = scheduler.evaluate_actions(
        dag_task_static_features, dag_adjs, dag_metadata,
        ready_task_indices, ready_task_dyn_features,
        proc_features, proc_states, task_dag_ids,
        actions_task, actions_proc
    )
    print(f"  logprobs shape: {logprobs.shape}")
    print(f"  values shape: {values.shape}")
    print(f"  entropy shape: {entropy.shape}")

    # 工厂函数测试
    print("\n[工厂函数测试]")
    scheduler2 = create_hierarchical_scheduler()
    print(f"  Scheduler created: {sum(p.numel() for p in scheduler2.parameters()):,} parameters")

    print("\n" + "=" * 60)
    print("Phase 3 测试通过！")
    print("=" * 60)