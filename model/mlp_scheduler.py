"""
纯 MLP + PPO 调度器 — 消融实验 Baseline

继承 HierarchicalGATScheduler，仅覆盖 forward() 和 estimate_value()，
将 GAT 替换为 MLP。PPO / joint matching / evaluate_actions_batch 全部复用。
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List

from .constants import (
    STATIC_DIM, DYN_DIM, DAG_META_DIM, PROC_DIM, NUM_PROCS,
    DEFAULT_HIDDEN_DIM, DEFAULT_DROPOUT
)
from .hierarchical_scheduler import HierarchicalGATScheduler


class MLPScheduler(HierarchicalGATScheduler):
    """
    纯 MLP 调度器 — 仅覆盖 forward()，其余全部复用 GAT 版本。

    改动：
    - Layer 1 (Intra-DAG GAT) → MLP (static_dim → hidden_dim)
    - Layer 2 (Inter-DAG Attention) → MLP (dag_meta_dim → hidden_dim)
    - Layer 3 (TwoStageScheduler) → 复用
    - Critic → 复用
    """

    def __init__(
        self,
        static_dim: int = STATIC_DIM,
        dyn_dim: int = DYN_DIM,
        dag_meta_dim: int = DAG_META_DIM,
        proc_dim: int = PROC_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_procs: int = NUM_PROCS,
        dropout: float = DEFAULT_DROPOUT,
        enable_hold: bool = True,
        max_attn_dags: int = 64,
        joint_matching: bool = True,
        fusion_mode: str = 'gate',
        use_pair_features: bool = False,
        use_load_features: bool = False,
    ):
        # 先调父类 init（会创建 GAT 模块，之后被 MLP 覆盖）
        super().__init__(
            static_dim=static_dim, dyn_dim=dyn_dim,
            dag_meta_dim=dag_meta_dim, proc_dim=proc_dim,
            hidden_dim=hidden_dim, num_procs=num_procs,
            dropout=dropout, enable_hold=enable_hold,
            max_attn_dags=max_attn_dags,
            joint_matching=joint_matching,
            fusion_mode=fusion_mode,
            use_pair_features=use_pair_features,
            use_load_features=use_load_features,
        )

        # -------- 覆盖 Layer 1: MLP 任务编码 --------
        self.task_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # -------- 覆盖 Layer 2: MLP DAG 编码（无 attention） --------
        self.dag_mlp = nn.Sequential(
            nn.Linear(dag_meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer 3 和 Critic 继承自父类，不动

    # ================================================================
    # 重载 hook 方法 → evaluate_actions_batch 也走 MLP 编码
    # ================================================================

    def _encode_tasks(self, features, adjs):
        return [self.task_mlp(f) for f in features]

    def _encode_dags(self, dag_embs, metadata):
        return self.dag_mlp(metadata)

    def _pool_dag(self, task_embs):
        return task_embs.mean(dim=0)

    def _encode_dags_batch(self, dag_embs, dag_meta, dag_adj):
        B, N, D = dag_meta.shape
        return self.dag_mlp(dag_meta.view(B * N, D)).view(B, N, -1)

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
        task_proc_pair_features: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        model_device = next(self.parameters()).device
        static_features = [f.to(model_device) for f in dag_task_static_features]
        metadata = dag_metadata.to(model_device)
        ready_idx = ready_task_indices.to(model_device)
        ready_dyn = ready_task_dyn_features.to(model_device)
        proc_feat = proc_features.to(model_device)
        proc_st = proc_states.to(model_device)
        dag_ids = task_dag_ids.to(model_device)

        # ===== Step 1: MLP 任务编码（替代 Intra-DAG GAT） =====
        dag_task_embs_list = [self.task_mlp(f) for f in static_features]

        # ===== Step 2: 均值池化 =====
        dag_embeddings = torch.stack([
            self.inter_dag_attention.attention_pool(emb) for emb in dag_task_embs_list
        ])

        # ===== Step 3: MLP DAG 编码（替代 Inter-DAG Attention） =====
        updated_dag_embeddings = self.dag_mlp(metadata)

        # ===== Step 4: 就绪任务嵌入 =====
        all_task_embs = torch.cat(dag_task_embs_list, dim=0)
        ready_task_embs = all_task_embs[ready_idx]
        ready_dag_ids = dag_ids[ready_idx]
        ready_dag_embs = updated_dag_embeddings[ready_dag_ids]
        ready_task_embs = self._fuse_task_dag_context(ready_task_embs, ready_dag_embs)

        # ===== Step 5: 两阶段决策（复用） =====
        task_logits, proc_logits = self.two_stage_scheduler(
            ready_task_embs,
            ready_dyn,
            proc_feat,
            proc_st,
            task_proc_pair_features,
        )

        info = {
            'dag_embeddings': dag_embeddings,
            'updated_dag_embeddings': updated_dag_embeddings,
            'ready_task_embs': ready_task_embs,
        }
        return task_logits, proc_logits, info


def create_mlp_scheduler(
    static_dim: int = STATIC_DIM, dyn_dim: int = DYN_DIM,
    dag_meta_dim: int = DAG_META_DIM, proc_dim: int = PROC_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM, num_procs: int = NUM_PROCS,
    dropout: float = DEFAULT_DROPOUT, enable_hold: bool = True,
    max_attn_dags: int = 64, joint_matching: bool = True,
    fusion_mode: str = 'gate',
    use_pair_features: bool = False,
    use_load_features: bool = False,
) -> MLPScheduler:
    return MLPScheduler(
        static_dim=static_dim, dyn_dim=dyn_dim, dag_meta_dim=dag_meta_dim,
        proc_dim=proc_dim, hidden_dim=hidden_dim, num_procs=num_procs,
        dropout=dropout, enable_hold=enable_hold,
        max_attn_dags=max_attn_dags, joint_matching=joint_matching,
        fusion_mode=fusion_mode, use_pair_features=use_pair_features,
        use_load_features=use_load_features,
    )
