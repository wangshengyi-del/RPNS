"""
GAT+PPO 消融调度器

A2 w/o Inter-DAG: 跳过 Layer 2 self-attention，仅池化 + DAG 元数据投影
A3 w/o 特征分离: 12+7 维混合特征输入 Layer 1 GAT，Layer 3 不再单独拼接动态特征
"""

import torch
from typing import List

from .constants import (
    STATIC_DIM, DYN_DIM, MIXED_FEATURE_DIM, DAG_META_DIM, PROC_DIM, NUM_PROCS,
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_HEADS, DEFAULT_INTRA_LAYERS, DEFAULT_INTER_LAYERS,
    DEFAULT_DROPOUT, DEFAULT_ADJ_DIRECTION,
)
from .hierarchical_gat import IntraDAGEncoder
from .hierarchical_gat_layer3 import TwoStageScheduler
from .hierarchical_scheduler import HierarchicalGATScheduler


class NoInterDAGScheduler(HierarchicalGATScheduler):
    """
    消融 A2: w/o Inter-DAG Attention

    保留 Intra-DAG GAT 与静态/动态特征分离；Layer 2 仅 dag_proj(pool ⊕ meta)，无 DAG 间 GNN。
    """

    ablation_mode = 'no_inter'

    def _encode_dags(self, dag_embs: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        dag_embs_with_meta = torch.cat([dag_embs, metadata], dim=-1)
        return self.inter_dag_attention.dag_proj(dag_embs_with_meta)

    def _encode_dags_batch(
        self,
        dag_embs: torch.Tensor,
        dag_meta: torch.Tensor,
        dag_adj: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = dag_meta.shape
        dag_embs_with_meta = torch.cat([dag_embs, dag_meta], dim=-1)
        flat = dag_embs_with_meta.reshape(b * n, -1)
        return self.inter_dag_attention.dag_proj(flat).view(b, n, -1)


class MixedFeaturesScheduler(HierarchicalGATScheduler):
    """
    消融 A3: w/o 静态/动态特征分离

    Layer 1 GAT 输入 19 维（static + dynamic）；Layer 3 Task Scorer 仅使用 GAT 嵌入。
    """

    ablation_mode = 'mixed_features'
    mixed_features = True

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
        max_dags: int = 10,
        max_candidates: int = 15,
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
        super().__init__(
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
        # 覆盖 Layer 1：混合特征维度
        self.intra_dag_encoder = IntraDAGEncoder(
            static_dim=MIXED_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_intra_layers,
            dropout=dropout,
        )
        # 覆盖 Layer 3：决策层不再单独接收动态特征
        self.two_stage_scheduler = TwoStageScheduler(
            hidden_dim=hidden_dim,
            dyn_dim=0,
            proc_dim=proc_dim,
            num_procs=num_procs,
            dropout=dropout,
            enable_hold=enable_hold,
            use_pair_features=use_pair_features,
            use_load_features=use_load_features,
        )


def create_no_inter_scheduler(
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
    adj_direction: str = DEFAULT_ADJ_DIRECTION,
    use_pair_features: bool = False,
    use_load_features: bool = False,
) -> NoInterDAGScheduler:
    return NoInterDAGScheduler(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        dag_meta_dim=dag_meta_dim,
        proc_dim=proc_dim,
        hidden_dim=hidden_dim,
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


def create_mixed_features_scheduler(
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
    adj_direction: str = DEFAULT_ADJ_DIRECTION,
    use_pair_features: bool = False,
    use_load_features: bool = False,
) -> MixedFeaturesScheduler:
    return MixedFeaturesScheduler(
        static_dim=static_dim,
        dyn_dim=dyn_dim,
        dag_meta_dim=dag_meta_dim,
        proc_dim=proc_dim,
        hidden_dim=hidden_dim,
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
