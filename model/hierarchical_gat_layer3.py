"""
分层图注意力调度网络 - Phase 2
Layer 3: Two-Stage Decision Scheduler

包含:
- TwoStageScheduler: 两阶段决策器（Task Scorer + Proc Selector）
- HoldVector: Global Hold 机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

from .constants import (
    DYN_DIM, PROC_DIM, NUM_PROCS, NUM_CLOUD, DEFAULT_HIDDEN_DIM, DEFAULT_DROPOUT,
    PROC_AVAILABLE_NORM_IDX, HOLD_NORM_CLIP, PAIR_FEATURE_DIM, GLOBAL_FEATURES_DIM,
)


# ============================================================
# Layer 3: Two-Stage Decision Scheduler
# ============================================================

class TwoStageScheduler(nn.Module):
    """
    Layer 3: 两阶段决策器

    阶段1: Task_Scorer（拼接动态特征后打分，共享参数输出维度永远为1）
    阶段2: Proc_Selector（含Hold槽位）

    支持动态M: Task_Scorer对每个任务独立打分，输出维度动态
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dyn_dim: int = DYN_DIM,
        proc_dim: int = PROC_DIM,
        num_procs: int = NUM_PROCS,
        dropout: float = DEFAULT_DROPOUT,
        enable_hold: bool = True,
        use_pair_features: bool = False,
        use_load_features: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dyn_dim = dyn_dim
        self.num_procs = num_procs
        self.enable_hold = enable_hold
        self.use_pair_features = use_pair_features
        self.use_load_features = use_load_features
        self.load_feat_dim = GLOBAL_FEATURES_DIM if use_load_features else 0

        scorer_in = hidden_dim + dyn_dim if dyn_dim > 0 else hidden_dim
        scorer_in += self.load_feat_dim
        # ========== Stage 1: 任务打分器（共享参数，输入拼接动态特征） ==========
        # dyn_dim=0 时仅使用结构嵌入（特征分离消融 A3）
        # use_load_features 时再拼接全局负载 [queue_pressure, ready_ratio, active_dag_ratio]
        self.task_scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出永远是1维
        )
        if use_load_features:
            # 负载通道零初始化：开启时初始行为接近无负载特征
            first = self.task_scorer[0]
            with torch.no_grad():
                first.weight[:, -GLOBAL_FEATURES_DIM:].zero_()

        # ========== Stage 2: 处理器选择器（含Hold） ==========
        self.proc_encoder = nn.Sequential(
            nn.Linear(proc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 显式编码 task×processor 的 EFT 构成，避免双塔 MLP 隐式学习除法、
        # 通信聚合和 LFT 裕量。仅新实验启用，旧 checkpoint 保持结构兼容。
        if use_pair_features:
            self.pair_encoder = nn.Sequential(
                nn.Linear(PAIR_FEATURE_DIM, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            nn.init.zeros_(self.pair_encoder[-1].weight)
            nn.init.zeros_(self.pair_encoder[-1].bias)
        else:
            self.pair_encoder = None

        # Hold向量编码器（全局等待成本：归一化等待时间的 min/mean/max/std）
        self.hold_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 处理器选择网络（双塔MLP）
        self.proc_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # N个物理处理器 + 1个Global Hold
        )

    @staticmethod
    def _compute_hold_features(proc_features: torch.Tensor) -> torch.Tensor:
        """
        从 per-processor 已归一化等待时间聚合 Hold 特征。

        proc_features[:, PROC_AVAILABLE_NORM_IDX] 与 Device.get_features 一致，
        为 log(wait+1) 后经 wait_time_mean/std 的 z-score。
        """
        wait_norms = proc_features[:, PROC_AVAILABLE_NORM_IDX]
        if wait_norms.numel() > 1:
            std_val = wait_norms.std(unbiased=False)
        else:
            std_val = wait_norms.new_zeros(())
        hold_features = torch.stack([
            wait_norms.min(),
            wait_norms.mean(),
            wait_norms.max(),
            std_val,
        ])
        return hold_features.clamp(-HOLD_NORM_CLIP, HOLD_NORM_CLIP)

    def build_proc_embedding_with_hold(
        self,
        proc_features: torch.Tensor,
        proc_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        构建含Hold槽位的处理器嵌入矩阵

        Args:
            proc_features: [N, proc_dim] 处理器特征（含 Log+Z-Score 等待时间）
            proc_states: 已弃用，保留仅为向后兼容；Hold 特征从 proc_features 派生

        Returns:
            proc_emb_full: [N+1 or N, hidden_dim] 含/不含Hold向量的处理器嵌入
        """
        proc_emb = self.proc_encoder(proc_features)  # [N, hidden_dim]

        if not self.enable_hold:
            return proc_emb

        hold_features = self._compute_hold_features(proc_features)
        hold_emb = self.hold_encoder(hold_features)  # [hidden_dim]

        proc_emb_full = torch.cat([proc_emb, hold_emb.unsqueeze(0)], dim=0)  # [N+1, hidden_dim]
        return proc_emb_full

    def apply_candidate_type_mask(
        self,
        proc_logits: torch.Tensor,
        task_type: int
    ) -> torch.Tensor:
        """
        应用candidate_type约束的mask

        Args:
            proc_logits: [N+1 or N] 处理器选择logits
            task_type: 0=cloud_only, 1=edge_only, 2=flexible

        Returns:
            masked_proc_logits: 应用mask后的logits
        """
        num_options = proc_logits.size(0)

        if task_type == 0:  # cloud_only
            proc_mask = torch.zeros(num_options, device=proc_logits.device)
            proc_mask[:NUM_CLOUD] = 1.0  # 云处理器
            if self.enable_hold:
                proc_mask[-1] = 1.0  # Hold
        elif task_type == 1:  # edge_only
            proc_mask = torch.zeros(num_options, device=proc_logits.device)
            proc_mask[NUM_CLOUD:] = 1.0  # 边缘处理器
            if self.enable_hold:
                proc_mask[-1] = 1.0  # Hold
        else:  # flexible
            proc_mask = torch.ones(num_options, device=proc_logits.device)

        return proc_logits + (1 - proc_mask) * -1e9

    def build_task_scorer_input(
        self,
        task_embs: torch.Tensor,
        task_dyn_features: Optional[torch.Tensor] = None,
        global_load_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        构造 task_scorer 输入: [emb | dyn?] | load?]

        global_load_features: [3] 或 [M, 3]；开启 use_load_features 且为 None 时填 0。
        """
        if self.dyn_dim > 0:
            if task_dyn_features is None:
                raise ValueError("dyn_dim>0 时必须提供 task_dyn_features")
            x = torch.cat([task_embs, task_dyn_features], dim=-1)
        else:
            x = task_embs

        if not self.use_load_features:
            return x

        m = x.size(0)
        if global_load_features is None:
            g = x.new_zeros(m, GLOBAL_FEATURES_DIM)
        else:
            g = global_load_features.to(device=x.device, dtype=x.dtype)
            if g.dim() == 1:
                g = g.unsqueeze(0).expand(m, -1)
            elif g.dim() == 2 and g.size(0) == 1 and m > 1:
                g = g.expand(m, -1)
            elif g.dim() != 2 or g.size(0) != m or g.size(-1) != GLOBAL_FEATURES_DIM:
                raise ValueError(
                    f"global_load_features 期望 [3] 或 [{m}, {GLOBAL_FEATURES_DIM}]，"
                    f"实际 {tuple(g.shape)}"
                )
        return torch.cat([x, g], dim=-1)

    def forward(
        self,
        task_embs: torch.Tensor,
        task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        task_proc_pair_features: Optional[torch.Tensor] = None,
        global_load_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_embs: [M, hidden_dim] 就绪任务结构嵌入（M动态变化）
            task_dyn_features: [M, dyn_dim] 任务动态运行时特征（9维）
            proc_features: [N, proc_dim] 处理器特征
            proc_states: [N] 处理器剩余等待时间（原始秒数，Layer3 不再直接使用）
            global_load_features: [3] 可选，全局负载特征（与 Critic 同源）

        Returns:
            task_logits: [M] 任务选择logits
            proc_logits: [M, N+1] 处理器选择logits（每行对应一个任务）
        """
        task_emb_with_dyn = self.build_task_scorer_input(
            task_embs, task_dyn_features, global_load_features
        )
        task_logits = self.task_scorer(task_emb_with_dyn).squeeze(-1)  # [M]

        # Stage 2: 处理器选择（向量化版本）
        proc_logits = self.score_processors(
            task_embs,
            proc_features,
            proc_states,
            task_proc_pair_features,
        )

        return task_logits, proc_logits

    def score_processors(
        self,
        task_embs: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        task_proc_pair_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算每个 ready task × processor 的分数。"""
        proc_emb_full = self.build_proc_embedding_with_hold(proc_features, proc_states)  # [N+1 or N, hidden_dim]
        num_options = proc_emb_full.size(0)
        M = task_embs.size(0)

        # 向量化：一次性计算所有任务的处理器分数
        # task_embs: [M, h] → [M, 1, h] → [M, N+1, h]
        task_expanded = task_embs.unsqueeze(1).expand(M, num_options, -1)
        # proc_emb_full: [N+1, h] → [1, N+1, h] → [M, N+1, h]
        proc_expanded = proc_emb_full.unsqueeze(0).expand(M, -1, -1)
        if self.use_pair_features:
            if task_proc_pair_features is None:
                raise ValueError(
                    "use_pair_features=True 时必须提供 task_proc_pair_features"
                )
            pair = task_proc_pair_features.to(
                device=task_embs.device, dtype=task_embs.dtype
            )
            if pair.shape[:2] != (M, self.num_procs):
                raise ValueError(
                    "task_proc_pair_features shape 应为 "
                    f"[{M}, {self.num_procs}, *]，实际为 {tuple(pair.shape)}"
                )
            pair_emb = self.pair_encoder(pair)
            if self.enable_hold:
                hold_pair = torch.zeros(
                    M, 1, self.hidden_dim,
                    device=task_embs.device,
                    dtype=task_embs.dtype,
                )
                pair_emb = torch.cat([pair_emb, hold_pair], dim=1)
            proc_expanded = proc_expanded + pair_emb
        # 拼接: [M, N+1, 2h]
        combined = torch.cat([task_expanded, proc_expanded], dim=-1)
        # 批量计算: [M, N+1]
        proc_logits = self.proc_selector(combined).squeeze(-1)

        return proc_logits

    def forward_from_cached(
        self,
        ready_task_embs: torch.Tensor,
        task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        task_proc_pair_features: Optional[torch.Tensor] = None,
        global_load_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从缓存的嵌入直接计算两阶段决策（跳过 Layer 1+2）

        Args:
            ready_task_embs: [M, hidden_dim] 预计算的就绪任务嵌入
            task_dyn_features: [M, dyn_dim] 任务动态特征
            proc_features: [N, proc_dim]
            proc_states: [N]
            valid_mask: [M] 可选，任务有效掩码
            global_load_features: [3] 可选全局负载特征

        Returns:
            task_logits: [M] 任务选择logits
            proc_logits: [M, N+1] 处理器选择logits
        """
        task_emb_with_dyn = self.build_task_scorer_input(
            ready_task_embs, task_dyn_features, global_load_features
        )
        task_logits = self.task_scorer(task_emb_with_dyn).squeeze(-1)

        # 应用有效掩码
        if valid_mask is not None:
            task_logits = task_logits + (1 - valid_mask.float()) * -1e9

        # Stage 2: 处理器选择
        proc_logits = self.score_processors(
            ready_task_embs,
            proc_features,
            proc_states,
            task_proc_pair_features,
        )

        return task_logits, proc_logits

    def get_action(
        self,
        task_embs: torch.Tensor,
        task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        candidate_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        两阶段采样: 先选任务，再选处理器

        Args:
            task_embs: [M, hidden_dim] 任务结构嵌入
            task_dyn_features: [M, dyn_dim] 任务动态特征
            proc_features: [N, proc_dim]
            proc_states: [N]
            valid_mask: [M] 任务有效掩码（被Hold的任务为False）
            candidate_types: [M] 任务类型 (0=cloud_only, 1=edge_only, 2=flexible)

        Returns:
            task_id: 选中的任务索引
            proc_id: 选中的处理器索引（N为Global Hold）
        """
        task_logits, proc_logits = self.forward(task_embs, task_dyn_features, proc_features, proc_states)

        # 应用有效掩码
        if valid_mask is not None:
            task_logits = task_logits + (1 - valid_mask.float()) * -1e9

        # Stage 1: 选择任务
        task_probs = F.softmax(task_logits, dim=-1)
        task_dist = torch.distributions.Categorical(task_probs)
        task_id = task_dist.sample()

        # Stage 2: 选择处理器（基于选中的任务，应用candidate_type约束）
        proc_logits_selected = proc_logits[task_id]

        # 应用candidate_type约束的mask
        if candidate_types is not None:
            task_type = candidate_types[task_id].item()
            proc_logits_selected = self.apply_candidate_type_mask(proc_logits_selected, task_type)

        proc_probs = F.softmax(proc_logits_selected, dim=-1)
        proc_dist = torch.distributions.Categorical(proc_probs)
        proc_id = proc_dist.sample()

        return task_id, proc_id

    def get_action_joint(
        self,
        task_embs: torch.Tensor,
        task_dyn_features: torch.Tensor,
        proc_features: torch.Tensor,
        proc_states: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        candidate_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        联合匹配采样: task_scorer + proc_selector → [M×(N+1)] → 一次采样

        joint_logits[i,j] = task_scorer(task_i) + proc_selector(task_i, proc_j)
        让任务选择时能感知处理器适配度。

        Args:
            task_embs: [M, hidden_dim] 任务结构嵌入
            task_dyn_features: [M, dyn_dim] 任务动态特征
            proc_features: [N, proc_dim]
            proc_states: [N]
            valid_mask: [M] 任务有效掩码
            candidate_types: [M] 任务类型 (0=cloud_only, 1=edge_only, 2=flexible)

        Returns:
            task_id: 选中的任务索引 (0..M-1)
            proc_id: 选中的处理器索引 (N=Global Hold)
        """
        task_logits, proc_logits = self.forward(task_embs, task_dyn_features, proc_features, proc_states)

        # 联合分数: urgency(task_i) + compatibility(task_i, proc_j)
        joint = task_logits.unsqueeze(1) + proc_logits  # [M, N+1]
        M, num_opts = joint.shape

        # 应用 candidate_type 约束（逐任务）
        if candidate_types is not None:
            for i in range(M):
                joint[i] = self.apply_candidate_type_mask(joint[i], candidate_types[i].item())

        # 应用有效掩码（被 Hold 的任务）
        if valid_mask is not None:
            joint = joint + (1 - valid_mask.float()).unsqueeze(1) * -1e9

        # 拍平为 [M * num_opts]，一次 Categorical 采样
        joint_flat = joint.flatten()
        joint_dist = torch.distributions.Categorical(logits=joint_flat)
        action_id = joint_dist.sample()

        task_id = action_id // num_opts
        proc_id = action_id % num_opts

        return task_id, proc_id

# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Layer 3 - TwoStageScheduler 测试")
    print("=" * 60)

    # 测试参数
    hidden_dim = 128
    dyn_dim = 8
    proc_dim = 4
    num_procs = 9
    max_candidates = 15

    # 模拟数据：proc_features[:,1] 为已归一化等待时间 (z-score)
    M = 8
    task_embs = torch.randn(M, hidden_dim)
    task_dyn_features = torch.randn(M, dyn_dim)
    proc_features = torch.randn(num_procs, proc_dim)
    proc_features[:, 1] = torch.tensor([-1.2, 0.0, 0.5, 1.1, -0.3, 0.8, 2.0, -0.5, 0.2])
    proc_states = torch.tensor([10.0, 50.0, 80.0, 120.0, 30.0, 90.0, 200.0, 40.0, 60.0])

    # ========== TwoStageScheduler 测试 ==========
    print("\n[TwoStageScheduler]")
    scheduler = TwoStageScheduler(
        hidden_dim=hidden_dim,
        dyn_dim=dyn_dim,
        proc_dim=proc_dim,
        num_procs=num_procs
    )

    # forward 测试
    task_logits, proc_logits = scheduler(task_embs, task_dyn_features, proc_features, proc_states)
    hold_feat = TwoStageScheduler._compute_hold_features(proc_features)
    print(f"  hold_features (normalized): {hold_feat.tolist()}")
    print(f"  task_logits shape: {task_logits.shape}")
    print(f"  proc_logits shape: {proc_logits.shape}")

    print("\n" + "=" * 60)
    print("Phase 2 测试通过！")
    print("=" * 60)