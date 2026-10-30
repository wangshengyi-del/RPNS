"""
分层GAT调度器 - PPO训练脚本
与 ARCHITECTURE_DESIGN.md 对齐

训练: 动态生成 DAG
验证: 固定池
测试: 固定池
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import random
from typing import Dict, Optional, Tuple
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.multi_dag_environment import (
    EVAL_ARRIVAL_SEED,
    TEST_ARRIVAL_SEED,
    MultiDAGSchedulerEnv,
    WAIT_TIME_MEAN,
    WAIT_TIME_STD,
    estimate_schedule_timing,
)
from env.dag_task import TaskStatus
from model.hierarchical_scheduler import HierarchicalGATScheduler, compute_global_features
from model.mlp_scheduler import MLPScheduler, create_mlp_scheduler
from model.ablation_scheduler import create_no_inter_scheduler, create_mixed_features_scheduler
from replaybuffer_hierarchical import HierarchicalReplayBuffer
from model.constants import (
    STATIC_DIM, DYN_DIM, DAG_META_DIM, PROC_DIM, NUM_PROCS, NUM_CLOUD,
    CLOUD_MIPS, CLOUD_POWER, PAIR_FEATURE_CLIP,
    ORIGINAL_TASK_DIM,
    DEFAULT_HIDDEN_DIM, DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA, DEFAULT_EPSILON,
    DEFAULT_VALUE_COEF, DEFAULT_MAX_GRAD_NORM,
    DEFAULT_LR_DECAY, DEFAULT_PPO_EPOCHS, GLOBAL_FEATURES_DIM
)


def resolve_device(requested: str = 'auto') -> str:
    """
    解析训练设备。

    auto / gpu: cuda > mps > cpu
    cpu / mps / cuda: 显式指定；不可用时回退 cpu 并打印警告
    """
    req = requested.lower().strip()
    if req in ('auto', 'gpu'):
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if req == 'cuda':
        if not torch.cuda.is_available():
            print('警告: CUDA 不可用，回退到 CPU')
            return 'cpu'
        return 'cuda'

    if req == 'mps':
        if not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
            print('警告: MPS (Apple GPU) 不可用，回退到 CPU')
            return 'cpu'
        return 'mps'

    if req == 'cpu':
        return 'cpu'

    raise ValueError(f"未知设备 {requested!r}，请使用 auto/cpu/mps/cuda/gpu")


def sync_device(device: str) -> None:
    """等待加速器完成（用于计时或调试）。"""
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def configure_cpu_threads(device: str, num_threads: int = 0) -> None:
    """CPU 线程数：GPU 训练时减少 CPU 争用；纯 CPU 训练可用更多线程。"""
    if num_threads > 0:
        n = num_threads
    elif device == 'cpu':
        n = min(20, os.cpu_count() or 8)
    else:
        n = min(6, os.cpu_count() or 4)
    torch.set_num_threads(n)


def set_seed(seed: int, device: str = 'cpu'):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def linear_entropy_coef(
    episode_idx: int,
    total_episodes: int,
    start_coef: float,
    end_coef: float,
    start_episode: int = 0,
    decay_episodes: int = 0,
) -> float:
    """熵系数调度。

    - decay_episodes > 0：自 start_episode 起前 decay_episodes 轮从 start→end 线性衰减，
      之后保持 end（例如 201 ep、decay=100 → ep0..99 衰减，ep100.. 固定为 end）。
    - decay_episodes <= 0：在整个 [start_episode, total_episodes) 上线性插值（旧行为）。

    start_episode>0 时按微调窗口计本地 episode，避免续训一开始就被拉到中段。
    """
    if start_coef == end_coef:
        return start_coef

    local = episode_idx - start_episode
    if local <= 0:
        return start_coef

    if decay_episodes > 0:
        if decay_episodes == 1 or local >= decay_episodes - 1:
            return end_coef
        progress = local / (decay_episodes - 1)
        return start_coef + (end_coef - start_coef) * progress

    # 全时长线性（兼容旧逻辑）
    if total_episodes <= 1:
        return end_coef
    denom = max(total_episodes - start_episode - 1, 1)
    progress = local / denom
    progress = min(max(progress, 0.0), 1.0)
    return start_coef + (end_coef - start_coef) * progress


def linear_aux_coef_to_zero(
    episode_idx: int,
    initial_coef: float,
    decay_episodes: int,
) -> float:
    """前 decay_episodes 轮将辅助损失系数从 initial_coef 线性降到 0。"""
    if initial_coef <= 0:
        return 0.0
    if decay_episodes <= 0:
        return initial_coef
    if decay_episodes == 1 or episode_idx >= decay_episodes - 1:
        return 0.0
    progress = max(0.0, episode_idx / (decay_episodes - 1))
    return initial_coef * (1.0 - progress)


def _uses_mixed_features(model) -> bool:
    return getattr(model, 'mixed_features', False)


RULE_TASK_POLICIES = ("edf", "est")


def rule_ready_task_index(
    env: MultiDAGSchedulerEnv,
    ready_candidate_indices: list,
    task_policy: str,
) -> int:
    """规则锁任务：返回 ready_candidate_indices 中的下标。"""
    policy = (task_policy or "edf").lower()
    best_i = 0
    best_key = None
    for i, cand_idx in enumerate(ready_candidate_indices):
        if cand_idx < 0 or cand_idx >= len(env.candidate_tasks):
            continue
        task = env.candidate_tasks[cand_idx]
        if task is None or getattr(task, "is_empty", False):
            continue
        if policy == "est":
            ready_at = float(getattr(task, "arrival_time", 0.0))
            est = float(getattr(task, "est", 0.0))
            key = (ready_at, est, i)
        else:
            key = (float(getattr(task, "lft", float("inf"))), i)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return best_i


def select_task_proc(
    model: HierarchicalGATScheduler,
    task_logits: torch.Tensor,
    proc_logits: torch.Tensor,
    candidate_types: Optional[torch.Tensor],
    deterministic: bool,
    device: str,
    task_policy: str = "ppo",
    env: Optional[MultiDAGSchedulerEnv] = None,
    ready_candidate_indices: Optional[list] = None,
):
    """
    统一的任务/处理器选择（采集采样、采集贪心、evaluate_policy 共用）。

    task_policy:
      - ppo: 任务与处理器都由策略决定（默认）
      - edf: 任务取候选中 LFT 最小者，仅处理器由策略决定（logprob 只含 proc）
      - est: 任务取候选中实际就绪时刻最早者（arrival_time，并列比静态 EST），仅处理器由策略决定

    Returns:
        task_id, proc_id, action_logprob (float)
    """
    num_opts = model.num_procs + (1 if model.enable_hold else 0)
    joint_matching = getattr(model, 'joint_matching', False)
    task_policy = (task_policy or "ppo").lower()

    if task_policy in RULE_TASK_POLICIES:
        if env is None or ready_candidate_indices is None:
            raise ValueError(f"task_policy={task_policy} 需要 env 与 ready_candidate_indices")
        if joint_matching:
            raise ValueError(f"task_policy={task_policy} 不支持 joint_matching=1")

        best_i = rule_ready_task_index(env, ready_candidate_indices, task_policy)
        task_id = torch.tensor(best_i, device=task_logits.device)

        proc_logits_selected = proc_logits[task_id]
        if candidate_types is not None:
            task_type = candidate_types[task_id].item()
            proc_logits_selected = model.two_stage_scheduler.apply_candidate_type_mask(
                proc_logits_selected, task_type
            )
        if model.enable_hold:
            proc_logits_selected = proc_logits_selected.clone()
            proc_logits_selected[-1] = -1e9

        if deterministic:
            proc_probs = F.softmax(proc_logits_selected, dim=-1)
            proc_id = torch.argmax(proc_probs, dim=-1)
            proc_logprob = torch.log(proc_probs[proc_id] + 1e-8)
        else:
            proc_probs = F.softmax(proc_logits_selected, dim=-1)
            proc_dist = torch.distributions.Categorical(proc_probs)
            proc_id = proc_dist.sample()
            proc_logprob = proc_dist.log_prob(proc_id)
        return task_id, proc_id, proc_logprob.item()

    if joint_matching:
        joint = task_logits.unsqueeze(1) + proc_logits  # [M, num_opts]
        M_val = joint.shape[0]
        if candidate_types is not None:
            for k in range(M_val):
                joint[k] = model.two_stage_scheduler.apply_candidate_type_mask(
                    joint[k], candidate_types[k].item()
                )
        # 环境不支持真正的 Hold：采集/评估/PPO 更新均须屏蔽
        if model.enable_hold:
            joint[:, -1] = -1e9
        joint_flat = joint.flatten()
        if deterministic:
            action_id = torch.argmax(joint_flat, dim=-1)
            joint_probs = F.softmax(joint_flat, dim=-1)
            action_logprob = torch.log(joint_probs[action_id] + 1e-8).item()
        else:
            joint_dist = torch.distributions.Categorical(logits=joint_flat)
            action_id = joint_dist.sample()
            action_logprob = joint_dist.log_prob(action_id).item()
        task_id = action_id // num_opts
        proc_id = action_id % num_opts
        return task_id, proc_id, action_logprob

    # 两阶段
    if deterministic:
        task_probs = F.softmax(task_logits, dim=-1)
        task_id = torch.argmax(task_probs, dim=-1)
        task_logprob = torch.log(task_probs[task_id] + 1e-8)
    else:
        task_probs = F.softmax(task_logits, dim=-1)
        task_dist = torch.distributions.Categorical(task_probs)
        task_id = task_dist.sample()
        task_logprob = task_dist.log_prob(task_id)

    proc_logits_selected = proc_logits[task_id]
    if candidate_types is not None:
        task_type = candidate_types[task_id].item()
        proc_logits_selected = model.two_stage_scheduler.apply_candidate_type_mask(
            proc_logits_selected, task_type
        )
    if model.enable_hold:
        proc_logits_selected = proc_logits_selected.clone()
        proc_logits_selected[-1] = -1e9

    if deterministic:
        proc_probs = F.softmax(proc_logits_selected, dim=-1)
        proc_id = torch.argmax(proc_probs, dim=-1)
        proc_logprob = torch.log(proc_probs[proc_id] + 1e-8)
    else:
        proc_probs = F.softmax(proc_logits_selected, dim=-1)
        proc_dist = torch.distributions.Categorical(proc_probs)
        proc_id = proc_dist.sample()
        proc_logprob = proc_dist.log_prob(proc_id)

    return task_id, proc_id, (task_logprob + proc_logprob).item()


def _masked_proc_logits_row(
    model: HierarchicalGATScheduler,
    proc_logits_row: torch.Tensor,
    candidate_type: Optional[int],
) -> torch.Tensor:
    """对单行处理器 logits 应用 type mask，并屏蔽 Hold。"""
    row = proc_logits_row
    if candidate_type is not None:
        row = model.two_stage_scheduler.apply_candidate_type_mask(
            row, int(candidate_type)
        )
    if model.enable_hold:
        row = row.clone()
        row[-1] = -1e9
    # 评估时不允许 Hold 动作
    if row.numel() > NUM_PROCS:
        row = row.clone()
        row[NUM_PROCS:] = -1e9
    return row


def select_task_proc_beam_highload(
    model: HierarchicalGATScheduler,
    task_logits: torch.Tensor,
    proc_logits: torch.Tensor,
    candidate_types: Optional[torch.Tensor],
    device: str,
    env: MultiDAGSchedulerEnv,
    ready_candidate_indices: list,
    *,
    beam_width: int = 8,
    load_threshold: int = 6,
    num_active_dags: int = 0,
):
    """
    高负载 beam 推理：active_dags >= threshold 时，
    取策略 top-k (task,proc)，再用 EFT end_time 重排（越小越好）；
    低负载仍走 greedy。

    Returns:
        task_id, proc_id, action_logprob, meta dict
    """
    meta = {
        'beam_fired': False,
        'beam_k': 0,
        'num_active_dags': int(num_active_dags),
    }
    if (
        beam_width <= 1
        or num_active_dags < load_threshold
        or getattr(model, 'joint_matching', False)
    ):
        task_id, proc_id, lp = select_task_proc(
            model, task_logits, proc_logits, candidate_types,
            deterministic=True, device=device,
        )
        return task_id, proc_id, lp, meta

    m = int(task_logits.shape[0])
    num_opts = int(proc_logits.shape[-1])
    # 只在物理机上做 beam（Hold 已屏蔽）
    n_procs = min(NUM_PROCS, num_opts)
    if m <= 0 or n_procs <= 0:
        task_id, proc_id, lp = select_task_proc(
            model, task_logits, proc_logits, candidate_types,
            deterministic=True, device=device,
        )
        return task_id, proc_id, lp, meta

    # 向量化 joint ≈ task_logit[i] + masked_proc[i,j]，避免 m×P 次 .item()
    proc_mat = proc_logits[:, :n_procs]
    if candidate_types is not None or model.enable_hold or num_opts > n_procs:
        rows = []
        for i in range(m):
            ctype = None if candidate_types is None else int(candidate_types[i].item())
            rows.append(_masked_proc_logits_row(model, proc_logits[i], ctype)[:n_procs])
        proc_mat = torch.stack(rows, dim=0)
    joint = task_logits.unsqueeze(1) + proc_mat  # [M, P]
    flat = joint.reshape(-1)
    k = min(int(beam_width), int(flat.numel()))
    top_vals, top_flat_idx = torch.topk(flat, k=k, largest=True, sorted=True)
    top_i = (top_flat_idx // n_procs).tolist()
    top_j = (top_flat_idx % n_procs).tolist()
    top_joint = top_vals.detach().cpu().tolist()

    best_i, best_j = int(top_i[0]), int(top_j[0])
    best_end = float('inf')
    best_joint = float('-inf')
    current_time = float(env.current_time)

    for i, j, joint_score in zip(top_i, top_j, top_joint):
        cand_idx = ready_candidate_indices[i]
        if cand_idx < 0 or cand_idx >= len(env.candidate_tasks):
            continue
        task = env.candidate_tasks[cand_idx]
        if task is None or getattr(task, 'is_empty', False):
            continue
        proc = env.processors[j]
        _, end_time, _, _ = estimate_schedule_timing(
            task, proc, env.processors, current_time
        )
        # 主关键：更早结束；平局取更高策略分
        if end_time < best_end - 1e-9 or (
            abs(end_time - best_end) <= 1e-9 and joint_score > best_joint
        ):
            best_end = end_time
            best_joint = float(joint_score)
            best_i, best_j = int(i), int(j)

    task_id = torch.tensor(best_i, device=task_logits.device)
    proc_id = torch.tensor(best_j, device=task_logits.device)

    # logprob 按最终选中动作的两阶段分解
    task_probs = F.softmax(task_logits, dim=-1)
    ctype = None if candidate_types is None else int(candidate_types[best_i].item())
    prow = _masked_proc_logits_row(model, proc_logits[best_i], ctype)
    proc_probs = F.softmax(prow[:n_procs], dim=-1)
    lp = (
        torch.log(task_probs[best_i] + 1e-8)
        + torch.log(proc_probs[best_j] + 1e-8)
    ).item()

    meta['beam_fired'] = True
    meta['beam_k'] = int(k)
    meta['chosen_end_time'] = float(best_end)
    return task_id, proc_id, lp, meta


def encode_env_action(candidate_idx: int, proc_idx: int) -> int:
    """环境动作编码。Hold 未接入 env 时 proc_idx 必须 ∈ [0, NUM_PROCS)。"""
    if proc_idx < 0 or proc_idx >= NUM_PROCS:
        raise ValueError(
            f'非法 proc_idx={proc_idx}（NUM_PROCS={NUM_PROCS}）；'
            f'Hold 槽位不能直接编码进 env.step'
        )
    return candidate_idx * NUM_PROCS + proc_idx


def extract_wait_time_from_ckpt(
    ckpt: dict,
    fallback_mean: float = WAIT_TIME_MEAN,
    fallback_std: float = WAIT_TIME_STD,
) -> Tuple[float, float]:
    """从 checkpoint 读取 wait_time；旧 ckpt 无字段则回退默认先验。"""
    if not isinstance(ckpt, dict):
        return fallback_mean, fallback_std
    mean = ckpt.get('wait_time_mean')
    std = ckpt.get('wait_time_std')
    if mean is None or std is None:
        cfg = ckpt.get('config')
        if isinstance(cfg, dict):
            if mean is None:
                mean = cfg.get('wait_time_mean')
            if std is None:
                std = cfg.get('wait_time_std')
    if mean is None:
        mean = fallback_mean
    if std is None:
        std = fallback_std
    return float(mean), float(std)


def extract_env_data(
    env: MultiDAGSchedulerEnv,
    max_attn_dags: int = 0,
    mixed_features: bool = False,
) -> Dict:
    """
    从环境提取分层调度器所需的所有数据。

    max_attn_dags > 0 且活跃 DAG 超量时，预筛选 DAG 子集:
    - 有就绪任务的 DAG 强制保留
    - 其余按 completion_ratio 紧迫度补足到 max_attn_dags
    - 被裁减的 DAG 不提取静态特征和邻接矩阵，节省 CPU 计算

    关键修复:
    - ready_task_indices 使用全局任务索引（跨DAG拼接后的位置），而非DAG局部task_id
    - task_dag_ids 使用过滤后的DAG索引，消除skip导致的gap
    """
    active_dags = env.active_dags
    if len(active_dags) == 0:
        return None

    current_time = env.current_time
    wait_time_mean = env.wait_time_mean
    wait_time_std = env.wait_time_std

    # 过滤无效 DAG，建立 id(dag) -> valid_idx 映射
    valid_dags = []
    dag_to_valid_idx = {}
    for dag in active_dags:
        if dag is not None and len(dag.tasks) > 0:
            dag_to_valid_idx[id(dag)] = len(valid_dags)
            valid_dags.append(dag)

    num_active_dags = len(valid_dags)
    if num_active_dags == 0:
        return None

    # ===== 预筛选：活跃 DAG 过多时只提取子集 =====
    if 0 < max_attn_dags < num_active_dags:
        # 找出有就绪任务的 DAG
        ready_dag_set = set()
        for task in env.candidate_tasks:
            if not task.is_empty:
                d_idx = dag_to_valid_idx.get(id(task.dag), -1)
                if d_idx >= 0:
                    ready_dag_set.add(d_idx)

        if len(ready_dag_set) > max_attn_dags:
            # 就绪 DAG 超过上限，无法裁剪，全量提取
            selected_indices = list(range(num_active_dags))
        else:
            # 计算紧迫度：-completion_ratio（越低越紧迫）
            urgency = np.zeros(num_active_dags)
            for i, dag in enumerate(valid_dags):
                meta = dag.get_metadata(current_time)
                urgency[i] = -meta[1]
            # 有就绪任务的 DAG 强制保留
            for d in ready_dag_set:
                urgency[d] += 100.0
            selected_indices = np.argsort(urgency)[::-1][:max_attn_dags].tolist()
            selected_indices.sort()
    else:
        selected_indices = list(range(num_active_dags))

    # 构建 old_idx -> new_idx 映射（用于重编号 task_dag_ids）
    old_to_new = {old: new for new, old in enumerate(selected_indices)}

    min_proc_avail = min(p.available_at for p in env.processors)

    # ===== 第一遍：只提取选中 DAG 的静态特征和邻接矩阵 =====
    dag_adjs = []
    dag_metadata_list = []
    dag_task_static_features = []
    task_dag_ids = []
    task_global_idx = {}  # id(task) -> 全局索引
    global_idx = 0

    for old_idx in selected_indices:
        dag = valid_dags[old_idx]
        dag_adjs.append(torch.from_numpy(dag.get_adjacency_matrix()))
        dag_metadata_list.append(dag.get_metadata(current_time))

        non_empty_tasks = [task for task in dag.tasks if not task.is_empty]
        if non_empty_tasks:
            feat_rows = []
            for task in non_empty_tasks:
                row = task.get_static_features()
                if mixed_features:
                    row = np.concatenate([
                        row,
                        task.get_dynamic_features(current_time, min_proc_avail),
                    ])
                feat_rows.append(row)
            static_features = np.array(feat_rows, dtype=np.float32)
            dag_task_static_features.append(torch.from_numpy(static_features))

            new_dag_idx = old_to_new[old_idx]
            for task in non_empty_tasks:
                task_global_idx[id(task)] = global_idx
                global_idx += 1
            task_dag_ids.extend([new_dag_idx] * len(non_empty_tasks))

    if global_idx == 0:
        return None

    # ===== 第二遍：就绪任务 =====
    ready_indices = []
    ready_candidate_indices = []
    candidate_types_list = []
    ready_task_dyn_features_list = []
    ready_tasks = []

    for candidate_idx, task in enumerate(env.candidate_tasks):
        if task.is_empty:
            continue
        g_idx = task_global_idx.get(id(task), -1)
        if g_idx < 0:
            continue

        ready_indices.append(g_idx)
        ready_candidate_indices.append(candidate_idx)
        ready_tasks.append(task)

        if task.candidate_type == 'cloud_only':
            candidate_types_list.append(0)
        elif task.candidate_type == 'edge_only':
            candidate_types_list.append(1)
        else:
            candidate_types_list.append(2)

        ready_task_dyn_features_list.append(
            task.get_dynamic_features(current_time, min_proc_avail)
        )

    if len(ready_indices) == 0:
        return None

    ready_task_indices = torch.tensor(ready_indices, dtype=torch.long)
    ready_task_dyn_features = torch.tensor(np.array(ready_task_dyn_features_list, dtype=np.float32))
    candidate_types = torch.tensor(candidate_types_list, dtype=torch.long)

    # ===== 处理器特征 =====
    proc_features_list = [proc.get_features(current_time, wait_time_mean, wait_time_std) for proc in env.processors]
    proc_states_list = [max(0.0, proc.available_at - current_time) for proc in env.processors]

    proc_features = torch.tensor(np.array(proc_features_list, dtype=np.float32))
    proc_states = torch.tensor(np.array(proc_states_list, dtype=np.float32))

    # ===== Task–Processor 显式成对特征 + EFT / SLA 专家标签 =====
    pair_rows = []
    eft_targets = []
    sla_laxities = []
    for task, candidate_type in zip(ready_tasks, candidate_types_list):
        effective_time = (
            max(current_time, float(task.arrival_time))
            if task.status == TaskStatus.WILL_READY
            else current_time
        )
        ms = max(float(task.dag.makespan_est), 1e-6)
        cloud_ref_energy = max(
            float(task.computation) / CLOUD_MIPS * CLOUD_POWER,
            1e-6,
        )
        task_pair_rows = []
        end_times = []
        valid_proc_ids = []
        for proc_idx, proc in enumerate(env.processors):
            start, end, comp, comm = estimate_schedule_timing(
                task, proc, env.processors, effective_time
            )
            energy = comp * proc.power
            pair = np.array([
                (start - effective_time) / ms,
                comp / ms,
                comm / ms,
                (end - effective_time) / ms,
                (task.lft - end) / ms,
                energy / cloud_ref_energy,
            ], dtype=np.float32)
            pair = np.nan_to_num(
                pair,
                nan=0.0,
                posinf=PAIR_FEATURE_CLIP,
                neginf=-PAIR_FEATURE_CLIP,
            )
            task_pair_rows.append(
                np.clip(pair, -PAIR_FEATURE_CLIP, PAIR_FEATURE_CLIP)
            )
            is_valid = (
                candidate_type == 2
                or (candidate_type == 0 and proc.proc_type == 0)
                or (candidate_type == 1 and proc.proc_type == 1)
            )
            if is_valid:
                valid_proc_ids.append(proc_idx)
                end_times.append(end)
        pair_rows.append(task_pair_rows)
        eft_targets.append(
            valid_proc_ids[int(np.argmin(end_times))]
            if valid_proc_ids
            else 0
        )
        # 任务监督使用 decision-time 最小可达 EFT 计算归一化 laxity。
        # 值越小表示越紧迫；它不依赖任务实际完成结果，不会引入下游信用污染。
        best_eft = min(end_times) if end_times else float('inf')
        normalized_laxity = (
            (float(task.lft) - float(best_eft)) / ms
            if np.isfinite(best_eft)
            else float('inf')
        )
        sla_laxities.append(normalized_laxity)

    task_proc_pair_features = torch.tensor(
        np.asarray(pair_rows, dtype=np.float32)
    )
    eft_targets_tensor = torch.tensor(eft_targets, dtype=torch.long)
    sla_task_target = torch.tensor(
        int(np.argmin(np.asarray(sla_laxities, dtype=np.float64)))
        if sla_laxities
        else 0,
        dtype=torch.long,
    )

    # ===== DAG元数据 =====
    dag_metadata = torch.tensor(np.array(dag_metadata_list, dtype=np.float32))
    task_dag_ids_tensor = torch.tensor(task_dag_ids, dtype=torch.long)

    return {
        'dag_task_static_features': dag_task_static_features,
        'dag_adjs': dag_adjs,
        'dag_metadata': dag_metadata,
        'ready_task_indices': ready_task_indices,
        'ready_candidate_indices': ready_candidate_indices,
        'ready_task_dyn_features': ready_task_dyn_features,
        'proc_features': proc_features,
        'proc_states': proc_states,
        'task_proc_pair_features': task_proc_pair_features,
        'eft_targets': eft_targets_tensor,
        'sla_task_target': sla_task_target,
        'task_dag_ids': task_dag_ids_tensor,
        'candidate_types': candidate_types,
        'num_active_dags': num_active_dags,  # 原始活跃 DAG 数，供 Critic 用
    }


def run_training_step(
    env: MultiDAGSchedulerEnv,
    model: HierarchicalGATScheduler,
    buffer: HierarchicalReplayBuffer,
    optimizer,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    eft_aux_coef: float = 0.0,
    task_aux_coef: float = 0.0,
    ppo_epochs: int = 4,
    episode_idx: int = 0,
    device: str = 'cpu',
    task_policy: str = 'ppo',
) -> dict:
    """
    收集经验并训练
    """
    total_reward = 0
    total_steps = 0
    valid_actions = 0
    episode_rewards = []
    buffer_size = buffer.max_buffer_size

    proc_selection_count = [0] * NUM_PROCS
    proc_task_time = [0.0] * NUM_PROCS

    obs = env.reset('train', episode_idx=episode_idx)
    model.train()  # 确保训练模式（evaluate_policy 可能将其设为 eval）
    model.intra_dag_encoder.clear_cache()
    model.proc_only_logprob = (task_policy in RULE_TASK_POLICIES)

    # 训练统计
    update_stats = {
        'policy_loss_sum': 0.0,
        'value_loss_sum': 0.0,
        'entropy_sum': 0.0,
        'total_loss_sum': 0.0,
        'grad_norm_sum': 0.0,
        'value_pred_sum': 0.0,
        'value_target_sum': 0.0,
        'value_target_raw_sum': 0.0,
        'debug_rewards_mean_sum': 0.0,
        'eft_aux_loss_sum': 0.0,
        'eft_top1_acc_sum': 0.0,
        'task_aux_loss_sum': 0.0,
        'task_top1_acc_sum': 0.0,
        'count': 0
    }

    train_times = 0
    start = datetime.now()

    while True:
        # 提取环境数据
        env_data = extract_env_data(
            env, max_attn_dags=model.max_attn_dags,
            mixed_features=_uses_mixed_features(model),
        )
        if env_data is None:
            break
        # 获取就绪任务在candidate_tasks中的索引映射
        ready_candidate_indices = env_data.get('ready_candidate_indices', list(range(len(env_data['ready_task_indices']))))
        candidate_types = env_data.get('candidate_types', None)

        # 前向传播获取logits（采集阶段不需要梯度）
        with torch.no_grad():
            static_f = [f.to(device) for f in env_data['dag_task_static_features']]
            adjs = [a.to(device) for a in env_data['dag_adjs']]
            task_logits, proc_logits, model_info = model(
                dag_task_static_features=static_f,
                dag_adjs=adjs,
                dag_metadata=env_data['dag_metadata'].to(device),
                ready_task_indices=env_data['ready_task_indices'].to(device),
                ready_task_dyn_features=env_data['ready_task_dyn_features'].to(device),
                proc_features=env_data['proc_features'].to(device),
                proc_states=env_data['proc_states'].to(device),
                task_dag_ids=env_data['task_dag_ids'].to(device),
                task_proc_pair_features=env_data[
                    'task_proc_pair_features'
                ].to(device),
                num_active_dags=env_data['num_active_dags'],
            )

        # 训练采集用采样；若外部把 model 设为 eval，则走贪心（一般不会）
        task_id, proc_id, action_logprob = select_task_proc(
            model, task_logits, proc_logits, candidate_types,
            deterministic=not model.training,
            device=device,
            task_policy=task_policy,
            env=env,
            ready_candidate_indices=ready_candidate_indices,
        )

        # Value 估计（用于 GAE 预计算）
        ready_task_embs = model_info['ready_task_embs']
        updated_dag_embeddings = model_info['updated_dag_embeddings']
        M = ready_task_embs.size(0)
        num_active_dags = env_data['num_active_dags']
        with torch.no_grad():
            global_feat = compute_global_features(
                env_data['proc_features'].to(device), M, num_active_dags
            )
            value = model.estimate_value(ready_task_embs, global_feat).item()

        # 转换为环境动作: task_idx * num_processors + proc_idx
        task_id_val = task_id.item()
        if task_id_val >= len(ready_candidate_indices):
            task_id_val = 0
        actual_candidate_idx = ready_candidate_indices[task_id_val]
        proc_idx = proc_id.item()

        action_np = encode_env_action(actual_candidate_idx, proc_idx)

        # 执行动作
        obs, reward, done, info = env.step(action_np)

        # 统计
        if info.get('valid', False):
            if 0 <= proc_idx < NUM_PROCS:
                proc_selection_count[proc_idx] += 1
                proc_task_time[proc_idx] += info.get('exec_time', 0.0)
            valid_actions += 1

        total_reward += reward
        total_steps += 1
        episode_rewards.append(reward)

        # 存储经验（环形缓冲区自动覆盖旧数据）
        buffer.store(env_data, task_id.item(), proc_id.item(), action_logprob, reward, done,
                     value=value,
                     ready_task_embs=ready_task_embs, dag_embeddings=updated_dag_embeddings)

        if total_steps > 0 and total_steps % buffer_size == 0:
             if len(buffer) >= buffer.batch_size:
                buffer.compute_gae(gamma, gae_lambda)
                start_time = datetime.now()
                n_mid_batches = len(buffer) // buffer.batch_size
                for _ in range(n_mid_batches):
                        batch_result = buffer.get_batch()
                        if batch_result is None:
                            break
                        batch_env_data, batch_actions, batch_extra = batch_result
                        try:
                            update_info = train_batch(
                                model, optimizer,
                                batch_env_data, batch_actions, batch_extra,
                                gamma, gae_lambda, epsilon, value_coef, entropy_coef, max_grad_norm,
                                device, eft_aux_coef, task_aux_coef
                            )
                            update_stats['policy_loss_sum'] += update_info['policy_loss']
                            update_stats['value_loss_sum'] += update_info['value_loss']
                            update_stats['entropy_sum'] += update_info['entropy']
                            update_stats['total_loss_sum'] += update_info['total_loss']
                            update_stats['grad_norm_sum'] += update_info.get('grad_norm', 0)
                            update_stats['value_pred_sum'] += update_info.get('value_pred', 0)
                            update_stats['value_target_sum'] += update_info.get('value_target', 0)
                            update_stats['value_target_raw_sum'] += update_info.get('value_target_raw', 0)
                            update_stats['debug_rewards_mean_sum'] += update_info.get('debug_rewards_mean', 0)
                            update_stats['eft_aux_loss_sum'] += update_info.get('eft_aux_loss', 0)
                            update_stats['eft_top1_acc_sum'] += update_info.get('eft_top1_acc', 0)
                            update_stats['task_aux_loss_sum'] += update_info.get('task_aux_loss', 0)
                            update_stats['task_top1_acc_sum'] += update_info.get('task_top1_acc', 0)
                            update_stats['count'] += 1
                        except RuntimeError as e:
                            print(f'  Training error: {e}')
                            raise
                execution_time = datetime.now() - start_time
                train_times += execution_time.total_seconds()

        if done:
            break

    # Episode 结束：预计算 GAE，然后 multi-epoch PPO，最后 reset
    n_samples = len(buffer)
    if n_samples >= buffer.batch_size:
        buffer.compute_gae(gamma, gae_lambda)
        start_time = datetime.now()
        n_batches = n_samples // buffer.batch_size
        for epoch in range(ppo_epochs):
            for _ in range(n_batches):
                batch_result = buffer.get_batch()
                if batch_result is None:
                    break
                batch_env_data, batch_actions, batch_extra = batch_result
                try:
                    update_info = train_batch(
                        model, optimizer,
                        batch_env_data, batch_actions, batch_extra,
                        gamma, gae_lambda, epsilon, value_coef, entropy_coef, max_grad_norm,
                        device, eft_aux_coef, task_aux_coef
                    )
                    update_stats['policy_loss_sum'] += update_info['policy_loss']
                    update_stats['value_loss_sum'] += update_info['value_loss']
                    update_stats['entropy_sum'] += update_info['entropy']
                    update_stats['total_loss_sum'] += update_info['total_loss']
                    update_stats['grad_norm_sum'] += update_info.get('grad_norm', 0)
                    update_stats['value_pred_sum'] += update_info.get('value_pred', 0)
                    update_stats['value_target_sum'] += update_info.get('value_target', 0)
                    update_stats['value_target_raw_sum'] += update_info.get('value_target_raw', 0)
                    update_stats['debug_rewards_mean_sum'] += update_info.get('debug_rewards_mean', 0)
                    update_stats['eft_aux_loss_sum'] += update_info.get('eft_aux_loss', 0)
                    update_stats['eft_top1_acc_sum'] += update_info.get('eft_top1_acc', 0)
                    update_stats['task_aux_loss_sum'] += update_info.get('task_aux_loss', 0)
                    update_stats['task_top1_acc_sum'] += update_info.get('task_top1_acc', 0)
                    update_stats['count'] += 1
                except RuntimeError as e:
                    print(f'  Training error: {e}')
                    raise
        execution_time = datetime.now() - start_time
        train_times += execution_time.total_seconds()

    buffer.reset()

    total_energy = sum(proc.total_energy for proc in env.processors)
    cost = datetime.now() - start
    print(f"训练耗时:{train_times};总耗时：{cost.total_seconds()}")

    return {
        'reward': total_reward,
        'steps': total_steps,
        'valid_actions': valid_actions,
        'total_dags': env.total_dags_count,
        'total_tasks': env.total_tasks,
        'completed_tasks': env.on_time_completed_tasks,
        'completed_dags': env.on_time_completed_dags,
        'completion_rate': env.on_time_completed_tasks / env.total_tasks if env.total_tasks > 0 else 0,
        'dag_completion_rate': env.on_time_completed_dags / env.total_dags_count if env.total_dags_count > 0 else 0,
        'avg_energy': total_energy / env.total_tasks if env.total_tasks > 0 else 0,
        'avg_task_delay': env.total_task_delay / env.total_tasks if env.total_tasks > 0 else 0,
        'proc_selection_count': proc_selection_count,
        'proc_task_time': proc_task_time,
        'reward_mean': np.mean(episode_rewards) if episode_rewards else 0,
        'reward_std': np.std(episode_rewards) if episode_rewards else 0,
        'reward_pos_ratio': np.mean([r > 0 for r in episode_rewards]) if episode_rewards else 0,
        'update_stats': update_stats,
        'priority_stats': {p: dict(ps) for p, ps in env._priority_stats.items()},
    }


def train_batch(
    model: HierarchicalGATScheduler,
    optimizer,
    batch_env_data: dict,
    batch_actions: dict,
    batch_extra: dict,
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    device: str,
    eft_aux_coef: float = 0.0,
    task_aux_coef: float = 0.0,
) -> dict:
    """
    训练一个batch — 批量前向传播（梯度流过所有层），使用预计算的GAE advantages。
    evaluate_actions_batch 一次性批量处理所有样本的 Layer 1+2+3。
    """
    batch_size = len(batch_env_data['dag_task_static_features'])

    # 跳过全是空就绪任务的batch（极少情况）
    if all(n == 0 for n in batch_env_data['num_ready']):
        return {
            'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
            'total_loss': 0.0, 'grad_norm': 0.0,
            'value_pred': 0.0, 'value_target': 0.0, 'value_target_raw': 0.0,
            'debug_rewards_mean': 0.0,
            'eft_aux_loss': 0.0,
            'eft_top1_acc': 0.0,
            'task_aux_loss': 0.0,
            'task_top1_acc': 0.0,
        }

    # 迁移索引张量到 device（evaluate_actions_batch 内部会用作 gather 索引）
    batch_actions['task'] = batch_actions['task'].to(device)
    batch_actions['proc'] = batch_actions['proc'].to(device)
    if batch_env_data.get('candidate_types'):
        batch_env_data['candidate_types'] = [ct.to(device) for ct in batch_env_data['candidate_types']]

    # 批量前向传播（一次性处理所有样本的 Layer 1+2+3）
    (
        action_logprobs,
        values,
        entropy,
        eft_aux_loss,
        eft_top1_acc,
        task_aux_loss,
        task_top1_acc,
    ) = model.evaluate_actions_batch(batch_env_data, batch_actions, device)

    old_logprobs = batch_actions['logprobs'].to(device)
    advantages_raw = batch_extra['advantages'].to(device)
    old_values = batch_extra['values'].to(device)

    # 标准化 + 裁剪 advantages（用于 policy loss）
    if advantages_raw.std() > 1e-8:
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
    else:
        advantages = advantages_raw.clone()
    advantages = torch.clamp(advantages, -3.0, 3.0)

    # returns = 原始 A + V（TD-λ target，不能混用标准化后的 advantage）
    returns = advantages_raw + old_values
    returns = torch.clamp(returns, -50.0, 50.0)

    # PPO clip 损失
    ratio = torch.exp(action_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value 损失（PPO-style value clipping 防止 value 更新过大）
    value_pred_clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
    v_loss1 = F.mse_loss(values, returns)
    v_loss2 = F.mse_loss(value_pred_clipped, returns)
    value_loss = torch.max(v_loss1, v_loss2)

    entropy_loss = -entropy.mean()

    loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
        + eft_aux_coef * eft_aux_loss
        + task_aux_coef * task_aux_loss
    )

    optimizer.zero_grad()
    loss.backward()
    sync_device(device)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    sync_device(device)

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.mean().item(),
        'total_loss': loss.item(),
        'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
        'value_pred': values.mean().item(),
        'value_target': returns.mean().item(),
        'value_target_raw': returns.mean().item(),
        'debug_rewards_mean': batch_extra['rewards'].mean().item(),
        'eft_aux_loss': eft_aux_loss.item(),
        'eft_top1_acc': eft_top1_acc.item(),
        'task_aux_loss': task_aux_loss.item(),
        'task_top1_acc': task_top1_acc.item(),
    }


def evaluate_policy(
    env: MultiDAGSchedulerEnv,
    model: HierarchicalGATScheduler,
    num_episodes: int = 10,
    device: str = 'cpu',
    verbose: bool = True,
    split: str = 'val',
    beam_width: int = 1,
    beam_load_threshold: int = 6,
    task_policy: str = 'ppo',
) -> dict:
    """
    评估策略
    split: 'val' 或 'test'
    beam_width>1 时：高负载（active_dags>=threshold）用 top-k + EFT 重排。
    task_policy=edf/est 时：任务按规则（LFT / 最早就绪），仅处理器走策略。
    """
    model.eval()
    task_policy = (task_policy or 'ppo').lower()

    total_reward = 0
    total_steps = 0
    completion_rates = []
    dag_completion_rates = []
    task_delays = []
    ep_energies = []
    ep_dag_counts = []
    ep_task_counts = []
    ep_priority_list = []  # 每个 episode 的 priority_stats 快照
    revision_metrics = []
    beam_steps = 0
    greedy_steps = 0

    # 紧迫度诊断统计
    urgency_dag_info = []  # [(urgency, completed_on_time), ...]
    # 跨 episode 累计 per-priority 统计（reset 会清空单 ep 桶）
    agg_priority_stats: Dict[float, Dict] = {}

    for ep in range(num_episodes):
        obs = env.reset(split, episode_idx=ep)

        ep_reward = 0
        ep_steps = 0
        ep_decision_s = 0.0

        # 收集这个 episode 的 DAG 信息用于诊断
        ep_dags_urgency = []  # 这个episode中每个DAG的紧迫度
        ep_negative_rewards = 0  # 负奖励计数（表示任务延迟）
        ep_total_rewards = 0

        while True:
            t0 = time.perf_counter()
            env_data = extract_env_data(
                env, max_attn_dags=model.max_attn_dags,
                mixed_features=_uses_mixed_features(model),
            )
            if env_data is None:
                # 与训练一致：无法决策则结束本 ep，避免 continue 死循环
                if getattr(env, '_is_done', lambda: True)():
                    break
                print('[evaluate_policy] extract_env_data=None，提前结束本 episode')
                break

            ready_candidate_indices = env_data.get(
                'ready_candidate_indices',
                list(range(len(env_data['ready_task_indices']))),
            )
            candidate_types = env_data.get('candidate_types', None)

            with torch.no_grad():
                static_f = [f.to(device) for f in env_data['dag_task_static_features']]
                adjs = [a.to(device) for a in env_data['dag_adjs']]
                task_logits, proc_logits, _ = model(
                    dag_task_static_features=static_f,
                    dag_adjs=adjs,
                    dag_metadata=env_data['dag_metadata'].to(device),
                    ready_task_indices=env_data['ready_task_indices'].to(device),
                    ready_task_dyn_features=env_data['ready_task_dyn_features'].to(device),
                    proc_features=env_data['proc_features'].to(device),
                    proc_states=env_data['proc_states'].to(device),
                    task_dag_ids=env_data['task_dag_ids'].to(device),
                    task_proc_pair_features=env_data[
                        'task_proc_pair_features'
                    ].to(device),
                    num_active_dags=env_data['num_active_dags'],
                )
                if beam_width > 1 and task_policy == 'ppo':
                    task_id, proc_id, _, beam_meta = select_task_proc_beam_highload(
                        model, task_logits, proc_logits, candidate_types, device,
                        env=env,
                        ready_candidate_indices=ready_candidate_indices,
                        beam_width=beam_width,
                        load_threshold=beam_load_threshold,
                        num_active_dags=int(env_data['num_active_dags']),
                    )
                    if beam_meta.get('beam_fired'):
                        beam_steps += 1
                    else:
                        greedy_steps += 1
                else:
                    task_id, proc_id, _ = select_task_proc(
                        model, task_logits, proc_logits, candidate_types,
                        deterministic=True, device=device,
                        task_policy=task_policy,
                        env=env,
                        ready_candidate_indices=ready_candidate_indices,
                    )
                    greedy_steps += 1
            ep_decision_s += time.perf_counter() - t0

            actual_candidate_idx = ready_candidate_indices[task_id.item()]
            action_np = encode_env_action(actual_candidate_idx, proc_id.item())
            obs, reward, done, info = env.step(action_np)

            # 记录选中任务的 DAG 紧迫度（用于诊断）
            if 'dag_urgency' in info:
                ep_dags_urgency.append(info['dag_urgency'])

            ep_reward += reward
            ep_steps += 1
            ep_total_rewards += 1
            if reward < 0:
                ep_negative_rewards += 1

            if beam_width > 1 and ep_steps % 2000 == 0:
                total_sel = beam_steps + greedy_steps
                ratio = beam_steps / max(total_sel, 1)
                print(
                    f"  [beam_progress] ep={ep + 1}/{num_episodes} "
                    f"steps={ep_steps} beam_ratio={ratio:.1%} "
                    f"active_dags={int(env_data['num_active_dags'])}",
                    flush=True,
                )

            if done:
                break

        # 收集这个 episode 的 DAG 完成情况用于诊断
        urgency_dag_info.append({
            'avg_dag_urgency': np.mean(ep_dags_urgency) if ep_dags_urgency else 0,
            'high_urgency_task_ratio': sum(1 for u in ep_dags_urgency if u > 0.1) / max(len(ep_dags_urgency), 1),
            'negative_reward_ratio': ep_negative_rewards / max(ep_total_rewards, 1),
            'on_time_dags': env.on_time_completed_dags,
            'total_dags': env.total_dags_count
        })

        total_reward += ep_reward
        total_steps += ep_steps

        # 计算这个 episode 的完成率（必须在 env 状态还在的时候计算）
        ep_completion_rate = env.on_time_completed_tasks / env.total_tasks if env.total_tasks > 0 else 0
        ep_dag_completion_rate = env.on_time_completed_dags / env.total_dags_count if env.total_dags_count > 0 else 0
        ep_task_delay = env.total_task_delay / env.total_tasks if env.total_tasks > 0 else 0
        ep_total_energy = sum(p.total_energy for p in env.processors)
        ep_avg_energy = ep_total_energy / env.total_tasks if env.total_tasks > 0 else 0
        ep_total_dags = env.total_dags_count
        ep_total_tasks = env.total_tasks
        completion_rates.append(ep_completion_rate)
        dag_completion_rates.append(ep_dag_completion_rate)
        task_delays.append(ep_task_delay)
        ep_energies.append(ep_avg_energy)
        ep_dag_counts.append(ep_total_dags)
        ep_task_counts.append(ep_total_tasks)

        ep_priority_stats = {}
        if hasattr(env, '_priority_stats'):
            for p, ps in env._priority_stats.items():
                ep_priority_stats[p] = {k: ps.get(k, 0) for k in (
                    'total_dags', 'on_time_dags',
                    'total_tasks', 'on_time_tasks',
                    'total_delay', 'total_energy',
                )}
                bucket = agg_priority_stats.setdefault(
                    p,
                    {
                        'total_dags': 0, 'on_time_dags': 0,
                        'total_tasks': 0, 'on_time_tasks': 0,
                        'total_delay': 0.0, 'total_energy': 0.0,
                    },
                )
                for k in bucket:
                    bucket[k] += ps.get(k, 0)
        ep_priority_list.append(ep_priority_stats)
        extra = env.snapshot_revision_metrics()
        extra['steps'] = ep_steps
        extra['mean_decision_s'] = (
            ep_decision_s / ep_steps if ep_steps > 0 else 0.0
        )
        revision_metrics.append(extra)

    return {
        'mean_reward': total_reward / total_steps if total_steps > 0 else 0,
        'mean_steps': total_steps / num_episodes,
        'mean_completion_rate': np.mean(completion_rates),
        'mean_dag_completion_rate': np.mean(dag_completion_rates),
        'std_completion_rate': float(np.std(completion_rates)) if completion_rates else 0.0,
        'std_dag_completion_rate': float(np.std(dag_completion_rates)) if dag_completion_rates else 0.0,
        'mean_task_delay': np.mean(task_delays),
        'mean_energy_per_task': np.mean(ep_energies) if ep_energies else 0,
        'total_dags': int(np.mean(ep_dag_counts)) if ep_dag_counts else 0,
        'total_tasks': int(np.mean(ep_task_counts)) if ep_task_counts else 0,
        'per_episode': [
            {
                'episode_idx': i,
                'completion_rate': completion_rates[i],
                'dag_completion_rate': dag_completion_rates[i],
                'avg_task_delay': task_delays[i],
                'avg_energy': ep_energies[i] if i < len(ep_energies) else 0,
                'priority_stats': ep_priority_list[i] if i < len(ep_priority_list) else {},
                **(revision_metrics[i] if i < len(revision_metrics) else {}),
            }
            for i in range(len(completion_rates))
        ],
        'priority_stats': agg_priority_stats,
        'urgency_diagnosis': {
            'episodes': urgency_dag_info,
            'avg_dag_urgency': np.mean([e['avg_dag_urgency'] for e in urgency_dag_info]) if urgency_dag_info else 0,
            'high_urgency_task_ratio': np.mean([e['high_urgency_task_ratio'] for e in urgency_dag_info]) if urgency_dag_info else 0,
            'negative_reward_ratio': np.mean([e['negative_reward_ratio'] for e in urgency_dag_info]) if urgency_dag_info else 0,
        },
        'beam_steps': beam_steps,
        'greedy_steps': greedy_steps,
        'beam_step_ratio': (
            float(beam_steps / (beam_steps + greedy_steps))
            if (beam_steps + greedy_steps) > 0 else 0.0
        ),
        'beam_width': beam_width,
        'beam_load_threshold': beam_load_threshold,
    }


def lookup_priority_stats(ps_dict: Optional[dict], p) -> dict:
    """按 priority 取值，兼容 float / str / 近似相等的 key。"""
    if not ps_dict:
        return {}
    if p in ps_dict:
        return ps_dict[p] or {}
    try:
        pf = float(p)
    except (TypeError, ValueError):
        return {}
    hit = ps_dict.get(pf) or ps_dict.get(f'{pf:.1f}') or ps_dict.get(str(pf))
    if hit:
        return hit
    for k, v in ps_dict.items():
        try:
            if abs(float(k) - pf) < 1e-9:
                return v or {}
        except (TypeError, ValueError):
            continue
    return {}


def priority_csv_cols(ps_dict: Optional[dict], priority_levels) -> list:
    """从 priority_stats 提取与 header 对齐的 CSV 列（dag/task/delay/energy × 各 priority）。"""
    cols = []
    for p in priority_levels:
        s = lookup_priority_stats(ps_dict, p)
        td = s.get('total_dags', 0)
        tt = s.get('total_tasks', 0)
        cols.append(f"{s.get('on_time_dags', 0) / td:.4f}" if td > 0 else 'N/A')
        cols.append(f"{s.get('on_time_tasks', 0) / tt:.4f}" if tt > 0 else 'N/A')
        cols.append(f"{s.get('total_delay', 0.0) / tt:.4f}" if tt > 0 else 'N/A')
        cols.append(f"{s.get('total_energy', 0.0) / tt:.4f}" if tt > 0 else 'N/A')
    return cols


def format_priority_line(priority_stats: Optional[dict]) -> str:
    """控制台一行摘要：p1.5: DAG=.. TASK=.. | ..."""
    if not priority_stats:
        return '(无)'
    keys = sorted(priority_stats.keys(), key=lambda x: float(x))
    parts = []
    for p in keys:
        ps = priority_stats[p] or {}
        td = ps.get('total_dags', 0)
        tt = ps.get('total_tasks', 0)
        if td == 0 and tt == 0:
            continue
        dag_r = ps.get('on_time_dags', 0) / td if td else float('nan')
        task_r = ps.get('on_time_tasks', 0) / tt if tt else float('nan')
        parts.append(f'p{float(p):.1f}: DAG={dag_r:.2%} TASK={task_r:.2%} (n={td})')
    return ' | '.join(parts) if parts else '(无)'


def build_training_log_row(
    row_type: str,
    episode,
    num_dags,
    dag_completion_rate,
    num_tasks,
    task_completion_rate,
    avg_reward,
    steps,
    avg_delay,
    avg_energy,
    priority_levels,
    priority_stats: Optional[dict] = None,
    eft_aux_loss='',
    eft_top1_acc='',
    task_aux_loss='',
    task_top1_acc='',
) -> list:
    """构造与 training_log.csv header 列数严格对齐的一行（含分 priority）。"""
    def _fmt_rate(x):
        if isinstance(x, str):
            return x
        return f'{float(x):.4f}'

    def _fmt_num(x):
        if isinstance(x, str):
            return x
        return f'{float(x):.4f}'

    row = [
        row_type,
        episode,
        num_dags,
        _fmt_rate(dag_completion_rate),
        num_tasks,
        _fmt_rate(task_completion_rate),
        _fmt_num(avg_reward),
        steps,
        _fmt_num(avg_delay),
        _fmt_num(avg_energy),
        eft_aux_loss,
        eft_top1_acc,
        task_aux_loss,
        task_top1_acc,
    ] + priority_csv_cols(priority_stats, priority_levels)
    return row


def main():
    parser = argparse.ArgumentParser(description='分层GAT调度器 PPO训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--episodes', type=int, default=201, help='训练episode数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度 (256→128 CPU优化)')
    parser.add_argument('--lr', type=float, default=4e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='batch大小 (128→64 CPU优化)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'mps', 'cuda', 'gpu'],
                        help='训练设备: auto/gpu=自动选择, cpu, mps(Apple), cuda(NVIDIA)')
    parser.add_argument('--num_threads', type=int, default=0,
                        help='PyTorch CPU 线程数 (0=按设备自动设置)')
    parser.add_argument('--enable_hold', type=int, default=0, help='是否启用Global Hold (1=yes, 0=no)')
    parser.add_argument('--max_attn_dags', type=int, default=64, help='Inter-DAG Attention 最大DAG数 (超量时裁剪)')
    parser.add_argument('--joint_matching', type=int, default=1, help='联合匹配 O(M×N) (1=yes, 0=两阶段)')
    parser.add_argument(
        '--use_pair_features', type=int, default=1, choices=[0, 1],
        help='处理器头使用显式 Task×Processor EFT/能耗成对特征',
    )
    parser.add_argument(
        '--use_load_features', type=int, default=0, choices=[0, 1],
        help='任务头拼接全局负载特征 queue/ready/active_dag（改结构，需从头训）',
    )
    parser.add_argument(
        '--eft_aux_coef', type=float, default=0.2,
        help='EFT processor 排序辅助损失系数；0 表示关闭',
    )
    parser.add_argument(
        '--task_aux_coef', type=float, default=0.0,
        help='SLA laxity 任务排序辅助损失初始系数；0 表示关闭',
    )
    parser.add_argument(
        '--task_aux_decay_episodes', type=int, default=50,
        help='任务排序辅助损失线性衰减到 0 的轮数；<=0 表示不衰减',
    )
    parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate', 'add'],
                        help='任务-DAG嵌入融合: gate=门控残差, add=固定加法(消融)')
    parser.add_argument('--adj_direction', type=str, default='down', choices=['down', 'up', 'both'],
                        help='Intra-DAG GAT消息方向: down=父 attend 子, up=子 attend 父, both=双向')
    parser.add_argument('--ppo_epochs', type=int, default=1, help='PPO epoch数 (4→2 CPU优化)')
    parser.add_argument('--model_type', type=str, default='gat', choices=['gat', 'mlp'],
                        help='模型类型: gat=异构图注意力, mlp=纯MLP (消融实验)')
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_inter', 'mixed_features'],
                        help='GAT 消融: none=Full, no_inter=w/o Inter-DAG, mixed_features=w/o 特征分离')
    parser.add_argument('--device_config', type=str, default='dataset/device/device_heterogeneous.json',
                        help='处理器配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='从checkpoint恢复训练 (best_model.pt路径)')
    parser.add_argument(
        '--resume_weights_only', type=int, default=0, choices=[0, 1],
        help='仅加载模型权重与 wait_time；optimizer 用 --lr 重新初始化（微调高峰 ckpt）',
    )
    parser.add_argument(
        '--start_episode', type=int, default=-1,
        help='覆盖训练起始 episode；-1=从 ckpt.episode 或 0；微调建议设为 50+ 跳过课程学习',
    )
    parser.add_argument(
        '--early_stop_min_dag', type=float, default=-1.0,
        help='eval DAG 低于该阈值计入失败（<0 关闭）；保住高峰时可设 0.80',
    )
    parser.add_argument(
        '--early_stop_patience', type=int, default=3,
        help='连续多少次 post-warmup eval 低于 early_stop_min_dag 则提前停止',
    )
    parser.add_argument(
        '--peak_guard', type=int, default=0, choices=[0, 1],
        help='峰值守护：Val best DAG 达 arm 后，若相对 Val best 下滑超过阈值则停训（保留 best_dag；peak_model 按 train DAG 另存）',
    )
    parser.add_argument(
        '--peak_guard_drop', type=float, default=0.03,
        help='相对 best_eval_dag 允许的最大回撤（如 0.03=掉 3 个点开始计数）',
    )
    parser.add_argument(
        '--peak_guard_patience', type=int, default=2,
        help='连续多少次 eval 低于 best-drop 则停止（默认 2，尽快保住高峰）',
    )
    parser.add_argument(
        '--peak_guard_arm_dag', type=float, default=0.82,
        help='best_eval_dag 达到该值后才启用峰值守护（避免过早停在低水平）',
    )
    parser.add_argument(
        '--entropy_after_peak', type=float, default=-1.0,
        help='峰值守护武装后将 entropy 固定为此值；<0 表示不改（建议 0 或 0.005）',
    )
    parser.add_argument(
        '--run_tag', type=str, default='',
        help='可选后缀写入 run 目录名（如 ft_ep61）',
    )
    parser.add_argument(
        '--freeze_pair_encoder', type=int, default=0, choices=[0, 1],
        help='冻结 pair_encoder（保住 Pair 特征，仅微调其余参数）',
    )
    parser.add_argument(
        '--task_policy', type=str, default='ppo', choices=['ppo', 'edf', 'est'],
        help='任务选择: ppo=策略; edf=候选中 LFT 最小; est=候选中实际就绪最早（均仅学/用处理器头）',
    )
    parser.add_argument(
        '--train_proc_only', type=int, default=0, choices=[0, 1],
        help='仅训练处理器相关头+critic（冻结 GAT 与 task_scorer）',
    )
    parser.add_argument(
        '--dag_reward_timing', type=str, default='complete',
        choices=['complete', 'schedule'],
        help='DAG 完成奖励时机: complete=全部任务执行完; schedule=末任务被调度时',
    )
    parser.add_argument(
        '--dag_early_miss_penalty', type=int, default=0, choices=[0, 1],
        help=(
            '调度后若 max(已调度 finish)>deadline，立即发 −DAG_BONUS（每 DAG 一次；'
            '与 complete/schedule 共用防重复，避免迟到再扣一次）'
        ),
    )
    parser.add_argument(
        '--late_dag_reward_scale', type=float, default=1.0,
        help=(
            '决策时 current_time>dag.deadline（DAG 已超时）则本步任务奖励×该系数；'
            '1.0=关闭，0.0=忽略已超时图上的任务奖励（DAG±bonus 仍照常）'
        ),
    )
    parser.add_argument(
        '--task_reward_mode', type=str, default='g2_p',
        choices=['g2', 'g2_p', 'lft'],
        help=(
            'TASK 奖励: g2=原始 G2（无 TASK 级 priority 权重）; '
            'g2_p=G2×priority_weight（p1.5=2.0, p2=1.0, p3=0.5）; '
            'lft=归一化 signed-LFT 奖励（同样乘 priority_weight）。'
            'DAG 完成奖励始终按 SLA 分级（p1.5 更大）。'
        ),
    )
    parser.add_argument(
        '--entropy_coef_start', type=float, default=0.05,
        help='PPO 熵系数初值',
    )
    parser.add_argument(
        '--entropy_coef_end', type=float, default=0.05,
        help='PPO 熵系数终值（与 start 相同则固定）',
    )
    parser.add_argument(
        '--entropy_decay_episodes', type=int, default=100,
        help=(
            '前多少 ep 将熵从 start 线性降到 end，之后保持 end；'
            '<=0 表示在整个训练时长上线性衰减（旧行为）'
        ),
    )
    parser.add_argument(
        '--lr_decay', type=float, default=DEFAULT_LR_DECAY,
        help='每 episode 学习率衰减系数（U1=0.995；G2 基线=1.0）',
    )
    parser.add_argument(
        '--eval_episodes', type=int, default=1,
        help='训练中选 best 时的 val episode 数（建议≥5；1=仅 ep0）',
    )
    parser.add_argument(
        '--test_episodes', type=int, default=20,
        help='训练结束后 test 的到达种子数（episode_idx=0..N-1）。验证仍用 eval_episodes',
    )
    parser.add_argument(
        '--wait_time_mean', type=float, default=WAIT_TIME_MEAN,
        help='旧 ckpt / 无统计量时的 test 回退 mean；训练中 val 使用 train EMA',
    )
    parser.add_argument(
        '--wait_time_std', type=float, default=WAIT_TIME_STD,
        help='旧 ckpt / 无统计量时的 test 回退 std',
    )
    args = parser.parse_args()

    if args.model_type == 'mlp' and args.ablation != 'none':
        raise ValueError('model_type=mlp 与 --ablation 不能同时使用；MLP 请用 --model_type mlp')
    if args.eft_aux_coef < 0:
        raise ValueError('--eft_aux_coef 必须 >= 0')
    if args.task_aux_coef < 0:
        raise ValueError('--task_aux_coef 必须 >= 0')

    device = resolve_device(args.device)
    configure_cpu_threads(device, args.num_threads)
    if device == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    print(args)
    print(f"训练设备: {device} (请求: {args.device})")
    if device == 'mps':
        print('提示: 环境仿真仍在 CPU；神经网络前向/反传在 Apple GPU (MPS)')

    set_seed(args.seed, device)
    ablation_tag = '' if args.ablation == 'none' else f'_{args.ablation}'
    dag_timing_tag = '_dagsched' if args.dag_reward_timing == 'schedule' else '_dagcomplete'
    early_miss_tag = '_earlymiss' if args.dag_early_miss_penalty else ''
    late_scale_tag = (
        f'_latescale{args.late_dag_reward_scale:g}'
        if args.late_dag_reward_scale != 1.0
        else ''
    )
    task_pol_tag = (
        '_edftask' if args.task_policy == 'edf'
        else ('_esttask' if args.task_policy == 'est' else '')
    )
    proc_only_tag = '_proconly' if args.train_proc_only else ''
    task_reward_tag = f'_{args.task_reward_mode}'
    proc_head_tag = (
        f"_peft{args.eft_aux_coef:g}"
        if args.use_pair_features or args.eft_aux_coef > 0
        else ""
    )
    task_aux_tag = (
        f"_tlax{args.task_aux_coef:g}d{args.task_aux_decay_episodes}"
        if args.task_aux_coef > 0
        else ""
    )
    peak_tag = "_peakguard" if args.peak_guard else ""
    load_tag = "_loadfeat" if args.use_load_features else ""
    run_tag = f"_{args.run_tag}" if args.run_tag else ""
    log_dir = (
        f"runs/hierarchical_ppo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_seed{args.seed}_{device}{ablation_tag}{dag_timing_tag}{early_miss_tag}"
        f"{late_scale_tag}{task_pol_tag}{proc_only_tag}"
        f"{task_reward_tag}{proc_head_tag}{task_aux_tag}{load_tag}{peak_tag}{run_tag}"
    )

    # 创建环境
    env = MultiDAGSchedulerEnv(
        task_dim=ORIGINAL_TASK_DIM,
        proc_dim=PROC_DIM,
        dag_dataset_dir='dataset/daggen_factory/dags',
        max_candidates=15,
        arrival_rate=0.55,
        seed=args.seed,
        dag_generator='notOur',
        device_config=args.device_config,
        dag_reward_timing=args.dag_reward_timing,
        task_reward_mode=args.task_reward_mode,
        dag_early_miss_penalty=bool(args.dag_early_miss_penalty),
        late_dag_reward_scale=float(args.late_dag_reward_scale),
    )
    env.set_eval_wait_time_stats(args.wait_time_mean, args.wait_time_std)
    print(f"DAG 奖励时机: {args.dag_reward_timing}")
    print(f"DAG 提前超时惩罚: {bool(args.dag_early_miss_penalty)}")
    print(f"已超时 DAG 任务奖励缩放: {args.late_dag_reward_scale:g}")
    print(f"TASK 奖励模式: {args.task_reward_mode}")
    print(f"任务策略 task_policy: {args.task_policy}")
    print(
        f"随机种子: train env.seed={args.seed} "
        f"(到达/ρ/训练DAG池)；val 钉 EVAL_ARRIVAL_SEED={EVAL_ARRIVAL_SEED}；"
        f"test 钉 TEST_ARRIVAL_SEED={TEST_ARRIVAL_SEED}"
    )
    print(f"train_proc_only: {bool(args.train_proc_only)}")
    print(
        f"处理器头: pair_features={bool(args.use_pair_features)}, "
        f"eft_aux_coef={args.eft_aux_coef}"
    )
    print(f"任务头负载特征: use_load_features={bool(args.use_load_features)}")
    print(
        f"任务排序辅助: initial_coef={args.task_aux_coef}, "
        f"decay_episodes={args.task_aux_decay_episodes}"
    )
    print(
        "wait_time: train/val 用训练 EMA；"
        f"test 从 ckpt 加载（回退 mean={args.wait_time_mean}, std={args.wait_time_std}）"
    )
    if args.entropy_coef_start != args.entropy_coef_end:
        if args.entropy_decay_episodes > 0:
            print(
                f"熵系数: 前 {args.entropy_decay_episodes} ep "
                f"{args.entropy_coef_start} → {args.entropy_coef_end}，之后保持 "
                f"{args.entropy_coef_end}"
            )
        else:
            print(
                f"熵系数线性衰减 (全时长): {args.entropy_coef_start} → "
                f"{args.entropy_coef_end} over {args.episodes} episodes"
            )
    else:
        print(f"熵系数固定: {args.entropy_coef_start}")
    if args.lr_decay < 1.0:
        final_lr_ratio = args.lr_decay ** max(args.episodes - 1, 0)
        print(
            f"学习率衰减 (U1): 每 ep ×{args.lr_decay} "
            f"(初始 {args.lr:.2e} → 约 {args.lr * final_lr_ratio:.2e} @ ep{args.episodes})"
        )
    else:
        print(f"学习率衰减: 关闭 (lr_decay={args.lr_decay})")

    # 创建模型
    if args.model_type == 'mlp':
        model = create_mlp_scheduler(
            static_dim=STATIC_DIM, dyn_dim=DYN_DIM, dag_meta_dim=DAG_META_DIM,
            proc_dim=PROC_DIM, hidden_dim=args.hidden_dim, num_procs=NUM_PROCS,
            enable_hold=(args.enable_hold == 1), max_attn_dags=args.max_attn_dags,
            joint_matching=(args.joint_matching == 1),
            fusion_mode=args.fusion_mode,
            use_pair_features=(args.use_pair_features == 1),
            use_load_features=(args.use_load_features == 1),
        ).to(device)
        print(f"模型类型: MLP (消融 A1), fusion_mode={args.fusion_mode}")
    elif args.ablation == 'no_inter':
        model = create_no_inter_scheduler(
            static_dim=STATIC_DIM, dyn_dim=DYN_DIM, dag_meta_dim=DAG_META_DIM,
            proc_dim=PROC_DIM, hidden_dim=args.hidden_dim, num_procs=NUM_PROCS,
            enable_hold=(args.enable_hold == 1), max_attn_dags=args.max_attn_dags,
            joint_matching=(args.joint_matching == 1),
            fusion_mode=args.fusion_mode,
            adj_direction=args.adj_direction,
            use_pair_features=(args.use_pair_features == 1),
            use_load_features=(args.use_load_features == 1),
        ).to(device)
        print(f"模型类型: GAT w/o Inter-DAG (消融 A2), fusion_mode={args.fusion_mode}")
    elif args.ablation == 'mixed_features':
        model = create_mixed_features_scheduler(
            static_dim=STATIC_DIM, dyn_dim=DYN_DIM, dag_meta_dim=DAG_META_DIM,
            proc_dim=PROC_DIM, hidden_dim=args.hidden_dim, num_procs=NUM_PROCS,
            enable_hold=(args.enable_hold == 1), max_attn_dags=args.max_attn_dags,
            joint_matching=(args.joint_matching == 1),
            fusion_mode=args.fusion_mode,
            adj_direction=args.adj_direction,
            use_pair_features=(args.use_pair_features == 1),
            use_load_features=(args.use_load_features == 1),
        ).to(device)
        print(f"模型类型: GAT w/o 特征分离 (消融 A3), GAT输入={STATIC_DIM + DYN_DIM}维")
    else:
        model = HierarchicalGATScheduler(
            static_dim=STATIC_DIM, dyn_dim=DYN_DIM, dag_meta_dim=DAG_META_DIM,
            proc_dim=PROC_DIM, hidden_dim=args.hidden_dim, num_procs=NUM_PROCS,
            enable_hold=(args.enable_hold == 1), max_attn_dags=args.max_attn_dags,
            joint_matching=(args.joint_matching == 1),
            fusion_mode=args.fusion_mode,
            adj_direction=args.adj_direction,
            use_pair_features=(args.use_pair_features == 1),
            use_load_features=(args.use_load_features == 1),
        ).to(device)
        print(f"模型类型: GAT+PPO Full, fusion_mode={args.fusion_mode}, adj_direction={args.adj_direction}")

    # 创建优化器
    if args.task_policy in RULE_TASK_POLICIES and args.joint_matching == 1:
        raise ValueError(
            f'--task_policy {args.task_policy} 与 --joint_matching 1 不能同时使用'
        )

    if args.train_proc_only:
        # 冻结 GAT + task_scorer；只训处理器头与 critic
        ts = model.two_stage_scheduler
        trainable_modules = [
            ts.proc_encoder,
            ts.proc_selector,
            ts.hold_encoder,
            model.critic_net,
        ]
        if getattr(ts, 'pair_encoder', None) is not None:
            trainable_modules.append(ts.pair_encoder)
        trainable_ids = {id(p) for m in trainable_modules for p in m.parameters()}
        frozen = 0
        for p in model.parameters():
            if id(p) in trainable_ids:
                p.requires_grad = True
            else:
                p.requires_grad = False
                frozen += p.numel()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr)
        print(
            f"train_proc_only: 冻结参数量={frozen:,}; "
            f"可训练={sum(p.numel() for p in trainable):,}"
        )
    elif args.freeze_pair_encoder:
        pe = getattr(model.two_stage_scheduler, 'pair_encoder', None)
        if pe is None:
            raise RuntimeError('--freeze_pair_encoder=1 但模型无 pair_encoder')
        frozen = 0
        for p in pe.parameters():
            p.requires_grad = False
            frozen += p.numel()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=args.lr)
        print(f"已冻结 pair_encoder 参数量={frozen:,}; 可训练={sum(p.numel() for p in trainable):,}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 从 checkpoint 恢复
    start_episode = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # 新格式: 完整 checkpoint
            ckpt_cfg = ckpt.get('config', {})
            ckpt_uses_pair = bool(
                ckpt_cfg.get('use_pair_features', 0)
                if isinstance(ckpt_cfg, dict)
                else 0
            )
            warm_start_pair_head = (
                args.use_pair_features == 1 and not ckpt_uses_pair
            )
            weights_only = bool(args.resume_weights_only) or warm_start_pair_head
            if warm_start_pair_head:
                missing, unexpected = model.load_state_dict(
                    ckpt['model_state_dict'], strict=False
                )
                illegal_missing = [
                    k for k in missing
                    if not k.startswith('two_stage_scheduler.pair_encoder.')
                ]
                if illegal_missing or unexpected:
                    raise RuntimeError(
                        "旧 checkpoint warm-start 仅允许缺少 pair_encoder；"
                        f"missing={illegal_missing}, unexpected={unexpected}"
                    )
                # 新增参数后旧 optimizer state 的参数组不兼容，从新 LR 开始。
                start_episode = 0
                print(
                    f"从 {args.resume} warm-start 主干权重；"
                    "pair_encoder 与 optimizer 从头训练"
                )
            elif weights_only:
                model.load_state_dict(ckpt['model_state_dict'])
                start_episode = int(ckpt.get('episode', 0))
                print(
                    f"从 {args.resume} 仅加载权重 (episode {start_episode})；"
                    f"optimizer 以 lr={args.lr} 重新初始化"
                )
            else:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_episode = ckpt.get('episode', 0)
                print(f"从 {args.resume} 恢复训练 (episode {start_episode})")
            wt_mean, wt_std = extract_wait_time_from_ckpt(
                ckpt, args.wait_time_mean, args.wait_time_std
            )
            env.load_train_wait_time_stats(wt_mean, wt_std)
            print(f"  恢复 wait_time EMA: mean={wt_mean:.4f}, std={wt_std:.4f}")
        else:
            # 旧格式: 裸 state_dict, 仅加载模型权重
            model.load_state_dict(ckpt)
            start_episode = 0
            print(f"从 {args.resume} 加载模型权重 (旧格式, 优化器从头初始化)")

    if args.start_episode >= 0:
        start_episode = args.start_episode
        print(f"覆盖 start_episode={start_episode}")

    # 创建buffer
    buffer = HierarchicalReplayBuffer(
        max_buffer_size=512,
        static_dim=STATIC_DIM,
        dyn_dim=DYN_DIM,
        dag_meta_dim=DAG_META_DIM,
        proc_dim=PROC_DIM,
        num_procs=NUM_PROCS,
        batch_size=args.batch_size
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)

    run_config = {
        'seed': args.seed,
        'hidden_dim': args.hidden_dim,
        'enable_hold': args.enable_hold,
        'max_attn_dags': args.max_attn_dags,
        'joint_matching': args.joint_matching,
        'use_pair_features': args.use_pair_features,
        'use_load_features': args.use_load_features,
        'eft_aux_coef': args.eft_aux_coef,
        'task_aux_coef': args.task_aux_coef,
        'task_aux_decay_episodes': args.task_aux_decay_episodes,
        'fusion_mode': args.fusion_mode,
        'adj_direction': args.adj_direction,
        'model_type': args.model_type,
        'ablation': args.ablation,
        'dag_reward_timing': args.dag_reward_timing,
        'dag_early_miss_penalty': int(args.dag_early_miss_penalty),
        'late_dag_reward_scale': float(args.late_dag_reward_scale),
        'task_reward_mode': args.task_reward_mode,
        'lr': args.lr,
        'lr_decay': args.lr_decay,
        'entropy_coef_start': args.entropy_coef_start,
        'entropy_coef_end': args.entropy_coef_end,
        'entropy_decay_episodes': args.entropy_decay_episodes,
        'ppo_epochs': args.ppo_epochs,
        'eval_episodes': args.eval_episodes,
        'resume': args.resume,
        'resume_weights_only': args.resume_weights_only,
        'start_episode': start_episode,
        'early_stop_min_dag': args.early_stop_min_dag,
        'early_stop_patience': args.early_stop_patience,
        'peak_guard': args.peak_guard,
        'peak_guard_drop': args.peak_guard_drop,
        'peak_guard_patience': args.peak_guard_patience,
        'peak_guard_arm_dag': args.peak_guard_arm_dag,
        'entropy_after_peak': args.entropy_after_peak,
        'run_tag': args.run_tag,
        'freeze_pair_encoder': args.freeze_pair_encoder,
        'task_policy': args.task_policy,
        'train_proc_only': int(args.train_proc_only),
        'wait_time_policy': 'train_ema_for_val_ckpt_for_test',
        'wait_time_fallback_mean': args.wait_time_mean,
        'wait_time_fallback_std': args.wait_time_std,
        'priority_ratios': list(env.priority_ratios),
    }
    with open(os.path.join(log_dir, 'run_config.json'), 'w') as f:
        import json
        json.dump(run_config, f, indent=2)

    def _save_ckpt(path, episode_num):
        wt_mean, wt_std = env.get_train_wait_time_stats()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode_num,
            'config': run_config,
            'wait_time_mean': wt_mean,
            'wait_time_std': wt_std,
        }, path)

    # CSV 日志文件
    import csv
    csv_path = os.path.join(log_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    priority_levels = list(env.priority_ratios)
    csv_header = ['type', 'episode', 'num_dags', 'dag_completion_rate', 'num_tasks',
                  'task_completion_rate', 'avg_reward', 'steps', 'avg_delay', 'avg_energy',
                  'eft_aux_loss', 'eft_top1_acc', 'task_aux_loss', 'task_top1_acc']
    for p in priority_levels:
        csv_header.extend([f'dag_p{p:.1f}', f'task_p{p:.1f}', f'delay_p{p:.1f}', f'energy_p{p:.1f}'])
    csv_writer.writerow(csv_header)

    def _write_log_row(row: list, writer=None):
        """写入一行并校验列数，防止 train/eval/test 漏写分 priority 列。"""
        if len(row) != len(csv_header):
            raise ValueError(
                f'training_log 行列数不一致: got {len(row)}, expect {len(csv_header)} '
                f'(type={row[0] if row else "?"}). 请检查是否漏写 priority 列。'
            )
        (writer or csv_writer).writerow(row)

    # 对已跳过的 episode 应用 LR 衰减（weights-only 微调用新 LR，不回溯衰减）
    if not args.resume_weights_only:
        for _ in range(start_episode):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

    # 训练循环
    best_eval_dag = -float('inf')
    best_eval_reward = -float('inf')
    best_train_dag = -float('inf')
    early_fail_count = 0
    peak_fail_count = 0
    peak_armed = False
    stopped_early = False
    stop_reason = ''

    # 微调：先落盘起点权重，避免 early-stop 时无 best ckpt
    if args.resume and args.resume_weights_only:
        _save_ckpt(os.path.join(log_dir, 'best_dag_model.pt'), start_episode)
        _save_ckpt(os.path.join(log_dir, 'best_reward_model.pt'), start_episode)
        _save_ckpt(os.path.join(log_dir, 'peak_model.pt'), start_episode)
        print(f"已保存微调起点 ckpt (episode={start_episode})")

    if args.peak_guard:
        print(
            f"峰值守护: arm≥{args.peak_guard_arm_dag:.0%}, "
            f"drop={args.peak_guard_drop:.0%}, "
            f"patience={args.peak_guard_patience}"
            + (
                f", entropy_after_peak={args.entropy_after_peak}"
                if args.entropy_after_peak >= 0
                else ""
            )
        )

    for episode in range(start_episode, args.episodes):
        current_entropy_coef = linear_entropy_coef(
            episode,
            args.episodes,
            args.entropy_coef_start,
            args.entropy_coef_end,
            start_episode=start_episode,
            decay_episodes=args.entropy_decay_episodes,
        )
        if peak_armed and args.entropy_after_peak >= 0:
            current_entropy_coef = args.entropy_after_peak
        current_task_aux_coef = linear_aux_coef_to_zero(
            episode,
            args.task_aux_coef,
            args.task_aux_decay_episodes,
        )
        train_stats = run_training_step(
            env, model, buffer, optimizer,
            gamma=DEFAULT_GAMMA,
            gae_lambda=DEFAULT_GAE_LAMBDA,
            epsilon=DEFAULT_EPSILON,
            value_coef=DEFAULT_VALUE_COEF,
            entropy_coef=current_entropy_coef,
            max_grad_norm=DEFAULT_MAX_GRAD_NORM,
            eft_aux_coef=args.eft_aux_coef,
            task_aux_coef=current_task_aux_coef,
            ppo_epochs=args.ppo_epochs,
            episode_idx=episode,
            device=device,
            task_policy=args.task_policy,
        )

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 获取更新统计
        us = train_stats['update_stats']
        update_count = us['count']

        # 计算各项平均值
        if update_count > 0:
            avg_loss = us['total_loss_sum'] / update_count
            avg_grad = us['grad_norm_sum'] / update_count
            avg_entropy = us['entropy_sum'] / update_count
            avg_value_pred = us['value_pred_sum'] / update_count
            avg_value_target = us['value_target_sum'] / update_count
            avg_value_target_raw = us['value_target_raw_sum'] / update_count
            avg_reward_mean = us.get('debug_rewards_mean_sum', 0) / update_count
            avg_eft_aux = us.get('eft_aux_loss_sum', 0) / update_count
            avg_eft_acc = us.get('eft_top1_acc_sum', 0) / update_count
            avg_task_aux = us.get('task_aux_loss_sum', 0) / update_count
            avg_task_acc = us.get('task_top1_acc_sum', 0) / update_count
        else:
            avg_loss = avg_grad = avg_entropy = avg_value_pred = avg_value_target = avg_value_target_raw = avg_reward_mean = avg_eft_aux = avg_eft_acc = avg_task_aux = avg_task_acc = 0

        # 计算奖励统计
        reward_mean = train_stats.get('reward_mean', 0)
        reward_std = train_stats.get('reward_std', 0)
        reward_pos_ratio = train_stats.get('reward_pos_ratio', 0)
        avg_task_delay = train_stats.get('avg_task_delay', 0)
        avg_energy = train_stats.get('avg_energy', 0)

        print(f"Episode {episode + 1}/{args.episodes} | "
              f"Reward: {train_stats['reward']/max(train_stats['steps'], 1):.4f} | "
              f"TASK: {train_stats['completion_rate']:.2%} | DELAY: {avg_task_delay:.1f} | "
              f"ENERGY: {avg_energy:.0f}J | "
              f"DAG: {train_stats['dag_completion_rate']:.2%} | "
              f"r+: {reward_pos_ratio:.1%} | "
              f"LR: {current_lr:.2e} | "
              f"EntCoef: {current_entropy_coef:.4f} | "
              f"Updates: {update_count} | "
              f"Loss: {avg_loss:.4f} | Grad: {avg_grad:.4f} | "
              f"EFTaux: {avg_eft_aux:.4f} | EFTacc: {avg_eft_acc:.1%} | "
              f"TaskAux({current_task_aux_coef:.4f}): {avg_task_aux:.4f} | "
              f"TaskAcc: {avg_task_acc:.1%} | "
              f"Entropy: {avg_entropy:.4f} | "
              f"Vpred: {avg_value_pred:.4f} | Vtarget(norm): {avg_value_target:.4f} | Vtarget(raw): {avg_value_target_raw:.2f}")

        _write_log_row(build_training_log_row(
            'train', episode + 1,
            train_stats.get('total_dags', 0),
            train_stats['dag_completion_rate'],
            train_stats.get('total_tasks', 0),
            train_stats['completion_rate'],
            train_stats['reward_mean'],
            train_stats['steps'],
            avg_task_delay,
            avg_energy,
            priority_levels,
            priority_stats=train_stats.get('priority_stats', {}),
            eft_aux_loss=f"{avg_eft_aux:.6f}",
            eft_top1_acc=f"{avg_eft_acc:.6f}",
            task_aux_loss=f"{avg_task_aux:.6f}",
            task_top1_acc=f"{avg_task_acc:.6f}",
        ))
        csv_file.flush()

        # peak_model：按 train ep 的 DAG 完成率存峰值（与 Val 选优的 best_dag_model 分离）
        train_dag_rate = float(train_stats.get('dag_completion_rate', 0.0))
        if episode + 1 > 50 and train_dag_rate > best_train_dag:
            best_train_dag = train_dag_rate
            _save_ckpt(os.path.join(log_dir, 'peak_model.pt'), episode + 1)
            print(
                f"  [peak_model] train DAG 创新高 {best_train_dag:.2%} "
                f"@ ep{episode + 1} → peak_model.pt"
            )

        # 评估
        if episode % 5 == 0:
            eval_stats = evaluate_policy(
                env, model, num_episodes=args.eval_episodes, device=device,
                task_policy=args.task_policy,
            )
            eval_energy = eval_stats.get('mean_energy_per_task', 0)
            print(f"\n Episode {episode + 1} 评估结果: Reward: {eval_stats['mean_reward']:.2f} | "
                  f"TASK: {eval_stats['mean_completion_rate']:.2%} | "
                  f"DELAY: {eval_stats.get('mean_task_delay', 0):.1f} | "
                  f"ENERGY: {eval_energy:.0f}J | "
                  f"DAG: {eval_stats['mean_dag_completion_rate']:.2%}")
            print(f"  分 priority: {format_priority_line(eval_stats.get('priority_stats'))}")

            # 紧迫度诊断
            diag = eval_stats.get('urgency_diagnosis', {})
            if diag:
                print(f"  [诊断] avg_dag_urgency={diag.get('avg_dag_urgency', 0):.2f} | "
                      f"high_urgency_task_ratio={diag.get('high_urgency_task_ratio', 0):.2%} | "
                      f"negative_reward_ratio={diag.get('negative_reward_ratio', 0):.2%}")

            _write_log_row(build_training_log_row(
                'eval', episode + 1,
                eval_stats.get('total_dags', 0),
                eval_stats['mean_dag_completion_rate'],
                eval_stats.get('total_tasks', 0),
                eval_stats['mean_completion_rate'],
                eval_stats['mean_reward'],
                int(eval_stats.get('mean_steps', 0)),
                eval_stats.get('mean_task_delay', 0),
                eval_stats.get('mean_energy_per_task', 0),
                priority_levels,
                priority_stats=eval_stats.get('priority_stats', {}),
            ))
            csv_file.flush()

            # 跳过 warmup 阶段（前 50 轮课程学习），避免低负载下过拟合模型被选为最优
            if episode + 1 > 50:
                dag_rate = eval_stats['mean_dag_completion_rate']
                if dag_rate > best_eval_dag:
                    best_eval_dag = dag_rate
                    _save_ckpt(os.path.join(log_dir, 'best_dag_model.pt'), episode + 1)
                    peak_fail_count = 0
                    if (
                        args.peak_guard
                        and not peak_armed
                        and best_eval_dag >= args.peak_guard_arm_dag
                    ):
                        peak_armed = True
                        print(
                            f"  [peak-guard] armed at ep{episode + 1}: "
                            f"best_DAG={best_eval_dag:.2%}"
                        )

                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    _save_ckpt(os.path.join(log_dir, 'best_reward_model.pt'), episode + 1)

                if args.early_stop_min_dag >= 0:
                    if dag_rate < args.early_stop_min_dag:
                        early_fail_count += 1
                        print(
                            f"  [early-stop] eval DAG={dag_rate:.2%} < "
                            f"{args.early_stop_min_dag:.2%} "
                            f"({early_fail_count}/{args.early_stop_patience})"
                        )
                        if early_fail_count >= args.early_stop_patience:
                            stopped_early = True
                            stop_reason = 'early_stop_min_dag'
                            print(
                                f"\n提前停止：连续 {early_fail_count} 次 eval DAG "
                                f"低于 {args.early_stop_min_dag:.2%}"
                            )
                            break
                    else:
                        early_fail_count = 0

                if args.peak_guard and peak_armed:
                    floor = best_eval_dag - args.peak_guard_drop
                    if dag_rate < floor:
                        peak_fail_count += 1
                        print(
                            f"  [peak-guard] eval DAG={dag_rate:.2%} < "
                            f"best-drop={floor:.2%} "
                            f"({peak_fail_count}/{args.peak_guard_patience}); "
                            f"keep peak={best_eval_dag:.2%}"
                        )
                        if peak_fail_count >= args.peak_guard_patience:
                            stopped_early = True
                            stop_reason = 'peak_guard'
                            print(
                                f"\n峰值守护停止：连续 {peak_fail_count} 次低于 "
                                f"best-{args.peak_guard_drop:.0%}；"
                                f"保留 best_dag_model "
                                f"(val best={best_eval_dag:.2%}); "
                                f"peak_model 为 train DAG 峰值 "
                                f"(train best={best_train_dag:.2%})"
                            )
                            break
                    else:
                        peak_fail_count = 0

        # 学习率衰减（U1: 默认每 ep ×0.995）
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay

    final_episode = episode + 1 if stopped_early else args.episodes
    _save_ckpt(os.path.join(log_dir, 'final_model.pt'), final_episode)
    csv_file.close()
    print(
        "\n训练完成！"
        + (f" (early stop: {stop_reason})" if stopped_early else "")
    )

    # ===== Test 集最终评估：对比 peak / best_DAG / best_Reward =====
    def run_test(model, name, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        wt_mean, wt_std = extract_wait_time_from_ckpt(
            ckpt, args.wait_time_mean, args.wait_time_std
        )
        env.set_eval_wait_time_stats(wt_mean, wt_std)
        model.eval()
        print(f"\n{'='*70}")
        print(f"Test 评估 — {name}")
        print(f"  wait_time from ckpt: mean={wt_mean:.4f}, std={wt_std:.4f}")
        print(f"{'='*70}")
        stats = evaluate_policy(
            env, model,
            num_episodes=args.test_episodes,
            device=device,
            verbose=False,
            split='test',
            task_policy=args.task_policy,
        )
        ep = stats.get('per_episode', [])
        tasks = [r['completion_rate'] for r in ep]
        dags = [r['dag_completion_rate'] for r in ep]
        delays = [r['avg_task_delay'] for r in ep]
        energies = [r['avg_energy'] for r in ep]
        print(f"  任务完成率: {np.mean(tasks):.2%} ± {np.std(tasks):.2%}")
        print(f"  DAG 完成率: {np.mean(dags):.2%} ± {np.std(dags):.2%}")
        print(f"  平均延迟:   {np.mean(delays):.2f} ± {np.std(delays):.2f}s")
        print(f"  平均能耗:   {np.mean(energies):.2f} ± {np.std(energies):.2f}J")
        print(f"  分 priority: {format_priority_line(stats.get('priority_stats'))}")
        if not stats.get('priority_stats'):
            print('  [WARN] priority_stats 为空，分档指标将写为 N/A')

        # 与 train/eval 共用 build_training_log_row，保证含 dag_p*/task_p*/delay_p*/energy_p*
        with open(csv_path, 'a', newline='') as test_csv:
            test_writer = csv.writer(test_csv)
            for ep_i, ep_r in enumerate(ep):
                row = build_training_log_row(
                    'test', f'{name}_{ep_i + 1}',
                    stats.get('total_dags', 0),
                    ep_r['dag_completion_rate'],
                    stats.get('total_tasks', 0),
                    ep_r['completion_rate'],
                    stats['mean_reward'],
                    int(stats.get('mean_steps', 0)),
                    ep_r['avg_task_delay'],
                    ep_r['avg_energy'],
                    priority_levels,
                    priority_stats=ep_r.get('priority_stats', {}),
                )
                _write_log_row(row, writer=test_writer)
        return stats

    for which, name in (
        ('peak_model.pt', 'peak'),
        ('best_dag_model.pt', 'best_DAG'),
        ('best_reward_model.pt', 'best_Reward'),
    ):
        ckpt_path = os.path.join(log_dir, which)
        if os.path.isfile(ckpt_path):
            run_test(model, name, ckpt_path)
        else:
            print(f"跳过 Test：缺少 {ckpt_path}")

    print(f"\nTest 结果已保存到 {log_dir}")


if __name__ == "__main__":
    main()

