"""
分层GAT调度器 - 经验回放缓冲区
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


class HierarchicalReplayBuffer:
    """
    分层GAT调度器的经验回放缓冲区

    存储结构对应HierarchicalGATScheduler的输入:
    - dag_task_static_features: List of List of [num_tasks_i, static_dim]
    - dag_adjs: List of List of [num_tasks_i, num_tasks_i]
    - dag_metadata: [num_dags, dag_meta_dim]
    - ready_task_indices: [max_ready]
    - ready_task_dyn_features: [max_ready, dyn_dim]
    - proc_features: [num_procs, proc_dim]
    - proc_states: [num_procs]
    - task_proc_pair_features: [max_ready, num_procs, pair_dim]
    - eft_targets: [max_ready]
    - sla_task_target: scalar，最小归一化 laxity 的候选任务索引
    - task_dag_ids: [total_tasks]

    动作:
    - actions_task: 任务选择 (0 to M-1)
    - actions_proc: 处理器选择 (0 to N or N for Hold)

    优化: 缓存预计算的 embeddings，避免训练时重复计算 Layer 1+2
    """

    def __init__(
        self,
        max_buffer_size: int = 4096,
        max_dags: int = 10,
        max_candidates: int = 15,
        static_dim: int = 12,
        dyn_dim: int = 7,
        dag_meta_dim: int = 5,
        proc_dim: int = 4,
        num_procs: int = 11,
        batch_size: int = 32,
        hidden_dim: int = 256
    ):
        self.max_buffer_size = max_buffer_size
        self.max_dags = max_dags
        self.max_candidates = max_candidates
        self.static_dim = static_dim
        self.dyn_dim = dyn_dim
        self.dag_meta_dim = dag_meta_dim
        self.proc_dim = proc_dim
        self.num_procs = num_procs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # 指针
        self.ptr = 0
        self.size = 0

        # 存储每条经验的各字段
        self.dag_counts = np.zeros(max_buffer_size, dtype=np.int32)
        self.actions_task = np.zeros(max_buffer_size, dtype=np.int64)
        self.actions_proc = np.zeros(max_buffer_size, dtype=np.int64)
        self.action_logprobs = np.zeros(max_buffer_size, dtype=np.float32)
        self.rewards = np.zeros(max_buffer_size, dtype=np.float32)
        self.dones = np.zeros(max_buffer_size, dtype=np.float32)
        self.values = np.zeros(max_buffer_size, dtype=np.float32)
        self.advantages = np.zeros(max_buffer_size, dtype=np.float32)

        # 存储每个样本的完整环境数据
        self.env_data_list = [None] * max_buffer_size

        # 缓存的 embeddings（预计算，避免训练时重复计算 Layer 1+2）
        self.cached_ready_task_embs = [None] * max_buffer_size  # [M, hidden_dim]
        self.cached_dag_embeddings = [None] * max_buffer_size  # [num_dags, hidden_dim]

    def store(self, env_data: dict, action_task: int, action_proc: int,
              action_logprob: float, reward: float, done: bool,
              value: float = 0.0,
              ready_task_embs: torch.Tensor = None, dag_embeddings: torch.Tensor = None):
        """
        存储一条经验

        Args:
            env_data: 环境数据字典
            action_task: 任务选择动作
            action_proc: 处理器选择动作
            action_logprob: 动作对数概率
            reward: 奖励
            done: 是否结束
            value: 状态价值估计（用于GAE预计算）
            ready_task_embs: 预计算的就绪任务嵌入 [M, hidden_dim]
            dag_embeddings: 预计算的DAG嵌入 [num_dags, hidden_dim]
        """
        idx = self.ptr

        self.dag_counts[idx] = len(env_data['dag_task_static_features'])
        self.actions_task[idx] = action_task
        self.actions_proc[idx] = action_proc
        self.action_logprobs[idx] = action_logprob
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value

        # 深拷贝环境数据（.cpu() 避免GPU显存泄漏）
        self.env_data_list[idx] = {
            'dag_task_static_features': [feat.cpu().clone() for feat in env_data['dag_task_static_features']],
            'dag_adjs': [adj.cpu().clone() for adj in env_data['dag_adjs']],
            'dag_metadata': env_data['dag_metadata'].cpu().clone(),
            'ready_task_indices': env_data['ready_task_indices'].cpu().clone(),
            'ready_task_dyn_features': env_data['ready_task_dyn_features'].cpu().clone(),
            'proc_features': env_data['proc_features'].cpu().clone(),
            'proc_states': env_data['proc_states'].cpu().clone(),
            'task_proc_pair_features': env_data[
                'task_proc_pair_features'
            ].cpu().clone(),
            'eft_targets': env_data['eft_targets'].cpu().clone(),
            'sla_task_target': env_data['sla_task_target'].cpu().clone(),
            'task_dag_ids': env_data['task_dag_ids'].cpu().clone(),
            'candidate_types': env_data['candidate_types'].cpu().clone(),
            'num_dags': len(env_data['dag_task_static_features']),
            'num_active_dags': env_data.get(
                'num_active_dags', len(env_data['dag_task_static_features'])
            ),
            'num_ready': env_data['ready_task_indices'].size(0),
        }

        # 缓存 embeddings（.cpu().detach() 断开计算图并移到CPU，避免显存泄漏）
        self.cached_ready_task_embs[idx] = ready_task_embs.detach().cpu().clone() if ready_task_embs is not None else None
        self.cached_dag_embeddings[idx] = dag_embeddings.detach().cpu().clone() if dag_embeddings is not None else None

        self.ptr = (self.ptr + 1) % self.max_buffer_size
        self.size = min(self.size + 1, self.max_buffer_size)

    def compute_gae(self, gamma: float, gae_lambda: float):
        """
        在缓冲区全部数据上（按时间顺序）预计算 GAE。
        调用时机：episode 中训练前 / episode 结束训练前。
        """
        if self.size == 0:
            return

        # 环形缓冲区的时间顺序：ptr 是最旧，ptr-1 是最新
        if self.size < self.max_buffer_size:
            indices = list(range(self.size))
        else:
            indices = list(range(self.ptr, self.max_buffer_size)) + list(range(self.ptr))

        n = len(indices)
        rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        masks = np.array([1.0 - self.dones[i] for i in indices], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            v_next = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + gamma * v_next * masks[t] - values[t]
            gae = delta + gamma * gae_lambda * masks[t] * gae
            advantages[t] = gae

        for idx, adv in zip(indices, advantages):
            self.advantages[idx] = adv

    def get_batch(self) -> Tuple[dict, dict, dict]:
        """
        获取一个batch的数据用于训练

        Returns:
            (batch_env_data, batch_actions, batch_extra)
        """
        if self.size < self.batch_size:
            return None

        # 随机采样
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        # 收集batch数据
        batch_static = []
        batch_adjs = []
        batch_metadata = []
        batch_ready_indices = []
        batch_ready_dyn = []
        batch_proc = []
        batch_proc_states = []
        batch_pair_features = []
        batch_eft_targets = []
        batch_sla_task_targets = []
        batch_task_dag_ids = []

        batch_actions_task = []
        batch_actions_proc = []
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_masks = []
        batch_candidate_types = []
        batch_num_ready = []
        batch_num_active_dags = []

        for idx in indices:
            data = self.env_data_list[idx]

            batch_static.append(data['dag_task_static_features'])
            batch_adjs.append(data['dag_adjs'])
            batch_metadata.append(data['dag_metadata'])
            batch_ready_indices.append(data['ready_task_indices'])
            batch_ready_dyn.append(data['ready_task_dyn_features'])
            batch_proc.append(data['proc_features'])
            batch_proc_states.append(data['proc_states'])
            batch_pair_features.append(data['task_proc_pair_features'])
            batch_eft_targets.append(data['eft_targets'])
            batch_sla_task_targets.append(data['sla_task_target'])
            batch_task_dag_ids.append(data['task_dag_ids'])
            batch_candidate_types.append(data.get('candidate_types', torch.zeros(0, dtype=torch.long)))
            batch_num_ready.append(data.get('num_ready', data['ready_task_indices'].size(0)))
            batch_num_active_dags.append(
                data.get('num_active_dags', data.get('num_dags', len(data['dag_metadata'])))
            )

            batch_actions_task.append(self.actions_task[idx])
            batch_actions_proc.append(self.actions_proc[idx])
            batch_logprobs.append(self.action_logprobs[idx])
            batch_rewards.append(self.rewards[idx])
            batch_dones.append(self.dones[idx])
            # mask: 1表示继续，0表示结束
            batch_masks.append(1.0 - self.dones[idx])

        # 转换为tensor（处理变长数据）
        # dag_metadata: [batch, num_dags, dag_meta_dim] - num_dags varies per sample
        # 记录每个样本的实际DAG数量
        batch_num_dags = [m.size(0) for m in batch_metadata]

        # 填充到max_dags
        max_dags_in_batch = max(batch_num_dags)
        dag_meta_padded = []
        for m in batch_metadata:
            if m.size(0) < max_dags_in_batch:
                padding = torch.zeros(max_dags_in_batch - m.size(0), m.size(1), dtype=m.dtype)
                m = torch.cat([m, padding], dim=0)
            dag_meta_padded.append(m)
        batch_metadata_tensor = torch.stack(dag_meta_padded)  # [batch, max_dags, dag_meta_dim]

        batch_ready_indices_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_ready_indices, batch_first=True, padding_value=0
        )  # [batch, max_ready]
        batch_ready_dyn_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_ready_dyn, batch_first=True, padding_value=0
        )  # [batch, max_ready, dyn_dim]
        batch_proc_tensor = torch.stack(batch_proc)  # [batch, num_procs, proc_dim]
        batch_proc_states_tensor = torch.stack(batch_proc_states)  # [batch, num_procs]
        batch_pair_features_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_pair_features, batch_first=True, padding_value=0
        )  # [batch, max_ready, num_procs, pair_dim]
        batch_eft_targets_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_eft_targets, batch_first=True, padding_value=0
        )  # [batch, max_ready]
        batch_sla_task_targets_tensor = torch.stack(
            batch_sla_task_targets
        ).long()  # [batch]
        batch_task_dag_ids_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_task_dag_ids, batch_first=True, padding_value=0
        )  # [batch, total_tasks]

        batch_env_data = {
            'dag_task_static_features': batch_static,
            'dag_adjs': batch_adjs,
            'dag_metadata': batch_metadata_tensor,
            'ready_task_indices': batch_ready_indices_tensor,
            'ready_task_dyn_features': batch_ready_dyn_tensor,
            'proc_features': batch_proc_tensor,
            'proc_states': batch_proc_states_tensor,
            'task_proc_pair_features': batch_pair_features_tensor,
            'eft_targets': batch_eft_targets_tensor,
            'sla_task_targets': batch_sla_task_targets_tensor,
            'task_dag_ids': batch_task_dag_ids_tensor,
            'candidate_types': batch_candidate_types,
            'num_ready': batch_num_ready,
            'num_active_dags': batch_num_active_dags,
            'num_dags': batch_num_dags,  # 每个样本提取进模型的 DAG 数（padding 用）
        }

        batch_actions = {
            'task': torch.tensor(batch_actions_task, dtype=torch.long),
            'proc': torch.tensor(batch_actions_proc, dtype=torch.long),
            'logprobs': torch.tensor(batch_logprobs, dtype=torch.float32),
        }

        batch_extra = {
            'rewards': torch.tensor(batch_rewards, dtype=torch.float32),
            'dones': torch.tensor(batch_dones, dtype=torch.float32),
            'masks': torch.tensor(batch_masks, dtype=torch.float32),
            'advantages': torch.tensor([self.advantages[idx] for idx in indices], dtype=torch.float32),
            'values': torch.tensor([self.values[idx] for idx in indices], dtype=torch.float32),
        }

        # 收集缓存的 embeddings
        cached_ready_embs = [self.cached_ready_task_embs[i] for i in indices]
        cached_dag_embs = [self.cached_dag_embeddings[i] for i in indices]

        batch_env_data['cached_ready_task_embs'] = cached_ready_embs
        batch_env_data['cached_dag_embeddings'] = cached_dag_embs

        return batch_env_data, batch_actions, batch_extra

    def reset(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
        self.env_data_list = [None] * self.max_buffer_size
        self.cached_ready_task_embs = [None] * self.max_buffer_size
        self.cached_dag_embeddings = [None] * self.max_buffer_size

    def __len__(self):
        return self.size