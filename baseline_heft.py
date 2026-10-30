"""
HEFT 在线调度算法 Baseline — 与 GAT+PPO 公平对比

多 DAG 在线调度架构:
1. 每个 DAG 到达时计算经典 HEFT Upward Rank
2. 每步调度时，对所有就绪任务计算 Global_Rank 选优先级最高的
3. 用 EFT 规则选处理器
4. 通过 env.step() 推进仿真，与 GAT+PPO 完全相同的环境和指标计算
"""

import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.multi_dag_environment import MultiDAGSchedulerEnv, estimate_schedule_timing
from env.dag_task import DAG, DAGTask, TaskStatus
from env.device import DeviceEnv
from model.constants import NUM_PROCS


class HEFTScheduler:
    """
    在线 HEFT 调度器

    与 GAT+PPO 使用相同的 env.step() 接口，每步做一次调度决策。
    处理器 EFT 与环境 estimate_schedule_timing / _execute_action 完全一致。
    """

    def __init__(self, device_env: DeviceEnv):
        self.processors = device_env.get_processor_list()
        self.num_procs = len(self.processors)

        # HEFT 平均参数（用于计算静态 rank）
        self.avg_cloud_cpu = device_env.AVG_CLOUD_CPU
        self.avg_edge_cpu = device_env.AVG_EDGE_CPU
        self.avg_cpu = device_env.AVG_CPU
        self.avg_bandwidth = device_env.AVG_BANDWIDTH

        # 已计算 rank 的 DAG 缓存: dag_id -> {task_id: rank_u}
        self._dag_ranks: Dict[str, Dict[int, float]] = {}

    def reset(self):
        self._dag_ranks.clear()

    # ================================================================
    # 静态 Upward Rank 计算
    # ================================================================

    def _avg_computation_time(self, task: DAGTask) -> float:
        """任务在所有候选处理器上的平均计算时间"""
        if task.candidate_type == 'cloud_only':
            return task.computation / self.avg_cloud_cpu
        elif task.candidate_type == 'edge_only':
            return task.computation / self.avg_edge_cpu
        else:
            return task.computation / self.avg_cpu

    def compute_upward_rank(self, dag: DAG) -> Dict[int, float]:
        """
        计算 DAG 内所有任务的经典 HEFT Upward Rank:

            Rank_u(T) = w_avg(T) + max_{child}(c_avg + Rank_u(child))

        从出口节点逆向向上计算。
        """
        if dag.dag_id in self._dag_ranks:
            return self._dag_ranks[dag.dag_id]

        sorted_ids = dag._topological_sort()[::-1]  # 逆向拓扑序
        rank = {}

        for task_id in sorted_ids:
            task = dag.tasks[task_id]
            w_i = self._avg_computation_time(task)

            if not task.children:
                rank[task_id] = w_i
            else:
                c_avg = task.data_size / self.avg_bandwidth
                max_child = max(c_avg + rank.get(c, 0) for c in task.children)
                rank[task_id] = w_i + max_child

        self._dag_ranks[dag.dag_id] = rank
        return rank

    # ================================================================
    # 全局优先级 (Global Rank)
    # ================================================================

    def compute_global_rank(self, task: DAGTask) -> float:
        """
        计算跨 DAG 的全局优先级:

            Global_Rank = A_dag + (Makespan_Est - Rank_u) * slack_ratio

        值越小优先级越高。
        slack_ratio = deadline / Makespan_Est, 用于区分不同紧迫度的 DAG。
        """
        dag = task.dag
        if dag.dag_id not in self._dag_ranks:
            self.compute_upward_rank(dag)

        rank_u = self._dag_ranks[dag.dag_id].get(task.task_id, 0)
        makespan_est = dag.makespan_est
        arrival_time = dag.arrival_time

        # 紧迫度: priority 越小（越紧）slack_ratio 越小 → Global_Rank 更小 → 更优先
        slack_ratio = dag.deadline / makespan_est if makespan_est > 0 else 1.5

        head_length = makespan_est - rank_u  # 入口到该任务的"头部"工作量
        return arrival_time + head_length * slack_ratio

    # ================================================================
    # 处理器选择 (与环境一致的 EFT)
    # ================================================================

    def select_processor_eft(
        self,
        task: DAGTask,
        current_time: float,
        processors: Optional[List] = None,
    ) -> Tuple[int, float]:
        """
        为任务选择环境真实完成时间最早的处理器。

        使用 env.estimate_schedule_timing，与 _execute_action 同一公式。
        WILL_READY：按 time_passed(arrival) 后的时刻估计（max(now, arrival)）。

        Returns:
            (proc_idx, start_time)
        """
        procs = processors if processors is not None else self.processors
        # 与 _execute_action 中 WILL_READY → time_passed(arrival) 对齐
        eff_now = current_time
        if task.status == TaskStatus.WILL_READY:
            eff_now = max(current_time, task.arrival_time)

        best_proc = -1
        best_eft = float('inf')
        best_start = 0.0

        for proc_idx, proc in enumerate(procs):
            if task.candidate_type == 'cloud_only' and proc.proc_type != 0:
                continue
            if task.candidate_type == 'edge_only' and proc.proc_type != 1:
                continue

            start_time, end_time, _, _ = estimate_schedule_timing(
                task, proc, procs, eff_now
            )
            if end_time < best_eft:
                best_eft = end_time
                best_proc = proc_idx
                best_start = start_time

        if best_proc < 0:
            # 无合法处理器时回退 0（不应发生）
            best_proc = 0

        return best_proc, best_start

    # ================================================================
    # 在线调度决策（每步调用一次）
    # ================================================================

    def select_action(
        self,
        env: MultiDAGSchedulerEnv,
    ) -> Optional[int]:
        """
        从当前就绪任务中选出 (task, processor)。

        1. 对所有就绪任务计算 Global_Rank，选最小的（最紧迫）
        2. 对选中的任务用环境一致 EFT 选处理器
        3. 返回编码后的 action
        """
        candidate_tasks = env.candidate_tasks
        if not candidate_tasks:
            return None

        current_time = env.current_time
        # 与环境共用同一批 Device，保证 available_at 同步
        self.processors = env.processors
        self.num_procs = len(self.processors)

        for dag in env.active_dags:
            if dag is not None and dag.dag_id not in self._dag_ranks:
                self.compute_upward_rank(dag)

        best_global_rank = float('inf')
        best_task_idx = -1
        best_proc = -1

        for candidate_idx, task in enumerate(candidate_tasks):
            if task.is_empty:
                continue

            global_rank = self.compute_global_rank(task)
            if global_rank < best_global_rank:
                best_global_rank = global_rank
                best_task_idx = candidate_idx
                best_proc, _ = self.select_processor_eft(
                    task, current_time, processors=env.processors
                )

        if best_task_idx < 0 or best_proc < 0:
            return None

        return best_task_idx * env.num_processors + best_proc


# ================================================================
# 在线运行（与 GAT+PPO 使用相同的 env 和指标）
# ================================================================

def run_online_heft(
    env: MultiDAGSchedulerEnv,
    num_episodes: int = 10,
    split: str = 'val',
    verbose: bool = True,
) -> Dict:
    """运行在线 HEFT 调度"""
    scheduler = HEFTScheduler(env.device_env)
    return run_baseline(env, scheduler, num_episodes, split, verbose, name='HEFT')


# ================================================================
# Earliest Deadline First 调度器（在线启发式基准）
# ================================================================

class EDFScheduler:
    """
    EDF 调度器: 每步选 LFT（Latest Finish Time）最小的就绪任务，EFT 选处理器。
    需要借助 HEFTScheduler 的 EFT 处理器选择逻辑。
    """

    def __init__(self, device_env: DeviceEnv):
        self.heft = HEFTScheduler(device_env)

    def reset(self):
        self.heft.reset()

    def select_action(self, env: MultiDAGSchedulerEnv) -> Optional[int]:
        candidate_tasks = env.candidate_tasks
        if not candidate_tasks:
            return None

        current_time = env.current_time

        # 选 LFT 最小的非空任务
        best_lft = float('inf')
        best_task_idx = -1
        for idx, task in enumerate(candidate_tasks):
            if task.is_empty:
                continue
            if task.lft < best_lft:
                best_lft = task.lft
                best_task_idx = idx

        if best_task_idx < 0:
            return None

        task = candidate_tasks[best_task_idx]
        best_proc, _ = self.heft.select_processor_eft(
            task, current_time, processors=env.processors
        )
        return best_task_idx * env.num_processors + best_proc


# ================================================================
# PCP-style：LFT 选任务，子截止内选最低功率核，否则 EFT
# ================================================================

class PCPStyleScheduler:
    """
    适配固定处理器池、Job 按时目标的 PCP 放置：
    任务仍取就绪集中 LFT 最小者；能在 LFT 前完成则选功率最低的可行核，
    否则退回与 HEFT/EDF 相同的 EFT。
    """

    def __init__(self, device_env: DeviceEnv):
        self.heft = HEFTScheduler(device_env)

    def reset(self):
        self.heft.reset()

    def select_processor_pcp(
        self,
        task: DAGTask,
        current_time: float,
        processors: List,
    ) -> int:
        eff_now = current_time
        if task.status == TaskStatus.WILL_READY:
            eff_now = max(current_time, task.arrival_time)

        feasible: List[Tuple[int, float, float]] = []
        for proc_idx, proc in enumerate(processors):
            if task.candidate_type == 'cloud_only' and proc.proc_type != 0:
                continue
            if task.candidate_type == 'edge_only' and proc.proc_type != 1:
                continue
            _, end_time, _, _ = estimate_schedule_timing(
                task, proc, processors, eff_now
            )
            feasible.append((proc_idx, end_time, float(proc.power)))

        if not feasible:
            return 0

        meet_lft = [row for row in feasible if row[1] <= task.lft]
        if meet_lft:
            meet_lft.sort(key=lambda row: (row[2], row[1], row[0]))
            return meet_lft[0][0]
        feasible.sort(key=lambda row: (row[1], row[0]))
        return feasible[0][0]

    def select_action(self, env: MultiDAGSchedulerEnv) -> Optional[int]:
        candidate_tasks = env.candidate_tasks
        if not candidate_tasks:
            return None

        best_lft = float('inf')
        best_task_idx = -1
        for idx, task in enumerate(candidate_tasks):
            if task.is_empty:
                continue
            if task.lft < best_lft:
                best_lft = task.lft
                best_task_idx = idx

        if best_task_idx < 0:
            return None

        task = candidate_tasks[best_task_idx]
        best_proc = self.select_processor_pcp(
            task, env.current_time, processors=env.processors
        )
        return best_task_idx * env.num_processors + best_proc


# ================================================================
# Random 随机调度器（性能下界）
# ================================================================

class RandomScheduler:
    """随机调度器：每步随机选任务、随机选合法处理器"""

    def __init__(self, device_env: DeviceEnv):
        self.processors = device_env.get_processor_list()
        self.num_procs = len(self.processors)

    def reset(self):
        pass

    def select_action(self, env: MultiDAGSchedulerEnv) -> Optional[int]:
        candidate_tasks = env.candidate_tasks
        if not candidate_tasks:
            return None

        procs = env.processors
        rng = env._rng if getattr(env, '_rng', None) is not None else np.random

        valid_pairs = []
        for idx, task in enumerate(candidate_tasks):
            if task.is_empty:
                continue
            for p, proc in enumerate(procs):
                if task.candidate_type == 'cloud_only' and proc.proc_type != 0:
                    continue
                if task.candidate_type == 'edge_only' and proc.proc_type != 1:
                    continue
                valid_pairs.append((idx, p))

        if not valid_pairs:
            return None

        choice = int(rng.randint(0, len(valid_pairs)))
        task_idx, proc_idx = valid_pairs[choice]
        return task_idx * env.num_processors + proc_idx


# ================================================================
# 通用 Baseline 在线运行
# ================================================================

def run_baseline(
    env: MultiDAGSchedulerEnv,
    scheduler,
    num_episodes: int = 10,
    split: str = 'val',
    verbose: bool = True,
    name: str = 'Baseline',
) -> Dict:
    """通用的在线调度评估循环，适用于任意 scheduler"""
    results = []

    for ep in range(num_episodes):
        obs = env.reset(split, episode_idx=ep)
        if hasattr(scheduler, 'reset'):
            scheduler.reset()

        ep_reward = 0
        ep_steps = 0
        ep_decision_s = 0.0

        while True:
            t0 = time.perf_counter()
            action = scheduler.select_action(env)
            ep_decision_s += time.perf_counter() - t0
            if action is None:
                event_type, next_time, min_proc_avail = env._get_next_event_time()
                if env._is_done():
                    break
                env.change_time(next_time)
                if event_type == "task_complete":
                    env._cleanup_completed_tasks()
                    env._update_candidate_tasks(min_proc_avail)
                    while event_type == "task_complete" and len(env.candidate_tasks) == 0:
                        if env._is_done():
                            break
                        event_type, next_time, min_proc_avail = env._get_next_event_time()
                        env.change_time(next_time)
                        if event_type == "task_complete":
                            env._cleanup_completed_tasks()
                        elif event_type == "dag_arrival":
                            env._add_new_dag()
                        env._update_candidate_tasks(min_proc_avail)
                elif event_type == "dag_arrival":
                    env._add_new_dag()
                    env._update_candidate_tasks(min_proc_avail)
                elif event_type == "processor_idle":
                    env._update_candidate_tasks(min_proc_avail)
                env._cleanup_completed_dags()
                if env._is_done():
                    break
                continue

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if done:
                break

        total_energy = sum(p.total_energy for p in env.processors)
        results.append({
            'steps': ep_steps,
            'total_reward': ep_reward,
            'completion_rate': env.on_time_completed_tasks / env.total_tasks if env.total_tasks > 0 else 0,
            'dag_completion_rate': env.on_time_completed_dags / env.total_dags_count if env.total_dags_count > 0 else 0,
            'avg_task_delay': env.total_task_delay / env.total_tasks if env.total_tasks > 0 else 0,
            'total_dags': env.total_dags_count,
            'total_tasks': env.total_tasks,
            'avg_energy': total_energy / env.total_tasks if env.total_tasks > 0 else 0,
            'priority_stats': {p: dict(ps) for p, ps in env._priority_stats.items()},
            'episode_idx': ep,
            'mean_decision_s': ep_decision_s / ep_steps if ep_steps > 0 else 0.0,
            **env.snapshot_revision_metrics(),
        })

        if verbose:
            r = results[-1]
            proc_parts = []
            for i, proc in enumerate(env.processors):
                busy_time = proc.total_energy / proc.power if proc.power > 0 else 0
                proc_parts.append(f"Proc{i}:{env._diag_proc_task_counts[i]}tasks,{busy_time:.0f}s busy")
            proc_info = ", ".join(proc_parts)

            print(f"\n  [{name} Ep {ep+1}] Reward={r['total_reward']:.2f} | "
                  f"TASK: {r['completion_rate']:.2%} | DAG: {r['dag_completion_rate']:.2%} | "
                  f"DELAY: {r['avg_task_delay']:.2f}s | ENERGY: {r['avg_energy']:.1f}J")
            print(f"    DAGs={r['total_dags']}, Tasks={r['total_tasks']}, Steps={r['steps']}")
            print(f"    Proc: {proc_info}")

    # 汇总 per-priority 统计
    priority_agg = {}
    for r in results:
        for p, ps in r.get('priority_stats', {}).items():
            if p not in priority_agg:
                priority_agg[p] = {k: 0 for k in ps}
            for k, v in ps.items():
                priority_agg[p][k] += v

    return {
        'avg_completion_rate': np.mean([r['completion_rate'] for r in results]),
        'avg_dag_completion_rate': np.mean([r['dag_completion_rate'] for r in results]),
        'avg_task_delay': np.mean([r['avg_task_delay'] for r in results]),
        'avg_energy': np.mean([r['avg_energy'] for r in results]),
        'avg_reward': np.mean([r['total_reward'] for r in results]),
        'per_episode': results,
        'priority_stats': priority_agg,
    }


def run_random(
    env: MultiDAGSchedulerEnv,
    num_episodes: int = 10,
    split: str = 'val',
    verbose: bool = True,
) -> Dict:
    scheduler = RandomScheduler(env.device_env)
    return run_baseline(env, scheduler, num_episodes, split, verbose, name='Random')


def run_edf(
    env: MultiDAGSchedulerEnv,
    num_episodes: int = 10,
    split: str = 'val',
    verbose: bool = True,
) -> Dict:
    scheduler = EDFScheduler(env.device_env)
    return run_baseline(env, scheduler, num_episodes, split, verbose, name='EDF')


def run_pcp_style(
    env: MultiDAGSchedulerEnv,
    num_episodes: int = 10,
    split: str = 'val',
    verbose: bool = True,
) -> Dict:
    scheduler = PCPStyleScheduler(env.device_env)
    return run_baseline(env, scheduler, num_episodes, split, verbose, name='PCP-style')


def main():
    print("=" * 70)
    print("HEFT 在线调度 Baseline")
    print("=" * 70)

    task_dim = 17 + NUM_PROCS
    proc_dim = 4
    env = MultiDAGSchedulerEnv(
        task_dim=task_dim,
        proc_dim=proc_dim,
        dag_dataset_dir='dataset/daggen_factory/dags',
        max_candidates=15,
        arrival_rate=0.60,
        dag_generator='notOur',
        device_config='dataset/device/device_heterogeneous.json',
    )
    env.set_eval_wait_time_stats(2.0, 3.5)

    print("\n" + "-" * 70)
    print("HEFT 在线调度")
    print("-" * 70)
    heft_results = run_online_heft(env, num_episodes=5, split='test')

    print("\n" + "-" * 70)
    print("EDF 在线调度")
    print("-" * 70)
    edf_results = run_edf(env, num_episodes=5, split='test')

    print("\n" + "-" * 70)
    print("Random 在线调度")
    print("-" * 70)
    random_results = run_random(env, num_episodes=5, split='test')

    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    for name, r in [('HEFT', heft_results), ('EDF', edf_results), ('Random', random_results)]:
        print(f"\n{name}:")
        print(f"  任务按时完成率:  {r['avg_completion_rate']:.2%}")
        print(f"  DAG 按时完成率:   {r['avg_dag_completion_rate']:.2%}")
        print(f"  平均延迟:         {r['avg_task_delay']:.2f}s")
        print(f"  平均能耗:         {r['avg_energy']:.2f}J/task")
        ps = r.get('priority_stats', {})
        if ps:
            print(f"  按 SLA 等级:")
            for p in sorted(ps.keys()):
                s = ps[p]; td = s['total_dags']; od = s['on_time_dags']
                tt = s['total_tasks']; ot = s['on_time_tasks']; dly = s['total_delay']
                eng = s.get('total_energy', 0.0)
                print(f"    p={p:.1f}: DAG={od}/{td}={od/td:.1%}" + (f", TASK={ot/tt:.1%}" if tt > 0 else "") + (f", delay={dly/tt:.2f}s" if tt > 0 else "") + (f", energy={eng/tt:.2f}J" if tt > 0 else ""))

    print("\n" + "=" * 70)
    print("与 GAT+PPO 对比")
    print("=" * 70)
    print(f"\n{'指标':<24} {'Random':<10} {'EDF':<10} {'HEFT':<10} {'GAT+PPO':<10}")
    print("-" * 54)
    for metric, key in [('任务按时完成率', 'avg_completion_rate'),
                         ('DAG 按时完成率', 'avg_dag_completion_rate')]:
        print(f"{metric:<24} {random_results[key]:<10.2%} {edf_results[key]:<10.2%} "
              f"{heft_results[key]:<10.2%} {'(待填入)':<10}")


if __name__ == '__main__':
    main()
