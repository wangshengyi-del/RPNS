"""
分层图注意力调度网络 - 全局常量
与 ARCHITECTURE_DESIGN.md 对齐

定义所有网络相关的固定维度参数，便于集中管理和修改
"""

# 环境参数
# 全局归一化参数（与研究方案 5.1.5 对齐）
CLOUD_MIPS = 5000
CLOUD_POWER = 300
EDGE_POWER = [5, 48]
EDGE_MIPS = [200, 1000]

CLOUD_ONLY_RATIO = 0.15
EDGE_ONLY_RATIO = 0.15

# 方案 4.4.1 全局带宽常量（单位：Mbps）
# [源处理器类型, 目标处理器类型]
# 0=云, 1=边
BANDWIDTH_MATRIX = {
    (0, 1): 300,         # 云↔边：300 Mbps
    (1, 0): 200,         # 边↤云：200 Mbps（不对称）
    (1, 1): 500,         # 边↔边：500 Mbps
    (0, 0): 1000,         # 云↔云：1000 Mbps
}

BANDWIDTH_NORM = [300/1000, 200/1000, 500/1000, 1000/1000]

BASELINE_DELAY = 1

# ============================================================
# 任务特征维度
# ============================================================

# 任务静态特征维度（Layer 1 Intra-DAG GAT 输入）
STATIC_DIM = 12

# 任务动态特征维度（Layer 3 两阶段决策输入）
DYN_DIM = 7

# 消融 A3：静态+动态混合输入 GAT 时的任务特征维度
MIXED_FEATURE_DIM = STATIC_DIM + DYN_DIM

# ============================================================
# DAG 元数据维度
# ============================================================

# DAG 元数据维度（Layer 2 Inter-DAG Attention 输入）
DAG_META_DIM = 5

# ============================================================
# 处理器特征维度
# ============================================================

# 处理器特征维度
PROC_DIM = 4

# Task–Processor 显式成对特征：
# [start_wait/ms, comp/ms, comm/ms, (eft-now)/ms, (lft-eft)/ms, energy/cloud_ref]
PAIR_FEATURE_DIM = 6
PAIR_FEATURE_CLIP = 3.0

# proc_features 列索引（与 env/device.py get_features 顺序一致）
# [mips_norm, available_at_norm, proc_type_norm, power_norm]
PROC_AVAILABLE_NORM_IDX = 1

# Hold 向量：归一化等待时间统计量的裁剪范围（z-score 空间）
HOLD_NORM_CLIP = 3.0

# Intra-DAG GAT 消息传递方向（见 apply_adj_direction）
ADJ_DIRECTIONS = ('down', 'up', 'both')
DEFAULT_ADJ_DIRECTION = 'down'

# 处理器数量
NUM_CLOUD = 3  # 云处理器数量
NUM_EDGE = 8   # 边缘处理器数量
NUM_PROCS = NUM_CLOUD + NUM_EDGE

# 原始任务特征维度（flat obs：12 静态 + 5 动态 + NUM_PROCS pending）
ORIGINAL_TASK_DIM = STATIC_DIM + 5 + NUM_PROCS

# ============================================================
# 网络架构参数
# ============================================================

# 隐藏层维度（默认值，可根据需要调整）
DEFAULT_HIDDEN_DIM = 256

# 注意力头数
DEFAULT_NUM_HEADS = 4

# Intra-DAG GAT 层数
DEFAULT_INTRA_LAYERS = 2

# Inter-DAG Attention 层数
DEFAULT_INTER_LAYERS = 2

# Dropout 概率
DEFAULT_DROPOUT = 0.0  # 在线 PPO：采集/更新须同一分布；非 0 会导致 old/new logprob 因 Dropout 错位


# ============================================================
# 调度器参数
# ============================================================

# 最大活跃 DAG 数量
MAX_DAGS = 10

# 最大候选任务数量
MAX_CANDIDATES = 15

# ============================================================
# GAE 参数
# ============================================================

DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_EPSILON = 0.1

# ============================================================
# 损失函数权重
# ============================================================

DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.05

# ============================================================
# 梯度裁剪
# ============================================================

DEFAULT_MAX_GRAD_NORM = 2.0

# ============================================================
# 学习率
# ============================================================

DEFAULT_LR_ACTOR = 3e-4
DEFAULT_LR_CRITIC = 3e-4
# U1: 每 episode 乘以该系数；0.995 → 201 ep 后约保留 36.6% 初始 LR
# G2 基线对照请用 --lr_decay 1.0
DEFAULT_LR_DECAY = 0.995
DEFAULT_PPO_EPOCHS = 4

# Critic 全局特征维度
GLOBAL_FEATURES_DIM = 3

# Critic 全局特征归一化上限（与 train / evaluate 统一，避免 ratio 过早饱和）
CRITIC_NORM_MAX_CANDIDATES = 1000
CRITIC_NORM_MAX_DAGS = 1000