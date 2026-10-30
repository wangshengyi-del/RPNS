# Pegasus / WfCommons OOD DAG

用 [WfCommons](https://wfcommons.org/) 生成 Montage / Epigenomics / Seismology（CyberShake 族）科学工作流拓扑，并转为与 `dataset/dag_factory/dags` 相同的 JSON，供零样本 OOD 评测。

## 1. 安装

```bash
# 建议用项目已有 venv
.venv/bin/pip install wfcommons
```

## 2. 生成数据集

```bash
# 默认：约 1000 个 test DAG（synthetic 任务属性，隔离结构 OOD）
.venv/bin/python dataset/pegasus_wfcommons/generate_pegasus_dags.py

# 小规模冒烟
.venv/bin/python dataset/pegasus_wfcommons/generate_pegasus_dags.py \
  --specs epigenomics:50:8,montage:100:8,seismology:200:4 \
  --out_dir dataset/pegasus_wfcommons/dags_smoke
```

输出目录含：

- `dag_test_*.json` — 单 DAG
- `dag_dataset.json` — 环境索引
- `generation_summary.json` — 规模统计

`--attribute_mode`：

| 模式 | 含义 |
|---|---|
| `synthetic`（默认） | 拓扑来自 WfCommons；computation/data_size 用本仓库 Pareto/Lognormal |
| `wfcommons` | 用 runtime×1500 与输出文件大小映射（并裁剪到本仓库范围） |

## 3. 规模说明

WfCommons recipe 有最小基图约束（约）：

- Epigenomics ≥ ~43
- Montage ≥ ~60
- Seismology ≥ ~103

实际节点数可能略低于 `requested_num_tasks`。相对本仓库合成 Test（均值 ~30），这是**更大结构的 OOD**，符合补充实验定位。
