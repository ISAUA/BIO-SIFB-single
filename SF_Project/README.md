# BIO-SFIB Pipeline 使用说明

本项目提供统一入口脚本用于预处理、训练与评估多组学空间数据，可在不同数据集之间灵活切换。

目前内置数据集：

- human
- mouse（P22）
- MISAR：e11/e13.5/e15.5/e18.5 的 s1/s2 共 8 组

## 依赖安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 0) 进入环境和文件夹
```bash
source ~/.bashrc
conda activate sc_bridge
cd SF_Project
```

### 1) 全流程运行

```bash
python run_pipeline.py --dataset misar_e18_5_s1
```

### 2) 仅运行部分阶段

`--steps` 逗号分隔，可选 `preprocess`, `train`, `evaluate`。

```bash
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess,train
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate
```

### 3) 评估阶段附加参数

- `--checkpoint`: 指定评估使用的 checkpoint key 或文件名（默认读取配置文件 `eval.checkpoint`）。
- `--n-clusters`: 指定 mclust 聚类簇数（不传时默认使用配置文件 `eval.n_clusters`，默认 7）。

```bash
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_best.pth
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14
```

## e18.5 常用命令

### 全流程

```bash
python run_pipeline.py --dataset misar_e18_5_s1
```

### 仅预处理 / 仅训练 / 仅评估（示例）

```bash
# preprocess
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess

# train
python run_pipeline.py --dataset misar_e18_5_s1 --steps train

# evaluate
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14
```

### 单阶段确定性启动（推荐）

```bash
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

可用 `SEED_OVERRIDE` 临时覆盖配置中的 seed：

```bash
SEED_OVERRIDE=123 ./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
```

### 兼容别名

`--dataset misar_e18` 仍可使用，默认指向 `misar_e18_5_s1` 配置。

## 范围评估（可选）

当你需要一次性评估一段轮次（例如 1500-2000）时：

```bash
python run_evaluate_range.py --config configs/e18_5_s1/config_misar_e18_5_s1.yaml --start 1500 --end 3000 --step 100 --best-epoch 3000 --n-clusters 14
```

说明：

- `--start/--end/--step` 可自定义评估区间。
- 当轮次等于 `--best-epoch` 时，自动使用 `ckpt_best.pth`。
- PDF 输出在 `results/misar/<dataset>/figures/`，h5ad 输出在 `results/misar/<dataset>/predictions/`（MISAR 数据集）。

## e15.5 自动调参（数据集隔离）

说明：该流程会自动管理 `KNN` 与 `ino_pre_smooth_alpha`，并固定 `ino_use_edge_weight=true`、`ino_pre_smooth_enable=true`。

```bash
source ~/.bashrc
conda activate sc_bridge
cd SF_Project
python run_tuning_e15_5_s1.py --dataset-tag e15_5_s1 --knn-list 8,9,10 --alpha-list 0.7,0.8,0.9
```

可选参数示例：

```bash
python run_tuning_e15_5_s1.py --dataset-tag e15_5_s1 --max-runs 1
python run_tuning_e15_5_s1.py --dataset-tag e15_5_s1 --skip-existing
```

全局实验追踪 CSV：`results/experiments_global.csv`。

## 数据集配置文件

数据集映射实现位于 `scripts/pipeline/run_pipeline.py` 的 `DATASET_CONFIG`。

## 脚本目录整理

为减少根目录复杂度，入口脚本已按功能归并到 `scripts/`：

- `scripts/pipeline/`：预处理、训练、评估、范围评估、总流程
- `scripts/tuning/`：自动调参脚本
- `scripts/diagnostics/`：诊断脚本

兼容性说明：

- 根目录仍保留同名启动脚本（如 `run_pipeline.py`、`run_train.py`），它们会转发到 `scripts/` 下的实现。
- 因此你现有命令无需修改。

- `configs/config_human.yaml`
- `configs/config_mouse.yaml`
- `configs/e11_0_s1/config_misar_e11_0_s1.yaml`
- `configs/e11_0_s2/config_misar_e11_0_s2.yaml`
- `configs/e13_5_s1/config_misar_e13_5_s1.yaml`
- `configs/e13_5_s2/config_misar_e13_5_s2.yaml`
- `configs/e15_5_s1/config_misar_e15_5_s1.yaml`
- `configs/e15_5_s2/config_misar_e15_5_s2.yaml`
- `configs/e18_5_s1/config_misar_e18_5_s1.yaml`
- `configs/e18_5_s2/config_misar_e18_5_s2.yaml`
- `configs/e18_5_s1/config_misar_e18.yaml`（兼容别名配置，指向 e18_5_s1）

## 目录约定

- MISAR 原始数据：`data/raw/misar/<dataset>/`
- 数据集隔离调优配置：`configs/<dataset_tag>/config_tune_*.yaml`
- 数据集隔离预处理产物：`data/processed/<dataset_tag>/tuning/<trial_id>/`
- 数据集隔离结果输出：`results/<dataset_tag>/tuning/<trial_id>/`
- e18.5 历史调优结果：`results/e18_5_s1/tuning/`
- 全局实验追踪：`results/experiments_global.csv`