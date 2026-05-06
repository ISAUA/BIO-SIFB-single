# BIO-SFIB Pipeline 使用说明

本项目提供统一入口脚本用于预处理、训练与评估多组学空间数据，可在不同数据集之间灵活切换。

目前内置数据集：

- human
- mouse（P22）
- renal（SM 替换 ATAC）
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
python run_pipeline.py --dataset mouse_brain_p22
```

### 2) 仅运行部分阶段

`--steps` 逗号分隔，可选 `preprocess`, `train`, `evaluate`。

```bash
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess,train
python run_pipeline.py --dataset mouse_brain_p22 --steps preprocess,train
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate
```

### 3) 评估阶段附加参数

- `--checkpoint`: 指定评估使用的 checkpoint key 或文件名（默认读取配置文件 `eval.checkpoint`）。
- `--n-clusters`: 指定 mclust 聚类簇数（不传时默认使用配置文件 `eval.n_clusters`，默认 7）。

```bash
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_best.pth
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_2100.pth --n-clusters 14
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
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_2100.pth --n-clusters 14
```

### 单阶段确定性启动（推荐）

```bash
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### 跨模态翻译训练（Stage 2: RNA -> ATAC）

先确保已进入环境：`conda activate sc_bridge`

当前推荐使用 S2 配置统一驱动翻译训练（主干来源、层数、loss 权重等由 `translation.stage2` 控制；训练数据来源由 `translation.stage2.data.config` 指向 S1）：

```bash
python run_train_translator.py --config configs/e18_5_s2/config_misar_e18_5_s2.yaml
```

如需临时覆盖配置（优先级高于 YAML）：

```bash
python run_train_translator.py \
	--config configs/e18_5_s2/config_misar_e18_5_s2.yaml \
	--backbone-config configs/e18_5_s1/config_misar_e18_5_s1.yaml \
	--backbone-checkpoint 2100 \
	--epochs 400 \
	--lr 1e-4 \
	--n-blocks 3 \
	--lambda-cosine 1.0 \
	--lambda-mse 1.0 \
	--lambda-recon 1.0
```

如需切换翻译器训练数据切片，请直接修改 `translation.stage2.data.config`（例如从 S1 切到 S2）。

输出默认保存在 `save_dir/translator_checkpoints/`，包含 best 与 last 权重文件。

### 跨模态翻译评估（Translation Evaluation）

当前评估为 Stage 3.5 潜变量评估（Lower/Translated/Upper 三组对照），输出 ARI / NMI / AMI / HOM / Cluster Moran's I。先激活环境：`conda activate sc_bridge`

```bash
python evaluate_translation.py --config configs/e18_5_s2/config_misar_e18_5_s2.yaml
```

如需覆盖默认配置（`translation.stage35`）：

```bash
python evaluate_translation.py \
	--config configs/e18_5_s2/config_misar_e18_5_s2.yaml \
	--backbone-config configs/e18_5_s1/config_misar_e18_5_s1.yaml \
	--backbone-checkpoint 2100 \
	--translator-checkpoint results/misar/misar_e18-5-s2/checkpoints/translator_checkpoints/translator_r2a_best.pth \
	--n-clusters 14 \
	--pca-dim 20 \
	--moran-k 6
```

评估结果会在 `project.eval_dir` 下输出汇总文件 `translation_eval_stage35.csv`。

可用 `SEED_OVERRIDE` 临时覆盖配置中的 seed：

```bash
SEED_OVERRIDE=123 ./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
```

### 兼容别名

`--dataset misar_e18` 仍可使用，默认指向 `misar_e18_5_s1` 配置。

## Y7_T（SM 替换 ATAC）常用命令

### 全流程

```bash
python run_pipeline.py --dataset Y7_T
```

### 仅预处理 / 仅训练 / 仅评估（示例）

```bash
# preprocess
python run_pipeline.py --dataset Y7_T --steps preprocess

# train
python run_pipeline.py --dataset Y7_T --steps train

# evaluate
python run_pipeline.py --dataset Y7_T --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9
```

### 单阶段确定性启动（推荐）

```bash
./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_preprocess.py
./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_train.py
./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_evaluate.py --checkpoint ckpt_best.pth --resolution 0.9

python run_evaluate_range.py --config configs/renal/config_renal_Y7_T.yaml --start 1000 --end 3000 --step 200 --best-epoch 3000 --resolution 0.9
```

## 范围评估（可选）

当你需要一次性评估一段轮次（例如 1500-2000）时：

```bash
python run_evaluate_range.py --config configs/e18_5_s1/config_misar_e18_5_s1.yaml --start 1500 --end 3000 --step 100 --best-epoch 3000 --n-clusters 14
```

小鼠 P22 专用范围评估（自动按 cluster Moran 指数选最优轮次，并仅输出最优轮次空间图与聚类图）：

```bash
python run_evaluate_range_p22.py --start 2500 --end 2600 --step 100
```

说明：

- `--start/--end/--step` 可自定义评估区间。
- 小鼠 P22 专用脚本固定使用 `configs/config_mouse_brain_p22.yaml`，无需再传配置路径。
- 小鼠 P22 专用脚本会输出区间内每个轮次的 cluster Moran 指数 CSV，并自动选择 cluster Moran 指数最高的轮次出图。
- 若需要自定义配置文件路径，可追加 `--config <path>`。
- 输出默认位于 `results/mouse_brain_p22/` 下（含图、h5ad 与区间评分 CSV）。

## 自动调参（自然语言工作流）

说明：项目不再提供 `run_tuning_e15_5_s1.py` 脚本入口，建议使用自然语言工作流执行调参。

- 推荐模板：`TUNING_NL_PROMPT_TEMPLATE.md`
- 全局实验追踪 CSV：`results/experiments_global.csv`

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

-- `configs/config_human.yaml`
-- `configs/config_mouse.yaml`
-- `configs/renal/config_renal_R114_T.yaml`
-- `configs/renal/config_renal_Y7_T.yaml`
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