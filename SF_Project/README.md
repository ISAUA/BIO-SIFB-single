# BIO-SFIB Pipeline 使用说明

本项目提供统一入口脚本，用于多组学空间数据的预处理、训练、评估、跨模态翻译和翻译后融合评估。

## 环境准备

```bash
source ~/.bashrc
conda activate sc_bridge
cd /root/autodl-tmp/BIO-SFIB-single/SF_Project
pip install -r requirements.txt
```

常用入口都保留在项目根目录，例如 `run_pipeline.py`、`run_train.py`、`run_evaluate.py`。根目录入口会转发到 `scripts/pipeline/` 下的实际实现，平时直接复制下面的命令即可。

## 通用规则

`run_pipeline.py` 支持三类阶段：

- `preprocess`
- `train`
- `evaluate`

```bash
# 全流程
python run_pipeline.py --dataset <dataset_name>

# 只跑部分阶段
python run_pipeline.py --dataset <dataset_name> --steps preprocess
python run_pipeline.py --dataset <dataset_name> --steps train
python run_pipeline.py --dataset <dataset_name> --steps evaluate
python run_pipeline.py --dataset <dataset_name> --steps preprocess,train

# 评估时覆盖 checkpoint / 聚类参数
python run_pipeline.py --dataset <dataset_name> --steps evaluate --checkpoint ckpt_best.pth
python run_pipeline.py --dataset <dataset_name> --steps evaluate --checkpoint ckpt_1500.pth --n-clusters 14
python run_pipeline.py --dataset <dataset_name> --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9
```

如果需要固定随机性环境，使用：

```bash
./run_deterministic.sh <config_yaml> <python_script> [extra args...]
```

也可以临时覆盖 seed：

```bash
SEED_OVERRIDE=123 ./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
```

## MISAR 数据集命令

### e11.0 s1

```bash
python run_pipeline.py --dataset misar_e11_0_s1
python run_pipeline.py --dataset misar_e11_0_s1 --steps preprocess
python run_pipeline.py --dataset misar_e11_0_s1 --steps train
python run_pipeline.py --dataset misar_e11_0_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e11_0_s1/config_misar_e11_0_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e11_0_s1/config_misar_e11_0_s1.yaml run_train.py
./run_deterministic.sh configs/e11_0_s1/config_misar_e11_0_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

PCA 30 评估配置：

```bash
python run_evaluate.py --config configs/e11_0_s1/config_eval_pca30.yaml --checkpoint ckpt_best.pth --n-clusters 14
```

### e11.0 s2

```bash
python run_pipeline.py --dataset misar_e11_0_s2
python run_pipeline.py --dataset misar_e11_0_s2 --steps preprocess
python run_pipeline.py --dataset misar_e11_0_s2 --steps train
python run_pipeline.py --dataset misar_e11_0_s2 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e11_0_s2/config_misar_e11_0_s2.yaml run_preprocess.py
./run_deterministic.sh configs/e11_0_s2/config_misar_e11_0_s2.yaml run_train.py
./run_deterministic.sh configs/e11_0_s2/config_misar_e11_0_s2.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### e13.5 s1

```bash
python run_pipeline.py --dataset misar_e13_5_s1
python run_pipeline.py --dataset misar_e13_5_s1 --steps preprocess
python run_pipeline.py --dataset misar_e13_5_s1 --steps train
python run_pipeline.py --dataset misar_e13_5_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e13_5_s1/config_misar_e13_5_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e13_5_s1/config_misar_e13_5_s1.yaml run_train.py
./run_deterministic.sh configs/e13_5_s1/config_misar_e13_5_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### e13.5 s2

```bash
python run_pipeline.py --dataset misar_e13_5_s2
python run_pipeline.py --dataset misar_e13_5_s2 --steps preprocess
python run_pipeline.py --dataset misar_e13_5_s2 --steps train
python run_pipeline.py --dataset misar_e13_5_s2 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e13_5_s2/config_misar_e13_5_s2.yaml run_preprocess.py
./run_deterministic.sh configs/e13_5_s2/config_misar_e13_5_s2.yaml run_train.py
./run_deterministic.sh configs/e13_5_s2/config_misar_e13_5_s2.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### e15.5 s1

```bash
python run_pipeline.py --dataset misar_e15_5_s1
python run_pipeline.py --dataset misar_e15_5_s1 --steps preprocess
python run_pipeline.py --dataset misar_e15_5_s1 --steps train
python run_pipeline.py --dataset misar_e15_5_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e15_5_s1/config_misar_e15_5_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e15_5_s1/config_misar_e15_5_s1.yaml run_train.py
./run_deterministic.sh configs/e15_5_s1/config_misar_e15_5_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### e15.5 s2

```bash
python run_pipeline.py --dataset misar_e15_5_s2
python run_pipeline.py --dataset misar_e15_5_s2 --steps preprocess
python run_pipeline.py --dataset misar_e15_5_s2 --steps train
python run_pipeline.py --dataset misar_e15_5_s2 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e15_5_s2/config_misar_e15_5_s2.yaml run_preprocess.py
./run_deterministic.sh configs/e15_5_s2/config_misar_e15_5_s2.yaml run_train.py
./run_deterministic.sh configs/e15_5_s2/config_misar_e15_5_s2.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

### e18.5 s1

```bash
python run_pipeline.py --dataset misar_e18_5_s1
python run_pipeline.py --dataset misar_e18_5_s1 --steps preprocess,train
python run_pipeline.py --dataset misar_e18_5_s1 --steps train
python run_pipeline.py --dataset misar_e18_5_s1 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_preprocess.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_train.py
./run_deterministic.sh configs/e18_5_s1/config_misar_e18_5_s1.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

兼容别名：

```bash
python run_pipeline.py --dataset misar_e18
```

### e18.5 s2

```bash
python run_pipeline.py --dataset misar_e18_5_s2
python run_pipeline.py --dataset misar_e18_5_s2 --steps preprocess
python run_pipeline.py --dataset misar_e18_5_s2 --steps train
python run_pipeline.py --dataset misar_e18_5_s2 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14

./run_deterministic.sh configs/e18_5_s2/config_misar_e18_5_s2.yaml run_preprocess.py
./run_deterministic.sh configs/e18_5_s2/config_misar_e18_5_s2.yaml run_train.py
./run_deterministic.sh configs/e18_5_s2/config_misar_e18_5_s2.yaml run_evaluate.py --checkpoint ckpt_best.pth --n-clusters 14
```

## Mouse P22 命令

```bash
python run_pipeline.py --dataset mouse_brain_p22
python run_pipeline.py --dataset mouse_brain_p22 --steps preprocess
python run_pipeline.py --dataset mouse_brain_p22 --steps train
python run_pipeline.py --dataset mouse_brain_p22 --steps evaluate --checkpoint ckpt_best.pth

./run_deterministic.sh configs/config_mouse_brain_p22.yaml run_preprocess.py
./run_deterministic.sh configs/config_mouse_brain_p22.yaml run_train.py
./run_deterministic.sh configs/config_mouse_brain_p22.yaml run_evaluate.py --checkpoint ckpt_best.pth
```

P22 专用范围评估会按 Cluster Moran's I 自动选择最优轮次，并只输出最优轮次图和 h5ad：

```bash
python run_evaluate_range_p22.py --start 2500 --end 2600 --step 100
python run_evaluate_range_p22.py --config configs/config_mouse_brain_p22.yaml --start 2500 --end 2600 --step 100
```

## Renal 数据集命令

### R114_T

```bash
python run_pipeline.py --dataset R114_T
python run_pipeline.py --dataset R114_T --steps preprocess,train
python run_pipeline.py --dataset R114_T --steps train
python run_pipeline.py --dataset R114_T --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9

./run_deterministic.sh configs/renal/config_renal_R114_T.yaml run_preprocess.py
./run_deterministic.sh configs/renal/config_renal_R114_T.yaml run_train.py
./run_deterministic.sh configs/renal/config_renal_R114_T.yaml run_evaluate.py --checkpoint ckpt_best.pth --resolution 0.9
```

### Y7_T

```bash
python run_pipeline.py --dataset Y7_T
python run_pipeline.py --dataset Y7_T --steps preprocess,train
python run_pipeline.py --dataset Y7_T --steps train
python run_pipeline.py --dataset Y7_T --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9

./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_preprocess.py
./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_train.py
./run_deterministic.sh configs/renal/config_renal_Y7_T.yaml run_evaluate.py --checkpoint ckpt_best.pth --resolution 0.9
```

Y7_T 范围评估：

```bash
python run_evaluate_range.py --config configs/renal/config_renal_Y7_T.yaml --start 1000 --end 3000 --step 200 --best-epoch 3000 --resolution 0.9
```

## Simulation 数据集命令

```bash
python run_pipeline.py --dataset simulation1
python run_pipeline.py --dataset simulation2
python run_pipeline.py --dataset simulation3
python run_pipeline.py --dataset simulation4
python run_pipeline.py --dataset simulation5
```

只跑预处理和训练：

```bash
python run_pipeline.py --dataset simulation1 --steps preprocess,train
python run_pipeline.py --dataset simulation2 --steps preprocess,train
python run_pipeline.py --dataset simulation3 --steps preprocess,train
python run_pipeline.py --dataset simulation4 --steps preprocess,train
python run_pipeline.py --dataset simulation5 --steps preprocess,train
```

## Human 数据集命令

```bash
python run_pipeline.py --dataset human
python run_pipeline.py --dataset human --steps preprocess
python run_pipeline.py --dataset human --steps train
python run_pipeline.py --dataset human --steps evaluate --checkpoint ckpt_best.pth

./run_deterministic.sh configs/config_human.yaml run_preprocess.py
./run_deterministic.sh configs/config_human.yaml run_train.py
./run_deterministic.sh configs/config_human.yaml run_evaluate.py --checkpoint ckpt_best.pth
```

## 范围评估命令

范围评估用于批量评估一段 checkpoint，例如 800 到 1500，每 100 轮评估一次：

```bash
python run_evaluate_range.py --config configs/e18_5_s1/config_misar_e18_5_s1.yaml --start 800 --end 1500 --step 100 --n-clusters 14
python run_evaluate_range.py --config configs/e11_0_s1/config_misar_e11_0_s1.yaml --start 800 --end 1500 --step 100 --n-clusters 14
python run_evaluate_range.py --config configs/renal/config_renal_Y7_T.yaml --start 1000 --end 3000 --step 200 --best-epoch 3000 --resolution 0.9
```

常用参数：

- `--start` / `--end` / `--step`：评估轮次范围。
- `--checkpoint`：单次评估指定权重。
- `--best-epoch`：范围评估后指定额外输出图的轮次。
- `--n-clusters`：mclust 聚类数。
- `--resolution`：Leiden 分辨率。

## 翻译模块命令

翻译模块是 Stage 2 / Stage 3.5，主要入口是：

- 训练：`run_train_translator.py`
- 评估：`evaluate_translation.py`

### 训练翻译器

默认推荐使用目标切片的 S2 配置统一驱动翻译训练。训练数据来源、主干来源、loss 权重和保存文件名由 YAML 的 `translation.stage2` 控制。

```bash
python run_train_translator.py --config configs/e18_5_s2/config_misar_e18_5_s2.yaml
```

覆盖 YAML 参数：

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

输出默认保存在：

```text
<project.save_dir>/translator_checkpoints/
```

### 评估翻译器

Stage 3.5 评估会输出 Lower / Translated / Upper 三组对照指标，包括 ARI / NMI / AMI / HOM / Cluster Moran's I。

```bash
python evaluate_translation.py --config configs/e18_5_s2/config_misar_e18_5_s2.yaml
```

覆盖 YAML 参数：

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

评估汇总默认输出到 `project.eval_dir` 下：

```text
translation_eval_stage35.csv
```

## 融合模块命令

融合模块入口是：

```text
scripts/pipeline/run_evaluate_translator_fusion.py
```

它会加载目标切片数据、冻结主干模型、加载训练好的翻译器，然后执行 RNA -> Translator -> SFIB Fusion 的推理和评估。

### 翻译后融合评估

```bash
python -m scripts.pipeline.run_evaluate_translator_fusion \
  --config configs/e18_5_s2/config_misar_e18_5_s2.yaml \
  --backbone-checkpoint results/misar/misar_e18-5-s1/checkpoints/ckpt_best.pth \
  --translator-checkpoint results/misar/misar_e18-5-s2/checkpoints/translator_checkpoints/translator_r2a_best.pth \
  --n-clusters 14
```

如需使用具体轮次权重，把 `--backbone-checkpoint` 改成对应的 `.pth` 文件路径：

```bash
python -m scripts.pipeline.run_evaluate_translator_fusion \
  --config configs/e18_5_s2/config_misar_e18_5_s2.yaml \
  --backbone-checkpoint results/misar/misar_e18-5-s1/checkpoints/ckpt_2100.pth \
  --translator-checkpoint results/misar/misar_e18-5-s2/checkpoints/translator_checkpoints/translator_r2a_best.pth \
  --n-clusters 14
```

## 配置文件索引

数据集映射位于 `scripts/pipeline/run_pipeline.py` 的 `DATASET_CONFIG`。

常用配置：

- `configs/config_human.yaml`
- `configs/config_mouse_brain_p22.yaml`
- `configs/renal/config_renal_R114_T.yaml`
- `configs/renal/config_renal_Y7_T.yaml`
- `configs/e11_0_s1/config_misar_e11_0_s1.yaml`
- `configs/e11_0_s2/config_misar_e11_0_s2.yaml`
- `configs/e13_5_s1/config_misar_e13_5_s1.yaml`
- `configs/e13_5_s2/config_misar_e13_5_s2.yaml`
- `configs/e15_5_s1/config_misar_e15_5_s1.yaml`
- `configs/e15_5_s2/config_misar_e15_5_s2.yaml`
- `configs/e18_5_s1/config_misar_e18_5_s1.yaml`
- `configs/e18_5_s2/config_misar_e18_5_s2.yaml`
- `configs/e18_5_s1/config_misar_e18.yaml`

评估专用配置：

- `configs/e11_0_s1/config_eval_pca30.yaml`

调参记录和模板：

- `TUNING_NL_PROMPT_TEMPLATE.md`
- `results/e11_0_s1/experiments_global.csv`
- `results/e13_5_s1/experiments_global.csv`
- `results/e15_5_s1/experiments_global.csv`

## 目录约定

- 原始数据：`data/raw/<dataset_group>/`
- 预处理产物：`data/processed/<dataset_tag>/`
- checkpoint：`results/<dataset_tag>/checkpoints/` 或 `results/misar/<dataset_tag>/checkpoints/`
- 评估图和 h5ad：`project.eval_dir`
- 翻译器权重：`<project.save_dir>/translator_checkpoints/`
- 训练日志：`<project.save_dir>/train.log`

## 脚本目录

- `scripts/pipeline/`：预处理、训练、评估、范围评估、翻译训练、翻译评估、融合评估。
- `scripts/diagnostics/`：诊断脚本。
- `sf_model/`：模型、训练器、预处理和工具函数。
