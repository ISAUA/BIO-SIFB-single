# BIO-SFIB Pipeline 使用说明

本项目提供统一入口脚本用于预处理、训练与评估多组学空间数据，可在不同数据集之间灵活切换，目前内置 human、P22 mouse 与 MISAR E18。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) 全流程运行

```bash
python run_pipeline.py --dataset human
python run_pipeline.py --dataset mouse
python run_pipeline.py --dataset misar_e18
```

### 2) 仅运行部分阶段

`--steps` 逗号分隔，可选 `preprocess`, `train`, `evaluate`。

示例：仅预处理
```bash
python run_pipeline.py --dataset mouse --steps preprocess
```

示例：预处理 + 训练
```bash
python run_pipeline.py --dataset mouse --steps preprocess,train
```

### 3) 评估阶段附加参数

- `--checkpoint`: 指定评估使用的 checkpoint key 或文件名（默认读取配置文件 eval.checkpoint）。
- `--resolution`: 调整 Leiden 聚类分辨率，示例：`--resolution 0.6`。

示例：仅评估并指定 checkpoint
```bash
python run_pipeline.py --dataset mouse --steps evaluate --checkpoint ckpt_best.pth --resolution 0.6
```

### 4) 手动开启范围评估（新增）

日常训练/评估流程仍使用 `run_pipeline.py` + `run_evaluate.py`，不受影响。

当你需要一次性评估一段轮次（例如 1500-2000）时，手动运行下面脚本：

```bash
python run_evaluate_range.py --config configs/config_misar_e18.yaml --start 1500 --end 2000 --step 100 --best-epoch 2000 --resolution 0.6
```

说明：

- `--start/--end/--step` 可自定义评估区间。
- 当轮次等于 `--best-epoch`（默认 2000）时，自动使用 `ckpt_best.pth`。
- PDF 输出在 `results/<dataset>/figures/`，并在文件名末尾附加轮次后缀，例如 `spatial_analysis_epoch_1700.pdf` 与 `spatial_analysis_epoch_2000_best.pdf`。
- h5ad 输出在 `results/<dataset>/predictions/`，同样附加轮次后缀。

## 数据集配置与扩展

- 数据集配置映射在 run_pipeline.py 的 `DATASET_CONFIG` 中注册。
- 内置配置文件：
	- human: configs/config_human.yaml
	- mouse: configs/config_mouse.yaml（P22 小鼠脑数据，结果输出至 results/mouse）。
	- misar_e18: configs/config_misar_e18.yaml（MISAR-seq h5 + csv 数据，结果输出至 results/misar_e18）。
- 预处理会根据所选数据集配置自动选择读取方式：
	- human/mouse: mtx + tsv/csv 分文件读取
	- misar_e18: 10x multiome h5 + 空间 csv 读取
- 如需新增数据集，准备对应 YAML 配置并加入映射后即可使用 `--dataset <new>` 运行。

## 目录约定

- 原始数据：data/raw/<dataset>/
- 预处理结果：data/processed/<dataset>/processed_data.pt
- 训练输出：results/<dataset>/checkpoints/
- 评估输出：results/<dataset>/predictions/ 与 results/<dataset>/figures/

## 常见命令速查

- 全流程（human）：`python run_pipeline.py --dataset human`
- 全流程（mouse）：`python run_pipeline.py --dataset mouse`
- 全流程（misar_e18）：`python run_pipeline.py --dataset misar_e18`
- 仅预处理：`python run_pipeline.py --dataset human --steps preprocess`
- 仅预处理（misar_e18）：`python run_pipeline.py --dataset misar_e18 --steps preprocess`
- 仅训练：`python run_pipeline.py --dataset misar_e18 --steps train`
- 仅评估：`python run_pipeline.py --dataset misar_e18 --steps evaluate --checkpoint ckpt_best.pth`
- 评估和绘图参数：`python run_pipeline.py --dataset human --steps evaluate --checkpoint ckpt_best.pth --resolution 0.9`
- 范围评估（手动开启）：`python run_evaluate_range.py --config configs/config_misar_e18.yaml --start 1500 --end 2000 --step 100 --best-epoch 2000 --resolution 0.5`