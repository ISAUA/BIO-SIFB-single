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
- `--n-clusters`: 指定 mclust 聚类簇数（不传时默认使用配置文件 `eval.n_clusters`，默认 7）。

示例：仅评估并指定 checkpoint
```bash
python run_pipeline.py --dataset mouse --steps evaluate --checkpoint ckpt_best.pth
```

示例：仅评估并指定簇数
```bash
python run_pipeline.py --dataset mouse --steps evaluate --checkpoint ckpt_best.pth --n-clusters 7
```

### 4) 手动开启范围评估（新增）

日常训练/评估流程仍使用 `run_pipeline.py` + `run_evaluate.py`，不受影响。

当你需要一次性评估一段轮次（例如 1500-2000）时，手动运行下面脚本：

```bash
python run_evaluate_range.py --config configs/config_misar_e18.yaml --start 1500 --end 2000 --step 100 --best-epoch 2000 --n-clusters 7
```

说明：

- `--start/--end/--step` 可自定义评估区间。
- 当轮次等于 `--best-epoch`（默认 2000）时，自动使用 `ckpt_best.pth`。
- PDF 输出在 `results/<dataset>/figures/`，并在文件名末尾附加轮次后缀，例如 `spatial_analysis_epoch_1700.pdf` 与 `spatial_analysis_epoch_2000_best.pdf`。
- h5ad 输出在 `results/<dataset>/predictions/`，同样附加轮次后缀。

## 配置化参数（推荐）

当前版本已将常用可调项集中到配置文件（以 `configs/config_misar_e18.yaml` 为例），建议优先改配置再运行。

### eval 区域

- `eval.checkpoint`: 默认评估 checkpoint key。
- `eval.checkpoints`: key 到 checkpoint 文件名映射。
- `eval.n_clusters`: mclust 聚类簇数。
- `eval.mclust_pca_dim`: mclust 前的 PCA 维度（默认 20，用于显著加速 mclust）。
- `eval.resolution`: 仅用于 `run_evaluate_frespa.py`（Leiden）或 mclust 失败回退时的分辨率参数。
- `eval.moran_k`: Moran's I 计算邻居数。
- `eval.plotting.*`: 绘图参数。

### eval.plotting 子参数

- `figure_dpi`: 绘图 DPI。
- `figure_size`: 画布大小。
- `panel_size`: Scanpy 面板默认大小。
- `umap_point_size`: UMAP 点大小。
- `spatial_point_size`: Spatial 点大小。
- `alpha`: 点透明度。
- `legend_loc`: 图例位置。
- `palette_mode`: 配色模式，推荐 `high_contrast`（相邻色块对比更强）。

### train 区域

- `train.log_interval`: 训练日志记录间隔。
- `train.log_tail_blank_lines`: 每次训练结束后在 `train.log` 末尾追加空行数量（用于区分多次运行，默认 5）。

说明：CLI 参数会覆盖配置值，例如传入 `--n-clusters` 时会覆盖 `eval.n_clusters`。

## 日志说明

- 统一日志文件：`results/<dataset>/train.log`
- 预处理/训练/评估阶段都会写入同一个日志文件。
- 每次阶段启动会在日志开头记录完整配置参数。
- 训练阶段终端显示 epoch 进度条；日志保留完整损失项（`total/rec_rna/rec_atac/clip/clip_weight/lr/best`）。
- 评估阶段在终端结尾输出产物路径（figure、embedding、log），便于直接点击查看。

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
- 评估和聚类参数：`python run_pipeline.py --dataset misar_e18 --steps evaluate --checkpoint ckpt_best.pth --n-clusters 14`
- 范围评估（手动开启）：`python run_evaluate_range.py --config configs/config_misar_e18.yaml --start 1500 --end 2000 --step 100 --best-epoch 3000 --n-clusters 14`