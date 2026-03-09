# Mono-SFINet 模型架构详解

> 基于 `sf_model/` 目录下的代码梳理，涵盖数据预处理、图构建、模型前向传播与损失计算的完整流程。

---

## 目录

1. [整体概览](#1-整体概览)
2. [数据预处理](#2-数据预处理)
   - 2.1 [RNA 预处理](#21-rna-预处理)
   - 2.2 [ATAC 预处理](#22-atac-预处理)
   - 2.3 [空间图与 GFT 基底构建](#23-空间图与-gft-基底构建)
3. [模型输入](#3-模型输入)
4. [Phase I — 模态投影 + 自适应融合门控](#4-phase-i--模态投影--自适应融合门控)
5. [Phase II — 频域软低通（全局基底提取）](#5-phase-ii--频域软低通全局基底提取)
6. [Phase III — 空域高频细节注入（GATv2）](#6-phase-iii--空域高频细节注入gatv2)
7. [Phase IV — 残差叠加 + 通道门控](#7-phase-iv--残差叠加--通道门控)
8. [解码器（DeepDecoder）](#8-解码器deepdecoder)
9. [损失函数](#9-损失函数)
10. [训练策略](#10-训练策略)
11. [架构参数汇总](#11-架构参数汇总)

---

## 1 整体概览

**Mono-SFINet**（Mono Spatial-Frequency Integration Network）是一个用于空间多组学数据整合的单塔模型，将空间转录组（RNA）和空间染色质可及性（ATAC）两种模态融合为统一的低维潜在表示，同时支持双模态重构。

核心设计思路：
- **频域（Frequency Domain）** 提取全局低频基底信号 → 捕捉组织级别的宏观表达模式
- **空域（Spatial Domain）** 注入局部高频细节 → 捕捉细胞微环境的精细差异
- **自适应门控融合** 在两个模态间动态分配权重，避免模态主导偏差

```
RNA 输入 [N, 3000]  ──┐
                      ├──▶ Phase I: 模态投影 + 融合门控 ──▶ fused [N, C]
ATAC 输入 [N, D] ──┘                                         │
                                                              ├──▶ Phase II: 频域软低通 ──▶ base [N, C]
                                                              │
空间图 edge_index                                             ├──▶ Phase III: 空域 GATv2 ──▶ detail [N, C]
GFT 基底 u_basis [N, N]                                       │
                                                              ▼
                                                       Phase IV: base + γ·detail ──▶ z_fused [N, C]
                                                              │
                                              ┌───────────────┴──────────────┐
                                              ▼                              ▼
                                    RNA 解码器                        ATAC 解码器
                                    rec_rna [N, 3000]              rec_atac [N, D]
```

其中 `N` = spot 数（细胞数），`C` = 融合隐空间维度（`sfib_dim`，默认 128），`D` = ATAC 特征维度（预处理后动态确定）。

---

## 2 数据预处理

预处理流程由 `run_preprocess.py` 统一调度，分为 RNA 处理、ATAC 处理和空间图构建三个子模块。

### 2.1 RNA 预处理

**文件**：`sf_model/preprocess/rna_process.py` → `process_rna_pipeline()`

| 步骤 | 操作 | 输入/输出数据 |
|------|------|--------------|
| ① 读取原始数据 | `io.read_mtx_to_adata()` 读取 MTX 稀疏矩阵，转置为 [cells × genes]，强制 float32 | 原始 MTX + genes.tsv + barcodes.tsv → AnnData [N, G_raw] |
| ② 添加空间坐标 | `io.add_spatial_info()` 读取 position.tsv，将 `imagecol/imagerow` 存入 `adata.obsm['spatial']` | 对齐坐标后 AnnData [N, G_raw]，附 obsm['spatial'] [N, 2] |
| ③ 基因过滤 | `sc.pp.filter_genes(min_cells=3)`：删除在少于 3 个细胞中表达的基因 | [N, G_raw] → [N, G_filtered] |
| ④ 高变基因筛选 | `sc.pp.highly_variable_genes(flavor="seurat_v3", n_top_genes=3000, subset=True)`：在原始计数上计算均值-方差关系，保留 Top 3000 高变基因 | [N, G_filtered] → [N, 3000] |
| ⑤ 归一化 | 手工实现：每细胞总计数归一化到目标值 10000（`X = X / counts_per_cell × 1e4`） | 各 spot 表达矩阵行归一化 |
| ⑥ 对数化 | `sc.pp.log1p()`：`X = log(X + 1)` | 消除计数分布的重尾效应 |

**输出**：AnnData，`adata_rna.X` 形状 `[N, 3000]`，数值为 log1p 归一化表达量。

---

### 2.2 ATAC 预处理

**文件**：`sf_model/preprocess/atac_process.py` → `process_atac_pipeline()`

| 步骤 | 操作 | 输入/输出数据 |
|------|------|--------------|
| ① 读取原始数据 | `io.read_mtx_to_adata()` 读取 ATAC MTX 矩阵 | 原始 MTX + peaks.tsv + barcodes.tsv → AnnData [N, P_raw] |
| ② Peak 基础过滤 | `sc.pp.filter_genes(min_cells=50)`：删除在少于 50 个细胞中开放的 peak | [N, P_raw] → [N, P_filtered] |
| ③ 全局高变 peak 筛选 | `sc.pp.highly_variable_genes(flavor="seurat", n_top_genes=50000)`：按 Seurat 离散度方法保留全局 Top 50000 可变 peak | [N, P_filtered] → [N, ~50000] |
| ④ TSS 物理距离筛选 | `filter_peaks_by_tss()`：解析 GTF 文件获取基因 TSS 位置，仅保留距 RNA 高变基因 TSS ±150 kb 内的 peak，同时构建 Gene-Peak 对应掩码矩阵 | [N, ~50000] → [N, P_tss]（约数千到数万） |
| ⑤ 二次截断（可选） | 若 TSS 筛选后 peak 数超过 30000，再用 Seurat 方差排序截断至 30000 | [N, P_tss] → [N, ≤30000] |
| ⑥ TF-IDF 变换 | `custom_tf_idf()`：`TF = X / row_sum`，`IDF = N / (col_sum + ε)`，`X_tfidf = TF × IDF`，其中 ε=1e-6 防止除零 | 增强稀疏信号的区分度 |
| ⑦ 归一化 + 对数化 | 手工实现行归一化至 10000，再 `log1p`，并清除 inf/nan | 最终 ATAC 特征矩阵 [N, atac_dim]，float32 |

**输出**：AnnData，`adata_atac.X` 形状 `[N, atac_dim]`（atac_dim 由 TSS 筛选动态决定，配置上限 30000）。

---

### 2.3 空间图与 GFT 基底构建

**文件**：`sf_model/utils.py` → `build_spatial_graph()`

| 步骤 | 操作 | 输入/输出 |
|------|------|---------|
| ① KNN 图构建 | `NearestNeighbors(k=10, algorithm='ball_tree')`：基于细胞物理坐标 [N, 2] 构建 K=10 的 KNN 图，提取 `src/dst` 对 | 坐标 [N, 2] → `edge_index` [2, E]，E = N×10 |
| ② 邻接矩阵对称化 | 将有向图转为无向图：`A = A + A^T`（去重） | 稀疏邻接矩阵 A [N, N] |
| ③ 归一化拉普拉斯矩阵 | `L = I - D^{-1/2} A D^{-1/2}`，其中 D 为度矩阵 | 规范化拉普拉斯矩阵 L [N, N] |
| ④ 特征分解 | `np.linalg.eigh(L.toarray())`：对 L 进行全量特征分解（对于 N≈2500 的 spot 数，使用 dense solver）；按升序排列（低频→高频） | `u_basis` [N, N]（特征向量矩阵，即 GFT 基底 U）；`evals` [N]（特征值，即频率轴 λ） |

**输出**：`edge_index [2, E]`，`u_basis [N, N]`，`evals [N]`，全部保存为 PyTorch Tensor。

最终将所有预处理结果保存至 `processed_data.pt`：

```python
{
    "rna_feat":   FloatTensor [N, 3000],
    "atac_feat":  FloatTensor [N, atac_dim],
    "coords":     FloatTensor [N, 2],
    "edge_index": LongTensor  [2, E],
    "u_basis":    FloatTensor [N, N],
    "evals":      FloatTensor [N],
    "atac_dim":   int
}
```

---

## 3 模型输入

**文件**：`sf_model/model/bio_sfinet.py` → `BioSFINet.forward()`

| 变量 | 形状 | 来源 |
|------|------|------|
| `x_rna` | `[N, 3000]` | RNA 预处理后的表达矩阵 |
| `x_atac` | `[N, atac_dim]` | ATAC 预处理后的可及性矩阵 |
| `edge_index` | `[2, E]` | 空间 KNN 图的边索引 |
| `u_basis` | `[N, N]` | 图拉普拉斯特征向量矩阵（GFT 基底） |
| `evals` | `[N]` | 图拉普拉斯特征值（频率轴） |

---

## 4 Phase I — 模态投影 + 自适应融合门控

**目标**：将高维的 RNA 与 ATAC 特征分别映射到共享的低维隐空间，再通过学习得到的门控权重自适应地融合两种模态信号。

### 4.1 模态投影（ModalityProjector）

两路投影网络结构完全相同，均为两层带归一化的 MLP：

```
Linear(in_dim → hidden_dim=512) → LayerNorm → GELU → Dropout(0.1)
→ Linear(512 → fusion_dim=128) → LayerNorm
```

| 分支 | 输入 | 输出 |
|------|------|------|
| RNA 投影 `rna_proj` | `x_rna [N, 3000]` | `h_rna [N, 128]` |
| ATAC 投影 `atac_proj` | `x_atac [N, atac_dim]` | `h_atac [N, 128]` |

### 4.2 自适应融合门控（Fusion Gate）

将两路投影结果拼接后，通过 MLP 产生软门控权重 `gate`，实现信息自适应分配：

```python
gate_input = cat([h_rna, h_atac], dim=1)  # [N, 256]

gate = Linear(256 → 256) → GELU → Dropout(0.1)
     → Linear(256 → 128) → Sigmoid          # gate ∈ (0, 1)^{N×128}

fused = gate * h_rna + (1 - gate) * h_atac  # [N, 128]
```

`gate` 的每个元素 ∈ (0, 1)，控制该维度上 RNA 与 ATAC 的贡献比例，兼顾信息完整性（可证明为 RNA 与 ATAC 的凸组合）。

**Phase I 输出**：`fused [N, 128]`

---

## 5 Phase II — 频域软低通（全局基底提取）

**目标**：利用图傅里叶变换（GFT）将 `fused` 投影到频域，通过可学习的软低通滤波器保留低频全局信号，滤除高频噪声，提取全局基底 `base`。

**文件**：`BioSFINet._spectral_base()`

```python
# 1. 计算衰减系数（保证为正值）
β = softplus(freq_decay)   # 标量，可学习，初始值 0.8

# 2. GFT：将节点特征投影到图的频域
h_hat = u_basis.T @ fused   # [N, N] × [N, 128] → [N, 128]
                             # h_hat[k] 是第 k 个频率成分的系数向量

# 3. 软低通滤波器（频率衰减）
filt = exp(-β × evals)       # [N]，对高频分量（大特征值）指数衰减
filt = filt.unsqueeze(1)     # [N, 1]（广播到特征维度）
h_hat_low = h_hat * filt     # 元素乘，高频成分被抑制

# 4. iGFT：反变换回节点空间
h_base = u_basis @ h_hat_low  # [N, N] × [N, 128] → [N, 128]

# 5. 归一化
base = LayerNorm(h_base)     # [N, 128]
```

**关键参数**：
- `freq_decay`（可学习标量）：初始值 0.8，控制低通截止强度
- `β = softplus(freq_decay)` 确保衰减系数始终为正，避免数值不稳定

**Phase II 输出**：`base [N, 128]`（平滑的全局低频基底信号）

---

## 6 Phase III — 空域高频细节注入（GATv2）

**目标**：直接在原始图结构上，通过动态图注意力机制从空间邻域聚合细胞微环境的局部高频细节信号。

**使用的原始融合特征**：`fused [N, 128]`（非 `base`，保留全部频率信息供 GAT 聚合）

```python
# 1. GATv2 动态注意力聚合（多头均值输出）
detail = GATv2Conv(
    in_channels=128, out_channels=128,
    heads=4, concat=False,   # 4 头注意力，输出取均值
    dropout=0.1
)(fused, edge_index)         # [N, 128]

# 2. 归一化
detail = LayerNorm(detail)   # [N, 128]

# 3. 前馈网络（特征精炼）
detail = Linear(128 → 128) → GELU → Dropout(0.1) → LayerNorm  # [N, 128]
```

**GATv2 与标准 GAT 的区别**：GATv2 在计算注意力系数时将 query 与 key 拼接后线性变换，可建模更一般的注意力函数，避免标准 GAT 的秩退化问题。

**Phase III 输出**：`detail [N, 128]`（局部邻域聚合的高频细节）

---

## 7 Phase IV — 残差叠加 + 通道门控

**目标**：以低频基底信号 `base` 为主体，通过可学习的通道门控参数 `γ` 自适应地注入高频细节 `detail`，生成最终的融合潜在表示。

```python
# 1. 通道门控（每维度独立控制）
γ = sigmoid(detail_gate)       # detail_gate 为 [128] 可学习参数，初始值 0
γ = γ.unsqueeze(0)             # [1, 128] 广播

# 2. 残差叠加
z_fused = base + γ * detail    # [N, 128]
```

`detail_gate` 初始化为全零，使得训练初期 `γ ≈ 0.5`，随训练逐步学习每个特征维度上细节信号的贡献量。

**Phase IV 输出**：`z_fused [N, 128]`（最终融合潜在表示，兼含低频全局基底与高频局部细节）

---

## 8 解码器（DeepDecoder）

**目标**：以 `z_fused` 为输入，分别重构 RNA 和 ATAC 原始特征，用于自监督训练。

**文件**：`BioSFINet` 中的 `rna_dec` 和 `atac_dec`，均为 `DeepDecoder` 类。

### ResidualBlock（基础组件）

```
x ──► [LayerNorm → Linear(H→H) → GELU → Dropout]
    ──► [LayerNorm → Linear(H→H) → GELU → Dropout] → + x
```

即两个前归一化子层叠加的残差连接，H 为该块的隐藏维度。

### DeepDecoder 结构

```
输入 [N, in_dim]
  → 投影层: Linear(in_dim → hidden_dim) → LayerNorm → GELU → Dropout
  → n_blocks × ResidualBlock(hidden_dim)
  → 输出归一化: LayerNorm(hidden_dim)
  → 线性输出: Linear(hidden_dim → out_dim)
输出 [N, out_dim]
```

| 解码器 | in_dim | hidden_dim | n_blocks | out_dim |
|--------|--------|-----------|---------|--------|
| RNA 解码器 `rna_dec` | 128 | 256 | 1 | 3000 |
| ATAC 解码器 `atac_dec` | 128 | 512 | 1 | atac_dim |

**输出**：
- `rec_rna [N, 3000]`：重构的 RNA 表达
- `rec_atac [N, atac_dim]`：重构的 ATAC 可及性

---

## 9 损失函数

**文件**：`sf_model/trainer.py` → `SFTrainer.train_epoch()`

训练目标为双模态加权重构损失，对正值位置（即有真实信号的稀疏位置）给予更高的惩罚权重：

### 加权 MSE（Weighted MSE）

```python
def weighted_mse(pred, target, pos_w):
    weight = where(target > 0, pos_w, 1.0)   # 正值权重放大
    return mean(weight * (pred - target)^2)
```

| 损失项 | 正值权重 `pos_w` | 含义 |
|--------|----------------|------|
| `L_rec_rna` | 10.0 | RNA 重构，加权 MSE |
| `L_rec_atac` | 20.0 | ATAC 重构，加权 MSE（更稀疏，权重更大） |

### 总损失

```
L_total = λ_rna × L_rec_rna + λ_atac × L_rec_atac
        = 1.0 × L_rec_rna + 1.0 × L_rec_atac
```

其中 `λ_rna = λ_atac = 1.0`（均可通过配置文件调整）。

---

## 10 训练策略

**文件**：`sf_model/trainer.py` → `SFTrainer`

| 超参数 | 值 | 说明 |
|--------|----|------|
| 优化器 | AdamW | 带权重衰减的自适应梯度优化器 |
| 学习率 | 1e-3 | 初始学习率 |
| 权重衰减 | 1e-4 | L2 正则化系数 |
| 训练轮数 | 1000 | 全量批次（batch_size=1，全部 N 个 spot 一次传入） |
| 检查点保存 | 每 50 epoch 一次 + 最优 loss 实时更新 | 保存为 `ckpt_{epoch}.pth` / `ckpt_best.pth` |
| 设备 | CUDA（自动回退 CPU） | GPU 加速全矩阵运算 |

**训练流程**：
1. 所有数据（RNA、ATAC、边索引、GFT 基底）一次性移至 GPU
2. 前向传播 → 计算 `z_fused`、`rec_rna`、`rec_atac`
3. 计算 `L_total` → 反向传播 → AdamW 更新
4. 若当前 `L_total` 小于历史最优，则保存最优模型

---

## 11 架构参数汇总

以下为默认配置（`configs/config_human.yaml`）下的完整参数表：

| 参数名 | 默认值 | 含义 |
|--------|--------|------|
| `rna_in_dim` | 3000 | RNA 输入维度（高变基因数） |
| `atac_dim` | 动态 | ATAC 输入维度（预处理后确定，最大 30000） |
| `hidden_dim` | 512 | 模态投影的中间隐藏维度 |
| `sfib_dim` (fusion_dim) | 128 | 共享隐空间维度（模型宽度） |
| `gate_hidden_dim` | 256 | 融合门控网络的隐藏维度 |
| `n_heads` | 4 | GATv2 注意力头数 |
| `dropout` | 0.1 | 通用 Dropout 率（投影/解码器） |
| `gat_dropout` | 0.1 | 空域细节分支的 Dropout 率 |
| `freq_decay_init` | 0.8 | 频域软低通衰减系数初始值 |
| RNA 解码器 `hidden_dim` | 256 | RNA 解码中间层宽度 |
| RNA 解码器 `n_blocks` | 1 | RNA 解码残差块数 |
| ATAC 解码器 `hidden_dim` | 512 | ATAC 解码中间层宽度 |
| ATAC 解码器 `n_blocks` | 1 | ATAC 解码残差块数 |
| KNN 近邻数 `knn_k` | 10 | 空间图构建近邻数 |
| `pos_weight_rna` | 10.0 | RNA 加权 MSE 正类权重 |
| `pos_weight_atac` | 20.0 | ATAC 加权 MSE 正类权重 |
| `lambda_rna` | 1.0 | RNA 损失项权重 |
| `lambda_atac` | 1.0 | ATAC 损失项权重 |
| `learning_rate` | 1e-3 | AdamW 学习率 |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `epochs` | 1000 | 训练轮数 |

---

## 完整前向传播数据流

```
输入:
  x_rna    [N, 3000]
  x_atac   [N, D]       D = atac_dim（动态）
  edge_index [2, E]
  u_basis  [N, N]
  evals    [N]

─────────────────── Phase I ─────────────────────
  h_rna  = rna_proj(x_rna)              [N, 128]
           Linear(3000→512)→LN→GELU→Dropout→Linear(512→128)→LN

  h_atac = atac_proj(x_atac)            [N, 128]
           Linear(D→512)→LN→GELU→Dropout→Linear(512→128)→LN

  gate   = fusion_gate(cat[h_rna, h_atac])  [N, 128]
           Linear(256→256)→GELU→Dropout→Linear(256→128)→Sigmoid

  fused  = gate * h_rna + (1-gate) * h_atac  [N, 128]

─────────────────── Phase II ────────────────────
  β      = softplus(freq_decay)          scalar
  h_hat  = u_basis.T @ fused             [N, 128]  ← GFT
  filt   = exp(-β × evals)               [N,   1]
  h_hat_low = h_hat * filt               [N, 128]
  h_base = u_basis @ h_hat_low           [N, 128]  ← iGFT
  base   = LayerNorm(h_base)             [N, 128]

─────────────────── Phase III ───────────────────
  detail = GATv2Conv(fused, edge_index)  [N, 128]   4头均值
  detail = LayerNorm(detail)             [N, 128]
  detail = FFN(detail)                   [N, 128]   Linear→GELU→Dropout→LN

─────────────────── Phase IV ────────────────────
  γ      = sigmoid(detail_gate)          [1, 128]   可学习
  z_fused = base + γ * detail            [N, 128]

─────────────────── 解码器 ──────────────────────
  rec_rna  = rna_dec(z_fused)            [N, 3000]
             Linear(128→256)→LN→GELU→Dropout→ResBlock(256)→LN→Linear(256→3000)

  rec_atac = atac_dec(z_fused)           [N, D]
             Linear(128→512)→LN→GELU→Dropout→ResBlock(512)→LN→Linear(512→D)

─────────────────── 损失 ────────────────────────
  L_total = WeightedMSE(rec_rna, x_rna, pos_w=10)
          + WeightedMSE(rec_atac, x_atac, pos_w=20)

输出:
  z_fused  [N, 128]   → 用于下游 UMAP 可视化、Leiden 聚类
  rec_rna  [N, 3000]  → RNA 重构验证
  rec_atac [N, D]     → ATAC 重构验证
```
