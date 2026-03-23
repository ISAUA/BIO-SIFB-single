import os
import argparse
import re
import torch
import yaml
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sf_model.utils import set_seed

# 引入模型
from sf_model.model.bio_sfinet import BioSFINet

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bio-SFINet & Plot UMAP/Spatial")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    # 可在 CLI 覆盖 checkpoint key；若不提供则采用 config 中的 eval.checkpoint
    parser.add_argument("--checkpoint", default=None, help="Checkpoint key or filename; defaults to config eval.checkpoint")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden clustering resolution")
    return parser.parse_args()

def infer_epoch_label(ckpt_name, total_epochs=None):
    """
    从 checkpoint 文件名中推断 epoch 标签，便于图上标注。
    例如 ckpt_150.pth -> "Epoch 150"；ckpt_best.pth -> "BEST (Epoch 1600)"。
    """
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    match = re.search(r"ckpt[_-]?(\d+)", base)
    if match:
        return f"Epoch {match.group(1)}"
    if "best" in base.lower():
        if total_epochs is not None:
            return f"BEST (Epoch {int(total_epochs)})"
        return "BEST"
    return base

# ==============================================================================
# 独立计算模块：纯数学矩阵实现的 Moran's I
# ==============================================================================
def calculate_spatial_morans_i(coords, features, k=6):
    """
    纯 Numpy/Scipy 实现的 Moran's I，完全避开图状态冲突。
    支持对连续 Embedding 或 One-hot 编码后的聚类标签进行评估。
    """
    N = coords.shape[0]
    # 1. 独立构建基于物理坐标的 KNN 空间权重矩阵
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # 提取边（排除自身）
    src = np.repeat(np.arange(N), k)
    dst = indices[:, 1:].flatten()
    
    # 构建稀疏矩阵并对称化（无向图）
    W = sp.coo_matrix((np.ones_like(src), (src, dst)), shape=(N, N))
    W = W.maximum(W.T)
    
    # 行归一化 (Row-normalize)
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = W.multiply(1.0 / row_sums[:, None])
    
    # 2. 计算莫兰指数
    features = np.asarray(features, dtype=np.float32)
    # 兼容一维特征输入
    if features.ndim == 1:
        features = features[:, np.newaxis]
        
    mean_feat = np.mean(features, axis=0)
    centered_feat = features - mean_feat
    
    # 方差项 (分母)
    var = np.sum(centered_feat ** 2, axis=0)
    var[var == 0] = 1e-10  # 防止除零
    
    # 空间自协方差项 (分子)
    cov = np.sum(centered_feat * (W_norm @ centered_feat), axis=0)
    
    morans_i_vals = cov / var
    return np.mean(morans_i_vals), morans_i_vals
# ==============================================================================


def visualize_and_save(z_final, coords, save_dir, resolution=0.5, epoch_label=None):
    """
    使用 Scanpy 进行降维、聚类和绘图
    z_final: [N, C] 最终的融合特征 (Tensor or Numpy)
    coords: [N, 2] 空间坐标
    """
    print(f"\n🎨 Starting Visualization (Leiden Res={resolution})...")
    
    # 确保转为 numpy
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    # 1. 构建 AnnData
    adata = sc.AnnData(X=z_final)
    adata.obsm['spatial'] = coords
    
    # 2. 基础分析流程 (Neighbors -> UMAP -> Leiden)
    print("   -> Computing Neighbors...")
    sc.pp.neighbors(adata, use_rep='X')
    
    print("   -> Computing UMAP...")
    sc.tl.umap(adata)
    
    print(f"   -> Clustering (Leiden)...")
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added='cluster',
            flavor='igraph',
            n_iterations=2,
            directed=False,
        )
    except Exception as e:
        print("   ⚠️ Leiden clustering failed (maybe install leidenalg?), falling back to louvain.")
        sc.tl.louvain(adata, resolution=resolution, key_added='cluster')
        
    # ==============================================================================
    # 无侵入式聚合度评估 (Moran's I) - 不打印终端，仅用于顶部图标题
    # ==============================================================================
    moran_title_str = ""
    try:
        # 评估视角 1：连续隐特征 z_final 的平均空间自相关性
        mi_latent_avg, _ = calculate_spatial_morans_i(coords, z_final, k=6)
        
        # 评估视角 2：离散聚类标签的空间连贯性
        cluster_labels = adata.obs['cluster'].values.astype(int)
        num_clusters = np.max(cluster_labels) + 1
        one_hot_clusters = np.eye(num_clusters)[cluster_labels]
        mi_cluster_avg, _ = calculate_spatial_morans_i(coords, one_hot_clusters, k=6)
        
        moran_title_str = f" | Latent Moran's I: {mi_latent_avg:.4f} | Cluster Moran's I: {mi_cluster_avg:.4f}"
    except Exception as e:
        moran_title_str = " | Moran's I Error"
    # ==============================================================================

    # 3. 绘图 (UMAP + Spatial)
    # 设置绘图风格
    # 统一高对比色与画布参数
    vivid_palette = plt.get_cmap("tab10").colors  # 高对比离散色
    sc.set_figure_params(dpi=180, figsize=(6, 6), frameon=True)
    
    print("   -> Plotting...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: UMAP (已按要求修改标题)
    sc.pl.umap(
        adata,
        color='cluster',
        ax=axs[0],
        show=False,
        title='UMAP',
        legend_loc='on data',
        frameon=True,
        size=60,  # 加大点径，减少缝隙
        palette=vivid_palette,
        alpha=1.0,
        edges=False,
    )
    
    # 右图: Spatial (物理空间)
    sc.pl.embedding(
        adata,
        basis='spatial',
        color='cluster',
        ax=axs[1],
        show=False,
        title='Spatial Map',
        size=80,  # 更大点径，形成致密块
        frameon=True,
        palette=vivid_palette,
        alpha=1.0,
        edges=False,
    )

    # 极简坐标轴：保留框体，隐藏刻度线与刻度标签
    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # 翻转 Y 轴以匹配常见的显微镜视角 (可选)
    # axs[1].invert_yaxis() 
    
    # 在图外标注使用的 epoch 信息以及莫兰指数 (全英文)
    if epoch_label:
        title_text = f"Weights: {epoch_label}{moran_title_str}"
        fig.text(
            0.5,
            0.99,
            title_text,
            ha="center",
            va="top",
            fontsize=12
        )
    
    # 4. 保存图片
    # 如果 save_dir 是 checkpoints 目录，我们把图存到上级的 figures 目录
    if save_dir.rstrip('/').endswith('checkpoints'):
        base_dir = os.path.dirname(save_dir.rstrip('/'))
        fig_dir = os.path.join(base_dir, "figures")
    else:
        fig_dir = os.path.join(save_dir, "figures")
        
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, "spatial_analysis.pdf")
    
    # 为顶部文字预留空间
    if epoch_label:
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"✅ Plots saved to: {plot_path}")
    
    # 5. 保存结果 h5ad (方便后续自定义分析)
    pred_dir = os.path.join(os.path.dirname(fig_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    h5ad_path = os.path.join(pred_dir, "embedding_joint.h5ad")
    adata.write(h5ad_path)
    print(f"✅ Embedding h5ad saved to: {h5ad_path}")

def main():
    print("🚀 [Phase 3] Starting Evaluation & Plotting...")
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    processed_dir = config['data']['processed_path']
    save_dir = config['project']['save_dir']
    
    # 2. 加载数据 (直接读预处理好的 Tensor)
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}")
        print("   -> Please run 'python run_preprocess.py' first.")
        return

    print(f"\n📦 Loading data from {data_path}...")
    data_dict = torch.load(data_path, map_location='cpu')
    
    # 提取 Tensor 并转到 GPU
    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    u_basis = data_dict["u_basis"].to(device)
    atac_dim = data_dict["atac_dim"]

    # 3. 初始化模型
    print("\n🧠 Initializing Bio-SFINet...")
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    
    # 4. 加载权重
    eval_cfg = config.get('eval', {})
    ckpt_key = args.checkpoint or eval_cfg.get('checkpoint', 'best')
    ckpt_map = eval_cfg.get('checkpoints', {})
    # 如果 key 存在映射则取映射，否则允许直接把文件名作为 key 使用
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        print("   -> Please train the model first using 'run_train.py'")
        return
        
    print(f"   -> Loading weights from {ckpt_path}...")
    # 加载参数 (处理可能的 key 不匹配问题)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    total_epochs = config.get('train', {}).get('epochs')
    epoch_label = infer_epoch_label(ckpt_name, total_epochs=total_epochs)
    
    # 5. 推理 (Inference)
    print("\n🔮 Running Inference...")
    with torch.no_grad():
        # Forward pass (single fused tower)
        outputs = model(rna_feat, atac_feat, edge_index, u_basis)
        z_final = outputs[0]
        
    print(f"   -> Extracted Latent Shape: {z_final.shape}")
    
    # 6. 可视化
    visualize_and_save(
        z_final, 
        coords, 
        save_dir, 
        resolution=args.resolution,
        epoch_label=epoch_label
    )
    
    print("\n🎉 Evaluation Complete!")

if __name__ == "__main__":
    main()