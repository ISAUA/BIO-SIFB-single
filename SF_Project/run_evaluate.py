import os
import argparse
import re
import logging
import warnings
import torch
import yaml
import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
)
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


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFEvaluate")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

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

def calculate_spatial_morans_i(coords, features, k=6):
    """
    基于 squidpy 计算 Moran's I。
    支持对连续 Embedding 或 One-hot 编码后的聚类标签进行评估。
    """
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    # 常量特征会导致 Moran's I 不可定义，先过滤掉并保留回填结构。
    valid_mask = np.var(features, axis=0) > 0
    if not np.any(valid_mask):
        morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
        return float(np.mean(morans_i_vals)), morans_i_vals

    adata_moran = sc.AnnData(X=features[:, valid_mask])
    adata_moran.obsm['spatial'] = coords

    sq.gr.spatial_neighbors(adata_moran, coord_type='generic', n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_moran,
        mode='moran',
        n_perms=None,
        show_progress_bar=False,
    )

    moran_df = adata_moran.uns['moranI']
    valid_vals = moran_df['I'].to_numpy(dtype=np.float32)
    morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
    morans_i_vals[valid_mask] = valid_vals
    return np.mean(morans_i_vals), morans_i_vals


def calculate_clustering_scores(cluster_labels, ground_truth):
    if ground_truth is None:
        return None

    gt = np.asarray(ground_truth)
    pred = np.asarray(cluster_labels)
    if gt.shape[0] != pred.shape[0]:
        return None

    adata_obs = sc.AnnData(X=np.zeros((gt.shape[0], 1), dtype=np.float32)).obs
    adata_obs["ground_truth"] = gt
    adata_obs["pred_cluster"] = pred
    adata_obs = adata_obs.dropna(subset=["ground_truth", "pred_cluster"])
    if adata_obs.shape[0] < 2:
        return None

    gt_valid = adata_obs["ground_truth"].astype(str).to_numpy()
    pred_valid = adata_obs["pred_cluster"].astype(str).to_numpy()

    non_empty = np.array([
        (g.strip() != "") and (g.lower() != "nan") and (p.strip() != "") and (p.lower() != "nan")
        for g, p in zip(gt_valid, pred_valid)
    ])
    gt_valid = gt_valid[non_empty]
    pred_valid = pred_valid[non_empty]
    if gt_valid.shape[0] < 2:
        return None

    if np.unique(gt_valid).size < 2:
        return None

    return {
        "ARI": float(adjusted_rand_score(gt_valid, pred_valid)),
        "NMI": float(normalized_mutual_info_score(gt_valid, pred_valid)),
        "AMI": float(adjusted_mutual_info_score(gt_valid, pred_valid)),
        "HOM": float(homogeneity_score(gt_valid, pred_valid)),
        "n_valid": int(gt_valid.shape[0]),
    }


def visualize_and_save(z_final, coords, save_dir, resolution=0.5, epoch_label=None, ground_truth=None, logger=None):
    """
    使用 Scanpy 进行降维、聚类和绘图
    z_final: [N, C] 最终的融合特征 (Tensor or Numpy)
    coords: [N, 2] 空间坐标
    """
    # 确保转为 numpy
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    # 1. 构建 AnnData
    adata = sc.AnnData(X=z_final)
    adata.obsm['spatial'] = coords
    
    # 2. 基础分析流程 (Neighbors -> UMAP -> Leiden)
    sc.pp.neighbors(adata, use_rep='X')

    sc.tl.umap(adata)

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
        if logger is not None:
            logger.warning("Leiden clustering failed, falling back to louvain.")
        sc.tl.louvain(adata, resolution=resolution, key_added='cluster')
        
    # ==============================================================================
    # 无侵入式聚合度评估 (Moran's I) - 不打印终端，仅用于顶部图标题
    # ==============================================================================
    moran_title_str = ""
    cluster_scores = None
    try:
        # 评估视角 1：连续隐特征 z_final 的平均空间自相关性
        mi_latent_avg, _ = calculate_spatial_morans_i(coords, z_final, k=6)
        
        # 评估视角 2：离散聚类标签的空间连贯性
        cluster_labels = adata.obs['cluster'].values.astype(int)
        num_clusters = np.max(cluster_labels) + 1
        one_hot_clusters = np.eye(num_clusters)[cluster_labels]
        mi_cluster_avg, _ = calculate_spatial_morans_i(coords, one_hot_clusters, k=6)
        
        moran_title_str = f" | Latent Moran's I: {mi_latent_avg:.4f} | Cluster Moran's I: {mi_cluster_avg:.4f}"

        cluster_scores = calculate_clustering_scores(cluster_labels, ground_truth)
        if cluster_scores is not None:
            moran_title_str += (
                f" | ARI: {cluster_scores['ARI']:.4f}"
                f" | NMI: {cluster_scores['NMI']:.4f}"
                f" | AMI: {cluster_scores['AMI']:.4f}"
                f" | HOM: {cluster_scores['HOM']:.4f}"
            )
    except Exception as e:
        moran_title_str = " | Moran's I Error"
    # ==============================================================================

    # 3. 绘图 (UMAP + Spatial)
    # 设置绘图风格
    # 统一高对比色与画布参数
    n_clusters = int(adata.obs['cluster'].nunique())
    cmap = plt.get_cmap("tab20")
    vivid_palette = [cmap(i % cmap.N) for i in range(max(1, n_clusters))]
    sc.set_figure_params(dpi=180, figsize=(6, 6), frameon=True)
    
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
    
    # 5. 保存结果 h5ad (方便后续自定义分析)
    pred_dir = os.path.join(os.path.dirname(fig_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    h5ad_path = os.path.join(pred_dir, "embedding_joint.h5ad")
    adata.write(h5ad_path)
    if logger is not None:
        logger.info("Artifacts: %s | %s", plot_path, h5ad_path)

    if cluster_scores is not None:
        if logger is not None:
            logger.info(
                "GT metrics | ARI=%.4f | NMI=%.4f | AMI=%.4f | HOM=%.4f | n_valid=%d",
                cluster_scores['ARI'],
                cluster_scores['NMI'],
                cluster_scores['AMI'],
                cluster_scores['HOM'],
                cluster_scores['n_valid'],
            )
    else:
        if logger is not None:
            logger.warning("Ground truth unavailable or invalid; skip ARI/NMI/AMI/HOM.")

    return plot_path, h5ad_path

def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    processed_dir = config['data']['processed_path']
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)

    logger.info("[Phase 3] Evaluation started.")
    
    # 2. 加载数据 (直接读预处理好的 Tensor)
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        logger.error("Data not found at %s", data_path)
        logger.error("Please run 'python run_preprocess.py' first.")
        return

    logger.info("Loading processed data...")
    data_dict = torch_load_compat(data_path, map_location='cpu', weights_only=False)
    
    # 提取 Tensor 并转到 GPU
    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    u_basis = data_dict["u_basis"].to(device)
    evals = data_dict.get("evals", None)
    if evals is not None:
        evals = evals.to(device)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = data_dict["atac_dim"]
    ground_truth = data_dict.get("ground_truth", None)

    # 3. 初始化模型
    logger.info("Initializing model...")
    config['model']['rna_in_dim'] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    
    # 4. 加载权重
    eval_cfg = config.get('eval', {})
    ckpt_key = args.checkpoint or eval_cfg.get('checkpoint', 'best')
    ckpt_map = eval_cfg.get('checkpoints', {})
    # 如果 key 存在映射则取映射，否则允许直接把文件名作为 key 使用
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        logger.error("Checkpoint not found at %s", ckpt_path)
        logger.error("Please train the model first using 'run_train.py'")
        return

    logger.info("Loading checkpoint: %s", ckpt_name)
    # 加载参数 (处理可能的 key 不匹配问题)
    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    total_epochs = config.get('train', {}).get('epochs')
    epoch_label = infer_epoch_label(ckpt_name, total_epochs=total_epochs)
    
    # 5. 推理 (Inference)
    logger.info("Inference running...")
    with torch.no_grad():
        # Forward pass (single fused tower)
        outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals)
        z_final = outputs[0]

    logger.info("Latent shape: %s", z_final.shape)
    
    # 6. 可视化
    plot_path, h5ad_path = visualize_and_save(
        z_final, 
        coords, 
        save_dir, 
        resolution=args.resolution,
        epoch_label=epoch_label,
        ground_truth=ground_truth,
        logger=logger,
    )

    logger.info("Evaluation complete.")
    print("\n✅ Evaluation complete")
    print(f"   Figure: {plot_path}")
    print(f"   Embedding: {h5ad_path}")
    print(f"   Log: {log_path}")

if __name__ == "__main__":
    main()