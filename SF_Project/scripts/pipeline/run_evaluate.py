import os
import argparse
import re
import logging
import warnings
import sys
import torch
import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")
warnings.filterwarnings("ignore", message="In the future, the default backend for leiden will be igraph instead of leidenalg.*", category=FutureWarning)

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
)
from sklearn.decomposition import PCA
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
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters for mclust (override config)")
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


def append_log_separator(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_r_runtime_available():
    """Ensure rpy2 can locate the R runtime in conda environments."""
    py_bin = os.path.dirname(sys.executable)
    r_bin = os.path.join(py_bin, "R")
    r_home = os.path.abspath(os.path.join(py_bin, "..", "lib", "R"))

    if os.path.exists(r_home) and not os.environ.get("R_HOME"):
        os.environ["R_HOME"] = r_home

    if os.path.exists(r_bin):
        current_path = os.environ.get("PATH", "")
        if py_bin not in current_path.split(":"):
            os.environ["PATH"] = f"{py_bin}:{current_path}" if current_path else py_bin


def extract_mclust_labels(res):
    """Robustly extract mclust classification labels from rpy2 result object."""
    try:
        labels = np.array(res.rx2("classification"))
        if labels.size > 0:
            return labels
    except Exception:
        pass

    labels = np.array(res[-2])
    if labels.size == 0:
        raise ValueError("mclust classification is empty.")
    return labels


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


def visualize_and_save(
    z_final,
    coords,
    save_dir,
    n_clusters=7,
    mclust_pca_dim=20,
    epoch_label=None,
    ground_truth=None,
    logger=None,
    moran_k=6,
    plot_cfg=None,
    checkpoint_name=None,
    seed=42,
):
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
    coords_plot = coords.copy()
    # Y 轴镜像翻转，修正切片方向中的镜面对称问题
    coords_plot[:, 1] = -1.0 * coords_plot[:, 1]
    adata.obsm['spatial'] = coords_plot
    
    # 2. 基础分析流程 (Neighbors -> UMAP -> Clustering)
    sc.pp.neighbors(adata, use_rep='X', random_state=int(seed))
    sc.tl.umap(adata, random_state=int(seed))

    ensure_r_runtime_available()
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    robjects.r['set.seed'](int(seed))
    robjects.r('suppressPackageStartupMessages(library(mclust))')
    rmclust = robjects.r['Mclust']

    # 使用 PCA 将高维特征降维以加速 mclust 计算（维度由配置控制）
    pca_target_dim = max(1, int(mclust_pca_dim))
    pca_dim = min(pca_target_dim, z_final.shape[1], z_final.shape[0])
    pca_model = PCA(n_components=pca_dim, random_state=int(seed))
    z_pca = pca_model.fit_transform(z_final)

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(z_pca), int(n_clusters), 'EEE')
    mclust_res = extract_mclust_labels(res)
    adata.obs['cluster'] = mclust_res

    # 强制聚类标签为 category，确保 Scanpy 使用离散类别配色。
    adata.obs['cluster'] = adata.obs['cluster'].astype(int).astype(str).astype('category')
    # 固定类别顺序（按数字顺序）确保图例与颜色映射稳定。
    cluster_list = sorted(adata.obs['cluster'].unique(), key=lambda x: int(x))
    adata.obs['cluster'] = pd.Categorical(adata.obs['cluster'], categories=cluster_list, ordered=True)
        
    # ==============================================================================
    # 无侵入式聚合度评估 (Moran's I) - 不打印终端，仅用于顶部图标题
    # ==============================================================================
    moran_title_str = ""
    mi_latent_avg = None
    mi_cluster_avg = None
    cluster_scores = None
    try:
        # 评估视角 1：连续隐特征 z_final 的平均空间自相关性
        mi_latent_avg, _ = calculate_spatial_morans_i(coords_plot, z_final, k=moran_k)
        
        # 评估视角 2：离散聚类标签的空间连贯性
        cluster_labels = adata.obs['cluster'].values.astype(int)
        num_clusters = np.max(cluster_labels) + 1
        one_hot_clusters = np.eye(num_clusters)[cluster_labels]
        mi_cluster_avg, _ = calculate_spatial_morans_i(coords_plot, one_hot_clusters, k=moran_k)
        
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
        if logger is not None:
            logger.warning("Moran's I computation failed: %s", str(e))
    # ==============================================================================

    # 3. 绘图 (UMAP + Spatial)
    # 设置绘图风格
    plot_cfg = plot_cfg or {}

    panel_size = plot_cfg.get("panel_size", [6, 6])
    fig_size = plot_cfg.get("figure_size", [14, 6])
    dpi = int(plot_cfg.get("figure_dpi", 180))
    umap_size = float(plot_cfg.get("umap_point_size", 60))
    spatial_size = float(plot_cfg.get("spatial_point_size", 80))
    alpha = float(plot_cfg.get("alpha", 1.0))
    legend_loc = plot_cfg.get("legend_loc", "on data")

    sc.set_figure_params(dpi=dpi, figsize=tuple(panel_size), frameon=True)

    fig, axs = plt.subplots(1, 2, figsize=tuple(fig_size))
    
    # 左图: UMAP (已按要求修改标题)
    sc.pl.umap(
        adata,
        color='cluster',
        ax=axs[0],
        show=False,
        title='UMAP',
        legend_loc=legend_loc,
        frameon=True,
        size=umap_size,
        alpha=alpha,
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
        size=spatial_size,
        frameon=True,
        alpha=alpha,
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
    abs_plot_path = os.path.abspath(plot_path)
    abs_h5ad_path = os.path.abspath(h5ad_path)
    if logger is not None:
        logger.info("Artifact plot: %s", abs_plot_path)
        logger.info("Artifact h5ad: %s", abs_h5ad_path)
    print(f"Artifact plot: {abs_plot_path}")
    print(f"Artifact h5ad: {abs_h5ad_path}")

    metric_tag = epoch_label or "unknown"
    ckpt_tag = checkpoint_name or "unknown"

    def _fmt_metric(value):
        return "NA" if value is None else f"{float(value):.4f}"

    if logger is not None:
        logger.info(
            "Eval metrics | checkpoint=%s | epoch=%s | latent_moran=%s | cluster_moran=%s",
            ckpt_tag,
            metric_tag,
            _fmt_metric(mi_latent_avg),
            _fmt_metric(mi_cluster_avg),
        )

    if cluster_scores is not None:
        if logger is not None:
            logger.info(
                "GT metrics | checkpoint=%s | epoch=%s | ARI=%.4f | NMI=%.4f | AMI=%.4f | HOM=%.4f | n_valid=%d",
                ckpt_tag,
                metric_tag,
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
    seed = int(config['project'].get('seed', 42))
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
    edge_weight = data_dict.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
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
    plot_cfg = eval_cfg.get('plotting', {})
    n_clusters = int(args.n_clusters if args.n_clusters is not None else eval_cfg.get('n_clusters', 7))
    mclust_pca_dim = int(eval_cfg.get('mclust_pca_dim', 20))
    moran_k = int(eval_cfg.get('moran_k', 6))
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
        outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        z_final = outputs[0]

    logger.info("Latent shape: %s", z_final.shape)
    
    # 6. 可视化
    plot_path, h5ad_path = visualize_and_save(
        z_final, 
        coords, 
        save_dir, 
        n_clusters=n_clusters,
        mclust_pca_dim=mclust_pca_dim,
        epoch_label=epoch_label,
        ground_truth=ground_truth,
        logger=logger,
        moran_k=moran_k,
        plot_cfg=plot_cfg,
        checkpoint_name=ckpt_name,
        seed=seed,
    )

    abs_log_path = os.path.abspath(log_path)
    logger.info("Evaluation complete.")
    logger.info("Eval log: %s", abs_log_path)
    print("\n✅ Evaluation complete")
    print(f"   Figure: {os.path.abspath(plot_path)}")
    print(f"   Embedding: {os.path.abspath(h5ad_path)}")
    print(f"   Log: {abs_log_path}")
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()