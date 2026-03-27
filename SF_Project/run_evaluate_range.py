import argparse
import os
import re
import logging
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import squidpy as sq
import torch
import yaml
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
)

from sf_model.model.bio_sfinet import BioSFINet
from sf_model.utils import set_seed


def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bio-SFINet on a checkpoint epoch range")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    parser.add_argument("--start", type=int, default=1500, help="Start epoch (inclusive)")
    parser.add_argument("--end", type=int, default=2000, help="End epoch (inclusive)")
    parser.add_argument("--step", type=int, default=100, help="Epoch step")
    parser.add_argument("--best-epoch", type=int, default=2000, help="Use best checkpoint when epoch equals this value")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden clustering resolution")
    return parser.parse_args()


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFEvaluateRange")
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


def calculate_spatial_morans_i(coords, features, k=6):
    """
    基于 squidpy 计算 Moran's I。
    """
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    valid_mask = np.var(features, axis=0) > 0
    if not np.any(valid_mask):
        morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
        return float(np.mean(morans_i_vals)), morans_i_vals

    adata_moran = sc.AnnData(X=features[:, valid_mask])
    adata_moran.obsm["spatial"] = coords

    sq.gr.spatial_neighbors(adata_moran, coord_type="generic", n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_moran,
        mode="moran",
        n_perms=None,
        show_progress_bar=False,
    )

    moran_df = adata_moran.uns["moranI"]
    valid_vals = moran_df["I"].to_numpy(dtype=np.float32)
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


def infer_epoch_label(ckpt_name, epoch):
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    if "best" in base.lower():
        return f"Epoch {epoch} (BEST)"

    match = re.search(r"ckpt[_-]?(\d+)", base)
    if match:
        return f"Epoch {match.group(1)}"

    return f"Epoch {epoch}"


def visualize_and_save(z_final, coords, save_dir, resolution=0.5, epoch_label=None, output_suffix=None, ground_truth=None, logger=None):
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    adata = sc.AnnData(X=z_final)
    adata.obsm["spatial"] = coords

    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)

    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="cluster",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except Exception:
        sc.tl.louvain(adata, resolution=resolution, key_added="cluster")

    moran_title_str = ""
    cluster_scores = None
    try:
        mi_latent_avg, _ = calculate_spatial_morans_i(coords, z_final, k=6)
        cluster_labels = adata.obs["cluster"].values.astype(int)
        num_clusters = np.max(cluster_labels) + 1
        one_hot_clusters = np.eye(num_clusters)[cluster_labels]
        mi_cluster_avg, _ = calculate_spatial_morans_i(coords, one_hot_clusters, k=6)
        moran_title_str = (
            f" | Latent Moran's I: {mi_latent_avg:.4f}"
            f" | Cluster Moran's I: {mi_cluster_avg:.4f}"
        )

        cluster_scores = calculate_clustering_scores(cluster_labels, ground_truth)
        if cluster_scores is not None:
            moran_title_str += (
                f" | ARI: {cluster_scores['ARI']:.4f}"
                f" | NMI: {cluster_scores['NMI']:.4f}"
                f" | AMI: {cluster_scores['AMI']:.4f}"
                f" | HOM: {cluster_scores['HOM']:.4f}"
            )
    except Exception:
        moran_title_str = " | Moran's I Error"

    n_clusters = int(adata.obs["cluster"].nunique())
    cmap = plt.get_cmap("tab20")
    vivid_palette = [cmap(i % cmap.N) for i in range(max(1, n_clusters))]
    sc.set_figure_params(dpi=180, figsize=(6, 6), frameon=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sc.pl.umap(
        adata,
        color="cluster",
        ax=axs[0],
        show=False,
        title="UMAP",
        legend_loc="on data",
        frameon=True,
        size=60,
        palette=vivid_palette,
        alpha=1.0,
        edges=False,
    )

    sc.pl.embedding(
        adata,
        basis="spatial",
        color="cluster",
        ax=axs[1],
        show=False,
        title="Spatial Map",
        size=80,
        frameon=True,
        palette=vivid_palette,
        alpha=1.0,
        edges=False,
    )

    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if epoch_label:
        fig.text(0.5, 0.99, f"Weights: {epoch_label}{moran_title_str}", ha="center", va="top", fontsize=12)

    if save_dir.rstrip("/").endswith("checkpoints"):
        base_dir = os.path.dirname(save_dir.rstrip("/"))
        fig_dir = os.path.join(base_dir, "figures")
    else:
        fig_dir = os.path.join(save_dir, "figures")

    os.makedirs(fig_dir, exist_ok=True)

    suffix = output_suffix or "default"
    plot_path = os.path.join(fig_dir, f"spatial_analysis_{suffix}.pdf")

    if epoch_label:
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    pred_dir = os.path.join(os.path.dirname(fig_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    h5ad_path = os.path.join(pred_dir, f"embedding_joint_{suffix}.h5ad")
    adata.write(h5ad_path)

    if cluster_scores is not None:
        if logger is not None:
            logger.info(
                "GT metrics (%s) | ARI=%.4f | NMI=%.4f | AMI=%.4f | HOM=%.4f | n_valid=%d",
                suffix,
                cluster_scores['ARI'],
                cluster_scores['NMI'],
                cluster_scores['AMI'],
                cluster_scores['HOM'],
                cluster_scores['n_valid'],
            )
    else:
        if logger is not None:
            logger.warning("Ground truth unavailable or invalid for %s; skip ARI/NMI/AMI/HOM.", suffix)

    return plot_path, h5ad_path


def resolve_checkpoint_name(eval_cfg, epoch, best_epoch):
    if epoch == best_epoch:
        ckpt_map = eval_cfg.get("checkpoints", {})
        return ckpt_map.get("best", "ckpt_best.pth")
    return f"ckpt_{epoch}.pth"


def build_epoch_list(start, end, step):
    if step <= 0:
        raise ValueError("--step 必须是正整数")
    if end < start:
        raise ValueError("--end 必须大于等于 --start")
    return list(range(start, end + 1, step))


def main():
    args = parse_args()

    epochs = build_epoch_list(args.start, args.end, args.step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    set_seed(config["project"].get("seed", 42))

    processed_dir = config["data"]["processed_path"]
    save_dir = config["project"]["save_dir"]
    eval_cfg = config.get("eval", {})
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)

    logger.info("[Range Evaluate] Started. epochs=%s", epochs)

    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        logger.error("Data not found at %s", data_path)
        logger.error("Please run 'python run_preprocess.py' first.")
        return

    logger.info("Loading processed data...")
    data_dict = torch_load_compat(data_path, map_location="cpu", weights_only=False)

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

    logger.info("Initializing Bio-SFINet...")
    config["model"]["rna_in_dim"] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    model.eval()

    success = 0
    failed = 0

    for epoch in epochs:
        ckpt_name = resolve_checkpoint_name(eval_cfg, epoch, args.best_epoch)
        ckpt_path = os.path.join(save_dir, ckpt_name)

        logger.info("Evaluating epoch %d with %s", epoch, ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.warning("Skip: checkpoint not found at %s", ckpt_path)
            failed += 1
            continue

        state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals)
            z_final = outputs[0]

        epoch_label = infer_epoch_label(ckpt_name, epoch)
        suffix = f"epoch_{epoch}"
        if epoch == args.best_epoch:
            suffix = f"epoch_{epoch}_best"

        plot_path, h5ad_path = visualize_and_save(
            z_final,
            coords,
            save_dir,
            resolution=args.resolution,
            epoch_label=epoch_label,
            output_suffix=suffix,
            ground_truth=ground_truth,
            logger=logger,
        )

        logger.info("Artifacts (%s): %s | %s", suffix, plot_path, h5ad_path)
        success += 1

    logger.info("Range evaluation complete. Success=%d | Failed/Skipped=%d", success, failed)


if __name__ == "__main__":
    main()
