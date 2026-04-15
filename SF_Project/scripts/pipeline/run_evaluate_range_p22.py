import argparse
import os


def _sanitize_thread_env_vars():
    """Prevent libgomp warnings from invalid thread env values (e.g. 0/empty)."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        try:
            valid = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[key] = "1"


_sanitize_thread_env_vars()

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from sf_model.model.bio_sfinet import BioSFINet
from sf_model.utils import set_seed

from scripts.pipeline.run_evaluate_range import (
    append_log_separator,
    build_epoch_list,
    calculate_spatial_morans_i,
    ensure_r_runtime_available,
    extract_mclust_labels,
    infer_epoch_label,
    load_config,
    resolve_train_log_path,
    setup_logger,
    torch_load_compat,
    visualize_and_save,
)


DEFAULT_P22_CONFIG = "configs/config_mouse_brain_p22.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Range evaluate mouse P22 and select best Moran's I")
    parser.add_argument("--start", type=int, required=True, help="Start epoch (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End epoch (inclusive)")
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Evaluate once every N epochs (default: 100)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_P22_CONFIG,
        help="Mouse P22 config path (default: configs/config_mouse_brain_p22.yaml)",
    )
    return parser.parse_args()


def resolve_prediction_dir(save_dir):
    if save_dir.rstrip("/").endswith("checkpoints"):
        return os.path.join(os.path.dirname(save_dir.rstrip("/")), "predictions")
    return os.path.join(save_dir, "predictions")


def calculate_cluster_moran_i(z_np, coords_np, n_clusters, mclust_pca_dim, moran_k, seed):
    ensure_r_runtime_available()
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](int(seed))
    robjects.r('suppressPackageStartupMessages(library(mclust))')
    rmclust = robjects.r["Mclust"]

    pca_target_dim = max(1, int(mclust_pca_dim))
    pca_dim = min(pca_target_dim, z_np.shape[1], z_np.shape[0])
    pca_model = PCA(n_components=pca_dim, random_state=int(seed))
    z_pca = pca_model.fit_transform(z_np)

    res = rmclust(
        rpy2.robjects.numpy2ri.numpy2rpy(z_pca),
        int(n_clusters),
        "EEE",
        verbose=False,
    )
    cluster_labels = extract_mclust_labels(res).astype(int)
    num_clusters = int(np.max(cluster_labels)) + 1
    one_hot_clusters = np.eye(num_clusters, dtype=np.float32)[cluster_labels]
    cluster_moran, _ = calculate_spatial_morans_i(coords_np, one_hot_clusters, k=moran_k)
    return float(cluster_moran)


def main():
    args = parse_args()
    epochs = build_epoch_list(args.start, args.end, args.step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    save_dir = config["project"]["save_dir"]
    processed_dir = config["data"]["processed_path"]
    eval_cfg = config.get("eval", {})
    n_clusters = int(eval_cfg.get("n_clusters", 7))
    mclust_pca_dim = int(eval_cfg.get("mclust_pca_dim", 20))
    moran_k = int(eval_cfg.get("moran_k", 6))
    plot_cfg = eval_cfg.get("plotting", {})
    compute_gt_metrics = bool(eval_cfg.get("compute_gt_metrics", False))

    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)

    logger.info("[Range Evaluate P22] Started. epochs=%s", epochs)
    logger.info("Config: %s", os.path.abspath(args.config))

    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        logger.error("Data not found at %s", data_path)
        logger.error("Please run 'python run_preprocess.py --config %s' first.", args.config)
        if os.environ.get("SF_PIPELINE_RUN") != "1":
            append_log_separator(log_path)
        return

    logger.info("Loading processed data...")
    data_dict = torch_load_compat(data_path, map_location="cpu", weights_only=False)

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
    if not compute_gt_metrics:
        ground_truth = None

    config["model"]["rna_in_dim"] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    model.eval()

    coords_np = coords.detach().cpu().numpy()
    moran_records = []

    best_epoch = None
    best_ckpt_name = None
    best_moran = -np.inf
    best_z = None

    for epoch in epochs:
        ckpt_name = f"ckpt_{epoch}.pth"
        ckpt_path = os.path.join(save_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            logger.warning("Skip epoch=%d. checkpoint not found: %s", epoch, ckpt_path)
            continue

        logger.info("Scoring epoch=%d with checkpoint=%s", epoch, ckpt_name)
        state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
            z_final = outputs[0]

        z_np = z_final.detach().cpu().numpy()
        cluster_moran = calculate_cluster_moran_i(
            z_np,
            coords_np,
            n_clusters=n_clusters,
            mclust_pca_dim=mclust_pca_dim,
            moran_k=moran_k,
            seed=seed,
        )
        moran_records.append(
            {
                "epoch": int(epoch),
                "checkpoint": ckpt_name,
                "cluster_moran": float(cluster_moran),
            }
        )

        logger.info("Epoch=%d | cluster_moran=%.6f", epoch, float(cluster_moran))

        if cluster_moran > best_moran:
            best_moran = float(cluster_moran)
            best_epoch = int(epoch)
            best_ckpt_name = ckpt_name
            best_z = z_final.detach().clone()

    if not moran_records:
        logger.error("No valid checkpoints found in requested range: %s", epochs)
        abs_log_path = os.path.abspath(log_path)
        print("\n❌ No checkpoints evaluated")
        print(f"   Log: {abs_log_path}")
        if os.environ.get("SF_PIPELINE_RUN") != "1":
            append_log_separator(log_path)
        return

    pred_dir = resolve_prediction_dir(save_dir)
    os.makedirs(pred_dir, exist_ok=True)
    score_csv = os.path.join(
        pred_dir,
        f"p22_range_moran_scores_start{args.start}_end{args.end}_step{args.step}.csv",
    )
    pd.DataFrame(moran_records).sort_values("epoch").to_csv(score_csv, index=False)
    logger.info("Moran range scores saved: %s", os.path.abspath(score_csv))

    epoch_label = infer_epoch_label(best_ckpt_name, best_epoch)
    epoch_label = f"{epoch_label} (Best Cluster Moran)"
    suffix = f"p22_best_cluster_moran_epoch_{best_epoch}"

    plot_path, h5ad_path = visualize_and_save(
        best_z,
        coords,
        save_dir,
        n_clusters=n_clusters,
        mclust_pca_dim=mclust_pca_dim,
        epoch_label=epoch_label,
        output_suffix=suffix,
        ground_truth=ground_truth,
        logger=logger,
        moran_k=moran_k,
        plot_cfg=plot_cfg,
        checkpoint_name=best_ckpt_name,
        seed=seed,
        moran_mode="cluster_only",
        precomputed_cluster_moran=best_moran,
    )

    abs_plot_path = os.path.abspath(plot_path)
    abs_h5ad_path = os.path.abspath(h5ad_path)
    abs_csv_path = os.path.abspath(score_csv)
    abs_log_path = os.path.abspath(log_path)

    logger.info(
        "Best epoch selected by Cluster Moran's I | epoch=%d | checkpoint=%s | cluster_moran=%.6f",
        best_epoch,
        best_ckpt_name,
        best_moran,
    )
    logger.info("Best artifact plot: %s", abs_plot_path)
    logger.info("Best artifact h5ad: %s", abs_h5ad_path)
    logger.info("Range eval scores csv: %s", abs_csv_path)
    logger.info("Range eval log: %s", abs_log_path)

    print("\n✅ Mouse P22 range evaluation complete")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best cluster Moran's I: {best_moran:.6f}")
    print(f"   Figure: {abs_plot_path}")
    print(f"   Embedding: {abs_h5ad_path}")
    print(f"   Moran Scores CSV: {abs_csv_path}")
    print(f"   Log: {abs_log_path}")

    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)


if __name__ == "__main__":
    main()