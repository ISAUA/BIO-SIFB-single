import argparse
import logging
import os


def _sanitize_thread_env():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = os.environ.get(key)
        if val is None:
            os.environ[key] = "1"
            continue
        val = val.strip()
        if not val.isdigit() or int(val) <= 0:
            os.environ[key] = "1"


_sanitize_thread_env()
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from .translation_runtime import (
    append_log_separator,
    load_config,
    load_processed_data,
    resolve_backbone_checkpoint,
    resolve_train_log_path,
    resolve_translator_checkpoint,
    setup_file_logger,
    torch_load_compat,
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_r_runtime_available():
    """Ensure rpy2 can locate the R runtime in conda environments."""
    import os
    import sys

    py_bin = os.path.dirname(sys.executable)
    r_bin = os.path.join(py_bin, "R")
    r_home = os.path.abspath(os.path.join(py_bin, "..", "lib", "R"))

    if os.path.exists(r_home) and not os.environ.get("R_HOME"):
        os.environ["R_HOME"] = r_home

    if os.path.exists(r_bin):
        current_path = os.environ.get("PATH", "")
        if py_bin not in current_path.split(":"):
            os.environ["PATH"] = f"{py_bin}:{current_path}" if current_path else py_bin


def _extract_mclust_labels(res):
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


def _try_load_repo_moran_impl():
    """Prefer existing repository Moran's I implementation when available."""
    try:
        from scripts.pipeline.run_evaluate import calculate_spatial_morans_i as repo_moran_impl

        return repo_moran_impl
    except Exception as e:
        print(f"[evaluate_translation] repo Moran interface unavailable: {e}")
        return None


def _fallback_moran_impl(coords, features, k=6):
    """
    Fallback Moran implementation.
    Uses squidpy if available; otherwise returns NaN placeholders.
    """
    try:
        import squidpy as sq
    except Exception as e:
        print(f"[evaluate_translation] squidpy unavailable, Moran returns NaN: {e}")
        features_np = _to_numpy(features)
        n_feat = 1 if features_np.ndim == 1 else int(features_np.shape[1])
        return float("nan"), np.full(n_feat, np.nan, dtype=np.float32)

    coords_np = _to_numpy(coords).astype(np.float32)
    feat_np = _to_numpy(features).astype(np.float32)
    if feat_np.ndim == 1:
        feat_np = feat_np[:, np.newaxis]

    valid_mask = np.var(feat_np, axis=0) > 0
    if not np.any(valid_mask):
        morans_i_vals = np.zeros(feat_np.shape[1], dtype=np.float32)
        return float(np.mean(morans_i_vals)), morans_i_vals

    adata_moran = sc.AnnData(X=feat_np[:, valid_mask])
    adata_moran.obsm["spatial"] = coords_np
    sq.gr.spatial_neighbors(adata_moran, coord_type="generic", n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_moran,
        mode="moran",
        n_perms=None,
        show_progress_bar=False,
    )
    moran_df = adata_moran.uns["moranI"]
    valid_vals = moran_df["I"].to_numpy(dtype=np.float32)
    morans_i_vals = np.zeros(feat_np.shape[1], dtype=np.float32)
    morans_i_vals[valid_mask] = valid_vals
    return float(np.mean(morans_i_vals)), morans_i_vals


def _load_processed_translation_inputs(
    config_path: str,
    backbone_checkpoint: Optional[str] = None,
    backbone_config: Optional[str] = None,
    translator_checkpoint: Optional[str] = None,
    device: str = "cpu",
):
    from sf_model.model.bio_sfinet import BioSFINet, SF_Translator_R2A

    # 1. 严格使用当前 config (如 S2) 加载测试数据
    config = load_config(config_path)
    data_dict = load_processed_data(config)

    rna_feat = data_dict["rna_feat"].to(device)   # S2 true RNA
    atac_feat = data_dict["atac_feat"].to(device) # S2 true ATAC
    coords = data_dict["coords"]                 # S2 coordinates (metric reference)
    edge_index = data_dict["edge_index"].to(device) # S2 graph topology
    edge_weight = data_dict.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    u_basis = data_dict["u_basis"].to(device)
    evals = data_dict.get("evals", None)
    if evals is not None:
        evals = evals.to(device)
    true_labels = data_dict.get("ground_truth", None) # S2 ground-truth labels

    # ==========================================
    # [核心修复]: 必须用预处理降维后的实际维度覆盖 YAML 配置的原始维度
    # ==========================================
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = int(data_dict["atac_dim"])
    config["model"]["rna_in_dim"] = rna_dim
    # ==========================================

    # 2. 初始化并加载冻结的 S1 权重
    backbone_path, backbone_name, backbone_source = resolve_backbone_checkpoint(
        target_config=config,
        backbone_checkpoint=backbone_checkpoint,
        backbone_config_path=backbone_config,
    )
    translator_path, translator_name = resolve_translator_checkpoint(config, translator_checkpoint)
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")
    if not os.path.exists(translator_path):
        raise FileNotFoundError(f"Translator checkpoint not found: {translator_path}")

    # 此时模型初始化就会正确使用 rna_in_dim = 512
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    model.load_state_dict(torch_load_compat(backbone_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    translation_cfg = config.get("translation", {})
    stage2_cfg = translation_cfg.get("stage2", {})
    stage2_model_cfg = stage2_cfg.get("model", {})
    translator_cfg = config.get("translator", {})
    translator_blocks = int(stage2_model_cfg.get("n_blocks", translator_cfg.get("n_blocks", 3)))
    translator = SF_Translator_R2A(hidden_dim=int(config["model"].get("sfib_dim", 128)), n_blocks=translator_blocks).to(device)
    translator.load_state_dict(torch_load_compat(translator_path, map_location=device, weights_only=True))
    translator.eval()

    # 3. 前向传播提取三组对照潜变量
    with torch.no_grad():
        # Lower Bound: S2 单模态基线 (f_rna)
        h_rna = model.rna_enc(rna_feat, edge_index)
        f_rna = model.rna_proj(h_rna)
        
        # Translated Result: S2 翻译补偿融合 (z_fused_hat)
        f_atac_hat = translator(f_rna)
        z_fused_hat, *_ = model.sfib(f_rna, f_atac_hat, edge_index, u_basis, evals, edge_weight=edge_weight)
        
        # Upper Bound: S2 真实双模态黄金标准 (z_fused_true)
        h_atac_true = model.atac_enc(atac_feat)
        f_atac_true = model.atac_proj(h_atac_true)
        z_fused_true, *_ = model.sfib(f_rna, f_atac_true, edge_index, u_basis, evals, edge_weight=edge_weight)

    return {
        "f_rna": f_rna,               # Lower Bound
        "f_atac_hat": f_atac_hat,     # Translated ATAC only
        "z_fused_hat": z_fused_hat,   # Translated Result
        "z_fused_true": z_fused_true, # Upper Bound
        "true_labels": true_labels,
        "coords": coords,
        "backbone_name": backbone_name,
        "backbone_source": backbone_source,
        "translator_name": translator_name,
    }


def _cluster_with_mclust_or_scanpy(
    features_pca: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    mclust_model_name: str = "EEE",
):
    """
    Default clustering pipeline:
    1) try mclust on PCA features;
    2) fallback to scanpy leiden when R/mclust is unavailable;
    3) fallback to KMeans when leiden dependencies are unavailable.
    """
    try:
        _ensure_r_runtime_available()
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        robjects.r["set.seed"](int(random_state))
        robjects.r("suppressPackageStartupMessages(library(mclust))")
        rmclust = robjects.r["Mclust"]

        res = rmclust(
            rpy2.robjects.numpy2ri.numpy2rpy(features_pca),
            int(n_clusters),
            str(mclust_model_name),
            verbose=False,
        )
        labels = _extract_mclust_labels(res)
        return np.asarray(labels).astype(int)
    except Exception as e:
        print(f"[evaluate_translation] mclust unavailable, fallback to scanpy leiden: {e}")

    try:
        adata = sc.AnnData(X=features_pca)
        sc.pp.neighbors(adata, use_rep="X", random_state=int(random_state))

        # Sweep resolution to approximate target number of clusters.
        best_labels = None
        best_gap = 10**9
        for res in np.linspace(0.2, 3.0, 15):
            sc.tl.leiden(adata, resolution=float(res), key_added="leiden_tmp", random_state=int(random_state))
            labels = adata.obs["leiden_tmp"].astype(int).to_numpy()
            gap = abs(np.unique(labels).shape[0] - int(n_clusters))
            if gap < best_gap:
                best_gap = gap
                best_labels = labels
            if gap == 0:
                break

        if best_labels is not None:
            return np.asarray(best_labels).astype(int)
    except Exception as e:
        print(f"[evaluate_translation] leiden unavailable, fallback to KMeans: {e}")

    km = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
    return km.fit_predict(features_pca).astype(int)


def evaluate_translation(
    latent_matrix,
    true_spatial_labels,
    spatial_coords,
    n_clusters: int = 7,
    pca_dim: int = 20,
    moran_k: int = 6,
    random_state: int = 42,
    cluster_fn: Optional[Callable[..., np.ndarray]] = None,
    moran_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """
    直接评估潜变量矩阵的聚类与空间特征。
    """
    latent = _to_numpy(latent_matrix).astype(np.float32)
    labels_true = _to_numpy(true_spatial_labels)
    coords = _to_numpy(spatial_coords).astype(np.float32)

    # 1) 聚类评估 (默认使用 mclust)
    # 潜变量维度通常为 128，此处 PCA 旨在统一聚类前的特征空间
    pca_target_dim = max(1, int(pca_dim))
    pca_n_components = min(pca_target_dim, latent.shape[1], latent.shape[0])
    pca_model = PCA(n_components=pca_n_components, random_state=int(random_state))
    latent_pca = pca_model.fit_transform(latent)

    if cluster_fn is None:
        pred_cluster_labels = _cluster_with_mclust_or_scanpy(
            latent_pca, n_clusters=int(n_clusters), random_state=int(random_state)
        )
    else:
        pred_cluster_labels = np.asarray(cluster_fn(latent_pca, n_clusters=int(n_clusters)))

    # 计算聚类指标
    ari = float(adjusted_rand_score(labels_true, pred_cluster_labels))
    nmi = float(normalized_mutual_info_score(labels_true, pred_cluster_labels))
    ami = float(adjusted_mutual_info_score(labels_true, pred_cluster_labels))
    hom = float(homogeneity_score(labels_true, pred_cluster_labels))

    # 2) 空间连贯性评估 (Cluster Moran's I)
    moran_impl = moran_fn or _try_load_repo_moran_impl() or _fallback_moran_impl
    
    # 将离散标签转为 one-hot 以计算 Cluster Moran's I
    num_unique = int(np.unique(pred_cluster_labels).shape[0])
    one_hot_clusters = np.eye(num_unique)[pd.Categorical(pred_cluster_labels).codes]
    mi_cluster_avg, _ = moran_impl(coords, one_hot_clusters, k=int(moran_k))

    return {
        "ari": ari, "nmi": nmi, "ami": ami, "hom": hom,
        "moran": float(mi_cluster_avg),
        "n_clusters": num_unique
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cross-modal translation metrics")
    parser.add_argument("--config", default=None, help="Optional config path for end-to-end translation evaluation")
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Backbone checkpoint key or filename when using --config; defaults to config eval.checkpoint",
    )
    parser.add_argument(
        "--backbone-config",
        default=None,
        help="Optional source config path for frozen backbone (e.g., S1 config while evaluating S2).",
    )
    parser.add_argument(
        "--translator-checkpoint",
        default=None,
        help="Optional translator checkpoint path; defaults to translator_checkpoints/translator_r2a_best.pth",
    )
    parser.add_argument("--pred", default=None, help="Path to predicted ATAC matrix .npy")
    parser.add_argument("--true", default=None, help="Path to ground-truth ATAC matrix .npy")
    parser.add_argument("--labels", default=None, help="Path to true spatial labels .npy")
    parser.add_argument("--coords", default=None, help="Path to spatial coordinates .npy")
    parser.add_argument("--n-clusters", type=int, default=None, help="Target number of clusters")
    parser.add_argument("--pca-dim", type=int, default=None, help="PCA dimension before clustering/Moran")
    parser.add_argument("--moran-k", type=int, default=None, help="KNN neighbors for Moran's I")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = None
    log_path = None

    if args.config is not None:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            cfg_for_log = yaml.safe_load(f)
        translation_cfg = cfg_for_log.get("translation", {})
        stage35_cfg = translation_cfg.get("stage35", {})
        stage35_backbone_cfg = stage35_cfg.get("backbone", {})
        stage35_metrics_cfg = stage35_cfg.get("metrics", {})
        stage35_io_cfg = stage35_cfg.get("io", {})

        effective_backbone_checkpoint = (
            args.backbone_checkpoint
            if args.backbone_checkpoint is not None
            else stage35_backbone_cfg.get("checkpoint", None)
        )
        effective_backbone_config = (
            args.backbone_config
            if args.backbone_config is not None
            else stage35_backbone_cfg.get("config", None)
        )
        effective_translator_checkpoint = (
            args.translator_checkpoint
            if args.translator_checkpoint is not None
            else stage35_io_cfg.get("translator_checkpoint", None)
        )

        save_dir = cfg_for_log["project"]["save_dir"]
        log_path = resolve_train_log_path(save_dir)
        logger = setup_file_logger("SFTranslationEvaluate", log_path, with_stream=False)
        logger.info("[Stage 3.5] Latent Translation Evaluation started.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        bundle = _load_processed_translation_inputs(
            config_path=args.config,
            backbone_checkpoint=effective_backbone_checkpoint,
            backbone_config=effective_backbone_config,
            translator_checkpoint=effective_translator_checkpoint,
            device=device,
        )
        z_fused_hat = _to_numpy(bundle["z_fused_hat"])
        z_fused_true = _to_numpy(bundle["z_fused_true"])
        f_rna = _to_numpy(bundle["f_rna"])
        f_atac_hat = _to_numpy(bundle["f_atac_hat"])
        labels = bundle["true_labels"]
        coords = _to_numpy(bundle["coords"])

        if labels is None:
            raise SystemExit("ground_truth is missing in processed_data.pt; Stage 3.5 metrics require S2 labels.")
        
        logger.info(
            "Using checkpoints | backbone=%s (source=%s) | translator=%s | device=%s",
            bundle.get("backbone_name", "unknown"),
            bundle.get("backbone_source", "unknown"),
            bundle.get("translator_name", "unknown"),
            device,
        )
    else:
        raise SystemExit("For latent evaluation, --config must be provided.")

    eval_params = {
        "true_spatial_labels": labels,
        "spatial_coords": coords,
        "n_clusters": int(args.n_clusters if args.n_clusters is not None else stage35_metrics_cfg.get("n_clusters", cfg_for_log.get("eval", {}).get("n_clusters", 7))),
        "pca_dim": int(args.pca_dim if args.pca_dim is not None else stage35_metrics_cfg.get("pca_dim", cfg_for_log.get("eval", {}).get("mclust_pca_dim", 20))),
        "moran_k": int(args.moran_k if args.moran_k is not None else stage35_metrics_cfg.get("moran_k", cfg_for_log.get("eval", {}).get("moran_k", 6))),
    }

    print("\n--- Latent Evaluation: Upper Bound (True Dual-Modal Fusion) ---")
    res_upper = evaluate_translation(latent_matrix=z_fused_true, **eval_params)
    
    print("\n--- Latent Evaluation: Translated Result (RNA + Translated ATAC) ---")
    res_translated = evaluate_translation(latent_matrix=z_fused_hat, **eval_params)

    print("\n--- Latent Evaluation: Translated ATAC Only ---")
    res_translated_atac = evaluate_translation(latent_matrix=f_atac_hat, **eval_params)
    
    print("\n--- Latent Evaluation: Lower Bound (RNA Only Base) ---")
    res_lower = evaluate_translation(latent_matrix=f_rna, **eval_params)

    summary_df = pd.DataFrame(
        [
            {"group": "lower_rna", **res_lower},
            {"group": "translated", **res_translated},
            {"group": "translated_atac", **res_translated_atac},
            {"group": "upper_true", **res_upper},
        ]
    )
    eval_out_dir = cfg_for_log["project"].get("eval_dir", os.path.dirname(save_dir.rstrip("/\\")))
    os.makedirs(eval_out_dir, exist_ok=True)
    summary_name = stage35_io_cfg.get("summary_csv_name", "translation_eval_stage35.csv")
    summary_path = os.path.join(eval_out_dir, summary_name)
    summary_df.to_csv(summary_path, index=False)

    # 日志输出对照表
    if logger is not None:
        def log_res(name, r):
            logger.info(
                "%-18s | ARI=%.4f | NMI=%.4f | AMI=%.4f | HOM=%.4f | Moran=%.4f",
                name,
                float(r["ari"]),
                float(r["nmi"]),
                float(r["ami"]),
                float(r["hom"]),
                float(r["moran"]),
            )

        logger.info("-" * 88)
        log_res("Upper Bound (True)", res_upper)
        log_res("Translated Result", res_translated)
        log_res("Translated ATAC", res_translated_atac)
        log_res("Lower Bound (RNA)", res_lower)
        logger.info("Saved summary CSV: %s", os.path.abspath(summary_path))
        logger.info("-" * 88)

        logger.info("Latent translation evaluation complete.")
        if os.environ.get("SF_PIPELINE_RUN") != "1" and log_path is not None:
            append_log_separator(log_path)
            
    # 终端对比输出
    print("\n=== Translation Performance Summary ===")
    print(
        f"{'Metric':<10} | {'Lower (RNA)':<12} | {'Translated':<12} | {'Trans ATAC':<12} | {'Upper (True)':<12}"
    )
    print("-" * 73)
    for m in ['ari', 'nmi', 'ami', 'hom', 'moran']:
        print(
            f"{m.upper():<10} | {float(res_lower[m]):<12.4f} | {float(res_translated[m]):<12.4f} | {float(res_translated_atac[m]):<12.4f} | {float(res_upper[m]):<12.4f}"
        )
    print("-" * 73)
    print(f"Saved CSV: {os.path.abspath(summary_path)}")
    print("=======================================\n")

if __name__ == "__main__":
    main()
