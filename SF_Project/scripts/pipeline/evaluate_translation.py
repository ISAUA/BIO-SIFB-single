import argparse
import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    mean_squared_error,
    normalized_mutual_info_score,
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


def _safe_spotwise_pcc(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute row-wise Pearson correlation and return mean over valid rows."""
    pcc_vals = []
    for i in range(pred.shape[0]):
        row_pred = pred[i]
        row_true = true[i]
        # pearsonr returns nan when one row is constant; skip invalid rows.
        corr, _ = pearsonr(row_pred, row_true)
        if np.isfinite(corr):
            pcc_vals.append(float(corr))

    if len(pcc_vals) == 0:
        return float("nan")
    return float(np.mean(pcc_vals))


def _cluster_with_mclust_or_scanpy(
    features_pca: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    mclust_model_name: str = "EEE",
):
    """
    Default clustering pipeline:
    1) try mclust on PCA features;
    2) fallback to scanpy leiden when R/mclust is unavailable.
    """
    try:
        ensure_r_runtime_available()
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

    return np.asarray(best_labels).astype(int)


def evaluate_translation(
    pred_atac_matrix,
    true_atac_matrix,
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
    Evaluate RNA->ATAC translation with reconstruction, clustering and spatial autocorrelation metrics.

    Required inputs:
      pred_atac_matrix: [N_spots, N_peaks]
      true_atac_matrix: [N_spots, N_peaks]
      true_spatial_labels: [N_spots]
      spatial_coords: [N_spots, 2]

    Optional interfaces:
      cluster_fn(features_pca, n_clusters, random_state) -> cluster labels
      moran_fn(coords, features, k) -> (avg_moran, per_feature_moran)
    """
    pred = _to_numpy(pred_atac_matrix).astype(np.float32)
    true = _to_numpy(true_atac_matrix).astype(np.float32)
    labels_true = _to_numpy(true_spatial_labels)
    coords = _to_numpy(spatial_coords).astype(np.float32)

    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")
    if pred.shape[0] != labels_true.shape[0]:
        raise ValueError(
            f"Spot mismatch: pred rows={pred.shape[0]}, true_spatial_labels={labels_true.shape[0]}"
        )
    if coords.shape[0] != pred.shape[0] or coords.shape[1] != 2:
        raise ValueError(f"spatial_coords must be [N_spots, 2], got {coords.shape}")

    # 1) Reconstruction metrics
    mse = mean_squared_error(true.reshape(-1), pred.reshape(-1))
    rmse = float(math.sqrt(float(mse)))
    spotwise_pcc = _safe_spotwise_pcc(pred, true)

    print(f"Translation Evaluation - RMSE: {rmse:.4f}, Spot-wise PCC: {spotwise_pcc:.4f}")

    # 2) Clustering metrics (PCA + clustering interface)
    pca_target_dim = max(1, int(pca_dim))
    pca_n_components = min(pca_target_dim, pred.shape[1], pred.shape[0])
    pca_model = PCA(n_components=pca_n_components, random_state=int(random_state))
    pred_pca = pca_model.fit_transform(pred)

    if cluster_fn is None:
        pred_cluster_labels = _cluster_with_mclust_or_scanpy(
            pred_pca,
            n_clusters=int(n_clusters),
            random_state=int(random_state),
            mclust_model_name="EEE",
        )
    else:
        # Interface placeholder: use caller-provided mclust/scanpy pipeline.
        pred_cluster_labels = np.asarray(
            cluster_fn(pred_pca, n_clusters=int(n_clusters), random_state=int(random_state))
        )

    labels_true_series = pd.Series(labels_true).astype(str)
    labels_pred_series = pd.Series(pred_cluster_labels).astype(str)

    ari = float(adjusted_rand_score(labels_true_series, labels_pred_series))
    nmi = float(normalized_mutual_info_score(labels_true_series, labels_pred_series))
    ami = float(adjusted_mutual_info_score(labels_true_series, labels_pred_series))
    hom = float(homogeneity_score(labels_true_series, labels_pred_series))

    print(
        "Translation Evaluation - Clustering: "
        f"ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, HOM={hom:.4f}"
    )

    # 3) Spatial autocorrelation (Moran's I interface)
    moran_impl = moran_fn
    if moran_impl is None:
        moran_impl = _try_load_repo_moran_impl() or _fallback_moran_impl
    mi_pred_avg, mi_pred_per_feature = moran_impl(coords, pred_pca, k=int(moran_k))

    print(f"Translation Evaluation - Moran's I: {float(mi_pred_avg):.4f}")

    result = {
        "reconstruction": {
            "rmse": rmse,
            "spotwise_pcc": float(spotwise_pcc),
        },
        "clustering": {
            "ari": ari,
            "nmi": nmi,
            "ami": ami,
            "hom": hom,
            "n_clusters": int(np.unique(pred_cluster_labels).shape[0]),
        },
        "spatial_autocorrelation": {
            "moran_i_avg": float(mi_pred_avg),
            "moran_i_per_feature": _to_numpy(mi_pred_per_feature).tolist(),
            "moran_features": "PCA features from predicted ATAC",
        },
    }

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cross-modal translation metrics")
    parser.add_argument("--pred", required=True, help="Path to predicted ATAC matrix .npy")
    parser.add_argument("--true", required=True, help="Path to ground-truth ATAC matrix .npy")
    parser.add_argument("--labels", required=True, help="Path to true spatial labels .npy")
    parser.add_argument("--coords", required=True, help="Path to spatial coordinates .npy")
    parser.add_argument("--n-clusters", type=int, default=7, help="Target number of clusters")
    parser.add_argument("--pca-dim", type=int, default=20, help="PCA dimension before clustering/Moran")
    parser.add_argument("--moran-k", type=int, default=6, help="KNN neighbors for Moran's I")
    return parser.parse_args()


def main():
    args = parse_args()
    pred = np.load(args.pred)
    true = np.load(args.true)
    labels = np.load(args.labels)
    coords = np.load(args.coords)

    results = evaluate_translation(
        pred_atac_matrix=pred,
        true_atac_matrix=true,
        true_spatial_labels=labels,
        spatial_coords=coords,
        n_clusters=int(args.n_clusters),
        pca_dim=int(args.pca_dim),
        moran_k=int(args.moran_k),
    )
    print("Translation Evaluation - Summary:")
    print(results)


if __name__ == "__main__":
    main()
