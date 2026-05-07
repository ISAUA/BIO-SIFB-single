import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="In the future, the default backend for leiden will be igraph instead of leidenalg.*", category=FutureWarning)

import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_load_compat(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_r_runtime_available():
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Y7_T ground-truth spatial map and optional mclust map")
    parser.add_argument(
        "--config",
        default="configs/renal/config_renal_Y7_T.yaml",
        help="Path to Y7_T config file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path; default: <eval_dir>/ground_truth_vs_mclust_spatial.pdf",
    )
    parser.add_argument(
        "--mclust-pca-dim",
        type=int,
        default=None,
        help="PCA dim before mclust; default uses eval.mclust_pca_dim",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override",
    )
    return parser.parse_args()


def _normalize_labels(arr):
    labels = np.asarray(arr).astype(str)
    labels = np.array([x.strip() for x in labels], dtype=object)
    valid = np.array([(x != "") and (x.lower() != "nan") and (x.lower() != "none") for x in labels])
    return labels, valid


def _sort_categories(values):
    uniq = pd.unique(values)

    def _key(v):
        s = str(v)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted([str(x) for x in uniq], key=_key)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42))
    np.random.seed(seed)

    processed_path = os.path.join(cfg["data"]["processed_path"], "processed_data.pt")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"processed_data.pt not found: {processed_path}")

    data = torch_load_compat(processed_path, map_location="cpu", weights_only=False)

    if "coords" not in data:
        raise KeyError("coords not found in processed_data.pt")
    coords = data["coords"]
    coords = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)

    gt_key_cfg = cfg.get("data", {}).get("parameters", {}).get("ground_truth_key", "ground_truth")
    gt = data.get("ground_truth", None)
    if gt is None:
        raise KeyError(
            "ground_truth not found in processed_data.pt. "
            f"Please ensure preprocess saved `{gt_key_cfg}` into ground_truth."
        )

    labels_raw, valid_mask = _normalize_labels(gt)
    if valid_mask.sum() < 2:
        raise ValueError("ground_truth valid labels are insufficient for plotting.")

    coords = coords[valid_mask]
    gt_labels = labels_raw[valid_mask]

    adata = sc.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    adata.obsm["spatial"] = coords
    gt_categories = _sort_categories(gt_labels)
    adata.obs["ground_truth"] = pd.Categorical(gt_labels, categories=gt_categories, ordered=True)

    n_clusters = len(pd.unique(gt_labels))
    print(f"[Info] ground_truth unique clusters: {n_clusters}")

    # Use RNA embedding as clustering input to align with existing evaluation logic.
    features = data.get("rna_feat", None)
    if features is None:
        raise KeyError("rna_feat not found in processed_data.pt, cannot run mclust.")
    features = features.cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    features = features[valid_mask]

    pca_dim_cfg = int(cfg.get("eval", {}).get("mclust_pca_dim", 20))
    pca_dim = int(args.mclust_pca_dim if args.mclust_pca_dim is not None else pca_dim_cfg)
    pca_dim = max(1, min(pca_dim, features.shape[0], features.shape[1]))
    z_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    mclust_labels = None
    mclust_ok = False
    try:
        ensure_r_runtime_available()
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        robjects.r["set.seed"](int(seed))
        robjects.r("suppressPackageStartupMessages(library(mclust))")
        rmclust = robjects.r["Mclust"]

        res = rmclust(
            rpy2.robjects.numpy2ri.numpy2rpy(z_pca),
            int(n_clusters),
            "EEE",
            verbose=False,
        )
        mclust_labels = extract_mclust_labels(res).astype(int).astype(str)
        mclust_ok = True
    except Exception as exc:
        print(f"[Warn] mclust unavailable, skip mclust panel: {exc}")

    plot_cfg = cfg.get("eval", {}).get("plotting", {})
    dpi = int(plot_cfg.get("figure_dpi", 180))
    panel_size = plot_cfg.get("panel_size", [6, 6])
    fig_w = float(panel_size[0]) * (2 if mclust_ok else 1)
    fig_h = float(panel_size[1])
    spatial_size = float(plot_cfg.get("spatial_point_size", 80))
    alpha = float(plot_cfg.get("alpha", 1.0))
    legend_loc = plot_cfg.get("legend_loc", "on data")

    sc.set_figure_params(dpi=dpi, figsize=tuple(panel_size), frameon=True)
    if mclust_ok:
        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        mclust_categories = _sort_categories(mclust_labels)
        adata.obs["mclust"] = pd.Categorical(mclust_labels, categories=mclust_categories, ordered=True)

        sc.pl.embedding(
            adata,
            basis="spatial",
            color="ground_truth",
            ax=axs[0],
            show=False,
            title="Ground Truth Spatial",
            size=spatial_size,
            alpha=alpha,
            legend_loc=legend_loc,
            frameon=True,
            edges=False,
        )
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="mclust",
            ax=axs[1],
            show=False,
            title=f"mclust Spatial (k={n_clusters})",
            size=spatial_size,
            alpha=alpha,
            legend_loc=legend_loc,
            frameon=True,
            edges=False,
        )
        for ax in axs:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="ground_truth",
            ax=ax,
            show=False,
            title="Ground Truth Spatial",
            size=spatial_size,
            alpha=alpha,
            legend_loc=legend_loc,
            frameon=True,
            edges=False,
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    eval_dir = cfg.get("project", {}).get("eval_dir", "./results")
    out_path = args.output or os.path.join(eval_dir, "ground_truth_vs_mclust_spatial.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Done] Saved figure: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
