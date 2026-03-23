import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

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


def calculate_spatial_morans_i(coords, features, k=6):
    """
    纯 Numpy/Scipy 实现的 Moran's I。
    """
    n_cells = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    src = np.repeat(np.arange(n_cells), k)
    dst = indices[:, 1:].flatten()

    w = sp.coo_matrix((np.ones_like(src), (src, dst)), shape=(n_cells, n_cells))
    w = w.maximum(w.T)

    row_sums = np.array(w.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    w_norm = w.multiply(1.0 / row_sums[:, None])

    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    mean_feat = np.mean(features, axis=0)
    centered_feat = features - mean_feat

    var = np.sum(centered_feat ** 2, axis=0)
    var[var == 0] = 1e-10

    cov = np.sum(centered_feat * (w_norm @ centered_feat), axis=0)
    morans_i_vals = cov / var
    return np.mean(morans_i_vals), morans_i_vals


def infer_epoch_label(ckpt_name, epoch):
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    if "best" in base.lower():
        return f"Epoch {epoch} (BEST)"

    match = re.search(r"ckpt[_-]?(\d+)", base)
    if match:
        return f"Epoch {match.group(1)}"

    return f"Epoch {epoch}"


def visualize_and_save(z_final, coords, save_dir, resolution=0.5, epoch_label=None, output_suffix=None):
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
    except Exception:
        moran_title_str = " | Moran's I Error"

    vivid_palette = plt.get_cmap("tab10").colors
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
    print("🚀 [Range Evaluate] Starting checkpoint-range evaluation...")

    epochs = build_epoch_list(args.start, args.end, args.step)
    print(f"   Target epochs: {epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    config = load_config(args.config)
    set_seed(config["project"].get("seed", 42))

    processed_dir = config["data"]["processed_path"]
    save_dir = config["project"]["save_dir"]
    eval_cfg = config.get("eval", {})

    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}")
        print("   -> Please run 'python run_preprocess.py' first.")
        return

    print(f"\n📦 Loading data from {data_path}...")
    data_dict = torch.load(data_path, map_location="cpu")

    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    u_basis = data_dict["u_basis"].to(device)
    atac_dim = data_dict["atac_dim"]

    print("\n🧠 Initializing Bio-SFINet...")
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    model.eval()

    success = 0
    failed = 0

    for epoch in epochs:
        ckpt_name = resolve_checkpoint_name(eval_cfg, epoch, args.best_epoch)
        ckpt_path = os.path.join(save_dir, ckpt_name)

        print(f"\n===== Evaluating epoch {epoch} with {ckpt_name} =====")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skip: checkpoint not found at {ckpt_path}")
            failed += 1
            continue

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            outputs = model(rna_feat, atac_feat, edge_index, u_basis)
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
        )

        print(f"✅ Saved figure: {plot_path}")
        print(f"✅ Saved h5ad: {h5ad_path}")
        success += 1

    print("\n🎉 Range evaluation complete!")
    print(f"   Success: {success} | Failed/Skipped: {failed}")


if __name__ == "__main__":
    main()
