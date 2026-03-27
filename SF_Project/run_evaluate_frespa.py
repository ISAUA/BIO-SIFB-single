import os
import argparse
import re
import warnings
import torch
import yaml

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import scanpy as sc
import matplotlib.pyplot as plt
from sf_model.utils import set_seed
from sf_model.model.bio_sfinet import BioSFINet


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFIB freq vs spa branches")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint key or filename; defaults to config eval.checkpoint")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden clustering resolution")
    return parser.parse_args()


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def infer_epoch_label(ckpt_name: str) -> str:
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    match = re.search(r"ckpt[_-]?(\d+)", base)
    if match:
        return f"epoch {match.group(1)}"
    if "best" in base.lower():
        return "best"
    return base


def visualize_and_save(z_embed, coords, save_root: str, resolution: float, label: str, epoch_label: str):
    """Run UMAP + Leiden and save UMAP/Spatial plots and h5ad for a given embedding."""
    if isinstance(z_embed, torch.Tensor):
        z_embed = z_embed.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    adata = sc.AnnData(X=z_embed)
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
        title=f"{label} UMAP",
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
        title=f"{label} Spatial",
        size=80,
        frameon=True,
        palette=vivid_palette,
        alpha=1.0,
        edges=False,
    )

    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.text(0.5, 0.99, f"{label} | {epoch_label}", ha="center", va="top", fontsize=12)

    fig_dir = os.path.join(save_root, "figures")
    pred_dir = os.path.join(save_root, "predictions")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = os.path.join(fig_dir, f"spatial_analysis_{label}.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    h5ad_path = os.path.join(pred_dir, f"embedding_{label}.h5ad")
    adata.write(h5ad_path)

    print(f"✅ Saved plots to {plot_path}")
    print(f"✅ Saved h5ad to {h5ad_path}")


def main():
    print("🚀 Evaluate SFIB frequency vs spatial embeddings")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    set_seed(config["project"].get("seed", 42))

    processed_dir = config["data"]["processed_path"]
    save_dir = config["project"]["save_dir"]

    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}; run preprocessing first.")
        return

    print(f"📦 Loading data from {data_path} ...")
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

    print("🧠 Initializing BioSFINet ...")
    config["model"]["rna_in_dim"] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)

    eval_cfg = config.get("eval", {})
    ckpt_key = args.checkpoint or eval_cfg.get("checkpoint", "best")
    ckpt_map = eval_cfg.get("checkpoints", {})
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)
    ckpt_path = os.path.join(save_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at {ckpt_path}")
        return

    print(f"   -> Loading weights from {ckpt_path}")
    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    epoch_label = infer_epoch_label(ckpt_name)

    print("🔮 Running inference ...")
    with torch.no_grad():
        z_fused, _, _, _, _, _, _, z_base, z_detail = model(rna_feat, atac_feat, edge_index, u_basis, evals)

    # Save and plot frequency branch (z_base) and spatial branch (z_detail)
    visualize_and_save(z_base, coords, save_dir, args.resolution, label="freq", epoch_label=epoch_label)
    visualize_and_save(z_detail, coords, save_dir, args.resolution, label="spa", epoch_label=epoch_label)

    print("🎉 Done.")


if __name__ == "__main__":
    main()
