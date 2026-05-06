import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from sf_model.model.bio_sfinet import BioSFINet
from sf_model.utils import set_seed
from scripts.pipeline.translation_runtime import resolve_checkpoint_path, torch_load_compat


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose BioSFINet intermediate tensors")
    parser.add_argument(
        "--config",
        default="configs/e18_5_s1/config_misar_e18_5_s1.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint key or filename; defaults to config eval.checkpoint",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., cuda, cpu). Default: auto",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots; default: <project results>/checkmodel",
    )
    parser.add_argument(
        "--umap-sample",
        type=int,
        default=5000,
        help="Max number of points for UMAP (default: 5000)",
    )
    parser.add_argument(
        "--scatter-sample",
        type=int,
        default=200000,
        help="Max number of points for scatter plot (default: 200000)",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_shape_compatible(model, state_dict):
    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    missing = [k for k in model_state.keys() if k not in filtered_state]
    unexpected = [k for k in state_dict.keys() if k not in model_state]

    model.load_state_dict(filtered_state, strict=False)

    if skipped_shape:
        print("[Warn] Skipped mismatched checkpoint params:")
        for key, ckpt_shape, model_shape in skipped_shape:
            print(f"  - {key}: ckpt={ckpt_shape}, model={model_shape}")

    if unexpected:
        print(f"[Warn] Unexpected checkpoint keys: {len(unexpected)}")

    if missing:
        print(f"[Warn] Missing model keys after filtered load: {len(missing)}")


def tensor_sample(flat_tensor, max_samples, seed=42):
    if flat_tensor.numel() <= max_samples:
        return flat_tensor
    g = torch.Generator(device=flat_tensor.device)
    g.manual_seed(int(seed))
    idx = torch.randperm(flat_tensor.numel(), generator=g, device=flat_tensor.device)[:max_samples]
    return flat_tensor.view(-1)[idx]


def plot_scatter_negative_rna(rna_feat, rec_rna, out_dir, max_samples):
    mask = rna_feat < 0
    if mask.sum().item() == 0:
        print("[Warn] No negative values found in rna_feat; skip scatter plot.")
        return

    x = rna_feat[mask]
    y = rec_rna[mask]
    x = tensor_sample(x, max_samples)
    y = tensor_sample(y, max_samples)

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_np, y_np, s=2, alpha=0.25, edgecolors="none")
    plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("rna_feat (<0)")
    plt.ylabel("rec_rna prediction")
    plt.title("Negative RNA PCA vs Reconstruction")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "scatter_negative_rna.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved scatter: {out_path}")


def plot_zero_freq_explosion(hat_rna, out_dir):
    freq_magnitude = hat_rna.abs().mean(dim=1)
    freq_np = freq_magnitude.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(freq_np)), freq_np, linewidth=1.2)
    plt.xlabel("Frequency index")
    plt.ylabel("Mean |hat_rna|")
    plt.title("Frequency Energy (mean abs)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "freq_energy_hat_rna.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved freq energy: {out_path}")


def plot_attention_heatmap(m_freq, out_dir):
    attn = m_freq.detach().cpu().float().numpy()
    diag_mean = float(np.diag(attn).mean())
    off_diag = attn[~np.eye(attn.shape[0], dtype=bool)]
    off_mean = float(off_diag.mean()) if off_diag.size > 0 else float("nan")

    plt.figure(figsize=(7, 6))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("m_freq (Spectral Attention)")
    plt.xlabel("Key index")
    plt.ylabel("Query index")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "m_freq_heatmap.png")
    plt.savefig(out_path, dpi=220)
    plt.close()

    stats_path = os.path.join(out_dir, "m_freq_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"diag_mean={diag_mean:.6f}\n")
        f.write(f"off_diag_mean={off_mean:.6f}\n")
    print(f"[Plot] Saved heatmap: {out_path}")
    print(f"[Stats] Saved m_freq stats: {stats_path}")


def plot_gamma_hist(gamma_spa, out_dir):
    gamma_np = gamma_spa.detach().cpu().numpy().reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(gamma_np, bins=60, color="#1f77b4", alpha=0.85)
    plt.xlabel("gamma_spa")
    plt.ylabel("Count")
    plt.title("Spatial Gate Distribution")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "gamma_spa_hist.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved gamma histogram: {out_path}")


def run_umap_panels(embeds, labels, out_dir, sample_n, seed=42):
    try:
        import umap
    except Exception as exc:
        print(f"[Warn] UMAP not available ({exc}); skip UMAP plots.")
        return

    n = embeds[0].shape[0]
    if n == 0:
        print("[Warn] Empty embeddings; skip UMAP.")
        return

    idx = np.arange(n)
    if n > sample_n:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=sample_n, replace=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (emb, label) in enumerate(zip(embeds, labels)):
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=int(seed))
        emb2d = reducer.fit_transform(emb[idx])
        ax = axes[i]
        ax.scatter(emb2d[:, 0], emb2d[:, 1], s=6, alpha=0.8)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(out_dir, "umap_comparison.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[Plot] Saved UMAP comparison: {out_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))

    device = resolve_device(args.device)
    print(f"[Info] Device: {device}")

    processed_dir = config["data"]["processed_path"]
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    data_dict = torch_load_compat(data_path, map_location="cpu", weights_only=False)
    rna_feat = data_dict["rna_feat"]
    atac_feat = data_dict["atac_feat"]
    edge_index = data_dict["edge_index"]
    edge_weight = data_dict.get("edge_weight", None)
    u_basis = data_dict["u_basis"]
    evals = data_dict.get("evals", None)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = int(data_dict["atac_dim"])

    print("[Info] Input shapes:")
    print(f"  rna_feat: {tuple(rna_feat.shape)}")
    print(f"  atac_feat: {tuple(atac_feat.shape)}")
    print(f"  edge_index: {tuple(edge_index.shape)}")
    print(f"  u_basis: {tuple(u_basis.shape)}")
    if evals is not None:
        print(f"  evals: {tuple(evals.shape)}")

    config["model"]["rna_in_dim"] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)

    save_dir = config["project"]["save_dir"]
    eval_cfg = config.get("eval", {})
    ckpt_key = args.checkpoint or eval_cfg.get("checkpoint", "best")
    ckpt_map = eval_cfg.get("checkpoints", {})
    ckpt_path, ckpt_name = resolve_checkpoint_path(str(ckpt_key), save_dir, ckpt_map)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    load_state_dict_shape_compatible(model, state_dict)
    model.eval()

    out_dir = args.output_dir
    if out_dir is None:
        project_root = os.path.dirname(os.path.abspath(save_dir.rstrip("/\\")))
        out_dir = os.path.join(project_root, "checkmodel")
    ensure_dir(out_dir)

    rna_feat = rna_feat.to(device)
    atac_feat = atac_feat.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    u_basis = u_basis.to(device)
    if evals is not None:
        evals = evals.to(device)

    with torch.no_grad():
        h_rna = model.rna_enc(rna_feat, edge_index)
        h_atac = model.atac_enc(atac_feat)
        f_rna = model.rna_proj(h_rna)
        f_atac = model.atac_proj(h_atac)

        hat_rna = torch.matmul(u_basis.t(), f_rna)
        hat_atac = torch.matmul(u_basis.t(), f_atac)

        outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        z_fused, p_rna, p_atac, rec_rna, rec_atac, m_freq, gamma_spa, z_base, z_detail = outputs

    plot_scatter_negative_rna(rna_feat, rec_rna, out_dir, args.scatter_sample)
    plot_zero_freq_explosion(hat_rna, out_dir)
    plot_attention_heatmap(m_freq, out_dir)
    plot_gamma_hist(gamma_spa, out_dir)

    embeds = [
        rna_feat.detach().cpu().numpy(),
        z_base.detach().cpu().numpy(),
        z_detail.detach().cpu().numpy(),
        z_fused.detach().cpu().numpy(),
    ]
    labels = ["rna_feat", "z_base", "z_detail", "z_fused"]
    run_umap_panels(embeds, labels, out_dir, args.umap_sample)

    print("[Done] Diagnostics complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
