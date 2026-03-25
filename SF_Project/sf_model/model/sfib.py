import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv


class INOUnit(nn.Module):
    """Bi-directional affine INO unit."""

    def __init__(self, dim: int):
        super().__init__()
        self.gcn_s1 = GCNConv(dim, dim)
        self.gcn_t1 = GCNConv(dim, dim)
        self.gcn_s2 = GCNConv(dim, dim)
        self.gcn_t2 = GCNConv(dim, dim)

    def forward(self, x_main: torch.Tensor, x_guide: torch.Tensor, edge_index: torch.Tensor):
        s1 = torch.tanh(self.gcn_s1(x_guide, edge_index))
        t1 = self.gcn_t1(x_guide, edge_index)
        x_main_star = x_main * torch.exp(s1) + t1

        s2 = torch.tanh(self.gcn_s2(x_main_star, edge_index))
        t2 = self.gcn_t2(x_main_star, edge_index)
        x_guide_star = x_guide * torch.exp(s2) + t2

        return x_main_star, x_guide_star


class SpectralTransformerGate(nn.Module):
    """Cross-attention in spectral domain with Laplacian-eigenvalue diagonal penalty."""

    def __init__(
        self,
        dim: int,
        debug_mode: bool = False,
        debug_epochs=None,
        debug_every_n_epochs: int = 0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.gamma_eig = nn.Parameter(torch.tensor(1.0))
        self.debug_mode = bool(debug_mode)
        self.debug_epochs = set(int(e) for e in (debug_epochs or []))
        self.debug_every_n_epochs = int(debug_every_n_epochs)
        self.current_epoch = None
        self._debug_plotted_epochs = set()
        self._forward_calls = 0

    def set_debug_epoch(self, epoch=None):
        self.current_epoch = None if epoch is None else int(epoch)

    def _should_run_debug(self):
        if not self.debug_mode:
            return False

        # 没有 epoch 上下文（如独立推理）时，默认只在第一次 forward 诊断
        if self.current_epoch is None:
            return self._forward_calls == 1

        if self.debug_epochs and self.current_epoch in self.debug_epochs:
            return True

        if self.debug_every_n_epochs > 0 and self.current_epoch % self.debug_every_n_epochs == 0:
            return True

        # 打开 debug_mode 但未显式配置时，默认首轮诊断
        if not self.debug_epochs and self.debug_every_n_epochs <= 0:
            return self.current_epoch == 1

        return False

    def forward(self, hat_rna: torch.Tensor, hat_atac: torch.Tensor, evals: torch.Tensor = None):
        self._forward_calls += 1
        run_debug_plot = self._should_run_debug()
        run_debug_stats = self.debug_mode
        q = self.q_proj(hat_rna)
        k = self.k_proj(hat_atac)
        v = self.v_proj(hat_atac)

        if run_debug_stats:
            with torch.no_grad():
                q_det = q.detach()
                k_det = k.detach()
                print(
                    "[SpectralTransformerGate][Q stats] "
                    f"mean={q_det.mean().item():.6f}, var={q_det.var(unbiased=False).item():.6f}, "
                    f"min={q_det.min().item():.6f}, max={q_det.max().item():.6f}"
                )
                print(
                    "[SpectralTransformerGate][K stats] "
                    f"mean={k_det.mean().item():.6f}, var={k_det.var(unbiased=False).item():.6f}, "
                    f"min={k_det.min().item():.6f}, max={k_det.max().item():.6f}"
                )

        scale = self.dim ** 0.5
        logits = torch.matmul(q, k.t()) / scale

        n_nodes = hat_rna.size(0)
        if evals is None:
            eig_mask = torch.zeros((n_nodes, n_nodes), device=hat_rna.device, dtype=hat_rna.dtype)
        else:
            evals = evals.to(device=hat_rna.device, dtype=hat_rna.dtype).view(-1)
            if evals.numel() != n_nodes:
                raise ValueError(f"evals size mismatch: expected {n_nodes}, got {evals.numel()}")
            penalty_vec = -self.gamma_eig * torch.log1p(torch.clamp(evals, min=0.0))
            eig_mask = torch.diag(penalty_vec)

        if run_debug_stats:
            with torch.no_grad():
                logits_det = logits.detach()
                eig_mask_det = eig_mask.detach()

                print(
                    "[SpectralTransformerGate][Score range] "
                    f"logits_min={logits_det.min().item():.6f}, logits_max={logits_det.max().item():.6f}, "
                    f"logits_mean={logits_det.mean().item():.6f}"
                )

                non_zero_mask_vals = eig_mask_det[eig_mask_det != 0]
                if non_zero_mask_vals.numel() > 0:
                    print(
                        "[SpectralTransformerGate][Mask stats] "
                        f"mask_min={non_zero_mask_vals.min().item():.6f}, "
                        f"mask_max={non_zero_mask_vals.max().item():.6f}, "
                        f"mask_mean={non_zero_mask_vals.mean().item():.6f}, "
                        f"non_zero={non_zero_mask_vals.numel()}"
                    )
                else:
                    print("[SpectralTransformerGate][Mask stats] all mask entries are zero.")

                combined_det = (logits + eig_mask).detach()
                print(
                    "[SpectralTransformerGate][Combined score range] "
                    f"combined_min={combined_det.min().item():.6f}, "
                    f"combined_max={combined_det.max().item():.6f}, "
                    f"combined_mean={combined_det.mean().item():.6f}"
                )

        attn = F.softmax(logits + eig_mask, dim=-1)

        epoch_key = self.current_epoch if self.current_epoch is not None else -1
        if run_debug_plot and (epoch_key not in self._debug_plotted_epochs):
            with torch.no_grad():
                try:
                    os.makedirs("./diagnostics", exist_ok=True)
                    attn_map = attn.detach().cpu().float().numpy()
                    plt.figure(figsize=(7, 6))
                    plt.imshow(attn_map, cmap="viridis", aspect="auto")
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title("Spectral Cross-Attention Map")
                    plt.xlabel("Key Index")
                    plt.ylabel("Query Index")
                    plt.tight_layout()
                    # 通用输出名（便于快速查看最新结果）
                    plt.savefig("./diagnostics/attention_map.png", dpi=220)
                    # 轮次输出名（便于纵向比较）
                    if self.current_epoch is not None:
                        plt.savefig(f"./diagnostics/attention_map_epoch_{self.current_epoch}.png", dpi=220)
                    plt.close()
                    print(
                        "[SpectralTransformerGate][Debug] saved attention heatmap to "
                        "./diagnostics/attention_map.png"
                    )
                except Exception as e:
                    print(f"[SpectralTransformerGate][Debug] attention map save failed: {e}")
                finally:
                    self._debug_plotted_epochs.add(epoch_key)

        hat_fused = torch.matmul(attn, v) + hat_rna
        return hat_fused, attn


class SymmetricSFIB(nn.Module):
    """Symmetric selective fusion with INO-based spatial branch."""

    def __init__(
        self,
        dim: int = 128,
        num_ino_layers: int = 3,
        debug_mode: bool = False,
        debug_epochs=None,
        debug_every_n_epochs: int = 0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_ino_layers = int(num_ino_layers)

        # Frequency gate
        self.freq_gate = SpectralTransformerGate(
            self.dim,
            debug_mode=debug_mode,
            debug_epochs=debug_epochs,
            debug_every_n_epochs=debug_every_n_epochs,
        )

        # Spatial INO stack
        self.ino_layers = nn.ModuleList([INOUnit(self.dim) for _ in range(self.num_ino_layers)])
        self.spa_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.out_norm = nn.LayerNorm(self.dim)

    def set_debug_epoch(self, epoch=None):
        self.freq_gate.set_debug_epoch(epoch)

    def frequency_branch(
        self,
        h_rna: torch.Tensor,
        h_atac: torch.Tensor,
        u_basis: torch.Tensor,
        evals: torch.Tensor = None,
    ):
        hat_rna = torch.matmul(u_basis.t(), h_rna)
        hat_atac = torch.matmul(u_basis.t(), h_atac)
        hat_fused, attn_freq = self.freq_gate(hat_rna, hat_atac, evals)
        z_base = torch.matmul(u_basis, hat_fused)
        return z_base, attn_freq

    def spatial_branch(self, h_rna: torch.Tensor, h_atac: torch.Tensor, edge_index: torch.Tensor):
        main, guide = h_rna, h_atac
        for ino in self.ino_layers:
            main, guide = ino(main, guide, edge_index)
        gamma_spa = self.spa_gate(torch.cat([main, guide], dim=1))
        z_detail = gamma_spa * main + (1.0 - gamma_spa) * guide
        return z_detail, gamma_spa

    def forward(
        self,
        h_rna: torch.Tensor,
        h_atac: torch.Tensor,
        edge_index: torch.Tensor,
        u_basis: torch.Tensor,
        evals: torch.Tensor = None,
    ):
        z_base, m_freq = self.frequency_branch(h_rna, h_atac, u_basis, evals)
        z_detail, gamma_spa = self.spatial_branch(h_rna, h_atac, edge_index)
        z_fused = self.out_norm(z_base + self.gamma * z_detail)
        return z_fused, z_base, z_detail, m_freq, gamma_spa