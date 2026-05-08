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

    def forward(
        self,
        x_main: torch.Tensor,
        x_guide: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ):
        s1 = torch.tanh(self.gcn_s1(x_guide, edge_index, edge_weight))
        t1 = self.gcn_t1(x_guide, edge_index, edge_weight)
        x_main_star = x_main * torch.exp(s1) + t1

        s2 = torch.tanh(self.gcn_s2(x_main_star, edge_index, edge_weight))
        t2 = self.gcn_t2(x_main_star, edge_index, edge_weight)
        x_guide_star = x_guide * torch.exp(s2) + t2

        return x_main_star, x_guide_star


class SpectralTransformerGate(nn.Module):
    """Adaptive frequency gating in spectral domain."""

    def __init__(
        self,
        dim: int,
        debug_mode: bool = False,
        debug_epochs=None,
        debug_every_n_epochs: int = 0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.norm_rna = nn.LayerNorm(self.dim)
        self.norm_atac = nn.LayerNorm(self.dim)
        self.gating_mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
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

        # Normalize spectral features before adaptive gating to prevent scale collapse.
        norm_rna_out = self.norm_rna(hat_rna)
        norm_atac_out = self.norm_atac(hat_atac)
        gate_input = torch.cat([norm_rna_out, norm_atac_out], dim=-1)
        gate = torch.sigmoid(self.gating_mlp(gate_input))

        if run_debug_stats:
            with torch.no_grad():
                gate_det = gate.detach()
                print(
                    "[SpectralTransformerGate][Gate stats] "
                    f"mean={gate_det.mean().item():.6f}, var={gate_det.var(unbiased=False).item():.6f}, "
                    f"min={gate_det.min().item():.6f}, max={gate_det.max().item():.6f}"
                )

        if run_debug_stats:
            with torch.no_grad():
                print(
                    "[SpectralTransformerGate][Fused stats] "
                    f"rna_mean={norm_rna_out.mean().item():.6f}, atac_mean={norm_atac_out.mean().item():.6f}"
                )

        epoch_key = self.current_epoch if self.current_epoch is not None else -1
        if run_debug_plot and (epoch_key not in self._debug_plotted_epochs):
            with torch.no_grad():
                try:
                    os.makedirs("./diagnostics", exist_ok=True)
                    gate_map = gate.detach().cpu().float().numpy()
                    plt.figure(figsize=(7, 6))
                    plt.imshow(gate_map, cmap="viridis", aspect="auto")
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title("Spectral Adaptive Gating Map")
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
                        "[SpectralTransformerGate][Debug] saved gating heatmap to "
                        "./diagnostics/attention_map.png"
                    )
                except Exception as e:
                    print(f"[SpectralTransformerGate][Debug] gating map save failed: {e}")
                finally:
                    self._debug_plotted_epochs.add(epoch_key)

        hat_fused = gate * norm_rna_out + (1.0 - gate) * norm_atac_out
        return hat_fused, gate


class SymmetricSFIB(nn.Module):
    """Symmetric selective fusion with INO-based spatial branch."""

    def __init__(
        self,
        dim: int = 128,
        num_ino_layers: int = 3,
        ino_use_edge_weight: bool = True,
        pre_smooth_enable: bool = True,
        pre_smooth_alpha: float = 1,
        debug_mode: bool = False,
        debug_epochs=None,
        debug_every_n_epochs: int = 0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_ino_layers = int(num_ino_layers)
        self.ino_use_edge_weight = bool(ino_use_edge_weight)
        self.pre_smooth_enable = bool(pre_smooth_enable)
        self.pre_smooth_alpha = float(max(0.0, min(1.0, pre_smooth_alpha)))

        # Frequency gate
        self.freq_gate = SpectralTransformerGate(
            self.dim,
            debug_mode=debug_mode,
            debug_epochs=debug_epochs,
            debug_every_n_epochs=debug_every_n_epochs,
        )

        # Spatial INO stack
        self.pre_smooth = GCNConv(self.dim, self.dim, bias=False)
        with torch.no_grad():
            self.pre_smooth.lin.weight.copy_(torch.eye(self.dim))
        self.pre_smooth.lin.weight.requires_grad_(False)

        self.ino_layers = nn.ModuleList([INOUnit(self.dim) for _ in range(self.num_ino_layers)])
        self.spa_gate = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
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

    def _smooth_with_fidelity(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        x_smooth = self.pre_smooth(x, edge_index, edge_weight)
        return (1.0 - self.pre_smooth_alpha) * x_smooth + self.pre_smooth_alpha * x

    def spatial_branch(
        self,
        h_rna: torch.Tensor,
        h_atac: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ):
        effective_edge_weight = edge_weight if self.ino_use_edge_weight else None

        main, guide = h_rna, h_atac
        if self.pre_smooth_enable:
            main = self._smooth_with_fidelity(main, edge_index, effective_edge_weight)
            guide = self._smooth_with_fidelity(guide, edge_index, effective_edge_weight)

        for ino in self.ino_layers:
            main, guide = ino(main, guide, edge_index, edge_weight=effective_edge_weight)
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
        edge_weight: torch.Tensor = None,
    ):
        z_base, m_freq = self.frequency_branch(h_rna, h_atac, u_basis, evals)
        z_detail, gamma_spa = self.spatial_branch(h_rna, h_atac, edge_index, edge_weight=edge_weight)
        z_fused = self.out_norm(z_base + self.gamma * z_detail)
        return z_fused, z_base, z_detail, m_freq, gamma_spa