import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class SymmetricSFIB(nn.Module):
    """
    Symmetric Selective Fusion block.
    Implements frequency-domain and spatial-domain competition where RNA/ATAC are treated equally.
    """

    def __init__(self, dim: int = 128, gat_dropout: float = 0.1, gat_heads: int = 1):
        super().__init__()
        self.dim = int(dim)

        # Frequency gate: decides per-frequency confidence between RNA and ATAC
        self.freq_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )

        # Spatial extractors (independent attention for each modality)
        self.gat_rna = GATv2Conv(self.dim, self.dim, heads=gat_heads, concat=False, dropout=gat_dropout)
        self.gat_atac = GATv2Conv(self.dim, self.dim, heads=gat_heads, concat=False, dropout=gat_dropout)

        # Spatial gate: decides per-node/feature winner after GAT aggregation
        self.spa_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )

        # Detail amplification factor
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Output normalization for stability
        self.out_norm = nn.LayerNorm(self.dim)

    def frequency_branch(self, h_rna: torch.Tensor, h_atac: torch.Tensor, u_basis: torch.Tensor):
        """Frequency competition with dual GFT and confidence gating."""
        hat_rna = torch.matmul(u_basis.t(), h_rna)
        hat_atac = torch.matmul(u_basis.t(), h_atac)

        m_freq = self.freq_gate(torch.cat([hat_rna, hat_atac], dim=1))
        hat_fused = m_freq * hat_rna + (1.0 - m_freq) * hat_atac
        z_base = torch.matmul(u_basis, hat_fused)
        return z_base, m_freq

    def spatial_branch(self, h_rna: torch.Tensor, h_atac: torch.Tensor, edge_index: torch.Tensor):
        """Spatial competition via dual GATv2 encoders and gating."""
        n_rna = self.gat_rna(h_rna, edge_index)
        n_atac = self.gat_atac(h_atac, edge_index)

        gamma_spa = self.spa_gate(torch.cat([n_rna, n_atac], dim=1))
        z_detail = gamma_spa * n_rna + (1.0 - gamma_spa) * n_atac
        return z_detail, gamma_spa

    def forward(self, h_rna: torch.Tensor, h_atac: torch.Tensor, edge_index: torch.Tensor, u_basis: torch.Tensor):
        # Frequency-domain symmetric selection
        z_base, m_freq = self.frequency_branch(h_rna, h_atac, u_basis)

        # Spatial-domain symmetric selection
        z_detail, gamma_spa = self.spatial_branch(h_rna, h_atac, edge_index)

        # Final single-tower representation
        z_fused = self.out_norm(z_base + self.gamma * z_detail)
        return z_fused, z_base, z_detail, m_freq, gamma_spa