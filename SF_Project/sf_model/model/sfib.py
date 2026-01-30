import torch
import torch.nn as nn
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


class SymmetricSFIB(nn.Module):
    """Symmetric selective fusion with INO-based spatial branch."""

    def __init__(self, dim: int = 128, num_ino_layers: int = 3):
        super().__init__()
        self.dim = int(dim)
        self.num_ino_layers = int(num_ino_layers)

        # Frequency gate
        self.freq_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
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

    def frequency_branch(self, h_rna: torch.Tensor, h_atac: torch.Tensor, u_basis: torch.Tensor):
        hat_rna = torch.matmul(u_basis.t(), h_rna)
        hat_atac = torch.matmul(u_basis.t(), h_atac)
        m_freq = self.freq_gate(torch.cat([hat_rna, hat_atac], dim=1))
        hat_fused = m_freq * hat_rna + (1.0 - m_freq) * hat_atac
        z_base = torch.matmul(u_basis, hat_fused)
        return z_base, m_freq

    def spatial_branch(self, h_rna: torch.Tensor, h_atac: torch.Tensor, edge_index: torch.Tensor):
        main, guide = h_rna, h_atac
        for ino in self.ino_layers:
            main, guide = ino(main, guide, edge_index)
        gamma_spa = self.spa_gate(torch.cat([main, guide], dim=1))
        z_detail = gamma_spa * main + (1.0 - gamma_spa) * guide
        return z_detail, gamma_spa

    def forward(self, h_rna: torch.Tensor, h_atac: torch.Tensor, edge_index: torch.Tensor, u_basis: torch.Tensor):
        z_base, m_freq = self.frequency_branch(h_rna, h_atac, u_basis)
        z_detail, gamma_spa = self.spatial_branch(h_rna, h_atac, edge_index)
        z_fused = self.out_norm(z_base + self.gamma * z_detail)
        return z_fused, z_base, z_detail, m_freq, gamma_spa