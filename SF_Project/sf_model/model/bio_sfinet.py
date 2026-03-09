import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# ==========================================
# 1. 解码器组件 (保留 DeepDecoder 以重构两模态)
# ==========================================


class ResidualBlock(nn.Module):
    """两层前归一化残差块，用于 DeepDecoder。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim)

        def _sublayer() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.block = nn.Sequential(
            _sublayer(),
            _sublayer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepDecoder(nn.Module):
    """Residual Deep Decoder (Projection + Residual Stack + Output)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        n_blocks = int(n_blocks)

        self.proj = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return self.out(x)


# ==========================================
# 2. Mono-SFINet: 单塔频域-空域信号分解
#    Phase I: 模态投影 + 自适应融合
#    Phase II: 频域软低通 (基底)
#    Phase III: 空域 GATv2 细节注入
#    Phase IV: 残差叠加 + 解码
# ==========================================


class ModalityProjector(nn.Module):
    """两层 MLP，将原模态映射到共享隐空间。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BioSFINet(nn.Module):
    """
    Mono-SFINet 主干：频域提基底 + 空域补细节的单塔结构。
    """

    def __init__(self, config, atac_dim: int):
        super().__init__()

        model_cfg = config["model"]
        rna_dim = int(model_cfg["rna_in_dim"])
        hidden_dim = int(model_cfg.get("hidden_dim", 512))
        fusion_dim = int(model_cfg.get("sfib_dim", hidden_dim))
        dropout = float(model_cfg.get("dropout", 0.1))

        gate_hidden = int(model_cfg.get("gate_hidden_dim", hidden_dim))
        gat_heads = int(model_cfg.get("n_heads", 4))
        gat_dropout = float(model_cfg.get("gat_dropout", dropout))
        freq_decay_init = float(model_cfg.get("freq_decay_init", 0.5))

        # Phase I: 模态投影 + 自适应融合
        self.rna_proj = ModalityProjector(rna_dim, hidden_dim, fusion_dim, dropout)
        self.atac_proj = ModalityProjector(atac_dim, hidden_dim, fusion_dim, dropout)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, fusion_dim),
            nn.Sigmoid(),
        )

        # Phase II: 频域软低通 (全局基底)
        self.freq_decay = nn.Parameter(torch.tensor(freq_decay_init))
        self.freq_norm = nn.LayerNorm(fusion_dim)

        # Phase III: 空域细节注入 (GATv2 动态注意力)
        self.detail_attn = GATv2Conv(
            in_channels=fusion_dim,
            out_channels=fusion_dim,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
        )
        self.detail_norm = nn.LayerNorm(fusion_dim)
        self.detail_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(gat_dropout),
            nn.LayerNorm(fusion_dim),
        )

        # Phase IV: 通道门控残差叠加
        self.detail_gate = nn.Parameter(torch.zeros(fusion_dim))

        # 解码器
        self.rna_dec = DeepDecoder(
            in_dim=fusion_dim,
            out_dim=rna_dim,
            hidden_dim=256,
            n_blocks=1,
            dropout=dropout,
        )

        self.atac_dec = DeepDecoder(
            in_dim=fusion_dim,
            out_dim=atac_dim,
            hidden_dim=512,
            n_blocks=1,
            dropout=dropout,
        )

        # 对比投影头: 将融合特征映射到对比空间 (不干扰重构)
        proj_dim = int(model_cfg.get("proj_dim", 128))
        self.contrastive_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    # -------------------------
    # Phase II: 频域基底提取
    # -------------------------
    def _spectral_base(self, x: torch.Tensor, u_basis: torch.Tensor, evals: torch.Tensor) -> torch.Tensor:
        # 如果缺少特征值，退化为恒等映射，保持兼容旧数据
        if evals is None:
            return self.freq_norm(x)

        beta = F.softplus(self.freq_decay)  # 保证衰减为正
        h_hat = torch.matmul(u_basis.t(), x)
        filt = torch.exp(-beta * evals).unsqueeze(1)  # [N,1]
        h_hat_low = h_hat * filt
        h_base = torch.matmul(u_basis, h_hat_low)
        return self.freq_norm(h_base)

    def forward(self, x_rna, x_atac, edge_index, u_basis, evals=None):
        # Phase I: 模态投影 + 自适应融合
        h_rna = self.rna_proj(x_rna)
        h_atac = self.atac_proj(x_atac)
        gate = self.fusion_gate(torch.cat([h_rna, h_atac], dim=1))
        fused = gate * h_rna + (1.0 - gate) * h_atac

        # Phase II: 全局低频基底 (频域软低通)
        base = self._spectral_base(fused, u_basis, evals)

        # Phase III: 空域高频细节 (GATv2 动态注意力)
        detail = self.detail_attn(fused, edge_index)
        detail = self.detail_norm(detail)
        detail = self.detail_ffn(detail)

        # Phase IV: 残差叠加 + 通道门控
        gamma = torch.sigmoid(self.detail_gate).unsqueeze(0)  # [1, C]
        z_fused = base + gamma * detail

        rec_rna = self.rna_dec(z_fused)
        rec_atac = self.atac_dec(z_fused)

        # 对比投影头：映射到对比空间并 L2 归一化
        h = self.contrastive_proj(z_fused)
        h = F.normalize(h, dim=1)

        return z_fused, rec_rna, rec_atac, h