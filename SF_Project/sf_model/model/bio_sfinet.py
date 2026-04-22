import torch
import torch.nn as nn
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SymmetricSFIB

# ==========================================
# 1. 新增组件: 从 GraphTransformer 迁移的深度解码器
# ==========================================

class ResidualBlock(nn.Module):
    """标准残差块（Pre-LN）：两层子层并做残差连接。"""
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
        x = x + self.block(x)
        return x


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

        # 1. 投影到高维隐空间
        self.proj = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. 残差堆叠
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        
        # 3. 输出映射
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return self.out(x)


class StreamProjector(nn.Module):
    """Two-layer MLP projector that preserves modality independence before SFIB."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SF_Translator_R2A(nn.Module):
    """Point-wise RNA->ATAC latent translator with residual skip."""

    def __init__(self, hidden_dim: int, n_blocks: int = 3):
        super().__init__()
        hidden_dim = int(hidden_dim)
        n_blocks = max(2, int(n_blocks))

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, f_rna: torch.Tensor) -> torch.Tensor:
        mapped = f_rna
        for block in self.blocks:
            mapped = block(mapped)
        # Preserve RNA latent base to reduce collapse risk.
        return f_rna + mapped


class BioSFINet(nn.Module):
    def __init__(self, config, atac_dim):
        """Single-tower Symmetric Selective Fusion network."""
        super().__init__()

        model_cfg = config['model']
        rna_dim = model_cfg['rna_in_dim']
        hidden_dim = model_cfg['hidden_dim']
        sfib_dim = model_cfg.get('sfib_dim', 128)
        rna_heads = model_cfg.get('rna_n_heads', model_cfg.get('n_heads', 4))
        rna_dropout = model_cfg.get('rna_dropout', model_cfg.get('dropout', 0.1))
        atac_dropout = model_cfg.get('atac_dropout', model_cfg.get('dropout', 0.1))
        proj_hidden = model_cfg.get('proj_hidden_dim', 64)
        proj_output = model_cfg.get('proj_output_dim', 64)
        num_ino_layers = model_cfg.get('sfib_ino_layers', 3)
        ino_use_edge_weight = bool(model_cfg.get('ino_use_edge_weight', True))
        ino_pre_smooth_enable = bool(model_cfg.get('ino_pre_smooth_enable', True))
        ino_pre_smooth_alpha = float(model_cfg.get('ino_pre_smooth_alpha', 0.2))
        sfib_debug_mode = bool(model_cfg.get('sfib_debug_mode', False))
        sfib_debug_epochs = model_cfg.get('sfib_debug_epochs', [])
        sfib_debug_every_n_epochs = int(model_cfg.get('sfib_debug_every_n_epochs', 0))

        # 1) Encoders
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim, n_heads=rna_heads, dropout=rna_dropout)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim, dropout=atac_dropout)

        # 2) Independent dual-stream projections into shared dimension d
        self.rna_proj = StreamProjector(in_dim=hidden_dim, out_dim=sfib_dim, dropout=rna_dropout)
        self.atac_proj = StreamProjector(in_dim=hidden_dim, out_dim=sfib_dim, dropout=atac_dropout)

        # 3) Symmetric SFIB (frequency + spatial competition)
        self.sfib = SymmetricSFIB(
            dim=sfib_dim,
            num_ino_layers=num_ino_layers,
            ino_use_edge_weight=ino_use_edge_weight,
            pre_smooth_enable=ino_pre_smooth_enable,
            pre_smooth_alpha=ino_pre_smooth_alpha,
            debug_mode=sfib_debug_mode,
            debug_epochs=sfib_debug_epochs,
            debug_every_n_epochs=sfib_debug_every_n_epochs,
        )

        # 4) Decoders reuse deep residual heads
        rna_dec_hidden = model_cfg.get('rna_dec_hidden', 512)
        rna_dec_blocks = model_cfg.get('rna_dec_blocks', 1)
        self.rna_dec = DeepDecoder(
            in_dim=sfib_dim,
            out_dim=rna_dim,
            hidden_dim=rna_dec_hidden,
            n_blocks=rna_dec_blocks,
            dropout=rna_dropout
        )

        atac_dec_hidden = model_cfg.get('atac_dec_hidden', 512)
        atac_dec_blocks = model_cfg.get('atac_dec_blocks', 1)
        self.atac_dec = DeepDecoder(
            in_dim=sfib_dim,
            out_dim=atac_dim,
            hidden_dim=atac_dec_hidden,
            n_blocks=atac_dec_blocks,
            dropout=atac_dropout
        )

        # 5) Optional contrastive head (pre-fusion) for alignment if needed
        self.proj_head = nn.Sequential(
            nn.Linear(sfib_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_output)
        )

    def set_debug_epoch(self, epoch=None):
        self.sfib.set_debug_epoch(epoch)

    def forward(self, x_rna, x_atac, edge_index, u_basis, evals=None, edge_weight=None):
        # Encode modalities independently
        h_rna = self.rna_enc(x_rna, edge_index)
        h_atac = self.atac_enc(x_atac)

        # Project to shared space (still separated)
        f_rna = self.rna_proj(h_rna)
        f_atac = self.atac_proj(h_atac)

        # Symmetric selective fusion (single tower)
        z_fused, z_base, z_detail, m_freq, gamma_spa = self.sfib(
            f_rna, f_atac, edge_index, u_basis, evals, edge_weight=edge_weight
        )

        # Decode both modalities from fused embedding
        rec_rna = self.rna_dec(z_fused)
        rec_atac = self.atac_dec(z_fused)

        # Pre-fusion contrastive heads (optional for losses)
        p_rna = self.proj_head(f_rna)
        p_atac = self.proj_head(f_atac)

        # Return fused embedding plus auxiliary maps for monitoring
        return z_fused, p_rna, p_atac, rec_rna, rec_atac, m_freq, gamma_spa, z_base, z_detail