import torch
import torch.nn as nn
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SFIB

class BioSFINet(nn.Module):
    def __init__(self, config, atac_dim):
        """
        Args:
            atac_dim: 运行时动态获取的 ATAC Peak 数
        """
        super().__init__()
        
        model_cfg = config['model']
        rna_dim = model_cfg['rna_in_dim']
        hidden_dim = model_cfg['hidden_dim']
        sfib_dim = model_cfg.get('sfib_dim', 128)
        rna_heads = model_cfg.get('rna_n_heads', model_cfg.get('n_heads', 4))
        rna_dropout = model_cfg.get('rna_dropout', model_cfg.get('dropout', 0.1))
        atac_dropout = model_cfg.get('atac_dropout', model_cfg.get('dropout', 0.1))
        n_layers = int(model_cfg.get('n_layers', 2))
        
        # 1. Encoders (Phase I)
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim, n_heads=rna_heads, dropout=rna_dropout)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim, dropout=atac_dropout)
        
        # 2. Projections (Phase II)
        self.rna_proj = nn.Linear(hidden_dim, sfib_dim)
        self.atac_proj = nn.Linear(hidden_dim, sfib_dim)
        self.ln_rna = nn.LayerNorm(sfib_dim)
        self.ln_atac = nn.LayerNorm(sfib_dim)
        
        # 3. Cascade SFIB Tower (ATAC Main guided by RNA)
        self.sfib_atac = nn.ModuleList([SFIB(dim=sfib_dim) for _ in range(n_layers)])
        
        # 4. Decoders (Phase IV)
        self.rna_dec = nn.Linear(sfib_dim, rna_dim)
        self.atac_dec = nn.Linear(sfib_dim, atac_dim)

    def forward(self, x_rna, x_atac, edge_index, u_basis):
        # 1. Encode [N, 512]
        h_rna = self.rna_enc(x_rna, edge_index)
        h_atac = self.atac_enc(x_atac)
        
        # 2. Project [N, 128]
        f_rna = self.ln_rna(self.rna_proj(h_rna))
        f_atac = self.ln_atac(self.atac_proj(h_atac))
        
        # 3. Cascade SFIB tower guided by RNA (static guide)
        curr_atac = f_atac
        for block in self.sfib_atac:
            updated = block(x_main=curr_atac, x_guide=f_rna, edge_index=edge_index, u_basis=u_basis)
            curr_atac = updated + curr_atac
        z_fused = curr_atac
        
        # 4. Decode (dual reconstruction from fused latent)
        rec_rna = self.rna_dec(z_fused)
        rec_atac = self.atac_dec(z_fused)

        return z_fused, rec_rna, rec_atac