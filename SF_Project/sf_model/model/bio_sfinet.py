import torch
import torch.nn as nn
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SFIB

# ==========================================
# 1. 新增: 从 GraphTransformer 迁移过来的解码器组件
# ==========================================

class ResidualBlock(nn.Module):
    """标准残差块（Pre-LN）：两层子层并做残差连接。
    来源: GraphTransformer
    """
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
    """Residual Deep Decoder (Projection + Residual Stack + Output).
    来源: GraphTransformer
    """
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


# ==========================================
# 2. 修改后的 BioSFINet 主模型
# ==========================================

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
        
        # 获取参数
        rna_heads = model_cfg.get('rna_n_heads', model_cfg.get('n_heads', 4))
        rna_dropout = model_cfg.get('rna_dropout', model_cfg.get('dropout', 0.1))
        atac_dropout = model_cfg.get('atac_dropout', model_cfg.get('dropout', 0.1))
        n_layers = int(model_cfg.get('n_layers', 2))
        
        # 1. Encoders (Phase I) - 保持不变
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim, n_heads=rna_heads, dropout=rna_dropout)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim, dropout=atac_dropout)
        
        # 2. Projections (Phase II) - 保持不变
        self.rna_proj = nn.Linear(hidden_dim, sfib_dim)
        self.atac_proj = nn.Linear(hidden_dim, sfib_dim)
        self.ln_rna = nn.LayerNorm(sfib_dim)
        self.ln_atac = nn.LayerNorm(sfib_dim)
        
        # 3. Cascade SFIB Tower - 保持不变
        self.sfib_atac = nn.ModuleList([SFIB(dim=sfib_dim) for _ in range(n_layers)])
        
        # 4. Decoders (Phase IV) - [核心修改区域]
        # 原代码:
        # self.rna_dec = nn.Linear(sfib_dim, rna_dim)
        # self.atac_dec = nn.Linear(sfib_dim, atac_dim)
        
        # 新代码: 使用 DeepDecoder
        # 参数说明: in_dim=sfib特征维度, out_dim=原始表达维度
        # hidden_dim 和 n_blocks 参考 GraphTransformer 的配置 (RNA=1024, ATAC=2048)
        self.rna_dec = DeepDecoder(
            in_dim=sfib_dim, 
            out_dim=rna_dim, 
            hidden_dim=256,  # 增强容量
            n_blocks=1, 
            dropout=rna_dropout
        )
        
        self.atac_dec = DeepDecoder(
            in_dim=sfib_dim, 
            out_dim=atac_dim, 
            hidden_dim=512,  # ATAC通常更稀疏，给予更大容量
            n_blocks=1, 
            dropout=atac_dropout
        )

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
        # DeepDecoder 的调用接口与 nn.Linear 一致，直接传入即可
        rec_rna = self.rna_dec(z_fused)
        rec_atac = self.atac_dec(z_fused)

        return z_fused, rec_rna, rec_atac