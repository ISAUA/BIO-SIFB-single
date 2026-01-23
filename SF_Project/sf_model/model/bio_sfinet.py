import torch
import torch.nn as nn
import torch.nn.functional as F  # [新增] 用于激活函数
from torch_geometric.nn import GCNConv  # [新增] 用于 RNA 动态更新
from .encoders import RNA_Encoder, ATAC_Encoder
from .sfib import SFIB

# ==========================================
# 1. 解码器组件 (保持你提供的 DeepDecoder)
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


# ==========================================
# 2. 修改后的 BioSFINet 主模型 (集成动态指导)
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
        
        # 1. Encoders (Phase I)
        self.rna_enc = RNA_Encoder(in_dim=rna_dim, hidden_dim=hidden_dim, n_heads=rna_heads, dropout=rna_dropout)
        self.atac_enc = ATAC_Encoder(in_dim=atac_dim, hidden_dim=hidden_dim, dropout=atac_dropout)
        
        # 2. Projections (Phase II)
        self.rna_proj = nn.Linear(hidden_dim, sfib_dim)
        self.atac_proj = nn.Linear(hidden_dim, sfib_dim)
        self.ln_rna = nn.LayerNorm(sfib_dim)
        self.ln_atac = nn.LayerNorm(sfib_dim)
        
        # 3. Cascade SFIB Tower
        self.sfib_atac = nn.ModuleList([SFIB(dim=sfib_dim) for _ in range(n_layers)])
        
        # --- [NEW] RNA Update Stream (Guide Evolution) ---
        # 使用 GCNConv 在图上更新 RNA 特征，模拟论文中的 feature flow
        # 需要 n_layers - 1 个更新层
        if n_layers > 1:
            self.rna_updates = nn.ModuleList([
                GCNConv(sfib_dim, sfib_dim) for _ in range(n_layers - 1)
            ])
        else:
            self.rna_updates = nn.ModuleList([])
        
        # 4. Decoders (Phase IV) - 使用 DeepDecoder
        self.rna_dec = DeepDecoder(
            in_dim=sfib_dim, 
            out_dim=rna_dim, 
            hidden_dim=256,
            n_blocks=1, 
            dropout=rna_dropout
        )
        
        self.atac_dec = DeepDecoder(
            in_dim=sfib_dim, 
            out_dim=atac_dim, 
            hidden_dim=512,
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
        
        # 3. Cascade SFIB tower (Dynamic Guide Loop)
        curr_atac = f_atac
        curr_rna = f_rna  # RNA 初始化为投影后的特征
        
        for i, block in enumerate(self.sfib_atac):
            # A. 执行当前层的融合
            # 使用当前的 curr_rna 指导 curr_atac
            updated_atac = block(x_main=curr_atac, x_guide=curr_rna, edge_index=edge_index, u_basis=u_basis)
            curr_atac = updated_atac + curr_atac # ATAC 残差更新
            
            # B. 准备下一层的 RNA Guide (如果是最后一层则跳过)
            # 逻辑：利用 GCN 聚合邻域信息，使 RNA 特征随层级加深而演化
            if i < len(self.rna_updates):
                update_layer = self.rna_updates[i]
                
                # GCN 更新
                delta_rna = update_layer(curr_rna, edge_index)
                
                # RNA 残差更新 + ELU 激活 (增加非线性)
                curr_rna = F.elu(delta_rna + curr_rna)
        
        z_fused = curr_atac
        
        # 4. Decode (dual reconstruction from fused latent)
        rec_rna = self.rna_dec(z_fused)
        rec_atac = self.atac_dec(z_fused)

        return z_fused, rec_rna, rec_atac