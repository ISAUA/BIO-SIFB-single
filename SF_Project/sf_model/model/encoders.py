import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAProjector(nn.Module):
    """轻量投影头：用于 PCA 后特征的统一映射。"""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        in_dim = int(in_dim)
        hidden_dim = int(hidden_dim)
        self.pre_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.activation(x)
        x = self.post_norm(x)
        return x

class RNA_Encoder(nn.Module):
    """RNA 模态编码器：接收 PCA 后特征并做轻量映射。"""

    def __init__(self, in_dim=3000, hidden_dim=512, n_heads=4, dropout=0.1):
        super(RNA_Encoder, self).__init__()
        self.projector = PCAProjector(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x, edge_index):
        # edge_index 保留在接口中以兼容现有调用链。
        _ = edge_index
        return self.projector(x)


class ATAC_Encoder(nn.Module):
    """ATAC 模态编码器：接收 PCA/SVD 后特征并做轻量映射。"""

    def __init__(self, in_dim, hidden_dim=512, dropout=0.1):
        super(ATAC_Encoder, self).__init__()
        self.projector = PCAProjector(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        return self.projector(x)