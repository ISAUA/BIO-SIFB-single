import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

# def build_spatial_graph(coords, k=10):
#     """
#     构建空间 KNN 图并计算 GFT 基底。
#     对应框架: Phase 1 - Global Graph Basis Construction
    
#     Args:
#         coords: [N, 2] 原始物理坐标 (numpy array)
#         k: 近邻数
#     Returns:
#         edge_index: [2, E] 图的边索引 (供 GAT/GNN 使用)
#         u_basis: [N, N] 拉普拉斯矩阵的特征向量矩阵 (供 GFT 使用)
#     """
#     N = coords.shape[0]
    
#     # 1. 构建 KNN 图
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
#     distances, indices = nbrs.kneighbors(coords)
    
#     # 构建边索引 (Edge Index)
#     # indices 第一列是自己，从第二列开始是邻居
#     src = np.repeat(np.arange(N), k)
#     dst = indices[:, 1:].flatten()
    
#     # 转为 PyG 格式的 edge_index [2, E]
#     edge_index = torch.tensor([src, dst], dtype=torch.long)
    
#     # 2. 构建归一化拉普拉斯矩阵 L
#     # 构造稀疏邻接矩阵
#     data = np.ones(len(src))
#     adj = sp.coo_matrix((data, (src, dst)), shape=(N, N))
    
#     # 对称化 (变为无向图)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
#     # 计算度矩阵 D
#     degree = np.array(adj.sum(1)).flatten()
#     d_inv_sqrt = np.power(degree, -0.5)
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
#     # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
#     normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
#     laplacian = sp.eye(N) - normalized_adj
    
#     # 3. 特征分解 (Eigen Decomposition)
#     # L = U * Lambda * U^T
#     # 对于 N=2500，dense solver (eigh) 速度很快
#     evals, evecs = np.linalg.eigh(laplacian.toarray())
    
#     # 排序 (低频 -> 高频)
#     idx = np.argsort(evals)
#     # evals = evals[idx]
#     evecs = evecs[:, idx]
    
#     # 转为 Tensor
#     u_basis = torch.FloatTensor(evecs)
    
#     return edge_index, u_basis


def build_spatial_graph(coords, features=None, k=6, device=None):
    """
    构建空间 KNN 图并计算 GFT 基底。
    - 默认：RNA 特征欧氏距离 + 高斯核衰减 (features 不为 None 时)
    - 备选：空间欧氏距离 + 高斯核衰减 (已注释，可手动切换)
    """
    if isinstance(coords, torch.Tensor):
        inferred_device = coords.device
        coords_np = coords.detach().cpu().numpy()
    else:
        inferred_device = torch.device("cpu")
        coords_np = np.asarray(coords)

    target_device = device if device is not None else inferred_device
    N = coords_np.shape[0]
    
    # 1. 构建物理 KNN 图 (确定谁是邻居)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords_np)
    distances, indices = nbrs.kneighbors(coords_np)
    
    # 排除自身 (第0列是自身)
    src = np.repeat(np.arange(N), k)
    dst = indices[:, 1:].flatten()
    
    # 2. 计算边权重 (默认：特征欧氏距离高斯衰减)
    if features is not None:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        else:
            features = np.asarray(features)

        f_src = features[src]
        f_dst = features[dst]

        if sp.issparse(features):
            f_src = f_src.toarray()
            f_dst = f_dst.toarray()

        dist_sq = np.sum((f_src - f_dst) ** 2, axis=1)
        sigma2 = np.median(dist_sq)
        if sigma2 == 0:
            sigma2 = 1e-4  # 防止除零
        weights = np.exp(-dist_sq / (2 * sigma2)) + 1e-4
    else:
        # # 备选方案：空间距离衰减（默认注释，可按需启用）
        # dist_sq = (distances[:, 1:] ** 2).flatten()
        # sigma2 = np.median(dist_sq)
        # if sigma2 == 0:
        #     sigma2 = 1e-4  # 防止除零
        # weights = np.exp(-dist_sq / (2 * sigma2)) + 1e-4

        # 当前退化为无权图；若需空间衰减，请取消上方注释
        weights = np.ones(len(src))

    #   如果同时使用特征和空间距离，可以在此处合成权重，例如：
    #     # 2. 计算边权重：特征核 × 空间核
    # spatial_dist_sq = (distances[:, 1:] ** 2).flatten()
    # sigma2_spa = np.median(spatial_dist_sq)
    # if sigma2_spa == 0:
    #     sigma2_spa = 1e-4
    # w_spa = np.exp(-spatial_dist_sq / (2 * sigma2_spa))

    # if features is not None:
    #     f_src = features[src]
    #     f_dst = features[dst]
    #     if sp.issparse(features):
    #         f_src = f_src.toarray()
    #         f_dst = f_dst.toarray()
    #     feat_dist_sq = np.sum((f_src - f_dst) ** 2, axis=1)
    #     sigma2_feat = np.median(feat_dist_sq)
    #     if sigma2_feat == 0:
    #         sigma2_feat = 1e-4
    #     w_feat = np.exp(-feat_dist_sq / (2 * sigma2_feat))
    # else:
    #     w_feat = np.ones(len(src))

    # # 合成权重并加微小正则保持连通
    # weights = w_spa * w_feat + 1e-4
    
    # 3. 构建稀疏邻接矩阵
    adj = sp.coo_matrix((weights, (src, dst)), shape=(N, N))
    
    # 对称化 (变为无向图，取节点间的最大相似度权重)
    adj = adj.maximum(adj.T) 
    
    # 4. 计算归一化拉普拉斯矩阵
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    laplacian = sp.eye(N) - normalized_adj
    
    # 5. 特征分解提取频率基底
    evals, evecs = np.linalg.eigh(laplacian.toarray())
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    u_basis = torch.FloatTensor(evecs)
    evals = torch.FloatTensor(evals).to(target_device)
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    return edge_index, u_basis, evals



def set_seed(seed: int = 42, deterministic: bool = True):
    """全局锁定随机种子，尽可能消除非确定性。"""
    seed = int(seed)
    # CUDA + CuBLAS 在 deterministic 模式下需要该环境变量。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    return seed

class CLIPLoss(nn.Module):
    """
    Phase 4: Contrastive Alignment Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        init = torch.log(torch.tensor(float(temperature)))
        self.log_temperature = nn.Parameter(init)

    def forward(self, z_rna, z_atac):
        # L2 Normalize
        z_rna = F.normalize(z_rna, dim=1)
        z_atac = F.normalize(z_atac, dim=1)
        
        # Similarity Matrix
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=10.0)
        logits = torch.matmul(z_rna, z_atac.T) / temperature
        
        # Labels: Diagonal is positive pair
        batch_size = z_rna.shape[0]
        labels = torch.arange(batch_size).to(z_rna.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2