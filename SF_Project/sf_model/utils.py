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


# 2026-03-11: 引入 RNA 特征权重机制的空间图构建函数
def build_spatial_graph(coords, features=None, k=6):
    """
    构建空间 KNN 图并计算 GFT 基底。引入 RNA 特征相似度作为边权重以保护生物学边界。
    """
    N = coords.shape[0]
    
    # 1. 构建物理 KNN 图 (确定谁是邻居)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 排除自身 (第0列是自身)
    src = np.repeat(np.arange(N), k)
    dst = indices[:, 1:].flatten()
    
    # 2. 计算边权重 (基于特征相似度)
    if features is not None:
        # 获取源节点和目标节点的特征向量
        f_src = features[src]
        f_dst = features[dst]
        
        # 处理稀疏矩阵格式
        if sp.issparse(features):
            f_src = f_src.toarray()
            f_dst = f_dst.toarray()
            
        # 计算特征空间中的欧式距离平方
        dist_sq = np.sum((f_src - f_dst) ** 2, axis=1)
        
        # 自适应估计高斯核带宽 sigma^2 (中位数启发式)
        sigma2 = np.median(dist_sq)
        if sigma2 == 0:
            sigma2 = 1e-4 # 防止除零
            
        # 计算高斯核权重，加入 1e-4 的微小正则项保证图的连通性
        weights = np.exp(-dist_sq / (2 * sigma2)) + 1e-4
    else:
        # 如果未提供特征，则退化为无权图
        weights = np.ones(len(src))
    
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
    evecs = evecs[:, idx]
    
    u_basis = torch.FloatTensor(evecs)
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    return edge_index, u_basis



def set_seed(seed: int = 42, deterministic: bool = True):
    """全局锁定随机种子，尽可能消除非确定性。"""
    seed = int(seed)
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