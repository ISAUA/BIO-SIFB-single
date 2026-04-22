import scanpy as sc
import squidpy as sq
import numpy as np

# 1. 加载您的 h5ad 文件
adata = sc.read_h5ad("/root/autodl-tmp/BIO-SFIB-single/SPADDM/spatialDDM_results_e15.h5ad")

# 2. 获取 mclust 列中唯一类别的数量
num_clusters = adata.obs['mclust'].nunique()
print(f"该文件中的 mclust 共划分了 {num_clusters} 个聚类。")

# 3. 如果您想详细看看每个聚类具体包含了多少个点位（spots），可以使用 value_counts
print("\n--- 每个聚类的具体点位数量统计 ---")
print(adata.obs['mclust'].value_counts())

# =====================================================================
# 新增功能：计算聚类结果的空间 Moran's I (莫兰指数)
# =====================================================================
print("\n--- 计算聚类结果的空间自相关性 (Moran's I) ---")

# 检查是否存在空间坐标 (通常储存在 obsm['spatial'] 中)
if 'spatial' not in adata.obsm:
    raise ValueError("❌ 错误：未在 adata.obsm 中找到 'spatial' 坐标，无法计算 Moran's I。请检查 h5ad 文件是否包含空间坐标。")

coords = adata.obsm['spatial']

# 1. 将离散的聚类标签转换为 One-hot 编码矩阵
# 使用 pandas 的 category codes 确保标签不论是数字还是字符串都能正确映射
categories = adata.obs['mclust'].astype('category').cat.categories
cluster_labels = adata.obs['mclust'].astype('category').cat.codes.values
num_unique = len(categories)

# 生成 One-hot 矩阵，形状为 [N_spots, N_clusters]
one_hot_clusters = np.eye(num_unique)[cluster_labels]

# 2. 构建一个用于计算 Moran's I 的临时 AnnData 对象
adata_moran = sc.AnnData(X=one_hot_clusters)
adata_moran.obsm['spatial'] = coords
# 将变量名设置为对应的聚类名称，方便查看结果
adata_moran.var_names = [f"Cluster_{cat}" for cat in categories]

# 3. 构建空间邻接图
# coord_type='generic' 适用于一般的 2D 坐标。
# n_neighs=6 通常适用于 Visium/Stereo-seq 等具有六边形/网格结构的邻居数
k_neighbors = 6 
sq.gr.spatial_neighbors(adata_moran, coord_type='generic', n_neighs=k_neighbors)

# 4. 计算 Moran's I
# 注意：这里我们只计算 I 值，跳过了 P-value 的置换检验(n_perms=None) 以大幅加快计算速度
sq.gr.spatial_autocorr(
    adata_moran,
    mode='moran',
    n_perms=None, 
    show_progress_bar=False
)

# 5. 提取并展示结果
moran_df = adata_moran.uns['moranI']
print("\n[各个聚类的 Moran's I 值]:")
print(moran_df[['I']])

# 计算综合平均 Moran's I (Cluster Moran's I)
mean_moran = moran_df['I'].mean()
print(f"\n⭐ 综合聚类 Moran's I (Mean): {mean_moran:.4f}")