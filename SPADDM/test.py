import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score, 
    adjusted_mutual_info_score, 
    homogeneity_score
)

# 引入 rpy2 用于在 Python 中无缝调用 R 语言代码
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

try:
    import squidpy as sq
except ImportError:
    sq = None


def calculate_spatial_morans_i(coords, features, k=6):
    """
    基于 squidpy 计算 Moran's I。
    与评估脚本保持一致：对常量特征做过滤并回填为 0。
    """
    if sq is None:
        raise ImportError("squidpy 未安装，无法计算 Moran's I。")

    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    valid_mask = np.var(features, axis=0) > 0
    if not np.any(valid_mask):
        morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
        return float(np.mean(morans_i_vals)), morans_i_vals

    adata_moran = sc.AnnData(X=features[:, valid_mask])
    adata_moran.obsm['spatial'] = coords

    sq.gr.spatial_neighbors(adata_moran, coord_type='generic', n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_moran,
        mode='moran',
        n_perms=None,
        show_progress_bar=False,
    )

    moran_df = adata_moran.uns['moranI']
    valid_vals = moran_df['I'].to_numpy(dtype=np.float32)
    morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
    morans_i_vals[valid_mask] = valid_vals
    return float(np.mean(morans_i_vals)), morans_i_vals

def calculate_clustering_metrics_mclust(h5ad_path, true_col, pred_col, moran_k=6):
    """
    基于 R 语言 mclust 包计算 ARI，并结合 sklearn 计算 NMI, AMI, HOM。
    """
    try:
        # 1. 加载包含结果的 h5ad 文件
        print(f"正在加载文件: {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path)
        
        # 2. 检查列名是否存在
        if true_col not in adata.obs.columns or pred_col not in adata.obs.columns:
            print(f"错误: 无法在 adata.obs 中找到指定的列名 '{true_col}' 或 '{pred_col}'。")
            return
            
        # 3. 提取真实标签和预测标签，并对齐去除空值 (NaN)
        df = pd.DataFrame({
            'y_true': adata.obs[true_col],
            'y_pred': adata.obs[pred_col]
        }).dropna()
        
        # 将标签统一转换为字符串类型
        y_true = df['y_true'].astype(str)
        y_pred = df['y_pred'].astype(str)

        # 与标签对齐的空间坐标（用于 Moran's I）
        if 'spatial' in adata.obsm:
            valid_pos = adata.obs.index.get_indexer(df.index)
            if np.any(valid_pos < 0):
                raise ValueError("部分有效标签索引无法映射到空间坐标。")
            coords = np.asarray(adata.obsm['spatial'])[valid_pos]
        else:
            coords = None
        
        print(f"成功提取并对齐 {len(df)} 个有效细胞/斑点(spots)的标签。\n")
        
        # 4. 计算评价指标
        # ---------------------------------------------------------
        # [核心修改]: 使用 R 语言的 mclust 计算 ARI
        # ---------------------------------------------------------
        print("正在初始化 R 环境并调用 mclust 包计算 ARI...")
        mclust = importr('mclust')
        
        # 将 Python 的 Series 数据转换为 R 语言认识的向量 (Vector)
        r_y_true = robjects.StrVector(y_true.values)
        r_y_pred = robjects.StrVector(y_pred.values)
        
        # 调用 mclust 中的 adjustedRandIndex
        ari_r_result = mclust.adjustedRandIndex(r_y_true, r_y_pred)
        ari = ari_r_result[0]  # 提取 R 返回的浮点数
        
        # ---------------------------------------------------------
        # mclust 不包含以下三个指标，因此保留 sklearn 计算逻辑
        # ---------------------------------------------------------
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        hom = homogeneity_score(y_true, y_pred)

        # 计算聚类标签的空间连贯性 (Cluster Moran's I)
        cluster_moran = None
        if coords is not None:
            pred_codes = pd.Categorical(y_pred).codes
            num_clusters = int(np.max(pred_codes)) + 1
            one_hot_clusters = np.eye(num_clusters, dtype=np.float32)[pred_codes]
            cluster_moran, _ = calculate_spatial_morans_i(coords, one_hot_clusters, k=moran_k)
        
        # 5. 格式化打印输出
        print(f"\n=== 聚类评估结果 ===")
        print(f"真实标签列: {true_col}")
        print(f"预测标签列: {pred_col}")
        print("-" * 20)
        print(f"Adjusted Rand Index (ARI) [mclust] : {ari:.4f}")
        print(f"Normalized Mutual Info (NMI)       : {nmi:.4f}")
        print(f"Adjusted Mutual Info (AMI)         : {ami:.4f}")
        print(f"Homogeneity (HOM)                  : {hom:.4f}")
        if cluster_moran is not None:
            print(f"Cluster Moran's I                  : {cluster_moran:.4f}")
        else:
            print("Cluster Moran's I                  : NA (未找到 adata.obsm['spatial'])")
        print("====================")
        
    except Exception as e:
        print(f"发生异常: {e}")
        print("\n【排错提示】: 如果出现 rpy2 或 mclust 相关的报错，请确保您的容器环境中：")
        print("1. 已经安装了 Python 包: pip install rpy2")
        print("2. 已经安装了 R 语言基础环境 (apt-get install r-base)")
        print("3. 已经在 R 环境中安装了 mclust 包 (R -e \"install.packages('mclust', repos='http://cran.us.r-project.org')\")")
        print("4. 如需 Moran's I，请安装: pip install squidpy esda libpysal")

# 执行计算
calculate_clustering_metrics_mclust(
    h5ad_path="spatialDDM_results_p22.h5ad", 
     #true_col="Combined_Clusters_annotation",
    true_col="Combined_Clusters", 
    pred_col="SpatialDDM"
)