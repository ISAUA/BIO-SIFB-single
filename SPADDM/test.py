import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

# 引入 rpy2 用于在 Python 中无缝调用 R 语言代码
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr


def _sanitize_thread_env_vars():
    """Prevent libgomp warnings from invalid thread env values."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        try:
            valid = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[key] = "1"


_sanitize_thread_env_vars()

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")
warnings.filterwarnings("ignore", message="In the future, the default backend for leiden will be igraph instead of leidenalg.*", category=FutureWarning)

import scanpy as sc

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


def _to_dense_matrix(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _resolve_save_paths(h5ad_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(h5ad_path))

    save_dir = os.path.abspath(save_dir)
    if os.path.basename(save_dir.rstrip("/\\")) == "checkpoints":
        base_dir = os.path.dirname(save_dir.rstrip("/\\"))
    else:
        base_dir = save_dir

    fig_dir = os.path.join(base_dir, "figures")
    pred_dir = os.path.join(base_dir, "predictions")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    return fig_dir, pred_dir


def _should_invert_spatial_axis(adata, save_dir=None):
    if 'slices_path' in adata.obs.columns:
        slices_path = adata.obs['slices_path'].astype(str).str.lower()
        if slices_path.str.contains('misar').any():
            return True
    if save_dir is not None and 'misar' in save_dir.lower():
        return True
    return False

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


def run_mclust_and_plot(
    h5ad_path,
    n_clusters=18,
    pca_dim=20,
    cluster_col="cluster",
    save_dir=None,
    seed=42,
    moran_k=6,
):
    """
    优先复用 h5ad 内已有的 SpatialDDM embedding / labels。
    如果不存在，再回退到对 AnnData.X 重新做 PCA + mclust。
    """
    print(f"正在加载文件用于 mclust 可视化: {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)

    use_existing_embedding = 'SpatialDDM' in adata.obsm and 'SpatialDDM' in adata.obs.columns
    if use_existing_embedding:
        embedding = np.asarray(adata.obsm['SpatialDDM'], dtype=np.float32)
        labels = adata.obs['SpatialDDM'].astype(str).to_numpy()
        adata.obs[cluster_col] = pd.Categorical(labels)
        if embedding.ndim != 2:
            raise ValueError(f"adata.obsm['SpatialDDM'] 维度异常: {embedding.shape}")
        print("检测到 h5ad 内已有 SpatialDDM embedding/labels，直接复用以复现来源方结果。")
    else:
        x = _to_dense_matrix(adata.X).astype(np.float32, copy=False)
        if x.ndim != 2:
            raise ValueError(f"adata.X 维度异常: {x.shape}")

        # mclust 前先做 PCA（与项目评估脚本一致）
        pca_dim = int(max(1, min(int(pca_dim), x.shape[0], x.shape[1])))
        z_pca = PCA(n_components=pca_dim, random_state=int(seed)).fit_transform(x)

        print("正在调用 R mclust 进行聚类...")
        rpy2.robjects.numpy2ri.activate()
        mclust = importr('mclust')
        r_z_pca = rpy2.robjects.numpy2ri.numpy2rpy(z_pca)
        res = mclust.Mclust(r_z_pca, int(n_clusters), 'EEE', verbose=False)
        labels = np.array(res.rx2('classification')).astype(int)

        adata.obs[cluster_col] = labels.astype(int).astype(str)
        cluster_list = sorted(pd.unique(adata.obs[cluster_col]), key=lambda value: int(value))
        adata.obs[cluster_col] = pd.Categorical(adata.obs[cluster_col], categories=cluster_list, ordered=True)
        embedding = z_pca

        # 计算 UMAP（用于聚类图）
        sc.pp.neighbors(adata, use_rep='X', random_state=int(seed))
        sc.tl.umap(adata, random_state=int(seed))

    if use_existing_embedding:
        sc.pp.neighbors(adata, use_rep='SpatialDDM', random_state=int(seed))
        sc.tl.umap(adata, random_state=int(seed))

    fig_dir, pred_dir = _resolve_save_paths(h5ad_path, save_dir=save_dir)
    out_h5ad = os.path.join(pred_dir, "embedding_joint.h5ad")
    plot_path = os.path.join(fig_dir, "spatial_analysis.pdf")

    adata.write(out_h5ad)

    cluster_moran = None
    if 'spatial' in adata.obsm:
        cluster_codes = adata.obs[cluster_col].astype('category').cat.codes.values
        num_clusters = int(np.max(cluster_codes)) + 1
        one_hot_clusters = np.eye(num_clusters, dtype=np.float32)[cluster_codes]
        cluster_moran, _ = calculate_spatial_morans_i(np.asarray(adata.obsm['spatial']), one_hot_clusters, k=moran_k)

    # 绘制并保存 UMAP + Spatial 双图
    sc.set_figure_params(dpi=180, figsize=(6, 6), frameon=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sc.pl.umap(
        adata,
        color=cluster_col,
        ax=axs[0],
        show=False,
        title='UMAP',
        legend_loc='on data',
        frameon=True,
        size=60,
        alpha=1.0,
        edges=False,
    )

    if 'spatial' in adata.obsm:
        sc.pl.embedding(
            adata,
            basis='spatial',
            color=cluster_col,
            ax=axs[1],
            show=False,
            title='Spatial Map',
            frameon=True,
            size=80,
            alpha=1.0,
            edges=False,
        )
    else:
        axs[1].text(0.5, 0.5, "No adata.obsm['spatial'] found", ha='center', va='center')
        axs[1].set_title('Spatial Map')
        axs[1].set_axis_off()

    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if 'spatial' in adata.obsm and _should_invert_spatial_axis(adata, save_dir=save_dir):
        axs[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Artifact plot: {os.path.abspath(plot_path)}")
    print(f"Artifact h5ad: {os.path.abspath(out_h5ad)}")
    if cluster_moran is not None:
        print(f"Cluster Moran's I                  : {cluster_moran:.4f}")

    return plot_path, out_h5ad


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpatialDDM result with run_evaluate-style outputs")
    parser.add_argument("--h5ad-path", default="/root/autodl-tmp/BIO-SFIB-single/SPADDM/SpatialDDM_results_p22.h5ad", help="Input h5ad file")
    parser.add_argument("--true-col", default="Combined_Clusters", help="Ground truth column name")
    parser.add_argument("--pred-col", default="SpatialDDM", help="Prediction column name for metric reporting")
    parser.add_argument("--n-clusters", type=int, default=18, help="Number of clusters for mclust")
    parser.add_argument("--pca-dim", type=int, default=20, help="PCA dimension before mclust")
    parser.add_argument("--moran-k", type=int, default=6, help="Neighbor count for Moran's I")
    parser.add_argument("--save-dir", default=None, help="Output directory; defaults to the h5ad parent directory")
    parser.add_argument("--cluster-col", default="cluster", help="Saved cluster column name")
    return parser.parse_args()


def main():
    args = parse_args()

    calculate_clustering_metrics_mclust(
        h5ad_path=args.h5ad_path,
        true_col=args.true_col,
        pred_col=args.pred_col,
        moran_k=args.moran_k,
    )

    run_mclust_and_plot(
        h5ad_path=args.h5ad_path,
        n_clusters=args.n_clusters,
        pca_dim=args.pca_dim,
        cluster_col=args.cluster_col,
        save_dir=args.save_dir,
        seed=42,
        moran_k=args.moran_k,
    )

if __name__ == "__main__":
    main()