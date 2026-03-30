import os
import argparse
import logging
import time
import torch
import yaml
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

# 引入预处理模块
from sf_model.preprocess.io import (
    read_mtx_to_adata,
    add_spatial_info,
    read_10x_h5_multiome,
    add_spatial_info_csv,
    read_h5ad_rna_atac,
)
from sf_model.preprocess.rna_process import process_rna_pipeline
from sf_model.preprocess.atac_process import process_atac_pipeline
from sf_model.utils import build_spatial_graph, set_seed

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw spatial multi-omics data")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    return parser.parse_args()


def infer_ground_truth_key(obs_df, preferred_key=None):
    if preferred_key and preferred_key in obs_df.columns:
        return preferred_key

    candidates = [
        'ground_truth', 'groundtruth', 'gt',
        'combined_clusters', 'combined_cluster',
        'combined_clusters_annotation', 'cluster', 'clusters',
        'cell_type', 'celltype', 'annotation', 'annot', 'label', 'labels',
    ]
    lower_to_col = {c.lower(): c for c in obs_df.columns}
    for c in candidates:
        if c in lower_to_col:
            return lower_to_col[c]
    return None


def _resolve_reduce_device(device_pref: str):
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dense_pca_lowrank(X_dense, n_components, seed, device, q_offset=8, niter=3):
    q = int(min(max(1, n_components + int(q_offset)), min(X_dense.shape[0], X_dense.shape[1])))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        x_t = torch.as_tensor(X_dense, dtype=torch.float32, device=device)
        U, S, _ = torch.pca_lowrank(x_t, q=q, center=True, niter=int(niter))
        z_t = U[:, :n_components] * S[:n_components].unsqueeze(0)
        z_np = z_t.detach().cpu().numpy()
        del x_t, U, S, z_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return z_np


def _torch_sparse_svd_lowrank(X_sparse, n_components, seed, device, q_offset=8, niter=3):
    x_coo = X_sparse.tocoo()
    indices = np.vstack((x_coo.row, x_coo.col))
    i_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    v_t = torch.as_tensor(x_coo.data, dtype=torch.float32, device=device)
    x_t = torch.sparse_coo_tensor(i_t, v_t, size=x_coo.shape, device=device).coalesce()

    q = int(min(max(1, n_components + int(q_offset)), min(x_coo.shape[0], x_coo.shape[1])))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    with torch.no_grad():
        U, S, _ = torch.svd_lowrank(x_t, q=q, niter=int(niter))
        z_t = U[:, :n_components] * S[:n_components].unsqueeze(0)
        z_np = z_t.detach().cpu().numpy()
        del i_t, v_t, x_t, U, S, z_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return z_np


def reduce_modality_features(X, n_components, seed, modality_name, reduce_cfg=None):
    """
    使用 PCA/TruncatedSVD 将模态特征降到固定维度。
    - 稀疏输入优先使用 TruncatedSVD，避免转 dense 导致内存开销过大。
    - 稠密输入使用 PCA。
    """
    reduce_cfg = reduce_cfg or {}
    reduce_mode = str(reduce_cfg.get('mode', 'gpu_safe')).lower()  # cpu|gpu_safe|gpu_full
    device_pref = str(reduce_cfg.get('device', 'auto')).lower()    # auto|cuda|cpu
    q_offset = int(reduce_cfg.get('q_offset', 8))
    niter = int(reduce_cfg.get('niter', 3))

    n_components = int(n_components)
    if n_components <= 0:
        raise ValueError(f"{modality_name} n_components must be positive, got {n_components}.")

    n_samples = X.shape[0]
    n_features = X.shape[1]
    max_rank = max(1, min(n_samples, n_features))
    if n_components > max_rank:
        print(
            f"   ⚠️ [{modality_name}] Requested n_components={n_components} exceeds max_rank={max_rank}; "
            f"using {max_rank} instead."
        )
        n_components = max_rank

    device = _resolve_reduce_device(device_pref)
    t0 = time.perf_counter()
    print(
        f"   [{modality_name}] Reducing features to {n_components} dims "
        f"(mode={reduce_mode}, device={device.type})..."
    )

    if sparse.issparse(X):
        # 稀疏模态默认保持 CPU TruncatedSVD，优先稳定性。
        use_sparse_gpu = (reduce_mode == 'gpu_full' and device.type == 'cuda')
        if use_sparse_gpu:
            try:
                Z = _torch_sparse_svd_lowrank(
                    X,
                    n_components=n_components,
                    seed=seed,
                    device=device,
                    q_offset=q_offset,
                    niter=niter,
                )
            except Exception as e:
                print(f"   ⚠️ [{modality_name}] GPU sparse SVD failed, fallback to CPU TruncatedSVD. reason={e}")
                reducer = TruncatedSVD(n_components=n_components, random_state=int(seed))
                Z = reducer.fit_transform(X)
        else:
            reducer = TruncatedSVD(n_components=n_components, random_state=int(seed))
            Z = reducer.fit_transform(X)
    else:
        X_dense = np.asarray(X, dtype=np.float32)
        use_dense_gpu = (reduce_mode in ('gpu_safe', 'gpu_full') and device.type == 'cuda')
        if use_dense_gpu:
            try:
                Z = _torch_dense_pca_lowrank(
                    X_dense,
                    n_components=n_components,
                    seed=seed,
                    device=device,
                    q_offset=q_offset,
                    niter=niter,
                )
            except Exception as e:
                print(f"   ⚠️ [{modality_name}] GPU dense PCA failed, fallback to CPU PCA. reason={e}")
                reducer = PCA(n_components=n_components, random_state=int(seed), svd_solver='auto')
                Z = reducer.fit_transform(X_dense)
        else:
            reducer = PCA(n_components=n_components, random_state=int(seed), svd_solver='auto')
            Z = reducer.fit_transform(X_dense)

    dt = time.perf_counter() - t0
    print(f"   [{modality_name}] Dim reduction done in {dt:.2f}s")

    return np.asarray(Z, dtype=np.float32)


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFPreprocess")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def append_log_separator(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")

def main():
    print("🚀 [Phase 1] Starting Data Preprocessing...")
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)
    logger.info("[Phase 1] Starting Data Preprocessing...")

    raw_dir = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    files = config['data']['files']
    params = config['data']['parameters']
    rna_min_cells = params.get('rna_min_cells', 3)
    rna_target_sum = params.get('rna_target_sum', 1e4)
    atac_min_cells = params.get('atac_min_cells', 50)
    atac_target_sum = params.get('atac_target_sum', 1e4)
    tfidf_eps = float(params.get('tfidf_eps', 1e-6))
    seed = int(config['project'].get('seed', 42))
    reduce_cfg = params.get('reduce', {})
    
    os.makedirs(processed_dir, exist_ok=True)

    # ==========================================
    # Step 1: 加载原始数据
    # ==========================================
    print("\n📦 Loading Raw Data...")
    logger.info("Loading raw data...")

    # 自动根据配置键选择预处理读取路径（优先级）：
    # 1) RNA+ATAC 双 h5ad
    # 2) MISAR: 10x h5 + spatial csv
    # 3) 旧版 mtx 输入
    default_rna_h5ad = os.path.join(raw_dir, "adata_RNA.h5ad")
    default_atac_h5ad = os.path.join(raw_dir, "adata_Peak.h5ad")
    has_pair_h5ad = (
        ('rna_h5ad' in files and 'atac_h5ad' in files) or
        (os.path.exists(default_rna_h5ad) and os.path.exists(default_atac_h5ad))
    )

    if has_pair_h5ad:
        logger.info("Data loader: paired h5ad")
        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        atac_h5ad = os.path.join(raw_dir, files.get('atac_h5ad', 'adata_Peak.h5ad'))
        adata_rna, adata_atac = read_h5ad_rna_atac(rna_h5ad, atac_h5ad)
    elif 'h5_matrix' in files and 'spatial_csv' in files:
        logger.info("Data loader: 10x h5 + spatial csv")
        print(f"   -> Reading RNA+ATAC from {files['h5_matrix']}...")
        adata_rna, adata_atac = read_10x_h5_multiome(
            os.path.join(raw_dir, files['h5_matrix'])
        )
        adata_rna = add_spatial_info_csv(
            adata_rna,
            os.path.join(raw_dir, files['spatial_csv'])
        )
    else:
        logger.info("Data loader: mtx/tsv/csv split files")
        print(f"   -> Reading RNA from {files['rna_mtx']}...")
        adata_rna = read_mtx_to_adata(
            os.path.join(raw_dir, files['rna_mtx']),
            os.path.join(raw_dir, files['rna_genes']),
            os.path.join(raw_dir, files['rna_barcodes'])
        )
        adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))

        print(f"   -> Reading ATAC from {files['atac_mtx']}...")
        adata_atac = read_mtx_to_adata(
            os.path.join(raw_dir, files['atac_mtx']),
            os.path.join(raw_dir, files['atac_peaks']),
            os.path.join(raw_dir, files['atac_barcodes'])
        )

    # 模态对齐：将 RNA 细胞顺序应用到 ATAC，并同步空间坐标
    adata_atac = adata_atac[adata_rna.obs_names, :].copy()
    adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    # ==========================================
    # Step 2: 严格对齐检查 (不取交集，仅验证)
    # ==========================================
    print("\n🔍 Verifying One-to-One Alignment...")
    n_rna = adata_rna.shape[0]
    n_atac = adata_atac.shape[0]
    
    # 1. 检查数量
    if n_rna != n_atac:
        raise ValueError(f"❌ Mismatch! RNA cells ({n_rna}) != ATAC cells ({n_atac}). Please check raw data.")
    
    # 2. 检查 Barcode 顺序 (简单抽查前5个和后5个)
    if not np.array_equal(adata_rna.obs_names[:5], adata_atac.obs_names[:5]):
        print("⚠️ Warning: Barcode order might mismatch in the first 5 cells!")
        # 如果您非常确定是对应的，可以忽略这个警告，或者在这里强制赋值索引
        # adata_atac.obs_names = adata_rna.obs_names 
    else:
        print("   ✅ Cell count and order look correct.")

    # ==========================================
    # Step 3: 执行预处理管线
    # ==========================================
    
    # --- RNA ---
    # 预防性措施：在传入 pipeline 之前，确保它是 float32
    # 这样即使不修改 rna_process.py，也能避免 normalize_total 报错
    if hasattr(adata_rna.X, "astype"):
        adata_rna.X = adata_rna.X.astype(np.float32)
        
    print("\n🧪 Processing RNA...")
    adata_rna = process_rna_pipeline(
        adata_rna,
        n_top_genes=params['n_top_genes'],
        min_cells=rna_min_cells,
        target_sum=rna_target_sum,
    )

    # --- ATAC ---
    print("\n🧪 Processing ATAC...")
    gtf_path = os.path.join(raw_dir, files['gtf'])
    rna_genes = adata_rna.var_names.tolist()
    
    adata_atac, _ = process_atac_pipeline(
        adata_atac, 
        rna_genes=rna_genes, 
        gtf_path=gtf_path,
        n_global=params['n_global_peaks'],
        n_final=params['n_final_peaks'],
        window=params['tss_window'],
        min_cells=atac_min_cells,
        target_sum=atac_target_sum,
        tfidf_eps=tfidf_eps,
    )

    # 预处理后再做一次强制对齐：按 RNA 条码对 ATAC 取子集并重排。
    # 这一步可兜底处理任一模态在上游发生了细胞过滤的情况。
    if adata_rna.n_obs != adata_atac.n_obs or not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
        if len(common_cells) == 0:
            raise ValueError("❌ No overlapping cells between RNA and ATAC after preprocessing.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_atac.n_obs:
            print(
                f"⚠️ Post-process alignment: RNA={adata_rna.n_obs}, ATAC={adata_atac.n_obs}, "
                f"keeping intersection={len(common_cells)}"
            )

        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    print(f"   ✅ Aligned final cells: RNA={adata_rna.n_obs}, ATAC={adata_atac.n_obs}")
    logger.info("Aligned final cells: RNA=%d, ATAC=%d", adata_rna.n_obs, adata_atac.n_obs)

    # ==========================================
    # Step 3.5: 模态 PCA/SVD 降维（统一入模特征维度）
    # ==========================================
    rna_pca_dim = int(params.get('rna_pca_dim', 512))
    atac_pca_dim = int(params.get('atac_pca_dim', 512))

    print("\n📉 Reducing RNA/ATAC features before model input...")
    rna_feat_np = reduce_modality_features(adata_rna.X, rna_pca_dim, seed, "RNA", reduce_cfg=reduce_cfg)
    atac_feat_np = reduce_modality_features(adata_atac.X, atac_pca_dim, seed, "ATAC", reduce_cfg=reduce_cfg)
    print(f"   ✅ Reduced RNA shape: {rna_feat_np.shape}")
    print(f"   ✅ Reduced ATAC shape: {atac_feat_np.shape}")
    logger.info("Reduced shapes: RNA=%s, ATAC=%s", rna_feat_np.shape, atac_feat_np.shape)

    # # ==========================================
    # # Step 4: 构建空间图 & 准备 Tensor
    # # ==========================================
    # print("\n🕸️ Building Spatial Graph & GFT Basis...")
    # coords = adata_rna.obsm['spatial']
    
    # # 计算图基底 (GFT Basis)
    # edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    # # 转换为 Tensor
    # def to_tensor(adata):
    #     if hasattr(adata.X, 'toarray'):
    #         return torch.FloatTensor(adata.X.toarray())
    #     return torch.FloatTensor(adata.X)

    # rna_feat = to_tensor(adata_rna)
    # atac_feat = to_tensor(adata_atac)
    # coords_tensor = torch.FloatTensor(coords)

    # ==========================================
    # Step 4: 构建空间图 & 准备 Tensor  #根据knn引入权重
    # ==========================================
    print("\n🕸️ Building Spatial Graph & GFT Basis with Feature Weights...")
    coords = adata_rna.obsm['spatial']
    # 默认：使用降维后的 RNA 特征构建加权图
    rna_features = rna_feat_np
    graph_outputs = build_spatial_graph(coords, features=rna_features, k=params['knn_k'])
    if len(graph_outputs) == 3:
        edge_index, u_basis, evals = graph_outputs
    else:
        edge_index, u_basis = graph_outputs
        evals = None
    # 备选：使用空间距离高斯衰减
    # edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    rna_feat = torch.FloatTensor(rna_feat_np)
    atac_feat = torch.FloatTensor(atac_feat_np)
    coords_tensor = torch.FloatTensor(coords)

    gt_key = infer_ground_truth_key(adata_rna.obs, preferred_key=params.get('ground_truth_key'))
    ground_truth = None
    if gt_key is not None:
        gt_vals = adata_rna.obs[gt_key].astype(str).to_numpy()
        valid_mask = np.array([
            (v is not None) and (str(v).strip() != "") and (str(v).lower() != "nan")
            for v in gt_vals
        ])
        if valid_mask.sum() > 0:
            ground_truth = gt_vals
            print(f"   ✅ Found ground truth in obs column: {gt_key}")
        else:
            print(f"   ⚠️ Ground truth column '{gt_key}' is empty after cleaning; skipping.")
    else:
        print("   ⚠️ No ground truth column found in RNA obs.")

    # ==========================================
    # Step 5: 保存处理好的数据
    # ==========================================
    save_path = os.path.join(processed_dir, "processed_data.pt")
    print(f"\n💾 Saving processed tensors to {save_path}...")
    logger.info("Saving processed tensors...")
    
    data_dict = {
        "rna_feat": rna_feat,
        "atac_feat": atac_feat,
        "coords": coords_tensor,
        "edge_index": edge_index,
        "u_basis": u_basis,
        "rna_dim": rna_feat.shape[1],
        "atac_dim": atac_feat.shape[1] # 记录动态 ATAC 维度
    }

    if evals is not None:
        data_dict["evals"] = evals

    if ground_truth is not None:
        data_dict["ground_truth"] = ground_truth
        data_dict["ground_truth_key"] = gt_key
    
    torch.save(data_dict, save_path)
    print("✅ Preprocessing Complete!")
    logger.info("Preprocessing complete: %s", save_path)
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()