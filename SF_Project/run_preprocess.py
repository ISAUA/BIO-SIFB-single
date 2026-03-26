import os
import argparse
import torch
import yaml
import numpy as np
import scanpy as sc

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

def main():
    print("🚀 [Phase 1] Starting Data Preprocessing...")
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    raw_dir = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    files = config['data']['files']
    params = config['data']['parameters']
    rna_min_cells = params.get('rna_min_cells', 3)
    rna_target_sum = params.get('rna_target_sum', 1e4)
    atac_min_cells = params.get('atac_min_cells', 50)
    atac_target_sum = params.get('atac_target_sum', 1e4)
    tfidf_eps = float(params.get('tfidf_eps', 1e-6))
    
    os.makedirs(processed_dir, exist_ok=True)

    # ==========================================
    # Step 1: 加载原始数据
    # ==========================================
    print("\n📦 Loading Raw Data...")

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
        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        atac_h5ad = os.path.join(raw_dir, files.get('atac_h5ad', 'adata_Peak.h5ad'))
        adata_rna, adata_atac = read_h5ad_rna_atac(rna_h5ad, atac_h5ad)
    elif 'h5_matrix' in files and 'spatial_csv' in files:
        print(f"   -> Reading RNA+ATAC from {files['h5_matrix']}...")
        adata_rna, adata_atac = read_10x_h5_multiome(
            os.path.join(raw_dir, files['h5_matrix'])
        )
        adata_rna = add_spatial_info_csv(
            adata_rna,
            os.path.join(raw_dir, files['spatial_csv'])
        )
    else:
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
    # 默认：RNA 特征欧氏距离权重；若需空间距离衰减，注释下行并启用备选行
    rna_features = adata_rna.X
    graph_outputs = build_spatial_graph(coords, features=rna_features, k=params['knn_k'])
    if len(graph_outputs) == 3:
        edge_index, u_basis, evals = graph_outputs
    else:
        edge_index, u_basis = graph_outputs
        evals = None
    # 备选：使用空间距离高斯衰减
    # edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    # 转换为 Tensor (保持原有逻辑)
    def to_tensor(adata):
        if hasattr(adata.X, 'toarray'):
            return torch.FloatTensor(adata.X.toarray())
        return torch.FloatTensor(adata.X)

    # 【核心修复】：这三行是将数据真正转为模型输入的关键，必须保留
    rna_feat = to_tensor(adata_rna)
    atac_feat = to_tensor(adata_atac)
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
    
    data_dict = {
        "rna_feat": rna_feat,
        "atac_feat": atac_feat,
        "coords": coords_tensor,
        "edge_index": edge_index,
        "u_basis": u_basis,
        "atac_dim": atac_feat.shape[1] # 记录动态 ATAC 维度
    }

    if evals is not None:
        data_dict["evals"] = evals

    if ground_truth is not None:
        data_dict["ground_truth"] = ground_truth
        data_dict["ground_truth_key"] = gt_key
    
    torch.save(data_dict, save_path)
    print("✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()