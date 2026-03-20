import pandas as pd
import anndata as ad
from scipy import io, sparse
import numpy as np
import os
import scanpy as sc

def read_mtx_to_adata(mtx_path, features_path, barcodes_path, transpose=True):
    """
    读取 MTX 文件并构建 AnnData (逻辑源自 prepare_adata.py)
    """
    print(f"Reading data from: {mtx_path}")
    
    # 1. 读取矩阵 (prepare_adata.py logic)
    mat = io.mmread(mtx_path)
    
    # 自动处理转置 (Your script: rna_mat = rna_mat.transpose())
    if transpose:
        mat = mat.transpose()
    
    # 转为 CSR，并强制为 float32 以避免后续 normalize_total 的 dtype 错误
    mat = sparse.csr_matrix(mat, dtype=np.float32)
    
    # 2. 读取元数据
    # header=None 对应您脚本中的 pd.read_csv(..., header=None)
    features = pd.read_csv(features_path, header=None, sep='\t')
    barcodes = pd.read_csv(barcodes_path, header=None, sep=',') # 注意您脚本中barcodes用的是逗号分隔
    
    # 3. 构建 AnnData
    adata = ad.AnnData(mat)
    adata.obs_names = barcodes.iloc[:, 0].astype(str)
    adata.var_names = features.iloc[:, 0].astype(str)
    
    # 确保唯一性 (Scanpy 常用操作)
    adata.var_names_make_unique()
    
    return adata

def add_spatial_info(adata, spatial_path):
    """
    添加空间坐标 (逻辑源自 prepare_adata.py)
    """
    # Your script: spatial = pd.read_csv("position.tsv", sep=',', index_col=0)
    spatial_df = pd.read_csv(spatial_path, sep=',', index_col=0)
    
    # 取交集 (防止报错)
    common_cells = adata.obs_names.intersection(spatial_df.index)
    if len(common_cells) < len(adata):
        print(f"Warning: Only {len(common_cells)} cells have spatial coords.")
        adata = adata[common_cells, :].copy()
    
    # Your script: coor_df = spatial.loc[..., ['imagecol','imagerow']]
    coords = spatial_df.loc[adata.obs_names, ['imagecol', 'imagerow']].values
    adata.obsm['spatial'] = coords
    
    return adata


def read_10x_h5_multiome(h5_path):
    """
    读取 10x multiome h5，并按 feature_types 拆分 RNA/ATAC。
    """
    print(f"Reading 10x multiome h5 from: {h5_path}")
    adata = sc.read_10x_h5(h5_path, gex_only=False)

    if 'feature_types' not in adata.var.columns:
        raise ValueError("Missing 'feature_types' in h5 var metadata.")

    feature_types = adata.var['feature_types'].astype(str)
    rna_mask = feature_types == 'Gene Expression'
    atac_mask = feature_types == 'Peaks'

    if rna_mask.sum() == 0:
        raise ValueError("No 'Gene Expression' features found in h5 file.")
    if atac_mask.sum() == 0:
        raise ValueError("No 'Peaks' features found in h5 file.")

    adata_rna = adata[:, rna_mask].copy()
    adata_atac = adata[:, atac_mask].copy()

    # 统一为 CSR + float32，兼容后续归一化与稀疏运算。
    adata_rna.X = sparse.csr_matrix(adata_rna.X, dtype=np.float32)
    adata_atac.X = sparse.csr_matrix(adata_atac.X, dtype=np.float32)

    adata_rna.obs_names = adata_rna.obs_names.astype(str)
    adata_rna.var_names = adata_rna.var_names.astype(str)
    adata_atac.obs_names = adata_atac.obs_names.astype(str)
    adata_atac.var_names = adata_atac.var_names.astype(str)

    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    return adata_rna, adata_atac


def add_spatial_info_csv(adata, spatial_path):
    """
    从 CSV 添加空间坐标，并兼容 barcode 是否带 '-1' 后缀。
    """
    spatial_df = pd.read_csv(spatial_path)
    if spatial_df.shape[1] < 2:
        raise ValueError("Spatial CSV must contain barcode and coordinate columns.")

    barcode_col = spatial_df.columns[0]
    spatial_df[barcode_col] = spatial_df[barcode_col].astype(str)
    spatial_df = spatial_df.set_index(barcode_col)

    lower_to_col = {c.lower(): c for c in spatial_df.columns}
    x_col = None
    y_col = None

    x_candidates = [
        'imagecol', 'array_col', 'arraycol', 'x',
        'pxl_col_in_fullres', 'col', 'coord_x'
    ]
    y_candidates = [
        'imagerow', 'array_row', 'arrayrow', 'y',
        'pxl_row_in_fullres', 'row', 'coord_y'
    ]

    for c in x_candidates:
        if c in lower_to_col:
            x_col = lower_to_col[c]
            break
    for c in y_candidates:
        if c in lower_to_col:
            y_col = lower_to_col[c]
            break

    if x_col is None or y_col is None:
        raise ValueError("Unable to infer spatial coordinate columns from CSV.")

    obs_names = adata.obs_names.astype(str)
    obs_has_suffix = obs_names.str.endswith('-1').all()
    csv_has_suffix = spatial_df.index.str.endswith('-1').all()

    if obs_has_suffix and not csv_has_suffix:
        obs_base = pd.Index(obs_names.str.replace(r'-1$', '', regex=True))
        common_base = obs_base.intersection(spatial_df.index)
        if len(common_base) < len(obs_names):
            print(f"Warning: Only {len(common_base)} cells have spatial coords.")
        keep_mask = obs_base.isin(common_base)
        adata = adata[keep_mask, :].copy()
        obs_base_kept = obs_base[keep_mask]
        coords = spatial_df.loc[obs_base_kept, [x_col, y_col]].to_numpy(dtype=np.float32)
        adata.obsm['spatial'] = coords
        return adata

    common_cells = obs_names.intersection(spatial_df.index)
    if len(common_cells) < len(obs_names):
        print(f"Warning: Only {len(common_cells)} cells have spatial coords.")
    adata = adata[common_cells, :].copy()
    coords = spatial_df.loc[adata.obs_names, [x_col, y_col]].to_numpy(dtype=np.float32)
    adata.obsm['spatial'] = coords

    return adata