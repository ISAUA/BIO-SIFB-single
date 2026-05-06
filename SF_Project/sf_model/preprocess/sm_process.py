import os
import warnings
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def resolve_sm_path(sm_data_path, raw_dir=None):
    if sm_data_path is None:
        return None
    path = os.path.expanduser(str(sm_data_path))
    if os.path.isabs(path):
        return path
    if raw_dir:
        candidate = os.path.join(raw_dir, path)
        if os.path.exists(candidate):
            return candidate
    return path


def read_sm_h5ad(sm_data_path):
    """Read SM AnnData and normalize identifiers."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
            category=UserWarning,
        )
        adata = sc.read_h5ad(sm_data_path)

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def _drop_dead_spots(adata):
    if sp.issparse(adata.X):
        sums = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        sums = np.asarray(adata.X, dtype=np.float32).sum(axis=1)
    keep_mask = sums > 0
    if np.all(keep_mask):
        return adata
    return adata[keep_mask, :].copy()


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _sanitize_matrix(X):
    if sp.issparse(X):
        X = X.tocsr(copy=False)
        data = X.data
        data[~np.isfinite(data)] = 0.0
        X.data = data
        return X
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, copy=False)


def process_sm_pipeline(
    adata,
    target_sum=1e4,
    apply_log1p=True,
    spatial_hvg_method="morans_i",
    n_top_metabolites=500,
    apply_scale=False,
    pval_threshold=0.05,
    drop_dead_spots=False,
    spatial_graph_method="knn",
    moran_k=6,
    radius=None,
    moran_permutations=0,
):
    if drop_dead_spots:
        adata = _drop_dead_spots(adata)

    if adata.shape[1] == 0:
        return adata

    if spatial_hvg_method not in (None, "hvg", "highly_variable", "morans_i"):
        warnings.warn(
            f"spatial_hvg_method='{spatial_hvg_method}' ignored; using HVG selection.",
            RuntimeWarning,
        )

    # Filter rare metabolites before HVG selection.
    sc.pp.filter_genes(adata, min_cells=3)
    if adata.shape[1] == 0:
        return adata

    # HVG selection on raw counts (before normalize/log1p).
    if n_top_metabolites is not None:
        try:
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=int(n_top_metabolites),
                subset=True,
            )
        except Exception as exc:
            warnings.warn(
                f"seurat_v3 HVG failed ({exc}); falling back to cell_ranger.",
                RuntimeWarning,
            )
            sc.pp.highly_variable_genes(
                adata,
                flavor="cell_ranger",
                n_top_genes=int(n_top_metabolites),
                subset=True,
            )

    if adata.shape[1] == 0:
        return adata

    adata.X = _sanitize_matrix(adata.X)
    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    if apply_log1p:
        sc.pp.log1p(adata)

    if apply_scale:
        warnings.warn("apply_scale is disabled for SM; skipping Z-score.", RuntimeWarning)

    adata.X = _to_dense_float32(adata.X)

    return adata
