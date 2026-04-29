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


def _compute_morans_i(X, coords, method, k, radius, permutations):
    try:
        import squidpy as sq
    except ImportError as exc:
        raise ImportError("squidpy is required for Moran's I. Install squidpy.") from exc

    permutations = int(permutations)
    coords = np.asarray(coords, dtype=np.float32)
    method = str(method).lower()

    adata_moran = sc.AnnData(X=X)
    adata_moran.obsm["spatial"] = coords

    neighbor_kwargs = {"coord_type": "generic"}
    if method == "knn":
        neighbor_kwargs["n_neighs"] = int(k)
    elif method in ("radius", "distance"):
        if radius is None:
            raise ValueError("radius must be provided when spatial_graph_method='radius'.")
        neighbor_kwargs["radius"] = float(radius)
    else:
        raise ValueError(f"Unsupported spatial_graph_method: {method}")

    try:
        sq.gr.spatial_neighbors(adata_moran, **neighbor_kwargs)
    except TypeError as exc:
        raise ValueError(
            "squidpy.spatial_neighbors does not support the requested neighbor configuration."
        ) from exc

    sq.gr.spatial_autocorr(
        adata_moran,
        mode="moran",
        n_perms=None if permutations <= 0 else int(permutations),
        show_progress_bar=False,
    )

    moran_df = adata_moran.uns["moranI"]
    moran_i = moran_df["I"].to_numpy(dtype=np.float32)
    if permutations > 0 and "pval_sim" in moran_df:
        p_vals = moran_df["pval_sim"].to_numpy(dtype=np.float32)
    elif "pval_norm" in moran_df:
        p_vals = moran_df["pval_norm"].to_numpy(dtype=np.float32)
    elif "pval" in moran_df:
        p_vals = moran_df["pval"].to_numpy(dtype=np.float32)
    else:
        p_vals = np.ones_like(moran_i, dtype=np.float32)

    return moran_i, p_vals


def process_sm_pipeline(
    adata,
    apply_log1p=True,
    spatial_hvg_method="morans_i",
    n_top_metabolites=500,
    apply_scale=True,
    pval_threshold=0.05,
    drop_dead_spots=False,
    spatial_graph_method="knn",
    moran_k=6,
    radius=None,
    moran_permutations=0,
):
    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial']; required for Moran's I.")

    if drop_dead_spots:
        adata = _drop_dead_spots(adata)

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    if apply_log1p:
        sc.pp.log1p(adata)

    if str(spatial_hvg_method).lower() != "morans_i":
        raise ValueError(f"Unsupported spatial_hvg_method: {spatial_hvg_method}")

    if adata.shape[1] == 0:
        return adata

    X_dense = _to_dense_float32(adata.X)
    X_dense = np.nan_to_num(X_dense, copy=False)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    moran_i, p_vals = _compute_morans_i(
        X_dense,
        coords,
        spatial_graph_method,
        moran_k,
        radius,
        moran_permutations,
    )

    sig_mask = p_vals <= float(pval_threshold)
    if not np.any(sig_mask):
        sig_mask = np.ones_like(p_vals, dtype=bool)

    candidates = np.flatnonzero(sig_mask)
    ranked = candidates[np.argsort(moran_i[candidates])[::-1]]
    if n_top_metabolites is not None:
        n_keep = min(int(n_top_metabolites), ranked.size)
        ranked = ranked[:n_keep]

    if ranked.size == 0:
        adata = adata[:, []].copy()
        adata.X = np.zeros((adata.n_obs, 0), dtype=np.float32)
        return adata

    adata = adata[:, ranked].copy()
    adata.var["morans_i"] = moran_i[ranked]
    adata.var["morans_pval"] = p_vals[ranked]

    if apply_scale:
        X_scaled = _to_dense_float32(adata.X)
        X_scaled = np.nan_to_num(X_scaled, copy=False)
        mean = X_scaled.mean(axis=0, keepdims=True)
        std = X_scaled.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X_scaled = (X_scaled - mean) / std
        adata.X = X_scaled.astype(np.float32)
    else:
        adata.X = _to_dense_float32(adata.X)

    return adata
