import os
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp


def resolve_adt_path(adt_data_path, raw_dir=None):
    if adt_data_path is None:
        return None
    path = os.path.expanduser(str(adt_data_path))
    if os.path.isabs(path):
        return path
    if raw_dir:
        candidate = os.path.join(raw_dir, path)
        if os.path.exists(candidate):
            return candidate
    return path


def read_adt_h5ad(adt_data_path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
            category=UserWarning,
        )
        adata = sc.read_h5ad(adt_data_path)

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _assert_finite(X, stage):
    if not np.isfinite(X).all():
        raise ValueError(f"ADT matrix contains NaN/Inf after {stage}.")


def process_adt_pipeline(adata, apply_clr=False, apply_scale=False):
    if adata.shape[1] == 0:
        return adata

    adata.X = _to_dense_float32(adata.X)
    _assert_finite(adata.X, "loading")

    if apply_clr:
        # CLR is optional; only run when explicitly enabled.
        X = np.log1p(adata.X)
        X = X - X.mean(axis=1, keepdims=True)
        adata.X = X.astype(np.float32)
        _assert_finite(adata.X, "CLR")

    if apply_scale:
        sc.pp.scale(adata, zero_center=True)
        adata.X = np.asarray(adata.X, dtype=np.float32)
        _assert_finite(adata.X, "scale")

    return adata
