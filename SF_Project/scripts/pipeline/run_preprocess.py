import os


def _sanitize_thread_env():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = os.environ.get(key)
        if val is None:
            os.environ[key] = "1"
            continue
        val = val.strip()
        if not val.isdigit() or int(val) <= 0:
            os.environ[key] = "1"


_sanitize_thread_env()

import argparse
import logging
import time
import warnings
import torch
import yaml
import numpy as np
import scanpy as sc
import joblib
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

# 引入预处理模块
from sf_model.preprocess.io import (
    read_mtx_to_adata,
    add_spatial_info,
    read_10x_h5_multiome,
    add_spatial_info_csv,
    read_h5ad_rna_atac,
    _ensure_spatial_from_obs,
)
from sf_model.preprocess.rna_process import process_rna_pipeline
from sf_model.preprocess.atac_process import process_atac_pipeline, custom_tf_idf
from sf_model.preprocess.adt_process import process_adt_pipeline, read_adt_h5ad, resolve_adt_path
from sf_model.preprocess.sm_process import read_sm_h5ad, process_sm_pipeline, resolve_sm_path
from sf_model.utils import build_spatial_graph, set_seed

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw spatial multi-omics data")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--save-pca-dir",
        default=None,
        help="Optional directory to save fitted modality reducers (S1 mode)",
    )
    parser.add_argument(
        "--load-pca-dir",
        default=None,
        help="Optional directory to load pre-fitted modality reducers and transform only (S2 mode)",
    )
    return parser.parse_args()


def infer_ground_truth_key(obs_df, preferred_key=None):
    if preferred_key and preferred_key in obs_df.columns:
        return preferred_key

    candidates = [
        'ground_truth', 'groundtruth', 'gt', 'true_label',
        'combined_clusters', 'combined_cluster',
        'combined_clusters_annotation', 'cluster', 'clusters',
        'cell_type', 'celltype', 'annotation', 'annot', 'label', 'labels',
    ]
    lower_to_col = {c.lower(): c for c in obs_df.columns}
    for c in candidates:
        if c in lower_to_col:
            return lower_to_col[c]
    return None


def align_adata_features_hard(adata, target_features, modality_name="modality"):
    """
    Hard-align adata features to target_features with strict 1:1 semantics:
    - Keep intersection features
    - Zero-pad missing features
    - Return matrix reordered exactly as target_features
    """
    target_features = [str(x) for x in target_features]
    if len(target_features) == 0:
        raise ValueError(f"[{modality_name}] target_features is empty.")

    target_set = set(target_features)
    current_set = set(adata.var_names.astype(str).tolist())

    present = [f for f in target_features if f in current_set]
    missing = [f for f in target_features if f not in current_set]

    adata_sub = adata[:, present].copy()
    X_sub = adata_sub.X

    if sparse.issparse(X_sub):
        X_sub = X_sub.tocsr().astype(np.float32)
    else:
        X_sub = np.asarray(X_sub, dtype=np.float32)

    if len(missing) > 0:
        X_missing = sparse.csr_matrix((adata_sub.n_obs, len(missing)), dtype=np.float32)
        if sparse.issparse(X_sub):
            X_aligned = sparse.hstack([X_sub, X_missing], format="csr")
        else:
            X_aligned = np.hstack([X_sub, np.zeros((adata_sub.n_obs, len(missing)), dtype=np.float32)])
    else:
        X_aligned = X_sub

    final_var_names = present + missing
    aligned = sc.AnnData(X=X_aligned, obs=adata_sub.obs.copy())
    aligned.var_names = final_var_names

    # Copy spatial coordinates (and any other obsm slots) for downstream graph build.
    for key, val in adata_sub.obsm.items():
        aligned.obsm[key] = val.copy()

    # Enforce exact target order (present+missing already follows target order; keep explicit for safety).
    aligned = aligned[:, target_features].copy()

    print(
        f"   [{modality_name}] Hard-align summary: target={len(target_features)} | "
        f"present={len(present)} | missing(padded)={len(missing)}"
    )
    return aligned


def reduce_modality_features(
    X,
    n_components,
    seed,
    modality_name,
    reduce_cfg=None,
    save_model_path=None,
    load_model_path=None,
):
    """
    使用 PCA/TruncatedSVD 将模态特征降到固定维度。
    - 稀疏输入优先使用 TruncatedSVD，避免转 dense 导致内存开销过大。
    - 稠密输入使用 PCA。
    """
    _ = reduce_cfg

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

    t0 = time.perf_counter()
    print(f"   [{modality_name}] Reducing features to {n_components} dims (deterministic-cpu)...")

    use_load_mode = load_model_path is not None and os.path.exists(load_model_path)

    if use_load_mode:
        print(f"   [{modality_name}] Loading reducer and transform-only: {load_model_path}")
        reducer = joblib.load(load_model_path)
        if sparse.issparse(X):
            Z = reducer.transform(X)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            Z = reducer.transform(X_dense)
    else:
        if load_model_path is not None and not os.path.exists(load_model_path):
            print(
                f"   ⚠️ [{modality_name}] Reducer file not found at {load_model_path}; "
                "falling back to fit_transform."
            )

        if sparse.issparse(X):
            reducer = TruncatedSVD(n_components=n_components, random_state=int(seed))
            Z = reducer.fit_transform(X)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            reducer = PCA(n_components=n_components, random_state=int(seed), svd_solver='auto')
            Z = reducer.fit_transform(X_dense)

        if save_model_path is not None:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            joblib.dump(reducer, save_model_path)
            print(f"   [{modality_name}] Saved reducer: {save_model_path}")

    # if modality_name.strip().lower() == "atac" and isinstance(reducer, TruncatedSVD):
    #     if Z.shape[1] <= 1:
    #         raise ValueError("ATAC SVD output has <=1 component; cannot drop PC1.")
    #     Z = Z[:, 1:]

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
    rna_target_sum = float(params.get('rna_target_sum', 1e4))
    atac_min_cells = params.get('atac_min_cells', 50)
    atac_target_sum = float(params.get('atac_target_sum', 1e4))
    tfidf_eps = float(params.get('tfidf_eps', 1e-6))
    seed = int(config['project'].get('seed', 42))
    reduce_cfg = params.get('reduce', {})
    n_freq_components = params.get('n_freq_components', config.get('model', {}).get('n_freq_components', None))
    use_rna_similarity_edge_weight = bool(params.get('use_rna_similarity_edge_weight', True))
    rna_use_pca = bool(params.get('rna_use_pca', True))
    sm_data_path = config.get('sm_data_path', None)
    sm_pre_cfg = config.get('sm_preprocess', {}) or {}
    sm_replace_atac = bool(sm_pre_cfg.get('replace_atac', False))
    sm_adata = None
    sm_feat_np = None
    adt_pre_cfg = config.get('adt_preprocess', {}) or {}
    use_adt = bool(adt_pre_cfg.get('use_adt', False))
    adt_data_path = adt_pre_cfg.get('adt_data_path', None)
    adt_apply_clr = bool(adt_pre_cfg.get('apply_clr', False))
    adt_apply_scale = bool(adt_pre_cfg.get('apply_scale', False))
    adt_clip_nonnegative = bool(adt_pre_cfg.get('clip_nonnegative', True))
    
    os.makedirs(processed_dir, exist_ok=True)

    save_pca_dir = args.save_pca_dir
    load_pca_dir = args.load_pca_dir
    if save_pca_dir is not None:
        os.makedirs(save_pca_dir, exist_ok=True)
    if load_pca_dir is not None:
        os.makedirs(load_pca_dir, exist_ok=True)

    rna_model_save_path = os.path.join(save_pca_dir, "pca_model_rna.pkl") if save_pca_dir else None
    atac_model_save_path = os.path.join(save_pca_dir, "pca_model_atac.pkl") if save_pca_dir else None
    rna_model_load_path = os.path.join(load_pca_dir, "pca_model_rna.pkl") if load_pca_dir else None
    atac_model_load_path = os.path.join(load_pca_dir, "pca_model_atac.pkl") if load_pca_dir else None

    fit_save_mode = (load_pca_dir is None) and (save_pca_dir is not None)
    load_align_mode = load_pca_dir is not None

    if sm_replace_atac and load_align_mode:
        raise ValueError("SM replace_atac mode is incompatible with --load-pca-dir.")
    if use_adt and sm_replace_atac:
        raise ValueError("ADT mode is incompatible with sm_preprocess.replace_atac.")
    if use_adt and load_align_mode:
        raise ValueError("ADT mode is incompatible with --load-pca-dir (expects ATAC feature alignment).")

    if load_pca_dir is not None:
        logger.info(
            "PCA mode: load-and-transform | dir=%s | rna_model=%s | atac_model=%s",
            load_pca_dir,
            rna_model_load_path,
            atac_model_load_path,
        )
    elif save_pca_dir is not None:
        logger.info(
            "PCA mode: fit-and-save | dir=%s | rna_model=%s | atac_model=%s",
            save_pca_dir,
            rna_model_save_path,
            atac_model_save_path,
        )
    else:
        logger.info("PCA mode: fit-only (no save/load reducer path provided)")

    if fit_save_mode:
        logger.info("Feature mode: fit-and-save vars (S1)")
    elif load_align_mode:
        logger.info("Feature mode: load-and-hard-align vars (S2)")
    else:
        logger.info("Feature mode: standalone preprocess (no cross-domain var alignment)")

    # ==========================================
    # Step 1: 加载原始数据
    # ==========================================
    print("\n📦 Loading Raw Data...")
    logger.info("Loading raw data...")

    # 自动根据配置键选择预处理读取路径（优先级）：
    # 1) RNA+ATAC 双 h5ad
    # 2) MISAR: 10x h5 + spatial csv
    # 3) 旧版 mtx 输入
    if use_adt:
        if not adt_data_path or str(adt_data_path).strip() == "":
            raise ValueError("adt_preprocess.adt_data_path is required when use_adt is true.")

        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        if os.path.exists(rna_h5ad):
            logger.info("Data loader: RNA h5ad + ADT h5ad")
            print(f"   -> Reading RNA from {rna_h5ad}...")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
                    category=UserWarning,
                )
                adata_rna = sc.read_h5ad(rna_h5ad)
            adata_rna.obs_names = adata_rna.obs_names.astype(str)
            adata_rna.var_names = adata_rna.var_names.astype(str)
            adata_rna.var_names_make_unique()
            adata_rna = _ensure_spatial_from_obs(adata_rna)
            if 'spatial' not in adata_rna.obsm:
                raise ValueError("RNA h5ad is missing spatial coordinates (obsm['spatial']).")
        else:
            logger.info("Data loader: RNA mtx + ADT h5ad")
            print(f"   -> Reading RNA from {files['rna_mtx']}...")
            adata_rna = read_mtx_to_adata(
                os.path.join(raw_dir, files['rna_mtx']),
                os.path.join(raw_dir, files['rna_genes']),
                os.path.join(raw_dir, files['rna_barcodes'])
            )
            adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))

        adt_data_path = resolve_adt_path(adt_data_path, raw_dir)
        print(f"   -> Reading ADT from {adt_data_path}...")
        adata_atac = read_adt_h5ad(adt_data_path)

        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping barcodes between RNA and ADT h5ad files.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_atac.n_obs:
            print(f"Warning: Keeping {len(common_cells)} intersected cells from RNA/ADT h5ad files.")

        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    elif sm_replace_atac:
        if not sm_data_path or str(sm_data_path).strip() == "":
            raise ValueError("sm_data_path is required when sm_preprocess.replace_atac is true.")

        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        if not os.path.exists(rna_h5ad):
            raise FileNotFoundError(f"RNA h5ad not found: {rna_h5ad}")

        logger.info("Data loader: RNA h5ad + SM (replace ATAC)")
        print(f"   -> Reading RNA from {rna_h5ad}...")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
                category=UserWarning,
            )
            adata_rna = sc.read_h5ad(rna_h5ad)

        adata_rna.obs_names = adata_rna.obs_names.astype(str)
        adata_rna.var_names = adata_rna.var_names.astype(str)
        adata_rna.var_names_make_unique()
        adata_rna = _ensure_spatial_from_obs(adata_rna)
        if 'spatial' not in adata_rna.obsm:
            raise ValueError("RNA h5ad is missing spatial coordinates (obsm['spatial']).")

        sm_data_path = resolve_sm_path(sm_data_path, raw_dir)
        print(f"   -> Reading SM from {sm_data_path}...")
        sm_adata = read_sm_h5ad(sm_data_path)

        common_cells = adata_rna.obs_names.intersection(sm_adata.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping barcodes between RNA and SM h5ad files.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < sm_adata.n_obs:
            print(f"Warning: Keeping {len(common_cells)} intersected cells from RNA/SM h5ad files.")

        adata_rna = adata_rna[common_cells, :].copy()
        sm_adata = sm_adata[common_cells, :].copy()
        sm_adata = sm_adata[adata_rna.obs_names, :].copy()
        if 'spatial' not in sm_adata.obsm:
            sm_adata.obsm['spatial'] = np.asarray(adata_rna.obsm['spatial'], dtype=np.float32)

        adata_atac = sm_adata
    # NOTE: fallback to standard RNA+ATAC loaders when not replacing ATAC.
    # ---------------------------------------------------------------
    # 1) paired h5ad
    # 2) 10x multiome h5
    # 3) mtx/tsv
    # ---------------------------------------------------------------
    # default_* used for standard loaders only
    default_rna_h5ad = os.path.join(raw_dir, "adata_RNA.h5ad")
    default_atac_h5ad = os.path.join(raw_dir, "adata_ATAC.h5ad")
    legacy_atac_h5ad = os.path.join(raw_dir, "adata_Peak.h5ad")
    has_pair_h5ad = (
        ('rna_h5ad' in files and 'atac_h5ad' in files) or
        (os.path.exists(default_rna_h5ad) and (os.path.exists(default_atac_h5ad) or os.path.exists(legacy_atac_h5ad)))
    )

    if use_adt:
        pass
    elif sm_replace_atac:
        pass
    elif has_pair_h5ad:
        logger.info("Data loader: paired h5ad")
        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        atac_h5ad = os.path.join(
            raw_dir,
            files.get('atac_h5ad', 'adata_ATAC.h5ad' if os.path.exists(default_atac_h5ad) else 'adata_Peak.h5ad')
        )
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

    # 模态对齐：将 RNA 细胞顺序应用到 ATAC/ADT，并同步空间坐标
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
    
    if load_align_mode:
        print("\n🧪 Processing RNA/ATAC in LOAD+ALIGN mode (skip HVG/peak selection)...")
        rna_vars_path = os.path.join(load_pca_dir, "rna_vars.txt")
        atac_vars_path = os.path.join(load_pca_dir, "atac_vars.txt")
        if not os.path.exists(rna_vars_path) or not os.path.exists(atac_vars_path):
            raise FileNotFoundError(
                "LOAD+ALIGN mode requires rna_vars.txt and atac_vars.txt under --load-pca-dir. "
                f"missing: rna={rna_vars_path}, atac={atac_vars_path}"
            )

        src_rna_vars = np.atleast_1d(np.loadtxt(rna_vars_path, dtype=str)).tolist()
        src_atac_vars = np.atleast_1d(np.loadtxt(atac_vars_path, dtype=str)).tolist()
        logger.info(
            "Loaded source feature lists | RNA=%d from %s | ATAC=%d from %s",
            len(src_rna_vars),
            rna_vars_path,
            len(src_atac_vars),
            atac_vars_path,
        )

        # 1) hard-align features to source-domain lists
        adata_rna = align_adata_features_hard(adata_rna, src_rna_vars, modality_name="RNA")
        adata_atac = align_adata_features_hard(adata_atac, src_atac_vars, modality_name="ATAC")

        # 2) manual normalization compensation after skipping default pipelines
        if sparse.issparse(adata_rna.X):
            adata_rna.X = adata_rna.X.tocsr().astype(np.float32)
        else:
            adata_rna.X = np.asarray(adata_rna.X, dtype=np.float32)
        sc.pp.normalize_total(adata_rna, target_sum=float(rna_target_sum))
        sc.pp.log1p(adata_rna)

        adata_atac = custom_tf_idf(adata_atac, eps=tfidf_eps)
        X_atac = adata_atac.X
        if sparse.issparse(X_atac):
            X_atac = X_atac.tocsr().astype(np.float32)
            data = X_atac.data
            data[~np.isfinite(data)] = 0.0
            X_atac.data = data
            adata_atac.X = X_atac
        else:
            X_atac = np.asarray(X_atac, dtype=np.float32)
            X_atac[~np.isfinite(X_atac)] = 0.0
            adata_atac.X = X_atac
        sc.pp.normalize_total(adata_atac, target_sum=float(atac_target_sum))
        sc.pp.log1p(adata_atac)

    else:
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

        # --- ATAC / ADT ---
        if use_adt:
            print("\n🧪 Processing ADT...")
            adata_atac = process_adt_pipeline(
                adata_atac,
                apply_clr=adt_apply_clr,
                apply_scale=adt_apply_scale,
                clip_nonnegative=adt_clip_nonnegative,
            )
        elif sm_replace_atac:
            logger.info("ATAC preprocessing skipped (SM replace_atac enabled)")
        else:
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

            gtf_stats = adata_atac.uns.get("gtf_filter_stats", None)
            if gtf_stats:
                logger.info(
                    "GTF filter stats | gtf=%s | window=%s | rna_genes=%s | matched_genes=%s | "
                    "peaks_before=%s | peaks_valid=%s | peaks_retained=%s | peaks_after_downsample=%s | "
                    "fallback_used=%s | status=%s",
                    gtf_stats.get("gtf_path"),
                    gtf_stats.get("window"),
                    gtf_stats.get("rna_gene_count"),
                    gtf_stats.get("matched_gene_count"),
                    gtf_stats.get("peaks_before"),
                    gtf_stats.get("peaks_valid"),
                    gtf_stats.get("peaks_retained"),
                    gtf_stats.get("peaks_after_downsample"),
                    gtf_stats.get("fallback_used"),
                    gtf_stats.get("status"),
                )

            # S1 fit&save mode: persist selected feature lists for cross-domain hard alignment.
            if fit_save_mode:
                rna_vars_path = os.path.join(save_pca_dir, "rna_vars.txt")
                atac_vars_path = os.path.join(save_pca_dir, "atac_vars.txt")
                np.savetxt(rna_vars_path, np.asarray(adata_rna.var_names, dtype=str), fmt="%s")
                np.savetxt(atac_vars_path, np.asarray(adata_atac.var_names, dtype=str), fmt="%s")
                logger.info(
                    "Saved selected feature lists | RNA=%d -> %s | ATAC=%d -> %s",
                    adata_rna.n_vars,
                    rna_vars_path,
                    adata_atac.n_vars,
                    atac_vars_path,
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
    # Step 3.2: 代谢组 (SM) 预处理（可选）
    # ==========================================
    if sm_data_path:
        if sm_adata is None:
            sm_data_path = resolve_sm_path(sm_data_path, raw_dir)
            print("\n🧪 Processing SM (metabolomics)...")
            logger.info("SM preprocess: %s", sm_data_path)
            sm_adata = read_sm_h5ad(sm_data_path)
        else:
            print("\n🧪 Processing SM (metabolomics)...")
            logger.info("SM preprocess: reuse loaded AnnData")

        sm_use_pca = bool(sm_pre_cfg.get('use_pca', False))

        sm_moran_k_default = config.get('eval', {}).get('moran_k', params.get('knn_k', 6))
        sm_n_top = sm_pre_cfg.get('n_top_metabolites', 500)
        sm_n_top = None if sm_n_top is None else int(sm_n_top)
        sm_pval = sm_pre_cfg.get('pval_threshold', 0.05)
        sm_pval = 0.05 if sm_pval is None else float(sm_pval)
        sm_moran_k = sm_pre_cfg.get('moran_k', sm_moran_k_default)
        sm_moran_k = sm_moran_k_default if sm_moran_k is None else int(sm_moran_k)
        sm_perms = sm_pre_cfg.get('moran_permutations', 0)
        sm_perms = 0 if sm_perms is None else int(sm_perms)
        sm_method = sm_pre_cfg.get('spatial_graph_method', 'knn') or 'knn'
        sm_hvg_method = sm_pre_cfg.get('spatial_hvg_method', 'morans_i') or 'morans_i'
        sm_target_sum = float(sm_pre_cfg.get('target_sum', 1e4))
        sm_adata = process_sm_pipeline(
            sm_adata,
            target_sum=sm_target_sum,
            apply_log1p=bool(sm_pre_cfg.get('apply_log1p', True)),
            spatial_hvg_method=sm_hvg_method,
            n_top_metabolites=sm_n_top,
            apply_scale=bool(sm_pre_cfg.get('apply_scale', True)),
            pval_threshold=sm_pval,
            drop_dead_spots=bool(sm_pre_cfg.get('drop_dead_spots', False)),
            spatial_graph_method=sm_method,
            moran_k=sm_moran_k,
            radius=sm_pre_cfg.get('radius', None),
            moran_permutations=sm_perms,
        )
        if sm_replace_atac:
            if sm_adata.n_obs != adata_rna.n_obs or not np.array_equal(sm_adata.obs_names, adata_rna.obs_names):
                common_cells = adata_rna.obs_names.intersection(sm_adata.obs_names)
                if len(common_cells) == 0:
                    raise ValueError("No overlapping cells between RNA and SM after preprocessing.")
                if len(common_cells) < adata_rna.n_obs or len(common_cells) < sm_adata.n_obs:
                    print(
                        f"⚠️ Post-SM alignment: RNA={adata_rna.n_obs}, SM={sm_adata.n_obs}, "
                        f"keeping intersection={len(common_cells)}"
                    )
                adata_rna = adata_rna[common_cells, :].copy()
                sm_adata = sm_adata[common_cells, :].copy()
                sm_adata = sm_adata[adata_rna.obs_names, :].copy()
        sm_feat_np = np.asarray(sm_adata.X, dtype=np.float32)
        print(f"   ✅ SM processed shape: {sm_feat_np.shape}")
        logger.info("SM processed shape: %s", sm_feat_np.shape)
        if sm_replace_atac:
            adata_atac = sm_adata
    else:
        logger.info("SM preprocess skipped (no sm_data_path).")

    # ==========================================
    # Step 3.5: 模态 PCA/SVD 降维（统一入模特征维度）
    # ==========================================
    rna_pca_dim = int(params.get('rna_pca_dim', 512))
    atac_pca_dim = int(params.get('atac_pca_dim', 512))

    print("\n📉 Reducing RNA/ATAC features before model input...")
    if rna_use_pca:
        rna_feat_np = reduce_modality_features(
            adata_rna.X,
            rna_pca_dim,
            seed,
            "RNA",
            reduce_cfg=reduce_cfg,
            save_model_path=rna_model_save_path,
            load_model_path=rna_model_load_path,
        )
    else:
        print("   [RNA] PCA disabled by config; using full feature matrix.")
        rna_feat_np = np.asarray(adata_rna.X, dtype=np.float32)
        if not np.isfinite(rna_feat_np).all():
            raise ValueError("RNA feature matrix contains NaN/Inf before model input.")
    if use_adt:
        atac_feat_np = np.asarray(adata_atac.X, dtype=np.float32)
        print(f"   ✅ Using ADT features as ATAC input: {atac_feat_np.shape}")
        logger.info("Using ADT features as ATAC input: %s", atac_feat_np.shape)
    elif sm_replace_atac:
        if sm_feat_np is None:
            raise ValueError("SM features are missing; cannot replace ATAC input.")
        if sm_use_pca:
            atac_feat_np = reduce_modality_features(
                sm_feat_np,
                atac_pca_dim,
                seed,
                "ATAC",
                reduce_cfg=reduce_cfg,
                save_model_path=atac_model_save_path,
                load_model_path=atac_model_load_path,
            )
            print(f"   ✅ Using PCA-reduced SM features as ATAC input: {atac_feat_np.shape}")
            logger.info("Using PCA-reduced SM features as ATAC input: %s", atac_feat_np.shape)
        else:
            atac_feat_np = np.asarray(sm_feat_np, dtype=np.float32)
            print(f"   ✅ Using SM features as ATAC input: {atac_feat_np.shape}")
            logger.info("Using SM features as ATAC input: %s", atac_feat_np.shape)
    else:
        atac_feat_np = reduce_modality_features(
            adata_atac.X,
            atac_pca_dim,
            seed,
            "ATAC",
            reduce_cfg=reduce_cfg,
            save_model_path=atac_model_save_path,
            load_model_path=atac_model_load_path,
        )

        # --- 请加上这段代码，切除被深度绑架的 PC1 ---
        # from sklearn.decomposition import TruncatedSVD
        # 由于你的 SVD 是在 reduce_modality_features 里做的，你可以在外面直接切掉第一列
        # print("   [ATAC] Dropping SVD PC1 due to high sequencing depth correlation.")
        # atac_feat_np = atac_feat_np[:, 1:]
        # ---------------------------------------------

        # if sparse.issparse(adata_atac.X):
        #     scaler = StandardScaler()
        #     atac_feat_np = scaler.fit_transform(atac_feat_np)
        #     atac_feat_np = np.asarray(atac_feat_np, dtype=np.float32)
        #     print("   ✅ StandardScaler applied to ATAC SVD features")
        #     logger.info("StandardScaler applied to ATAC SVD features")
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
    print("\n🕸️ Building Spatial Graph & GFT Basis...")
    coords = adata_rna.obsm['spatial']
    # 可选：使用降维后的 RNA 特征计算边权重；关闭时退化为仅拓扑（无 RNA 相似度加权）
    rna_features = rna_feat_np if use_rna_similarity_edge_weight else None
    if use_rna_similarity_edge_weight:
        print("   ✅ Edge weight mode: RNA feature similarity")
        logger.info("Edge weight mode: RNA feature similarity")
    else:
        print("   ✅ Edge weight mode: topology only (RNA similarity disabled)")
        logger.info("Edge weight mode: topology only (RNA similarity disabled)")

    graph_outputs = build_spatial_graph(
        coords,
        features=rna_features,
        k=params['knn_k'],
        n_freq_components=n_freq_components,
    )
    if len(graph_outputs) == 4:
        edge_index, edge_weight, u_basis, evals = graph_outputs
    elif len(graph_outputs) == 3:
        edge_index, u_basis, evals = graph_outputs
        edge_weight = None
    else:
        edge_index, u_basis = graph_outputs
        edge_weight = None
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

    if edge_weight is not None:
        data_dict["edge_weight"] = edge_weight

    if evals is not None:
        data_dict["evals"] = evals

    data_dict["n_freq_components"] = int(u_basis.shape[1])

    if ground_truth is not None:
        data_dict["ground_truth"] = ground_truth
        data_dict["ground_truth_key"] = gt_key

    if sm_feat_np is not None and sm_adata is not None:
        sm_feat = torch.FloatTensor(sm_feat_np)
        data_dict["sm_feat"] = sm_feat
        data_dict["sm_input_dim"] = int(sm_feat.shape[1])
        data_dict["sm_var_names"] = np.asarray(sm_adata.var_names, dtype=str)
        data_dict["sm_obs_names"] = np.asarray(sm_adata.obs_names, dtype=str)
        if "spatial" in sm_adata.obsm:
            sm_coords = np.asarray(sm_adata.obsm["spatial"], dtype=np.float32)
            data_dict["sm_coords"] = torch.FloatTensor(sm_coords)
    
    torch.save(data_dict, save_path)
    print("✅ Preprocessing Complete!")
    logger.info("Preprocessing complete: %s", save_path)
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()
