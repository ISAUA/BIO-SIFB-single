import os
import argparse
import torch
import yaml
import warnings

# 忽略不必要的警告，与原流水线保持一致
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

# 引入核心模型与工具
from sf_model.model.bio_sfinet import BioSFINet, SF_Translator_R2A
from sf_model.utils import set_seed

# 完美复用 run_evaluate 中的评估和绘图逻辑，无需重写任何聚类代码
from scripts.pipeline.run_evaluate import visualize_and_save, torch_load_compat

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Translation Module with Frozen Backbone Fusion")
    # 1. 当前目标切片的配置 (例如 S2 的 config)
    parser.add_argument("--config", required=True, help="Path to target slice YAML config (e.g., S2 config)")
    # 2. 提供融合模块的金标准主干权重 (例如 S1 训练好的 ckpt_best.pth)
    parser.add_argument("--backbone-checkpoint", required=True, help="Path to frozen backbone (e.g., S1's ckpt_best.pth)")
    # 3. 提供目标切片训练好的翻译器权重 (例如 S2 的 translator_r2a_best.pth)
    parser.add_argument("--translator-checkpoint", required=True, help="Path to translator weights")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters (override config)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # ==========================================
    # 1. 加载目标切片 (S2) 配置与数据
    # ==========================================
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    seed = int(config['project'].get('seed', 42))

    processed_dir = config['data']['processed_path']
    save_dir = config['project']['save_dir']
    eval_cfg = config.get('eval', {})
    n_clusters = int(args.n_clusters if args.n_clusters is not None else eval_cfg.get('n_clusters', 7))
    mclust_pca_dim = int(eval_cfg.get('mclust_pca_dim', 20))
    moran_k = int(eval_cfg.get('moran_k', 6))
    plot_cfg = eval_cfg.get('plotting', {})

    data_path = os.path.join(processed_dir, "processed_data.pt")
    print(f"📦 Loading processed data for evaluation from {data_path}...")
    data_dict = torch_load_compat(data_path, map_location='cpu', weights_only=False)

    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    edge_weight = data_dict.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    u_basis = data_dict["u_basis"].to(device)
    evals = data_dict.get("evals", None)
    if evals is not None:
        evals = evals.to(device)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = data_dict["atac_dim"]
    ground_truth = data_dict.get("ground_truth", None)

    # ==========================================
    # 2. 加载冻结的主干网络 (S1 提供基准融合)
    # ==========================================
    print(f"🧠 Initializing Backbone and loading: {args.backbone_checkpoint}")
    config['model']['rna_in_dim'] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    state_dict = torch_load_compat(args.backbone_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval() # 开启评估模式

    # ==========================================
    # 3. 加载翻译模块 (S2 翻译器)
    # ==========================================
    print(f"🗣️ Initializing Translator and loading: {args.translator_checkpoint}")
    hidden_dim = int(config["model"].get("sfib_dim", 128))
    translator_cfg = config.get("translator", {})
    translator_blocks = int(translator_cfg.get("n_blocks", 3))
    translator = SF_Translator_R2A(hidden_dim=hidden_dim, n_blocks=translator_blocks).to(device)
    trans_state_dict = torch_load_compat(args.translator_checkpoint, map_location=device, weights_only=True)
    translator.load_state_dict(trans_state_dict)
    translator.eval() # 开启评估模式

    # ==========================================
    # 4. 核心推理：翻译并融合 (Inference)
    # ==========================================
    print("⏳ Running Forward Pass (RNA -> Translator -> SFIB Fusion)...")
    with torch.no_grad():
        # 4.1 真实 RNA 提取特征
        h_rna = model.rna_enc(rna_feat, edge_index)
        f_rna = model.rna_proj(h_rna)

        # 4.2 翻译器生成预测 ATAC 特征
        f_atac_hat = translator(f_rna)

        # 4.3 预测 ATAC 与真实 RNA 进入冻结的 SFIB 进行交叉注意力融合
        z_fused_hat, *_ = model.sfib(
            f_rna, f_atac_hat, edge_index, u_basis, evals, edge_weight=edge_weight
        )

    # ==========================================
    # 5. 评估并计算 ARI 等指标
    # ==========================================
    print("📊 Evaluating Fused Features against Ground Truth...")
    
    # 复用原始流水线的画图与打分函数，自动打印 ARI/NMI/AMI/HOM 和 Moran's I
    plot_path, h5ad_path = visualize_and_save(
        z_final=z_fused_hat,
        coords=coords,
        save_dir=save_dir,
        n_clusters=n_clusters,
        mclust_pca_dim=mclust_pca_dim,
        epoch_label="Translated_Fusion",
        ground_truth=ground_truth,
        logger=None,
        moran_k=moran_k,
        plot_cfg=plot_cfg,
        checkpoint_name="Translator_Evaluation",
        seed=seed,
    )

    print("\n✅ Evaluation successfully completed!")
    print(f"   Saved Plot: {os.path.abspath(plot_path)}")
    print(f"   Saved h5ad: {os.path.abspath(h5ad_path)}")

if __name__ == "__main__":
    main()