import os
import argparse
import torch
import yaml
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sf_model.utils import set_seed

# 引入模型
from sf_model.model.bio_sfinet import BioSFINet

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bio-SFINet & Plot UMAP/Spatial")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    # 可在 CLI 覆盖 checkpoint key；若不提供则采用 config 中的 eval.checkpoint
    parser.add_argument("--checkpoint", default=None, help="Checkpoint key or filename; defaults to config eval.checkpoint")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden clustering resolution")
    return parser.parse_args()

def visualize_and_save(z_final, coords, save_dir, resolution=0.5):
    """
    使用 Scanpy 进行降维、聚类和绘图
    z_final: [N, C] 最终的融合特征 (Tensor or Numpy)
    coords: [N, 2] 空间坐标
    """
    print(f"\n🎨 Starting Visualization (Leiden Res={resolution})...")
    
    # 确保转为 numpy
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    # 1. 构建 AnnData
    adata = sc.AnnData(X=z_final)
    adata.obsm['spatial'] = coords
    
    # 2. 基础分析流程 (Neighbors -> UMAP -> Leiden)
    print("   -> Computing Neighbors...")
    sc.pp.neighbors(adata, use_rep='X')
    
    print("   -> Computing UMAP...")
    sc.tl.umap(adata)
    
    print(f"   -> Clustering (Leiden)...")
    try:
        sc.tl.leiden(adata, resolution=resolution, key_added='cluster')
    except Exception as e:
        print("   ⚠️ Leiden clustering failed (maybe install leidenalg?), falling back to louvain.")
        sc.tl.louvain(adata, resolution=resolution, key_added='cluster')
    
    # 3. 绘图 (UMAP + Spatial)
    # 设置绘图风格
    sc.set_figure_params(dpi=150, figsize=(6, 6))
    
    print("   -> Plotting...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: UMAP
    sc.pl.umap(
        adata, 
        color='cluster', 
        ax=axs[0], 
        show=False, 
        title='Bio-SFINet Joint Embedding (UMAP)',
        legend_loc='on data',
        frameon=False,
        size=20
    )
    
    # 右图: Spatial (物理空间)
    sc.pl.embedding(
        adata, 
        basis='spatial', 
        color='cluster', 
        ax=axs[1], 
        show=False, 
        title='Spatial Map',
        size=40, # 点的大小，可根据细胞密度调整
        frameon=False
    )
    # 翻转 Y 轴以匹配常见的显微镜视角 (可选)
    # axs[1].invert_yaxis() 
    
    # 4. 保存图片
    # 如果 save_dir 是 checkpoints 目录，我们把图存到上级的 figures 目录
    if save_dir.rstrip('/').endswith('checkpoints'):
        base_dir = os.path.dirname(save_dir.rstrip('/'))
        fig_dir = os.path.join(base_dir, "figures")
    else:
        fig_dir = os.path.join(save_dir, "figures")
        
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, "spatial_analysis.pdf")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"✅ Plots saved to: {plot_path}")
    
    # 5. 保存结果 h5ad (方便后续自定义分析)
    pred_dir = os.path.join(os.path.dirname(fig_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    h5ad_path = os.path.join(pred_dir, "embedding_joint.h5ad")
    adata.write(h5ad_path)
    print(f"✅ Embedding h5ad saved to: {h5ad_path}")

def main():
    print("🚀 [Phase 3] Starting Evaluation & Plotting...")
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    processed_dir = config['data']['processed_path']
    save_dir = config['project']['save_dir']
    
    # 2. 加载数据 (直接读预处理好的 Tensor)
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}")
        print("   -> Please run 'python run_preprocess.py' first.")
        return

    print(f"\n📦 Loading data from {data_path}...")
    data_dict = torch.load(data_path, map_location='cpu')
    
    # 提取 Tensor 并转到 GPU
    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    u_basis = data_dict["u_basis"].to(device)
    atac_dim = data_dict["atac_dim"]

    # 3. 初始化模型
    print("\n🧠 Initializing Bio-SFINet...")
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    
    # 4. 加载权重
    eval_cfg = config.get('eval', {})
    ckpt_key = args.checkpoint or eval_cfg.get('checkpoint', 'best')
    ckpt_map = eval_cfg.get('checkpoints', {})
    # 如果 key 存在映射则取映射，否则允许直接把文件名作为 key 使用
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        print("   -> Please train the model first using 'run_train.py'")
        return
        
    print(f"   -> Loading weights from {ckpt_path}...")
    # 加载参数 (处理可能的 key 不匹配问题)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 5. 推理 (Inference)
    print("\n🔮 Running Inference...")
    with torch.no_grad():
        # Forward pass
        outputs = model(rna_feat, atac_feat, edge_index, u_basis)
        z_fused = outputs[0]
        
        # 级联 SFIB 后的融合潜变量直接作为最终表征
        z_final = z_fused
        
    print(f"   -> Extracted Latent Shape: {z_final.shape}")
    
    # 6. 可视化
    visualize_and_save(
        z_final, 
        coords, 
        save_dir, 
        resolution=args.resolution
    )
    
    print("\n🎉 Evaluation Complete!")

if __name__ == "__main__":
    main()