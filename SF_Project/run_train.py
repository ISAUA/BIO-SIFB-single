import os
import argparse
import torch
import yaml
from sf_model.model.bio_sfinet import BioSFINet
from sf_model.trainer import SFTrainer
from sf_model.utils import set_seed

def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bio-SFINet")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    return parser.parse_args()

def main():
    print("🚀 [Phase 2] Starting Model Training...")
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    processed_dir = config['data']['processed_path']
    data_path = os.path.join(processed_dir, "processed_data.pt")
    
    # 2. 加载预处理好的数据
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("   -> Please run 'python run_preprocess.py' first.")
        return

    print(f"\n📦 Loading data from {data_path}...")
    # 使用 cpu 加载，trainer 会自动搬运到 cuda
    data_dict = torch.load(data_path, map_location='cpu')
    
    rna_feat = data_dict["rna_feat"]
    atac_feat = data_dict["atac_feat"]
    coords = data_dict["coords"]
    edge_index = data_dict["edge_index"]
    u_basis = data_dict["u_basis"]
    evals = data_dict.get("evals", None)
    atac_dim = data_dict["atac_dim"]
    
    print(f"   -> RNA Shape: {rna_feat.shape}")
    print(f"   -> ATAC Shape: {atac_feat.shape}")
    print(f"   -> Graph Edges: {edge_index.shape[1]}")
    if evals is None:
        print("   -> evals: not found in processed_data.pt (fallback without eigenvalue penalty)")
    else:
        print(f"   -> evals Shape: {evals.shape}")

    # 3. 初始化模型
    print("\n🧠 Initializing Bio-SFINet...")
    # 将 ATAC 维度传给模型
    model = BioSFINet(config, atac_dim=atac_dim)
    
    # 4. 初始化训练器
    trainer = SFTrainer(model, config)
    
    # 5. 开始训练
    print("\n🟢 STARTING TRAINING...")
    trainer.run(rna_feat, atac_feat, edge_index, u_basis, evals)

if __name__ == "__main__":
    main()