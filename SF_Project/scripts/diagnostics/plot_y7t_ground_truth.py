import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
import scanpy as sc
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def torch_load_compat(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _sort_categories(values):
    uniq = pd.unique(values)
    def _key(v):
        s = str(v)
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted([str(x) for x in uniq], key=_key)

def main():
    # 1. 基础配置加载
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/renal/config_renal_Y7_T.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 2. 数据读取
    processed_path = os.path.join(cfg["data"]["processed_path"], "processed_data.pt")
    data = torch_load_compat(processed_path, map_location="cpu")
    
    coords = data["coords"].cpu().numpy() if isinstance(data["coords"], torch.Tensor) else np.asarray(data["coords"])
    gt = data.get("ground_truth", None)
    
    # 3. 预处理标签（去除无效值）
    labels = np.asarray(gt).astype(str)
    valid_mask = np.array([(x != "") and (x.lower() != "nan") and (x.lower() != "none") for x in labels])
    
    coords = coords[valid_mask]
    gt_labels = labels[valid_mask]

    # 4. 构建 AnnData 对象进行绘图
    adata = sc.AnnData(X=np.zeros((coords.shape[0], 1)))
    adata.obsm["spatial"] = coords
    gt_categories = _sort_categories(gt_labels)
    adata.obs["Ground Truth"] = pd.Categorical(gt_labels, categories=gt_categories, ordered=True)

    # 5. 精细化绘图设置
    sc.set_figure_params(dpi=200, frameon=False, facecolor='white')
    
    # 调整这些参数来控制“好看”的程度
    point_size = 120    # 点的大小，视切片密度而定
    point_alpha = 0.9   # 点的透明度
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="Ground Truth",
        ax=ax,
        show=False,
        title="Y7_T Spatial Organization (Ground Truth)",
        size=point_size,
        alpha=point_alpha,
        legend_loc="right margin",  
        legend_fontsize=10,
        
        # --- 核心修改：去掉单引号，直接调用 Scanpy 内置的颜色列表对象 ---
        palette=sc.pl.palettes.zeileis_28, 
        
        frameon=False               
    )

    # 强制隐藏坐标刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 6. 保存
    eval_dir = cfg.get("project", {}).get("eval_dir", "./results")
    out_path = os.path.join(eval_dir, "Y7_T_Ground_Truth_Clean_Map.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"[Done] 绝美空间图已保存至: {out_path}")

if __name__ == "__main__":
    main()