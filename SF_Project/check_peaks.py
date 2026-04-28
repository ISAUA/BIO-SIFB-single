import scanpy as sc
import numpy as np

def main():
    # 根据您日志中的路径定义 S1 和 S2 的原始 ATAC 数据路径
    s1_atac_path = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/data/raw/misar/misar_e18-5-s1/adata_Peak.h5ad"
    s2_atac_path = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/data/raw/misar/misar_e18-5-s2/adata_Peak.h5ad"

    print("=== 加载 S1 和 S2 的原始 ATAC 数据 ===")
    try:
        adata_s1 = sc.read_h5ad(s1_atac_path)
        adata_s2 = sc.read_h5ad(s2_atac_path)
    except Exception as e:
        print(f"读取 h5ad 失败，请检查路径是否正确: {e}")
        return

    # 提取 Peak 名称（通常存在 var_names 中）
    peaks_s1 = adata_s1.var_names.astype(str).tolist()
    peaks_s2 = adata_s2.var_names.astype(str).tolist()

    print(f"\nS1 包含的原始 Peak 数量: {len(peaks_s1)}")
    print(f"S2 包含的原始 Peak 数量: {len(peaks_s2)}")

    print("\n=== 1. 命名格式与前 10 个 Peak 对比 ===")
    print("S1 的前 10 个 Peak:")
    for p in peaks_s1[:10]:
        print(f"  - {p}")
    
    print("\nS2 的前 10 个 Peak:")
    for p in peaks_s2[:10]:
        print(f"  - {p}")

    print("\n=== 2. 严格重合度分析 (Exact Intersection) ===")
    intersection = set(peaks_s1).intersection(set(peaks_s2))
    print(f"S1 和 S2 字符串完全一致的 Peak 数量: {len(intersection)}")
    if len(peaks_s1) > 0 and len(peaks_s2) > 0:
        print(f"占 S1 的比例: {len(intersection) / len(peaks_s1) * 100:.2f}%")
        print(f"占 S2 的比例: {len(intersection) / len(peaks_s2) * 100:.2f}%")

    print("\n=== 3. 模糊近邻分析 (看位点是否发生偏移) ===")
    # 提取 S1 和 S2 在 chr1 上的前 5 个 Peak 看看具体数值差异
    s1_chr1 = [p for p in peaks_s1 if "chr1" in p or p.startswith("1:")]
    s2_chr1 = [p for p in peaks_s2 if "chr1" in p or p.startswith("1:")]

    print("\nS1 中 chr1 的前 5 个 Peak:")
    for p in s1_chr1[:5]:
        print(f"  - {p}")
        
    print("\nS2 中 chr1 的前 5 个 Peak:")
    for p in s2_chr1[:5]:
        print(f"  - {p}")

if __name__ == "__main__":
    main()