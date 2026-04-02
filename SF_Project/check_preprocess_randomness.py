import torch
import numpy as np

def compare_pt_files(file1_path, file2_path):
    print(f"正在比对文件 1: {file1_path}")
    print(f"正在比对文件 2: {file2_path}\n")
    
    try:
        # 兼容当前代码中的 torch.load 配置
        data1 = torch.load(file1_path, map_location='cpu', weights_only=False)
        data2 = torch.load(file2_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ 读取文件失败，请检查路径: {e}")
        return

    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        print(f"❌ 键值集合不一致!\n文件1特有键: {keys1 - keys2}\n文件2特有键: {keys2 - keys1}")
    else:
        print("✅ 顶层字典键值完全一致。")

    all_match = True
    for key in keys1.intersection(keys2):
        v1 = data1[key]
        v2 = data2[key]
        
        if type(v1) != type(v2):
            print(f"❌ 键 '{key}' 类型不一致: {type(v1)} vs {type(v2)}")
            all_match = False
            continue
            
        if isinstance(v1, torch.Tensor):
            if v1.shape != v2.shape:
                print(f"❌ 键 '{key}' Tensor形状不一致: {v1.shape} vs {v2.shape}")
                all_match = False
            else:
                is_equal = torch.equal(v1, v2)
                if is_equal:
                    print(f"  ✅ '{key}' (Tensor): 严格一致。")
                else:
                    diff = torch.abs(v1 - v2).max().item()
                    print(f"❌ '{key}' (Tensor): 数值不一致! 最大绝对误差: {diff}")
                    all_match = False
        elif isinstance(v1, np.ndarray):
            if v1.shape != v2.shape:
                print(f"❌ 键 '{key}' ndarray形状不一致: {v1.shape} vs {v2.shape}")
                all_match = False
            else:
                is_equal = np.array_equal(v1, v2)
                if is_equal:
                    print(f"  ✅ '{key}' (ndarray): 严格一致。")
                else:
                    print(f"❌ '{key}' (ndarray): 数值不一致!")
                    all_match = False
        else:
            if v1 == v2:
                print(f"  ✅ '{key}' ({type(v1).__name__}): 严格一致。")
            else:
                print(f"❌ '{key}' ({type(v1).__name__}): 值不一致: {v1} vs {v2}")
                all_match = False
                
    if all_match:
        print("\n🎉 结论：两个预处理数据文件严格一致，预处理阶段不存在未受控的随机性。")
    else:
        print("\n⚠️ 结论：预处理数据存在差异，随机性源头在预处理阶段（如外部包内的高变基因筛选或降维算法未传递种子）。")

if __name__ == "__main__":
    # 请确保 file_1 是您“主文件夹”中 processed_data.pt 的准确绝对路径
    file_1 = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/processed_data.pt" 
    
    # file_2 已替换为您提供的绝对路径
    file_2 = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/data/processed/misar/misar_e18-5-s1/processed_data.pt"
    
    compare_pt_files(file_1, file_2)