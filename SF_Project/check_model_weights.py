import torch

def compare_initial_weights(file1, file2):
    print(f"正在比对: {file1} 和 {file2}\n")
    
    try:
        w1 = torch.load(file1, map_location='cpu', weights_only=True)
        w2 = torch.load(file2, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"❌ 读取权重文件失败: {e}")
        return

    all_match = True
    for key in w1.keys():
        if not torch.equal(w1[key], w2[key]):
            diff = torch.abs(w1[key] - w2[key]).max().item()
            print(f"❌ 权重 '{key}' 不一致! 最大绝对误差: {diff}")
            all_match = False
            
    if all_match:
        print("✅ 结论 1：两次初始化的模型权重完全严格一致。随机性不在初始化阶段。")
    else:
        print("⚠️ 结论 1：模型初始化存在随机性！可能是网络中存在未受控的模块。")

if __name__ == "__main__":
    # 请替换为您实际生成的绝对路径
    path1 = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/results/misar/misar_e18-5-s1/checkpoints/init_weights_test.pth"
    path2 = "/root/autodl-tmp/BIO-SFIB-single/SF_Project/results/misar/misar_e18-5-s1/checkpoints/init_weights_test_2.pth"
    compare_initial_weights(path1, path2)