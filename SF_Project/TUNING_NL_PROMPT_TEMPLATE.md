# 自动调参自然语言指令模板（可迁移）

## 使用方式
- 把下面模板中的占位符替换成你的目标数据集信息，然后直接发给助手执行。
- 推荐在每次执行后要求助手给出最优参数、最优 checkpoint、五项指标和覆盖结论。

## 模板 1：全量调参执行
请你先把 [DATASET_TAG] 的基础 config 按 [REFERENCE_DATASET] 的关键设置对齐，然后在 sc_bridge 环境里按确定性流程执行自动调参。

要求如下：
1. 固定参数：
- ino_use_edge_weight = true
- ino_pre_smooth_enable = true
2. 调参网格：
- knn: [KNN_LIST]（例如 7,8,9）
- ino_pre_smooth_alpha: [ALPHA_LIST]（例如 0.7,0.8,0.9）
3. 聚类数：
- n_clusters = [N_CLUSTERS]
4. 每个 trial 必须完整执行：
- 预处理 -> 训练 -> 评估 -> 范围评估
- 范围评估区间：start=[RANGE_START]，end=[RANGE_END]，step=[RANGE_STEP]
5. 结果选择规则：
- 基于 cluster_moran、ARI、NMI、AMI、HOM 五项综合尽可能大进行选优
6. 记录要求：
- 每个 trial 都要写入结果记录 CSV（数据集级 + 全局）

## 模板 2：最优覆盖验证
请根据范围评估选出的最优 trial 和 checkpoint，把基础配置覆盖为最优参数后，重新执行一轮基底实验（预处理 -> 训练 -> 评估 -> 范围评估），并对比是否达到或超过调参最优；若达到则确认覆盖完成。

## 模板 3：空间清理
请只保留最优 trial 的结果目录、processed/tuning 目录和 config_tune 文件，删除同数据集下其它非最优 trial 的结果与调参代码配置（不要删除数据集原始数据和基础 config）。

## 建议占位符示例
- [DATASET_TAG]: e11_0_s1
- [REFERENCE_DATASET]: e18_5_s1
- [KNN_LIST]: 7,8,9
- [ALPHA_LIST]: 0.7,0.8,0.9
- [N_CLUSTERS]: 7
- [RANGE_START]: 1500
- [RANGE_END]: 3000
- [RANGE_STEP]: 100
