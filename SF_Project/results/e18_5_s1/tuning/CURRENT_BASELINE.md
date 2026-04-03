# Current Tuning Baseline

- experiment_id: e12
- config: configs/tuning/config_tune_e12.yaml
- best_suffix: epoch_2800
- checkpoint: ckpt_2800.pth
- score: 3.1331
- metrics:
  - cluster_moran: 0.6943
  - ARI: 0.5347
  - NMI: 0.6227
  - AMI: 0.6158
  - HOM: 0.6656
- constraint:
  - ARI >= 0.4: pass

## Param Snapshot
- knn_k: 9
- ino_use_edge_weight: true
- ino_pre_smooth_enable: true
- ino_pre_smooth_alpha: 0.8
