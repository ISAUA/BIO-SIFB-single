import argparse
import os
import subprocess
import sys
import logging
from pathlib import Path
import yaml

# 数据集到配置文件的映射，新增数据集时在此注册
DATASET_CONFIG = {
    "human": "configs/config_human.yaml",
    "mouse": "configs/config_mouse.yaml",
    "mouse_brain_p22": "configs/config_mouse_brain_p22.yaml",
    # renal 原先为单一数据集，已拆分为 R114_T 与 Y7_T
    "renal": "configs/renal/config_renal_R114_T.yaml",
    "R114_T": "configs/renal/config_renal_R114_T.yaml",
    "Y7_T": "configs/renal/config_renal_Y7_T.yaml",
    "misar_e11_0_s1": "configs/e11_0_s1/config_misar_e11_0_s1.yaml",
    "misar_e11_0_s2": "configs/e11_0_s2/config_misar_e11_0_s2.yaml",
    "misar_e13_5_s1": "configs/e13_5_s1/config_misar_e13_5_s1.yaml",
    "misar_e13_5_s2": "configs/e13_5_s2/config_misar_e13_5_s2.yaml",
    "misar_e15_5_s1": "configs/e15_5_s1/config_misar_e15_5_s1.yaml",
    "misar_e15_5_s2": "configs/e15_5_s2/config_misar_e15_5_s2.yaml",
    "misar_e18_5_s1": "configs/e18_5_s1/config_misar_e18_5_s1.yaml",
    "misar_e18_5_s2": "configs/e18_5_s2/config_misar_e18_5_s2.yaml",
    "misar_e18": "configs/e18_5_s1/config_misar_e18.yaml",
    "simulation1": "configs/simulation/config_simulation1.yaml",
}

ALL_STEPS = ["preprocess", "train", "evaluate"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing, training, and evaluation for Bio-SFINet")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG.keys()), default="human", help="选择数据集")
    parser.add_argument(
        "--steps",
        default="preprocess,train,evaluate",
        help="要执行的步骤，逗号分隔，可选 preprocess/train/evaluate，例如: preprocess,train",
    )
    parser.add_argument("--checkpoint", default=None, help="评估阶段使用的 checkpoint key 或文件名，可选")
    parser.add_argument("--n-clusters", type=int, default=None, help="评估阶段 mclust 聚类簇数，可选")
    parser.add_argument("--resolution", type=float, default=None, help="评估阶段 Leiden 分辨率，可选")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFPipeline")
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


def build_deterministic_env(seed):
    env = os.environ.copy()
    seed = str(int(seed))
    env["PYTHONHASHSEED"] = seed
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    return env


def run_cmd(label, cmd, seed, logger=None, cwd=None):
    print(f"\n===== {label}: {' '.join(cmd)} =====")
    if logger is not None:
        logger.info("Step start: %s | seed=%d", label, int(seed))
    env = build_deterministic_env(seed)
    env["SF_PIPELINE_RUN"] = "1"
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        if logger is not None:
            logger.error("Step failed: %s | return_code=%d", label, result.returncode)
        sys.exit(result.returncode)
    if logger is not None:
        logger.info("Step done: %s", label)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.dataset not in DATASET_CONFIG:
        print(f"未注册的数据集: {args.dataset}. 请在 DATASET_CONFIG 中添加映射。")
        sys.exit(1)

    config_path = DATASET_CONFIG[args.dataset]
    config = load_config(str(project_root / config_path))
    seed = int(os.environ.get("SEED_OVERRIDE", config['project'].get('seed', 42)))
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)
    logger.info("Pipeline start: dataset=%s", args.dataset)
    logger.info("Deterministic seed=%d", seed)
    logger.info("Config:\n%s", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    requested_steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    for step in requested_steps:
        if step not in ALL_STEPS:
            print(f"不支持的步骤: {step}. 仅支持 {ALL_STEPS}")
            sys.exit(1)

    steps = []

    if "preprocess" in requested_steps:
        steps.append((
            "Preprocess",
            [sys.executable, str(project_root / "run_preprocess.py"), "--config", config_path],
        ))

    if "train" in requested_steps:
        steps.append((
            "Train",
            [sys.executable, str(project_root / "run_train.py"), "--config", config_path],
        ))

    if "evaluate" in requested_steps:
        eval_cmd = [sys.executable, str(project_root / "run_evaluate.py"), "--config", config_path]
        if args.checkpoint:
            eval_cmd.extend(["--checkpoint", args.checkpoint])
        if args.n_clusters is not None:
            eval_cmd.extend(["--n-clusters", str(args.n_clusters)])
        if args.resolution is not None:
            eval_cmd.extend(["--resolution", str(args.resolution)])
        steps.append(("Evaluate", eval_cmd))

    if not steps:
        print("未选择任何步骤，已退出。")
        return

    print(f"使用配置: {config_path}")
    print(f"执行顺序: {[name for name, _ in steps]}")
    logger.info("Pipeline steps: %s", [name for name, _ in steps])

    try:
        for name, cmd in steps:
            run_cmd(name, cmd, seed=seed, logger=logger, cwd=str(project_root))
        logger.info("Pipeline finished successfully.")
    finally:
        append_log_separator(log_path)


if __name__ == "__main__":
    main()
