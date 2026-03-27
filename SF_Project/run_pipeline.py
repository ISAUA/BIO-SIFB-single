import argparse
import os
import subprocess
import sys
import logging
import yaml

# 数据集到配置文件的映射，新增数据集时在此注册
DATASET_CONFIG = {
    "human": "configs/config_human.yaml",
    "mouse": "configs/config_mouse.yaml",
    "misar_e18": "configs/config_misar_e18.yaml",
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
    parser.add_argument("--resolution", type=float, default=None, help="评估阶段的 Leiden 分辨率，可选")
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


def run_cmd(label, cmd, logger=None):
    print(f"\n===== {label}: {' '.join(cmd)} =====")
    if logger is not None:
        logger.info("Step start: %s", label)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        if logger is not None:
            logger.error("Step failed: %s | return_code=%d", label, result.returncode)
        sys.exit(result.returncode)
    if logger is not None:
        logger.info("Step done: %s", label)


def main():
    args = parse_args()

    if args.dataset not in DATASET_CONFIG:
        print(f"未注册的数据集: {args.dataset}. 请在 DATASET_CONFIG 中添加映射。")
        sys.exit(1)

    config_path = DATASET_CONFIG[args.dataset]
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config = load_config(os.path.join(base_dir, config_path))
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)
    logger.info("Pipeline start: dataset=%s", args.dataset)

    requested_steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    for step in requested_steps:
        if step not in ALL_STEPS:
            print(f"不支持的步骤: {step}. 仅支持 {ALL_STEPS}")
            sys.exit(1)

    steps = []

    if "preprocess" in requested_steps:
        steps.append((
            "Preprocess",
            [sys.executable, os.path.join(base_dir, "run_preprocess.py"), "--config", config_path],
        ))

    if "train" in requested_steps:
        steps.append((
            "Train",
            [sys.executable, os.path.join(base_dir, "run_train.py"), "--config", config_path],
        ))

    if "evaluate" in requested_steps:
        eval_cmd = [sys.executable, os.path.join(base_dir, "run_evaluate.py"), "--config", config_path]
        if args.checkpoint:
            eval_cmd.extend(["--checkpoint", args.checkpoint])
        if args.resolution is not None:
            eval_cmd.extend(["--resolution", str(args.resolution)])
        steps.append(("Evaluate", eval_cmd))

    if not steps:
        print("未选择任何步骤，已退出。")
        return

    print(f"使用配置: {config_path}")
    print(f"执行顺序: {[name for name, _ in steps]}")
    logger.info("Pipeline steps: %s", [name for name, _ in steps])

    for name, cmd in steps:
        run_cmd(name, cmd, logger=logger)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
