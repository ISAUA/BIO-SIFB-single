import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_load_compat(path: str, map_location: Any = "cpu", weights_only: bool = False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_train_log_path(save_dir: str) -> str:
    save_dir = save_dir.rstrip("/\\")
    if os.path.basename(save_dir) == "checkpoints":
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_file_logger(name: str, log_path: str, with_stream: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if with_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def append_log_separator(log_path: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")


def resolve_checkpoint_path(ckpt_key: str, save_dir: str, checkpoint_map: Optional[dict] = None) -> Tuple[str, str]:
    checkpoint_map = checkpoint_map or {}
    ckpt_name = checkpoint_map.get(ckpt_key, checkpoint_map.get(str(ckpt_key), ckpt_key))
    ckpt_name = str(ckpt_name)

    candidates = []
    if os.path.isabs(ckpt_name):
        candidates.append(ckpt_name)
    else:
        candidates.append(os.path.join(save_dir, ckpt_name))

    if "/" not in ckpt_name and "\\" not in ckpt_name:
        if ckpt_name.isdigit():
            candidates.append(os.path.join(save_dir, f"ckpt_{ckpt_name}.pth"))
        elif ckpt_name.startswith("ckpt_") and not ckpt_name.endswith(".pth"):
            candidates.append(os.path.join(save_dir, f"{ckpt_name}.pth"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, os.path.basename(candidate)

    return candidates[0], ckpt_name


def resolve_backbone_checkpoint(
    target_config: Dict[str, Any],
    backbone_checkpoint: Optional[str] = None,
    backbone_config_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolve backbone checkpoint for Stage 2/3.5.

    Priority:
    1) absolute path in --backbone-checkpoint
    2) resolve from backbone config when --backbone-config is provided
    3) resolve from target config eval/checkpoints
    """
    if backbone_checkpoint and os.path.isabs(backbone_checkpoint):
        return backbone_checkpoint, os.path.basename(backbone_checkpoint), "cli-absolute"

    source_cfg = target_config
    source_tag = "target-config"
    if backbone_config_path:
        source_cfg = load_config(backbone_config_path)
        source_tag = os.path.basename(backbone_config_path)

    eval_cfg = source_cfg.get("eval", {})
    checkpoint_map = eval_cfg.get("checkpoints", {})
    ckpt_key = backbone_checkpoint or eval_cfg.get("checkpoint", "best")
    save_dir = source_cfg["project"]["save_dir"]

    path, name = resolve_checkpoint_path(str(ckpt_key), save_dir, checkpoint_map)
    return path, name, source_tag


def resolve_translator_checkpoint(
    target_config: Dict[str, Any],
    translator_checkpoint: Optional[str] = None,
) -> Tuple[str, str]:
    if translator_checkpoint is not None:
        if os.path.isabs(translator_checkpoint):
            return translator_checkpoint, os.path.basename(translator_checkpoint)
        return translator_checkpoint, os.path.basename(translator_checkpoint)

    save_dir = target_config["project"]["save_dir"]
    translator_cfg = target_config.get("translator", {})
    best_name = translator_cfg.get("best_name", "translator_r2a_best.pth")
    default_path = os.path.join(save_dir, "translator_checkpoints", best_name)
    return default_path, os.path.basename(default_path)


def load_processed_data(config: Dict[str, Any]) -> Dict[str, Any]:
    data_path = os.path.join(config["data"]["processed_path"], "processed_data.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    data_dict = torch_load_compat(data_path, map_location="cpu", weights_only=False)
    data_dict["_data_path"] = data_path
    return data_dict
