import os
import argparse
import logging
import torch
import yaml

from sf_model.model.bio_sfinet import BioSFINet, SF_Translator_R2A
from sf_model.trainer import SFTrainer
from sf_model.utils import set_seed


def load_config(config_path="configs/config_human.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 training for SF_Translator_R2A")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Backbone checkpoint key or filename. Defaults to config eval.checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Translator training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Translator learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Translator weight decay")
    parser.add_argument("--n-blocks", type=int, default=None, help="Number of MLP blocks in SF_Translator_R2A")
    parser.add_argument("--lambda-cosine", type=float, default=None, help="Weight for latent cosine loss")
    parser.add_argument("--lambda-mse", type=float, default=None, help="Weight for latent mse loss")
    parser.add_argument("--lambda-recon", type=float, default=None, help="Weight for cross-reconstruction loss")
    parser.add_argument("--save-every", type=int, default=None, help="Save translator checkpoint every N epochs")
    parser.add_argument(
        "--translator-save-name",
        default=None,
        help="Best translator checkpoint filename (saved under translator_checkpoints)",
    )
    return parser.parse_args()


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip("/\\")
    if os.path.basename(save_dir) == "checkpoints":
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFTranslatorTrain")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def append_log_separator(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(args_ckpt, config, save_dir):
    eval_cfg = config.get("eval", {})
    ckpt_key = args_ckpt or eval_cfg.get("checkpoint", "best")
    ckpt_map = eval_cfg.get("checkpoints", {})
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)

    if os.path.isabs(ckpt_name):
        ckpt_path = ckpt_name
    else:
        ckpt_path = os.path.join(save_dir, ckpt_name)

    return ckpt_path, ckpt_name


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    save_dir = config["project"]["save_dir"]
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)

    translator_cfg = config.get("translator", {})
    translator_epochs = int(args.epochs if args.epochs is not None else translator_cfg.get("epochs", 300))
    translator_lr = float(args.lr if args.lr is not None else translator_cfg.get("learning_rate", 1e-4))
    translator_wd = float(args.weight_decay if args.weight_decay is not None else translator_cfg.get("weight_decay", 1e-4))
    translator_blocks = int(args.n_blocks if args.n_blocks is not None else translator_cfg.get("n_blocks", 3))
    lambda_cosine = float(args.lambda_cosine if args.lambda_cosine is not None else translator_cfg.get("lambda_cosine", 1.0))
    lambda_mse = float(args.lambda_mse if args.lambda_mse is not None else translator_cfg.get("lambda_mse", 1.0))
    lambda_recon = float(args.lambda_recon if args.lambda_recon is not None else translator_cfg.get("lambda_recon", 1.0))
    save_every = int(args.save_every if args.save_every is not None else translator_cfg.get("save_every", 50))

    translator_dir = os.path.join(save_dir, "translator_checkpoints")
    os.makedirs(translator_dir, exist_ok=True)
    best_name = args.translator_save_name or translator_cfg.get("best_name", "translator_r2a_best.pth")
    best_path = os.path.join(translator_dir, best_name)
    last_path = os.path.join(translator_dir, "translator_r2a_last.pth")

    logger.info("[Stage 2] Starting translator training")
    logger.info(
        "Config: epochs=%d lr=%.3e wd=%.3e n_blocks=%d lambdas=(%.3f, %.3f, %.3f)",
        translator_epochs,
        translator_lr,
        translator_wd,
        translator_blocks,
        lambda_cosine,
        lambda_mse,
        lambda_recon,
    )

    processed_dir = config["data"]["processed_path"]
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        logger.error("Data file not found at %s", data_path)
        logger.error("Please run preprocess first.")
        return

    logger.info("Loading processed data from %s", data_path)
    data_dict = torch_load_compat(data_path, map_location="cpu", weights_only=False)

    rna_feat = data_dict["rna_feat"]
    atac_feat = data_dict["atac_feat"]
    edge_index = data_dict["edge_index"]
    edge_weight = data_dict.get("edge_weight", None)
    u_basis = data_dict["u_basis"]
    evals = data_dict.get("evals", None)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = data_dict["atac_dim"]

    config["model"]["rna_in_dim"] = rna_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BioSFINet(config, atac_dim=atac_dim).to(device)

    ckpt_path, ckpt_name = resolve_checkpoint_path(args.backbone_checkpoint, config, save_dir)
    if not os.path.exists(ckpt_path):
        logger.error("Backbone checkpoint not found at %s", ckpt_path)
        logger.error("Please train backbone first or pass --backbone-checkpoint.")
        return

    logger.info("Loading frozen backbone checkpoint: %s", ckpt_name)
    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    hidden_dim = int(config["model"].get("sfib_dim", 128))
    translator = SF_Translator_R2A(hidden_dim=hidden_dim, n_blocks=translator_blocks).to(device)
    translator_optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=translator_lr,
        weight_decay=translator_wd,
    )

    trainer = SFTrainer(model, config, device=device)

    best_loss = float("inf")
    for epoch in range(1, translator_epochs + 1):
        metrics = trainer.train_translator_epoch(
            translator=translator,
            translator_optimizer=translator_optimizer,
            rna_feat=rna_feat,
            atac_feat=atac_feat,
            edge_index=edge_index,
            u_basis=u_basis,
            evals=evals,
            edge_weight=edge_weight,
            lambda_cosine=lambda_cosine,
            lambda_mse=lambda_mse,
            lambda_recon=lambda_recon,
        )

        if metrics["total"] < best_loss:
            best_loss = metrics["total"]
            torch.save(translator.state_dict(), best_path)

        if save_every > 0 and epoch % save_every == 0:
            epoch_path = os.path.join(translator_dir, f"translator_r2a_ckpt_{epoch}.pth")
            torch.save(translator.state_dict(), epoch_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == translator_epochs:
            logger.info(
                "Translator Epoch %03d | total %.4f | cosine %.4f | mse %.4f | recon %.4f | best %.4f",
                epoch,
                metrics["total"],
                metrics["cosine"],
                metrics["mse"],
                metrics["recon"],
                best_loss,
            )

    torch.save(translator.state_dict(), last_path)
    logger.info("Translator training finished. best=%.4f", best_loss)
    logger.info("Saved best translator: %s", best_path)
    logger.info("Saved last translator: %s", last_path)

    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)


if __name__ == "__main__":
    main()
