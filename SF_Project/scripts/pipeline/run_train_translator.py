import os


def _sanitize_thread_env():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = os.environ.get(key)
        if val is None:
            os.environ[key] = "1"
            continue
        val = val.strip()
        if not val.isdigit() or int(val) <= 0:
            os.environ[key] = "1"


_sanitize_thread_env()
import argparse
import torch
from tqdm.auto import trange

from sf_model.model.bio_sfinet import BioSFINet, SF_Translator_R2A
from sf_model.trainer import SFTrainer
from sf_model.utils import set_seed
from .translation_runtime import (
    append_log_separator,
    load_config,
    load_processed_data,
    resolve_backbone_checkpoint,
    resolve_train_log_path,
    setup_file_logger,
    torch_load_compat,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 training for SF_Translator_R2A")
    parser.add_argument("--config", default="configs/config_human.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Backbone checkpoint key or filename. Can be key in --backbone-config eval.checkpoints",
    )
    parser.add_argument(
        "--backbone-config",
        default=None,
        help="Optional source config path for frozen backbone (e.g., S1 config when training S2 translator)",
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


def main():
    args = parse_args()
    config = load_config(args.config)
    translation_cfg = config.get("translation", {})
    stage2_cfg = translation_cfg.get("stage2", {})
    stage2_backbone_cfg = stage2_cfg.get("backbone", {})
    stage2_data_cfg = stage2_cfg.get("data", {})
    stage2_model_cfg = stage2_cfg.get("model", {})
    stage2_train_cfg = stage2_cfg.get("train", {})
    stage2_save_cfg = stage2_cfg.get("save", {})

    seed = int(config["project"].get("seed", 42))
    set_seed(seed)

    save_dir = config["project"]["save_dir"]
    log_path = resolve_train_log_path(save_dir)
    logger = setup_file_logger("SFTranslatorTrain", log_path, with_stream=False)

    translator_cfg = config.get("translator", {})
    translator_epochs = int(
        args.epochs
        if args.epochs is not None
        else stage2_train_cfg.get("epochs", translator_cfg.get("epochs", 300))
    )
    translator_lr = float(
        args.lr
        if args.lr is not None
        else stage2_train_cfg.get("learning_rate", translator_cfg.get("learning_rate", 1e-4))
    )
    translator_wd = float(
        args.weight_decay
        if args.weight_decay is not None
        else stage2_train_cfg.get("weight_decay", translator_cfg.get("weight_decay", 1e-4))
    )
    translator_blocks = int(
        args.n_blocks
        if args.n_blocks is not None
        else stage2_model_cfg.get("n_blocks", translator_cfg.get("n_blocks", 3))
    )
    lambda_cosine = float(
        args.lambda_cosine
        if args.lambda_cosine is not None
        else stage2_train_cfg.get("lambda_cosine", translator_cfg.get("lambda_cosine", 1.0))
    )
    lambda_mse = float(
        args.lambda_mse
        if args.lambda_mse is not None
        else stage2_train_cfg.get("lambda_mse", translator_cfg.get("lambda_mse", 1.0))
    )
    lambda_recon = float(
        args.lambda_recon
        if args.lambda_recon is not None
        else stage2_train_cfg.get("lambda_recon", translator_cfg.get("lambda_recon", 1.0))
    )
    save_every = int(
        args.save_every
        if args.save_every is not None
        else stage2_save_cfg.get("save_every", translator_cfg.get("save_every", 50))
    )

    translator_dir = os.path.join(save_dir, "translator_checkpoints")
    os.makedirs(translator_dir, exist_ok=True)
    best_name = (
        args.translator_save_name
        or stage2_save_cfg.get("best_name")
        or translator_cfg.get("best_name", "translator_r2a_best.pth")
    )
    best_path = os.path.join(translator_dir, best_name)
    last_path = os.path.join(translator_dir, "translator_r2a_last.pth")

    backbone_checkpoint = (
        args.backbone_checkpoint
        if args.backbone_checkpoint is not None
        else stage2_backbone_cfg.get("checkpoint", None)
    )
    backbone_config = (
        args.backbone_config
        if args.backbone_config is not None
        else stage2_backbone_cfg.get("config", None)
    )

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
    logger.info("Backbone source: config=%s checkpoint=%s", str(backbone_config), str(backbone_checkpoint))

    log_interval = int(config.get("train", {}).get("log_interval", 10))

    stage2_data_config = stage2_data_cfg.get("config", None)
    data_config = config
    if stage2_data_config:
        data_config = load_config(stage2_data_config)
        logger.info("Stage 2 data source: %s", stage2_data_config)
    else:
        logger.info("Stage 2 data source: %s", args.config)

    try:
        data_dict = load_processed_data(data_config)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please run preprocess first.")
        raise SystemExit(1)
    logger.info("Loading processed data from %s", data_dict["_data_path"])

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

    ckpt_path, ckpt_name, ckpt_source = resolve_backbone_checkpoint(
        target_config=config,
        backbone_checkpoint=backbone_checkpoint,
        backbone_config_path=backbone_config,
    )
    if not os.path.exists(ckpt_path):
        logger.error("Backbone checkpoint not found at %s", ckpt_path)
        logger.error(
            "Please train backbone first or pass --backbone-config/--backbone-checkpoint (e.g. --backbone-config S1 --backbone-checkpoint best)."
        )
        raise SystemExit(1)

    logger.info("Loading frozen backbone checkpoint: %s (source=%s)", ckpt_name, ckpt_source)
    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    hidden_dim = int(stage2_model_cfg.get("hidden_dim", config["model"].get("sfib_dim", 128)))
    translator = SF_Translator_R2A(hidden_dim=hidden_dim, n_blocks=translator_blocks).to(device)
    translator_optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=translator_lr,
        weight_decay=translator_wd,
    )

    trainer = SFTrainer(model, config, device=device)

    best_loss = float("inf")
    progress = trange(1, translator_epochs + 1, desc="Training Translator", unit="epoch", dynamic_ncols=True)
    for epoch in progress:
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

        progress.set_postfix(
            total=f"{metrics['total']:.4f}",
            cosine=f"{metrics['cosine']:.4f}",
            mse=f"{metrics['mse']:.4f}",
            recon=f"{metrics['recon']:.4f}",
            best=f"{best_loss:.4f}",
        )

        if epoch == 1 or epoch % max(1, log_interval) == 0 or epoch == translator_epochs:
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
