import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "config_human.yaml"
SWEEP_ROOT = ROOT / "results" / "human_loss_weight_sweep"
CONFIG_DIR = ROOT / "configs" / "sweeps" / "human_loss_weight_sweep"

TARGET_COMBOS = [
    # Current best neighborhood from the first six completed runs:
    # clip=0.75, atac=7, rna=5 reached ARI=0.4935.
    (1.0, 7.0, 5.0),
    (0.5, 7.0, 5.0),
    (0.75, 9.0, 5.0),
]


def tag_value(value):
    return str(value).replace(".", "p")


def run_cmd(cmd, env):
    print("\n===== " + " ".join(str(x) for x in cmd) + " =====", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def build_env(seed):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    return env


def main():
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    seed = int(base.get("project", {}).get("seed", 42))
    env = build_env(seed)
    summary_path = SWEEP_ROOT / "summary.csv"
    if summary_path.exists():
        summary_records = pd.read_csv(summary_path).to_dict("records")
    else:
        summary_records = []
    completed_tags = {
        str(row["tag"])
        for row in summary_records
        if str(row.get("status", "")).startswith("ok")
    }

    combos = TARGET_COMBOS
    for idx, (lambda_clip, lambda_atac, lambda_rna) in enumerate(combos, start=1):
        tag = (
            f"clip{tag_value(lambda_clip)}"
            f"_atac{tag_value(lambda_atac)}"
            f"_rna{tag_value(lambda_rna)}"
        )
        run_dir = SWEEP_ROOT / tag
        cfg_path = CONFIG_DIR / f"config_human_{tag}.yaml"

        if tag in completed_tags:
            print(f"\n### [{idx}/{len(combos)}] {tag} already completed; skipping.", flush=True)
            continue

        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
        cfg["project"]["save_dir"] = str(run_dir / "checkpoints")
        cfg["project"]["eval_dir"] = str(run_dir / "predictions")
        cfg["data"]["processed_path"] = str(run_dir / "processed")
        cfg["train"]["lambda_clip"] = float(lambda_clip)
        cfg["train"]["lambda_atac"] = float(lambda_atac)
        cfg["train"]["lambda_rna"] = float(lambda_rna)

        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(
            f"\n### [{idx}/{len(combos)}] {tag} "
            f"(lambda_clip={lambda_clip}, lambda_atac={lambda_atac}, lambda_rna={lambda_rna})",
            flush=True,
        )

        try:
            run_cmd([sys.executable, "run_preprocess.py", "--config", str(cfg_path)], env)
            run_cmd([sys.executable, "run_train.py", "--config", str(cfg_path)], env)
            run_cmd(
                [
                    sys.executable,
                    "run_evaluate_range.py",
                    "--config",
                    str(cfg_path),
                    "--start",
                    "800",
                    "--end",
                    "1500",
                    "--step",
                    "100",
                ],
                env,
            )

            metrics_csv = run_dir / "predictions" / "range_metrics_start800_end1500_step100.csv"
            df = pd.read_csv(metrics_csv)
            ok = df[df["ARI"].notna()].copy()
            if ok.empty:
                best = {
                    "best_epoch": None,
                    "best_ARI": None,
                    "best_NMI": None,
                    "best_AMI": None,
                    "best_HOM": None,
                    "status": "no_valid_ari",
                }
            else:
                row = ok.loc[ok["ARI"].idxmax()]
                best = {
                    "best_epoch": int(row["requested_epoch"]),
                    "best_ARI": float(row["ARI"]),
                    "best_NMI": float(row["NMI"]),
                    "best_AMI": float(row["AMI"]),
                    "best_HOM": float(row["HOM"]),
                    "status": str(row["status"]),
                }
        except Exception as exc:
            best = {
                "best_epoch": None,
                "best_ARI": None,
                "best_NMI": None,
                "best_AMI": None,
                "best_HOM": None,
                "status": f"failed:{type(exc).__name__}",
            }
            print(f"FAILED {tag}: {exc}", flush=True)

        record = {
            "tag": tag,
            "lambda_clip": float(lambda_clip),
            "lambda_atac": float(lambda_atac),
            "lambda_rna": float(lambda_rna),
            **best,
            "run_dir": str(run_dir),
            "config": str(cfg_path),
        }
        summary_records.append(record)

        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(summary_path, index=False)
        print(f"Current summary: {summary_path}", flush=True)
        print(summary_df.sort_values("best_ARI", ascending=False, na_position="last").head(10), flush=True)

    summary_df = pd.DataFrame(summary_records)
    summary_path = SWEEP_ROOT / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nDONE. Summary: {summary_path}", flush=True)
    print(summary_df.sort_values("best_ARI", ascending=False, na_position="last"), flush=True)


if __name__ == "__main__":
    main()
