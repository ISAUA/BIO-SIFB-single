#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

EVAL_PATTERN = re.compile(
    r"Eval metrics \(([^)]+)\) \| checkpoint=([^|]+) \| epoch=([^|]+) \| latent_moran=([^|]+) \| cluster_moran=([^|]+)"
)
GT_PATTERN = re.compile(
    r"GT metrics \(([^)]+)\) \| checkpoint=([^|]+) \| epoch=([^|]+) \| ARI=([0-9.\-]+) \| NMI=([0-9.\-]+) \| AMI=([0-9.\-]+) \| HOM=([0-9.\-]+) \| n_valid=(\d+)"
)

CSV_FIELDS = [
    "timestamp",
    "dataset",
    "trial_id",
    "config_path",
    "knn_k",
    "ino_use_edge_weight",
    "ino_pre_smooth_enable",
    "ino_pre_smooth_alpha",
    "best_suffix",
    "checkpoint",
    "cluster_moran",
    "ARI",
    "NMI",
    "AMI",
    "HOM",
    "score",
    "ari_pass",
    "status",
    "is_best",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Auto tuning for misar_e15_5_s1 with global CSV tracking")
    parser.add_argument("--dataset-tag", default="e15_5_s1", help="Dataset tag used in config/result folders")
    parser.add_argument("--base-config", default="configs/e15_5_s1/config_misar_e15_5_s1.yaml", help="Base config YAML path")
    parser.add_argument("--knn-list", default="8,9,10", help="Comma separated knn list")
    parser.add_argument("--alpha-list", default="0.7,0.8,0.9", help="Comma separated alpha list")
    parser.add_argument("--start-index", type=int, default=1, help="Reserved for compatibility, no effect when trial_id is param-based")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runs (0 means run all)")
    parser.add_argument("--n-clusters", type=int, default=14, help="mclust cluster count for evaluate/evaluate_range")
    parser.add_argument("--range-start", type=int, default=1500)
    parser.add_argument("--range-end", type=int, default=3000)
    parser.add_argument("--range-step", type=int, default=100)
    parser.add_argument("--ari-threshold", type=float, default=0.4)
    parser.add_argument("--skip-existing", action="store_true", help="Skip trials that already have finished range evaluation")
    return parser.parse_args()


def fmt_alpha(alpha: float) -> str:
    s = ("%.3f" % alpha).rstrip("0").rstrip(".")
    return s.replace(".", "p")


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_layout(dataset_tag: str):
    (ROOT / "configs" / dataset_tag).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / dataset_tag / "tuning").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed" / dataset_tag / "tuning").mkdir(parents=True, exist_ok=True)


def build_trial_config(base_cfg, dataset_tag: str, trial_id: str, knn_k: int, alpha: float, n_clusters: int):
    cfg = deepcopy(base_cfg)

    cfg.setdefault("project", {})
    cfg.setdefault("data", {}).setdefault("parameters", {})
    cfg.setdefault("model", {})
    cfg.setdefault("eval", {})

    cfg["project"]["name"] = f"TUNE_{dataset_tag}_{trial_id}"
    cfg["project"]["save_dir"] = f"./results/{dataset_tag}/tuning/{trial_id}/checkpoints/"
    cfg["project"]["eval_dir"] = f"./results/{dataset_tag}/tuning/{trial_id}/predictions/"

    cfg["data"]["processed_path"] = f"data/processed/{dataset_tag}/tuning/{trial_id}"
    cfg["data"]["parameters"]["knn_k"] = int(knn_k)

    cfg["model"]["ino_use_edge_weight"] = True
    cfg["model"]["ino_pre_smooth_enable"] = True
    cfg["model"]["ino_pre_smooth_alpha"] = float(alpha)

    cfg["eval"]["n_clusters"] = int(n_clusters)
    return cfg


def run_stage(config_rel: str, script_name: str, extra_args=None):
    cmd = ["./run_deterministic.sh", config_rel, script_name]
    if extra_args:
        cmd.extend(extra_args)
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_best_metrics(train_log: Path, ari_threshold: float):
    if not train_log.exists():
        return None

    rows = {}
    lines = train_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        m_eval = EVAL_PATTERN.search(line)
        if m_eval:
            suffix, checkpoint, epoch, _latent_moran, cluster_moran = [x.strip() for x in m_eval.groups()]
            row = rows.setdefault(suffix, {"suffix": suffix, "checkpoint": checkpoint, "epoch": epoch})
            row["cluster_moran"] = None if cluster_moran == "NA" else float(cluster_moran)

        m_gt = GT_PATTERN.search(line)
        if m_gt:
            suffix, checkpoint, epoch, ari, nmi, ami, hom, n_valid = m_gt.groups()
            suffix = suffix.strip()
            row = rows.setdefault(suffix, {"suffix": suffix, "checkpoint": checkpoint.strip(), "epoch": epoch.strip()})
            row.update(
                {
                    "ARI": float(ari),
                    "NMI": float(nmi),
                    "AMI": float(ami),
                    "HOM": float(hom),
                    "n_valid": int(n_valid),
                }
            )

    candidates = []
    for row in rows.values():
        keys_ok = all(k in row for k in ("cluster_moran", "ARI", "NMI", "AMI", "HOM"))
        if not keys_ok or row["cluster_moran"] is None:
            continue
        row["score"] = row["cluster_moran"] + row["ARI"] + row["NMI"] + row["AMI"] + row["HOM"]
        if row["ARI"] >= ari_threshold:
            candidates.append(row)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


def read_csv_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def update_best_flag(rows, dataset: str):
    for row in rows:
        if row.get("dataset") == dataset:
            row["is_best"] = "0"

    valid = []
    for idx, row in enumerate(rows):
        if row.get("dataset") != dataset:
            continue
        if row.get("status") != "ok":
            continue
        if row.get("ari_pass") != "1":
            continue
        try:
            score = float(row.get("score", ""))
        except ValueError:
            continue
        valid.append((score, idx))

    if valid:
        valid.sort(reverse=True)
        best_idx = valid[0][1]
        rows[best_idx]["is_best"] = "1"


def upsert_csv_row(csv_path: Path, row):
    rows = read_csv_rows(csv_path)
    rows = [r for r in rows if not (r.get("dataset") == row["dataset"] and r.get("trial_id") == row["trial_id"])]
    rows.append(row)
    update_best_flag(rows, row["dataset"])
    write_csv_rows(csv_path, rows)


def is_trial_finished(train_log: Path):
    if not train_log.exists():
        return False
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    return "Range evaluation complete" in text


def main():
    args = parse_args()
    os.chdir(ROOT)
    ensure_layout(args.dataset_tag)

    base_cfg_path = ROOT / args.base_config
    base_cfg = load_yaml(base_cfg_path)

    knn_list = [int(x.strip()) for x in args.knn_list.split(",") if x.strip()]
    alpha_list = [float(x.strip()) for x in args.alpha_list.split(",") if x.strip()]
    grid = list(itertools.product(knn_list, alpha_list))
    if args.max_runs > 0:
        grid = grid[: args.max_runs]

    csv_path = ROOT / "results" / "experiments_global.csv"

    print(f"[INFO] dataset={args.dataset_tag} runs={len(grid)}")
    print(f"[INFO] base_config={args.base_config}")

    for _idx, (knn_k, alpha) in enumerate(grid, start=args.start_index):
        trial_id = f"k{knn_k}_a{fmt_alpha(alpha)}"
        cfg = build_trial_config(base_cfg, args.dataset_tag, trial_id, knn_k, alpha, args.n_clusters)

        config_path = ROOT / "configs" / args.dataset_tag / f"config_tune_{trial_id}.yaml"
        save_yaml(config_path, cfg)
        config_rel = str(config_path.relative_to(ROOT))
        train_log = ROOT / "results" / args.dataset_tag / "tuning" / trial_id / "train.log"

        if args.skip_existing and is_trial_finished(train_log):
            print(f"[SKIP] {trial_id} already finished.")
            continue

        status = "ok"
        note = ""
        best = None

        try:
            run_stage(config_rel, "run_preprocess.py")
            run_stage(config_rel, "run_train.py")
            run_stage(config_rel, "run_evaluate.py", ["--checkpoint", "best", "--n-clusters", str(args.n_clusters)])
            run_stage(
                config_rel,
                "run_evaluate_range.py",
                [
                    "--start",
                    str(args.range_start),
                    "--end",
                    str(args.range_end),
                    "--step",
                    str(args.range_step),
                    "--n-clusters",
                    str(args.n_clusters),
                ],
            )
            best = parse_best_metrics(train_log, args.ari_threshold)
            if best is None:
                status = "no_candidate"
                note = f"No checkpoint meets ARI >= {args.ari_threshold}"
        except subprocess.CalledProcessError as exc:
            status = "failed"
            note = f"Command failed with code {exc.returncode}"

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset_tag,
            "trial_id": trial_id,
            "config_path": config_rel,
            "knn_k": str(knn_k),
            "ino_use_edge_weight": "true",
            "ino_pre_smooth_enable": "true",
            "ino_pre_smooth_alpha": f"{alpha:.4f}",
            "best_suffix": "",
            "checkpoint": "",
            "cluster_moran": "",
            "ARI": "",
            "NMI": "",
            "AMI": "",
            "HOM": "",
            "score": "",
            "ari_pass": "0",
            "status": status,
            "is_best": "0",
            "note": note,
        }

        if best is not None:
            row.update(
                {
                    "best_suffix": best["suffix"],
                    "checkpoint": best["checkpoint"],
                    "cluster_moran": f"{best['cluster_moran']:.4f}",
                    "ARI": f"{best['ARI']:.4f}",
                    "NMI": f"{best['NMI']:.4f}",
                    "AMI": f"{best['AMI']:.4f}",
                    "HOM": f"{best['HOM']:.4f}",
                    "score": f"{best['score']:.4f}",
                    "ari_pass": "1",
                }
            )

        upsert_csv_row(csv_path, row)
        print(f"[DONE] {trial_id} status={status} score={row['score'] or 'NA'}")

    print(f"[INFO] Global CSV updated: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
