#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_deterministic.sh <config_yaml> <python_script> [extra args...]
# Example:
#   ./run_deterministic.sh configs/config_misar_e18_5_s1.yaml run_preprocess.py
#   ./run_deterministic.sh configs/config_misar_e18_5_s1.yaml run_train.py
#   ./run_deterministic.sh configs/config_misar_e18_5_s1.yaml run_evaluate.py --checkpoint best --n-clusters 14

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_yaml> <python_script> [extra args...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
PY_SCRIPT="$2"
shift 2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Python script not found: $PY_SCRIPT" >&2
  exit 1
fi

# Prefer environment override, otherwise read project.seed from YAML.
if [[ -n "${SEED_OVERRIDE:-}" ]]; then
  SEED="$SEED_OVERRIDE"
else
  SEED="$(awk '
    $0 ~ /^project:[[:space:]]*$/ { in_project=1; next }
    in_project && $0 ~ /^[^[:space:]]/ { in_project=0 }
    in_project && $0 ~ /^[[:space:]]*seed:[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*seed:[[:space:]]*/, "", line)
      sub(/[[:space:]]*#.*/, "", line)
      gsub(/[[:space:]]/, "", line)
      print line
      exit
    }
  ' "$CONFIG_PATH")"
fi

if [[ -z "$SEED" ]]; then
  SEED="42"
fi

# Determinism-critical env vars must be present before Python starts.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

echo "[Deterministic Launch]"
echo "  Config: $CONFIG_PATH"
echo "  Script: $PY_SCRIPT"
echo "  Seed:   $SEED"

exec python "$PY_SCRIPT" --config "$CONFIG_PATH" "$@"
