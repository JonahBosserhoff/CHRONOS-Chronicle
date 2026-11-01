#!/usr/bin/env bash
#SBATCH --job-name=chronos-validate
#SBATCH --partition=gpu_nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chronos-validate-%j.out
# ^ relative to the directory sbatch is called from (the repo root), which must hold logs/ BEFORE
#   submitting: mkdir -p logs
#
# External validation (src/chronos/validate_os.py): scale a RAW external cohort with the training
# cohort's scalers and run inference with a trained model. The external cohort needs no split.
# Paths and options come from a YAML file (template: configs/validate_external.yaml).
#
# From the repository root:
#
#   sbatch --export=ALL,CONFIG=configs/validate_external.yaml slurm/validate.sbatch
#
# Options after the script override the YAML, e.g. `slurm/validate.sbatch --seed 1`.
#
# Environment variables (default):
#   CONFIG   (configs/validate_external.yaml) the YAML file
#   PYTHON   (python) interpreter of the env (needs PyYAML for the config)

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
[ -f src/chronos/validate_os.py ] || { echo "$REPO is not the CHRONOS repository root; submit from there or pass REPO." >&2; exit 1; }

CONFIG=${CONFIG:-configs/validate_external.yaml}
PYTHON=${PYTHON:-python}

[ -f "$CONFIG" ] || { echo "CONFIG=$CONFIG does not exist." >&2; exit 1; }
if ! "$PYTHON" -c "import torch, pandas, sklearn, joblib, pyarrow, yaml" 2>/dev/null; then
    echo "$PYTHON cannot import torch, pandas, scikit-learn, joblib, pyarrow and PyYAML on $(hostname). Pass PYTHON=<env>/bin/python." >&2
    exit 1
fi
"$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
    || echo "⚠️ No GPU visible on $(hostname); inference runs on the CPU." >&2

echo "Config $CONFIG  (host $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a))"
cat "$CONFIG"

"$PYTHON" src/chronos/validate_os.py --config "$CONFIG" "$@"
