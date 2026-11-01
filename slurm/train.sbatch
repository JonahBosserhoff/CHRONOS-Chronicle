#!/usr/bin/env bash
#SBATCH --job-name=chronos-train
#SBATCH --partition=gpu_nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/chronos-train-%j.out
# ^ relative to the directory sbatch is called from (the repo root), which must hold logs/ BEFORE
#   submitting: mkdir -p logs
#
# Train CHRONOS on one target of a Chronicle cohort (src/chronos/train_os.py) and evaluate it on
# the test split. DATA is the cohort directory Chronicle trains on: its split CSVs and scalers are
# read as they are, nothing is split or scaled here. One GPU; train_os.py runs on a single device.
#
# From the repository root:
#
#   sbatch --export=ALL,DATA=<cohort dir>,GLOBAL_FORMAT=<schema.csv>,TARGET=OS slurm/train.sbatch
#
# One job per target:
#
#   for T in OS TTNT label_min_lab_hemoglobin_60_days label_febrile_neutropenia_21d; do
#       sbatch --job-name=chronos-$T --export=ALL,DATA=<dir>,GLOBAL_FORMAT=<schema.csv>,TARGET=$T slurm/train.sbatch
#   done
#
# Environment variables (default):
#   DATA               (required) Chronicle's cohort directory: <prefix>_{train,val,test}.csv and scalers/
#   GLOBAL_FORMAT      (required) Chronicle feature schema CSV
#   TARGET             (OS) any target of the cohort, see RUN_CHRONICLE.md
#   OUT_DIR            (results_os/$TARGET) refused if it already holds a trained model, unless OVERWRITE=1
#   BINARY_LABEL_COLS  (unset = train_os.py's default) 0/1 targets, spaces or commas; as in Chronicle's run
#   PYTHON             (python) interpreter of the env
#   OVERWRITE          (0)
# Further train_os.py options go after the script, e.g. `slurm/train.sbatch --epochs 20 --seed 1`.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
[ -f src/chronos/train_os.py ] || { echo "$REPO is not the CHRONOS repository root; submit from there or pass REPO." >&2; exit 1; }

: "${DATA:?Pass the cohort directory: sbatch --export=ALL,DATA=<dir>,GLOBAL_FORMAT=<schema.csv> ...}"
: "${GLOBAL_FORMAT:?Pass the feature schema: sbatch --export=ALL,DATA=<dir>,GLOBAL_FORMAT=<schema.csv> ...}"
PYTHON=${PYTHON:-python}
TARGET=${TARGET:-OS}
OUT_DIR=${OUT_DIR:-results_os/$TARGET}
OVERWRITE=${OVERWRITE:-0}

if ! "$PYTHON" -c "import torch, pandas, sklearn, joblib, pyarrow" 2>/dev/null; then
    echo "$PYTHON cannot import torch, pandas, scikit-learn, joblib and pyarrow on $(hostname). Pass PYTHON=<env>/bin/python." >&2
    exit 1
fi
"$PYTHON" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
    || echo "⚠️ No GPU visible on $(hostname); training runs on the CPU." >&2
if [ -e "$OUT_DIR/checkpoints/best_model.pt" ] && [ "$OVERWRITE" != 1 ]; then
    echo "$OUT_DIR already holds a trained model. Pass another OUT_DIR, or OVERWRITE=1." >&2
    exit 1
fi

EXTRA=()
if [ -n "${BINARY_LABEL_COLS+set}" ]; then
    COLS=${BINARY_LABEL_COLS// /,}
    EXTRA+=(--binary_label_cols "$COLS")
fi

echo "Target $TARGET -> $OUT_DIR  (host $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a))"

"$PYTHON" src/chronos/train_os.py \
    --data "$DATA" \
    --global_format "$GLOBAL_FORMAT" \
    --target "$TARGET" \
    --out_dir "$OUT_DIR" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@"

echo "Done. Results in $OUT_DIR"
