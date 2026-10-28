#!/usr/bin/env bash
set -euo pipefail

: "${TACTILE_DATA_ROOT:?Set TACTILE_DATA_ROOT to the LMT_108 directory}"
TACTILE_PYTHON=${TACTILE_PYTHON:-python}
TACTILE_OUTPUT_ROOT=${TACTILE_OUTPUT_ROOT:-runs/lmt108}
TACTILE_TRAIN_WINDOW_POLICY=${TACTILE_TRAIN_WINDOW_POLICY:-random}
TACTILE_EPOCHS=${TACTILE_EPOCHS:-100}
mkdir -p "$TACTILE_OUTPUT_ROOT"

for ratio in 2 4 8 16 32 48; do
    if [[ -f "$TACTILE_OUTPUT_ROOT/cr_$ratio/test.json" ]]; then
        echo "CR=$ratio already has a completed test.json; skipping"
        continue
    fi
    resume_args=()
    if [[ -f "$TACTILE_OUTPUT_ROOT/cr_$ratio/last.pt" ]]; then
        resume_args+=(--resume)
    fi
    echo "Training Tactile-JSCC on LMT_108 at CR=$ratio"
    "$TACTILE_PYTHON" main_tactile_JSCC.py \
        --data-root "$TACTILE_DATA_ROOT" \
        --output-root "$TACTILE_OUTPUT_ROOT" \
        --compression-ratio "$ratio" \
        --epochs "$TACTILE_EPOCHS" --batch-size 8 --learning-rate 1e-5 \
        --step-size 100 --snr-db 10 --beta 1.75 \
        --train-window-policy "$TACTILE_TRAIN_WINDOW_POLICY" "${resume_args[@]}" \
        >> "$TACTILE_OUTPUT_ROOT/cr_$ratio.log" 2>&1
done
