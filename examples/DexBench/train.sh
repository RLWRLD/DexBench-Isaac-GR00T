#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RBY1_ACTION_MODE="${RBY1_ACTION_MODE:-abs}"
NUM_GPUS="${NUM_GPUS:-4}"
RUN_NAME="rby1-wuji2-${RBY1_ACTION_MODE}"
if [[ -n "${RBY1_CAMERAS:-}" ]]; then
    CAMERAS="${RBY1_CAMERAS//camera_/}"
    RUN_NAME+="-${CAMERAS//,/-}"
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    --module gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/DexBench/rby1_wuji2_config.py \
    --output-dir "outputs/${RUN_NAME}" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size 256 \
    --max-steps 30000 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 8 \
    --episode-sampling-rate 1.0 \
    --num-shards-per-epoch 32 \
    "$@"
