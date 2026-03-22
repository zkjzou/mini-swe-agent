#!/usr/bin/env bash
set -euo pipefail

for i in 5 6 7 8 9 10 11 12 13 14 15; do
    MODEL_DIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_minimax_2_5_${i}"

    mini-extra evaluation-client sync-result \
        --metadata-file "${MODEL_DIR}/evaluation_submission.json" \
        --output-dir "${MODEL_DIR}" \
        --wait \
        --poll-interval "${EVAL_POLL_INTERVAL:-30}"
done
