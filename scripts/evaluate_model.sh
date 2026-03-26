#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_minimax_2_5_5"

mini-extra evaluation-client submit-preds \
    "${MODEL_DIR}/preds.json" \
    --server-url "${EVAL_SERVER_URL:-http://laplace.eecs.umich.edu:8000}" \
    --subset "${EVAL_SUBSET:-swe-bench_verified}" \
    --split "${EVAL_SPLIT:-test}" \
    ${EVAL_RERUN:+--rerun} \
    --output-dir "${MODEL_DIR}"
