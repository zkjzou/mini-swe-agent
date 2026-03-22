#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$1"

mini-extra evaluation-client submit-preds \
    "${MODEL_DIR}/preds.json" \
    --server-url "${EVAL_SERVER_URL:-http://laplace.eecs.umich.edu:8000}" \
    --subset "${EVAL_SUBSET:-swe-bench_verified}" \
    --split "${EVAL_SPLIT:-test}" \
    --output-dir "${MODEL_DIR}"
