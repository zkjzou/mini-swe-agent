#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
SERVER_URL="${EVAL_SERVER_URL:-http://laplace.eecs.umich.edu:8000}"
SUBSET="${EVAL_SUBSET:-swe-bench_verified}"
SPLIT="${EVAL_SPLIT:-test}"

mini-extra evaluation-client submit-preds \
    "${RUN_DIR}/preds.json" \
    --server-url "${SERVER_URL}" \
    --subset "${SUBSET}" \
    --split "${SPLIT}" \
    --output-dir "${RUN_DIR}"
