#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${EVAL_SERVER_URL:-http://laplace.eecs.umich.edu:8000}"
SUBSET="${EVAL_SUBSET:-swe-bench_verified}"
SPLIT="${EVAL_SPLIT:-test}"

usage() {
    cat <<'EOF'
Simple wrapper for the mini-swe-agent evaluation server client.

Environment overrides:
  EVAL_SERVER_URL   default: http://laplace.eecs.umich.edu:8000
  EVAL_SUBSET       default: swe-bench_verified
  EVAL_SPLIT        default: test

Usage:
  scripts/eval_server.sh submit-run OUTPUT_DIR [extra submit-preds args]
  scripts/eval_server.sh submit-preds PREDS_JSON [extra submit-preds args]
  scripts/eval_server.sh submit-instance [submit-instance args]
  scripts/eval_server.sh sync-run OUTPUT_DIR [extra sync-result args]
  scripts/eval_server.sh sync METADATA_JSON [extra sync-result args]
  scripts/eval_server.sh poll-job [poll-job args]
  scripts/eval_server.sh poll-run [poll-run args]
  scripts/eval_server.sh start-run [start-run args]

Examples:
  scripts/eval_server.sh submit-run /path/to/run_dir
  scripts/eval_server.sh submit-preds /path/to/preds.json --run-id my-run
  scripts/eval_server.sh sync-run /path/to/run_dir --wait --poll-interval 30
  scripts/eval_server.sh submit-instance \
    --instance-id sympy__sympy-20590 \
    --model-name my-model \
    --patch-file /path/to/patch.diff \
    --run-id verified-run
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    submit-run)
        if [[ $# -lt 1 ]]; then
            usage
            exit 1
        fi
        OUTPUT_DIR="$1"
        shift
        PREDS_PATH="${OUTPUT_DIR}/preds.json"
        mini-extra evaluation-client submit-preds \
            "${PREDS_PATH}" \
            --server-url "${SERVER_URL}" \
            --subset "${SUBSET}" \
            --split "${SPLIT}" \
            --output-dir "${OUTPUT_DIR}" \
            "$@"
        ;;
    submit-preds)
        if [[ $# -lt 1 ]]; then
            usage
            exit 1
        fi
        PREDS_PATH="$1"
        shift
        OUTPUT_DIR="$(dirname "${PREDS_PATH}")"
        mini-extra evaluation-client submit-preds \
            "${PREDS_PATH}" \
            --server-url "${SERVER_URL}" \
            --subset "${SUBSET}" \
            --split "${SPLIT}" \
            --output-dir "${OUTPUT_DIR}" \
            "$@"
        ;;
    submit-instance)
        mini-extra evaluation-client submit-instance \
            --server-url "${SERVER_URL}" \
            --subset "${SUBSET}" \
            --split "${SPLIT}" \
            "$@"
        ;;
    sync-run)
        if [[ $# -lt 1 ]]; then
            usage
            exit 1
        fi
        OUTPUT_DIR="$1"
        shift
        METADATA_PATH="${OUTPUT_DIR}/evaluation_submission.json"
        mini-extra evaluation-client sync-result \
            --metadata-file "${METADATA_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            "$@"
        ;;
    sync)
        if [[ $# -lt 1 ]]; then
            usage
            exit 1
        fi
        METADATA_PATH="$1"
        shift
        OUTPUT_DIR="$(dirname "${METADATA_PATH}")"
        mini-extra evaluation-client sync-result \
            --metadata-file "${METADATA_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            "$@"
        ;;
    poll-job)
        mini-extra evaluation-client poll-job --server-url "${SERVER_URL}" "$@"
        ;;
    poll-run)
        mini-extra evaluation-client poll-run --server-url "${SERVER_URL}" "$@"
        ;;
    start-run)
        mini-extra evaluation-client start-run --server-url "${SERVER_URL}" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        usage
        exit 1
        ;;
esac
