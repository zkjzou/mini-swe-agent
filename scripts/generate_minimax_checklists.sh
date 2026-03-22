#!/usr/bin/env bash
set -euo pipefail

export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

INPUT_JSON="${1:-/home/zkjzou/SWE-PRM/mini-swe-agent/success_row.json}"
OUTPUT_JSONL="${2:-/home/zkjzou/SWE-PRM/mini-swe-agent/minimax_checklists.jsonl}"
MODEL_NAME="${MODEL_NAME:-minimax-2.5}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"

mini-extra generate-trajectory-checklists \
    --input "${INPUT_JSON}" \
    --output "${OUTPUT_JSONL}" \
    --mode trajectory_success \
    --prompt-name static_success_v2 \
    --model "${MODEL_NAME}" \
    --model-class litellm \
    --config verifier.yaml \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/checklist_generator \
    --config agent.verifier.model.model_kwargs.api_base="${API_BASE}" \
    --config agent.verifier.model.model_kwargs.api_key="${API_KEY}"
