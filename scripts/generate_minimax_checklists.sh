#!/usr/bin/env bash
set -euo pipefail

export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

mini-extra generate-trajectory-checklists \
    --input /home/zkjzou/SWE-PRM/mini-swe-agent/success_row.json \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/minimax_checklists.jsonl \
    --mode trajectory_success \
    --prompt-name static_success_v2 \
    --model minimax-2.5 \
    --model-class litellm \
    --config verifier.yaml \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/checklist_generator \
    --config agent.verifier.model.model_kwargs.api_base=http://localhost:8000/v1 \
    --config agent.verifier.model.model_kwargs.api_key=EMPTY \
    "$@"
