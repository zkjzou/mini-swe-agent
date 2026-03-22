#!/usr/bin/env bash
set -euo pipefail

export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

mini-extra generate-trajectory-checklists \
    --trajectory "$1" \
    --output "$2" \
    --step-index "$3" \
    --mode trajectory_dynamic \
    --prompt-name dynamic_success_v2 \
    --model openai/MiniMaxAI/MiniMax-M2.5 \
    --model-class litellm \
    --config swebench.yaml \
    --config agent.verifier.model.model_name="openai/MiniMaxAI/MiniMax-M2.5" \
    --config agent.verifier.history_message_format="multi_turn_chat" \
    --config agent.verifier.model.strip_think_tags=true \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/checklist_generator \
    --config agent.verifier.model.model_kwargs.api_base=http://localhost:8080/v1 \
    --config agent.verifier.model.model_kwargs.api_key=EMPTY
