#!/usr/bin/env bash
set -euo pipefail

module load singularity

export LITELLM_MODEL_REGISTRY_PATH="registry.json"
export MSWEA_COST_TRACKING="ignore_errors"
export SINGULARITY_CACHEDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity"
export SINGULARITY_TMPDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp"

AGENT_MODEL_PROFILE="qwen3_5_instruct"
INPUT_JSONL="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged_grouped_latest.jsonl"
OUTPUT_DIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/monte_carlo_rollouts/qwen3_5_35b"

mini-extra monte-carlo-rollout \
    "${INPUT_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --subset verified \
    --split test \
    --samples-per-action 4 \
    --max-rollout-steps 20 \
    --max-workers 4 \
    --show-progress \
    --config swebench.yaml \
    --config verifier.enabled=false \
    --config model.model_class="litellm" \
    --config agent_model_profile="${AGENT_MODEL_PROFILE}" \
    --config model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1"

# Optional extra flags to append manually:
#   --limit-rows 10
#   --step-index 3
#   --instance astropy__astropy-13453
