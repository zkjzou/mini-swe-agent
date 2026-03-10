#!/usr/bin/env bash
set -euo pipefail

module load singularity
export LITELLM_MODEL_REGISTRY_PATH="registry.json"
export MSWEA_COST_TRACKING="ignore_errors"
export SINGULARITY_CACHEDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity"
export SINGULARITY_TMPDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp"

OUTPUT_ROOT="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_coder"
INPUT_JSONL="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged_grouped_latest.jsonl"
MODEL_NAME="hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct"
API_BASE="http://localhost:8080/v1"

mkdir -p "${OUTPUT_ROOT}"

for verifier_variant in \
    checklist_verifier \
    checklist_v2_verifier \
    dynamic_checklist_regenerate_verifier \
    dynamic_checklist_modify_verifier
do
    mini-extra evaluate-verifier-actions \
        --input-jsonl "${INPUT_JSONL}" \
        --output-jsonl "${OUTPUT_ROOT}/${verifier_variant}_rows.jsonl" \
        --output-summary "${OUTPUT_ROOT}/${verifier_variant}_summary.json" \
        --output-distribution-csv "${OUTPUT_ROOT}/${verifier_variant}_predicted_action_distribution.csv" \
        --config swebench.yaml \
        --config verifier_model_profile=qwen3_coder \
        --config agent.enable_verbal_feedback=false \
        --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
        --config agent.verifier.model.model_name="${MODEL_NAME}" \
        --config agent.verifier.model.model_kwargs.api_base="${API_BASE}" \
        --verifier-variant "${verifier_variant}" \
        --max-workers 1 \
        --strict-five-actions \
        --overwrite
done
