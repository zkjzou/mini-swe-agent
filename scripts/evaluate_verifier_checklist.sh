#!/usr/bin/env bash
set -euo pipefail

module load singularity
export LITELLM_MODEL_REGISTRY_PATH="registry.json"
export MSWEA_COST_TRACKING="ignore_errors"
export SINGULARITY_CACHEDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity"
export SINGULARITY_TMPDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp"

mkdir -p /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_5_35b

for verifier_variant in \
    checklist_verifier \
    checklist_v2_verifier \
    dynamic_checklist_regenerate_verifier \
    dynamic_checklist_modify_verifier \
    checklist_reward \
    checklist_v2_reward \
    dynamic_checklist_regenerate_reward \
    dynamic_checklist_modify_reward
do
    mini-extra evaluate-verifier-actions \
        --input-jsonl /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged_grouped_latest.jsonl \
        --output-jsonl "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_5_35b/${verifier_variant}_rows.jsonl" \
        --output-summary "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_5_35b/${verifier_variant}_summary.json" \
        --output-distribution-csv /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_5_35b/predicted_action_distribution.csv \
        --config swebench.yaml \
        --config verifier_model_profile=qwen3_5_35b \
        --config agent.enable_verbal_feedback=false \
        --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
        --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
        --config agent.verifier.model.model_kwargs.api_base="http://localhost:8080/v1" \
        --verifier-variant "${verifier_variant}" \
        --max-workers 8 \
        --strict-five-actions \
        --overwrite
done
