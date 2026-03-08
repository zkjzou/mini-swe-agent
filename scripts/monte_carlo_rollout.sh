#!/usr/bin/env bash
set -euo pipefail

# Optional cluster environment setup.
module load singularity

export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp}"

# Input merged verifier rows.
INPUT_JSONL="${INPUT_JSONL:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged_grouped_latest.jsonl}"

# SWE-bench dataset selection.
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"

# Rollout output.
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/monte_carlo_rollouts/qwen3_5_35b}"
SAMPLES_PER_ACTION="${SAMPLES_PER_ACTION:-4}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-20}"
MAX_WORKERS="${MAX_WORKERS:-4}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
STEP_INDEX="${STEP_INDEX:-}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"

# Actor continuation model overrides.
AGENT_MODEL_PROFILE="${AGENT_MODEL_PROFILE:-qwen3_5_instruct}"
MODEL_CLASS="${MODEL_CLASS:-litellm}"
MODEL_NAME="${MODEL_NAME:-openai/Qwen/Qwen3.5-35B-A3B}"
MODEL_API_BASE="${MODEL_API_BASE:-http://localhost:8080/v1}"

# Optional instance filters. Leave empty to run all rows.
INSTANCE_IDS=(
  # "astropy__astropy-13453"
)

mkdir -p "${OUTPUT_DIR}"

rollout_cmd=(
  mini-extra monte-carlo-rollout
  "${INPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --subset "${SUBSET}"
  --split "${SPLIT}"
  --samples-per-action "${SAMPLES_PER_ACTION}"
  --max-rollout-steps "${MAX_ROLLOUT_STEPS}"
  --max-workers "${MAX_WORKERS}"
  -c swebench.yaml
  -c "verifier.enabled=false"
  -c "model.model_class=${MODEL_CLASS}"
  -c "agent_model_profile=${AGENT_MODEL_PROFILE}"
  -c "model.model_name=${MODEL_NAME}"
  -c "model.model_kwargs.api_base=${MODEL_API_BASE}"
)

if [[ "${SHOW_PROGRESS}" == "true" ]]; then
  rollout_cmd+=(--show-progress)
else
  rollout_cmd+=(--no-show-progress)
fi

if [[ -n "${LIMIT_ROWS}" ]]; then
  rollout_cmd+=(--limit-rows "${LIMIT_ROWS}")
fi

if [[ -n "${STEP_INDEX}" ]]; then
  rollout_cmd+=(--step-index "${STEP_INDEX}")
fi

for instance_id in "${INSTANCE_IDS[@]}"; do
  rollout_cmd+=(--instance "${instance_id}")
done

"${rollout_cmd[@]}"

echo "Monte Carlo rollout finished:"
echo "  ${OUTPUT_DIR}/results.jsonl"
echo "  ${OUTPUT_DIR}/summary.json"
