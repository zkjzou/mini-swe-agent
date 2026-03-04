#!/usr/bin/env bash
set -euo pipefail

# Optional environment setup (uncomment if needed)
# module load singularity
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

# Input data (defaults from your GPT-5-2 trajectories)
OUTPUT_JSON="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/docent_downloads/GPT-5-2 high reasoning/output.json"
TRANSCRIPTS_DIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/docent_downloads/GPT-5-2 high reasoning/transcripts"

# Sampling output
SAMPLE_OUTPUT_DIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/run_01"
NUM_SAMPLES=2
MAX_WORKERS=8
LIMIT_RUNS=""             # e.g. 50 (leave empty for all)
LIMIT_STEPS_PER_RUN=""    # e.g. 30 (leave empty for all)
EXCLUDE_PARALLEL_TOOL_CALLS=true

mkdir -p "${SAMPLE_OUTPUT_DIR}"
SAMPLER_CONFIG="${SAMPLE_OUTPUT_DIR}/sampler_models.yaml"

# Edit model list here.
cat > "${SAMPLER_CONFIG}" <<'YAML'
models:
  - id: gpt5mini
    model_name: openai/gpt-5-mini
    sampling_kwargs:
      temperature: 0.8
  - id: qwen32b
    model_name: openrouter/qwen/qwen3-32b
    sampling_kwargs:
      temperature: 1.0
YAML

sample_cmd=(
  mini-extra sample-verifier-actions
  --output-json "${OUTPUT_JSON}"
  --transcripts-dir "${TRANSCRIPTS_DIR}"
  --sampler-config "${SAMPLER_CONFIG}"
  --output-dir "${SAMPLE_OUTPUT_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --max-workers "${MAX_WORKERS}"
  "$( [[ "${EXCLUDE_PARALLEL_TOOL_CALLS}" == "true" ]] && echo --exclude-parallel-tool-calls || echo --allow-parallel-tool-calls )"
  --overwrite
)

if [[ -n "${LIMIT_RUNS}" ]]; then
  sample_cmd+=(--limit-runs "${LIMIT_RUNS}")
fi
if [[ -n "${LIMIT_STEPS_PER_RUN}" ]]; then
  sample_cmd+=(--limit-steps-per-run "${LIMIT_STEPS_PER_RUN}")
fi

"${sample_cmd[@]}"

echo "Sampling finished:"
echo "  ${SAMPLE_OUTPUT_DIR}/candidates.jsonl"
echo "  ${SAMPLE_OUTPUT_DIR}/summary.json"

# Optional merge section (set RUN_MERGE=true to enable)
RUN_MERGE=false
MERGED_OUTPUT_JSONL="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged/merged_candidates.jsonl"
MERGE_SUMMARY_JSON="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged/merged_summary.json"
DEDUPE="semantic_key"        # none | exact | semantic_key
CONFLICT_POLICY="keep_first" # keep_first | keep_last | error

MERGE_INPUTS=(
  "${SAMPLE_OUTPUT_DIR}/candidates.jsonl"
  # "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/run_02/candidates.jsonl"
  # "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/run_03/candidates.jsonl"
)

if [[ "${RUN_MERGE}" == "true" ]]; then
  mkdir -p "$(dirname "${MERGED_OUTPUT_JSONL}")"
  merge_cmd=(
    mini-extra merge-verifier-actions
    --output-jsonl "${MERGED_OUTPUT_JSONL}"
    --output-summary "${MERGE_SUMMARY_JSON}"
    --dedupe "${DEDUPE}"
    --conflict-policy "${CONFLICT_POLICY}"
    --require-gold
    --overwrite
  )
  for input_path in "${MERGE_INPUTS[@]}"; do
    merge_cmd+=(--inputs "${input_path}")
  done
  "${merge_cmd[@]}"
  echo "Merge finished:"
  echo "  ${MERGED_OUTPUT_JSONL}"
  echo "  ${MERGE_SUMMARY_JSON}"
fi
