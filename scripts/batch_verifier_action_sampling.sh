#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
SHOW_PROGRESS=true

mkdir -p "${SAMPLE_OUTPUT_DIR}"
SAMPLER_CONFIG="${SAMPLE_OUTPUT_DIR}/sampler_models.yaml"

# Build sampler config from swebench.yaml profiles.
SWEBENCH_CONFIG="${REPO_ROOT}/src/minisweagent/config/benchmarks/swebench.yaml"
MODEL_PROFILES=(
  gpt5_mini
  gpt5_2
)

# Optional model_name override for profiles with empty model_name in swebench.yaml.
# Example:
# PROFILE_MODEL_NAME_OVERRIDES_JSON='{"qwen3_5_instruct":"openrouter/qwen/qwen3-32b-instruct"}'
PROFILE_MODEL_NAME_OVERRIDES_JSON='{}'

# Optional per-profile sampling kwargs.
# Keys are profile names; values are kwargs passed to model.query.
PROFILE_SAMPLING_KWARGS_JSON='{
  "gpt5_mini": {"temperature": 0.8},
  "gpt5_2": {"temperature": 0.8}
}'

# Default used when a profile key is missing in PROFILE_SAMPLING_KWARGS_JSON.
DEFAULT_SAMPLING_KWARGS_JSON='{}'

python - "${SWEBENCH_CONFIG}" "${SAMPLER_CONFIG}" "${PROFILE_MODEL_NAME_OVERRIDES_JSON}" "${PROFILE_SAMPLING_KWARGS_JSON}" "${DEFAULT_SAMPLING_KWARGS_JSON}" "${MODEL_PROFILES[@]}" <<'PY'
import copy
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
overrides = json.loads(sys.argv[3])
sampling_map = json.loads(sys.argv[4])
default_sampling = json.loads(sys.argv[5])
requested_profiles = sys.argv[6:]

if not isinstance(overrides, dict):
    raise ValueError("PROFILE_MODEL_NAME_OVERRIDES_JSON must be a JSON object")
if not isinstance(sampling_map, dict):
    raise ValueError("PROFILE_SAMPLING_KWARGS_JSON must be a JSON object")
if not isinstance(default_sampling, dict):
    raise ValueError("DEFAULT_SAMPLING_KWARGS_JSON must be a JSON object")
if not requested_profiles:
    raise ValueError("MODEL_PROFILES is empty")

cfg = yaml.safe_load(config_path.read_text())
if not isinstance(cfg, dict):
    raise ValueError(f"Invalid swebench config: expected mapping, got {type(cfg).__name__}")
profiles_root = (cfg.get("profiles") or {}).get("model_profiles") or {}
if not isinstance(profiles_root, dict):
    raise ValueError("Invalid swebench config: profiles.model_profiles must be a mapping")

models = []
seen_ids = set()
for profile_name in requested_profiles:
    if profile_name not in profiles_root:
        available = ", ".join(sorted(profiles_root)) or "<none>"
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")
    raw_profile = profiles_root[profile_name]
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Profile '{profile_name}' must be a mapping")

    profile = copy.deepcopy(raw_profile)
    model_name = str(profile.get("model_name") or "").strip()
    if not model_name:
        model_name = str(overrides.get(profile_name) or "").strip()
    if not model_name:
        raise ValueError(
            f"Profile '{profile_name}' has empty model_name. "
            f"Set PROFILE_MODEL_NAME_OVERRIDES_JSON for this profile."
        )

    entry = {"id": profile_name, **profile, "model_name": model_name}
    sampling_kwargs = sampling_map.get(profile_name, default_sampling)
    if not isinstance(sampling_kwargs, dict):
        raise ValueError(f"Sampling kwargs for profile '{profile_name}' must be a mapping")
    entry["sampling_kwargs"] = sampling_kwargs

    if entry["id"] in seen_ids:
        raise ValueError(f"Duplicate sampler id '{entry['id']}'")
    seen_ids.add(entry["id"])
    models.append(entry)

out_path.write_text(yaml.safe_dump({"models": models}, sort_keys=False))
print(f"Wrote sampler config: {out_path}")
print("Profiles:", ", ".join(requested_profiles))
PY

sample_cmd=(
  mini-extra sample-verifier-actions
  --output-json "${OUTPUT_JSON}"
  --transcripts-dir "${TRANSCRIPTS_DIR}"
  --sampler-config "${SAMPLER_CONFIG}"
  --output-dir "${SAMPLE_OUTPUT_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --max-workers "${MAX_WORKERS}"
  --overwrite
)

if [[ "${EXCLUDE_PARALLEL_TOOL_CALLS}" == "true" ]]; then
  sample_cmd+=(--exclude-parallel-tool-calls)
else
  sample_cmd+=(--allow-parallel-tool-calls)
fi
if [[ "${SHOW_PROGRESS}" == "true" ]]; then
  sample_cmd+=(--show-progress)
else
  sample_cmd+=(--no-show-progress)
fi

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
