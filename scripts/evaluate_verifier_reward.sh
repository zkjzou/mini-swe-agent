#!/usr/bin/env bash
set -euo pipefail

module load singularity
export LITELLM_MODEL_REGISTRY_PATH="${LITELLM_MODEL_REGISTRY_PATH:-registry.json}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/merged_grouped_latest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/verifier_samples/qwen3_5_35b}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/verifier_eval_reward_rows.jsonl}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-${OUTPUT_DIR}/verifier_eval_reward_summary.json}"
DISTRIBUTION_CSV="${DISTRIBUTION_CSV:-${OUTPUT_DIR}/predicted_action_distribution.csv}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MODEL_NAME="${MODEL_NAME:-openai/Qwen/Qwen3.5-35B-A3B}"
API_BASE="${API_BASE:-http://localhost:8080/v1}"

mkdir -p "${OUTPUT_DIR}"

mini-extra evaluate-verifier-actions \
    --input-jsonl "${INPUT_JSONL}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --output-summary "${OUTPUT_SUMMARY}" \
    --config swebench.yaml \
    --config verifier_model_profile=qwen3_5_35b \
    --config agent.verifier.prompt_dir="${REPO_ROOT}/prompts/verifier" \
    --config agent.verifier.model.model_name="${MODEL_NAME}" \
    --config agent.verifier.model.model_kwargs.api_base="${API_BASE}" \
    --verifier-type reward_model \
    --max-workers "${MAX_WORKERS}" \
    --strict-five-actions \
    --overwrite

OUTPUT_JSONL="${OUTPUT_JSONL}" DISTRIBUTION_CSV="${DISTRIBUTION_CSV}" VERIFIER_TYPE="reward_model" python - <<'PY'
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

output_jsonl = Path(os.environ["OUTPUT_JSONL"])
distribution_csv = Path(os.environ["DISTRIBUTION_CSV"])
verifier_type = os.environ["VERIFIER_TYPE"]

counts = defaultdict(Counter)
totals = Counter()
with output_jsonl.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "evaluated":
            continue
        variant = str(row.get("verifier_variant") or verifier_type)
        label = row.get("selected_label")
        if not isinstance(label, str) or not label:
            label = f"index_{row.get('selected_index')}"
        counts[variant][label] += 1
        totals[variant] += 1

fieldnames = [
    "timestamp_utc",
    "verifier_type",
    "verifier_variant",
    "selected_label",
    "count",
    "fraction",
    "rows_evaluated",
    "output_jsonl",
]
distribution_csv.parent.mkdir(parents=True, exist_ok=True)
write_header = not distribution_csv.exists() or distribution_csv.stat().st_size == 0
timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

with distribution_csv.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    for variant in sorted(counts):
        total = totals[variant]
        for label, count in sorted(counts[variant].items()):
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "verifier_type": verifier_type,
                    "verifier_variant": variant,
                    "selected_label": label,
                    "count": count,
                    "fraction": f"{count / total:.6f}" if total else "0.000000",
                    "rows_evaluated": total,
                    "output_jsonl": str(output_jsonl),
                }
            )
PY

echo "Wrote evaluation rows: ${OUTPUT_JSONL}"
echo "Wrote evaluation summary: ${OUTPUT_SUMMARY}"
echo "Appended predicted action distribution: ${DISTRIBUTION_CSV}"
