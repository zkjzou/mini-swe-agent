#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${script_dir}/evaluate_verifier_checklist_verifier.sh"
"${script_dir}/evaluate_verifier_checklist_reward.sh"
