import json
from pathlib import Path
from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.extra import rejected_actions as rejected_actions_cli
from minisweagent.run.utils import rejected_actions as rejected_actions_utils


def test_normalize_model_pool_excludes_expert():
    pool = [
        {"model_name": "expert-model", "model_class": "litellm"},
        {"model_name": "alt-model", "model_class": "litellm"},
    ]
    normalized = rejected_actions_utils.normalize_model_pool(
        pool,
        expert_model_name="expert-model",
        expert_model_class=None,
    )
    assert [spec.model_name for spec in normalized] == ["alt-model"]


def test_rejected_action_sampler_records(tmp_path: Path):
    sampler = rejected_actions_utils.RejectedActionSampler(
        model_pool=[
            {
                "model_name": "deterministic",
                "model_class": "deterministic",
                "model_kwargs": {
                    "outputs": [
                        "```bash\nls -la\n```",
                        "```bash\npwd\n```",
                    ]
                },
            }
        ],
        action_regex=rejected_actions_utils.DEFAULT_ACTION_REGEX,
        k_per_step=2,
        output_path=tmp_path / "rejected.jsonl",
        overwrite=True,
        mode="online",
    )

    records = sampler.sample_records(
        prompt_messages=[{"role": "system", "content": "sys"}],
        step_index=0,
        expert_action="echo hi",
    )

    assert len(records) == 2
    assert records[0]["expert_action"] == "echo hi"
    assert records[0]["rejected_action"] == "ls -la"
    assert records[1]["rejected_action"] == "pwd"


def test_offline_cli_writes_rejected_actions(tmp_path: Path):
    traj_path = tmp_path / "sample.traj.json"
    traj_data = {
        "info": {
            "config": {
                "agent": {"action_regex": rejected_actions_utils.DEFAULT_ACTION_REGEX},
                "model": {"model_name": "expert-model"},
                "model_type": "minisweagent.models.test_models.DeterministicModel",
            }
        },
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```bash\nls\n```"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": "```bash\npwd\n```"},
        ],
    }
    traj_path.write_text(json.dumps(traj_data))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
rejected_action_sampling:
  enabled: true
  mode: offline
  k_per_step: 1
  selection_policy: round_robin
  model_pool:
    - model_name: deterministic
      model_class: deterministic
      model_kwargs:
        outputs:
          - "```bash\\nalt1\\n```"
          - "```bash\\nalt2\\n```"
"""
    )

    outputs = ["```bash\nalt1\n```", "```bash\nalt2\n```"]

    with patch("minisweagent.run.utils.rejected_actions.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=outputs, cost_per_call=0.0)
        rejected_actions_cli.main(path=traj_path, config=config_path, overwrite=True, output_dir=None, run_id="")

    output_file = tmp_path / "sample.rejected.jsonl"
    assert output_file.exists()
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["rejected_action"] == "alt1"
    assert first["mode"] == "offline"
