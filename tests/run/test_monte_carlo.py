import json
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.extra.monte_carlo import main as monte_carlo_main
from minisweagent.run.extra.utils.trajectory_replay import RegexActionSelector, TrajectoryReplayer
from minisweagent.run.utils.trajectory import build_message_history


def test_build_message_history_excludes_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "thought"},
        {"role": "user", "content": "obs"},
    ]
    filtered = build_message_history(messages, include_thoughts=False)
    assert all(message["role"] != "assistant" for message in filtered)
    assert [message["role"] for message in filtered] == ["system", "user"]


def test_replay_step_zero_excludes_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "```bash\necho replay\n```"},
        {"role": "user", "content": "obs"},
    ]
    env = LocalEnvironment()
    agent_config = {
        "action_regex": r"```bash\s*\n(.*?)\n```",
        "action_observation_template": "",
    }
    replayer = TrajectoryReplayer(
        messages,
        agent_config,
        env,
        include_thoughts=True,
        action_selector=RegexActionSelector(agent_config["action_regex"]),
    )
    replay_result = replayer.replay_to_step(0)
    assert replay_result.replayed_steps == 0
    assert all(message["role"] != "assistant" for message in replay_result.history_messages)


def test_monte_carlo_fixed_action(tmp_path: Path):
    trajectory_path = tmp_path / "sample.traj.json"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "```bash\necho replay\n```"},
        {"role": "user", "content": "<returncode>0</returncode>"},
    ]
    trajectory_path.write_text(json.dumps(messages))

    output_dir = tmp_path / "out"
    monte_carlo_main(
        trajectory=trajectory_path,
        step=1,
        rollouts=1,
        rollout_steps=1,
        output_dir=output_dir,
        workers=1,
        include_thoughts=True,
        verify_observations=False,
        config=Path("src/minisweagent/config/default.yaml"),
        model=None,
        model_class="deterministic",
        model_kwargs_json=json.dumps({"outputs": ["noop"]}),
        environment_class="local",
        environment_config_json="{}",
        env_startup_command="",
        rollout_action="echo rollout",
        rollout_actions_json="",
    )

    rollout_path = output_dir / "sample" / "rollout_0000.traj.json"
    assert rollout_path.exists()
    rollout_data = json.loads(rollout_path.read_text())
    rollout_info = rollout_data.get("info", {}).get("rollout", {})
    assert rollout_info.get("include_thoughts") is True
    assert rollout_info.get("source_step") == 1
    actions = rollout_info.get("actions", [])
    assert any(action.get("phase") == "replay" for action in actions)
    assert any(action.get("phase") == "rollout" for action in actions)
