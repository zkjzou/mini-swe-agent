from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_DOCENT_IMPORT_ERROR = (
    "The docent-python package is required to upload to Docent. "
    "Install it with: pip install docent-python"
)


@dataclass(frozen=True)
class DocentUploadResult:
    collection_id: str | None
    created_collection: bool
    n_runs: int
    trajectory_files: list[Path]


def _import_docent():
    try:
        from docent import Docent
        from docent.data_models import AgentRun, Transcript
        from docent.data_models.chat import ToolCall, parse_chat_message
    except ImportError as exc:
        raise ImportError(_DOCENT_IMPORT_ERROR) from exc
    return Docent, AgentRun, Transcript, ToolCall, parse_chat_message


def _resolve_trajectory_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.traj.json"))
    raise FileNotFoundError(f"Path does not exist: {path}")


def _load_resolved_lookup(evaluation_result_path: Path) -> Callable[[str | None], bool | None]:
    data = json.loads(evaluation_result_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Evaluation result file must be a JSON object.")

    resolved_ids = set(data.get("resolved_ids", []) or [])
    unresolved_ids = set(data.get("unresolved_ids", []) or [])

    def lookup(instance_id: str | None) -> bool | None:
        if not instance_id:
            return None
        if instance_id in resolved_ids:
            return True
        if instance_id in unresolved_ids:
            return False
        return None

    return lookup


def _normalize_message(
    msg: dict[str, Any],
    *,
    ToolCall: type,
) -> dict[str, Any]:
    role = msg.get("role")
    content = msg.get("content", "")
    message_data: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id is not None:
        message_data["tool_call_id"] = tool_call_id

    if role == "tool":
        name = msg.get("name") or msg.get("tool_name")
        if name is not None:
            message_data["name"] = name

    raw_tool_calls = msg.get("tool_calls")
    if role == "assistant" and raw_tool_calls:
        parsed_tool_calls: list[Any] = []
        for tc in raw_tool_calls:
            if isinstance(tc, ToolCall):
                parsed_tool_calls.append(tc)
                continue
            if not isinstance(tc, dict):
                raise ValueError("Unexpected tool call format")
            function = tc.get("function", {}) or {}
            arguments = function.get("arguments", {})
            parsed_tool_calls.append(
                ToolCall(
                    id=tc.get("id"),
                    function=function.get("name"),
                    arguments=arguments,
                    type=tc.get("type", "function"),
                    parse_error=tc.get("parse_error"),
                )
            )
        message_data["tool_calls"] = parsed_tool_calls

    return message_data


def _build_agent_run(
    path: Path,
    *,
    resolved_lookup: Callable[[str | None], bool | None] | None,
    AgentRun: type,
    Transcript: type,
    ToolCall: type,
    parse_chat_message: Callable[[dict[str, Any]], Any],
) -> Any:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        messages = data
        info = {}
        trajectory_format = None
        instance_id = None
    elif isinstance(data, dict):
        messages = data.get("messages", [])
        info = data.get("info", {})
        trajectory_format = data.get("trajectory_format")
        instance_id = data.get("instance_id")
    else:
        raise ValueError("Unrecognized trajectory format")

    parsed_messages = [
        parse_chat_message(_normalize_message(msg, ToolCall=ToolCall)) for msg in messages  # type: ignore[arg-type]
    ]

    info_config = info.get("config", {}) if isinstance(info, dict) else {}
    agent_config = info_config.get("agent", {}) if isinstance(info_config, dict) else {}
    env_config = info_config.get("environment", {}) if isinstance(info_config, dict) else {}
    env_env = env_config.get("env", {}) if isinstance(env_config, dict) else {}
    model_stats = info.get("model_stats", {}) if isinstance(info, dict) else {}

    resolved = resolved_lookup(instance_id) if resolved_lookup else None

    metadata = {
        "exit_status": info.get("exit_status") if isinstance(info, dict) else None,
        "instance_id": instance_id,
        "mini_version": info.get("mini_version") if isinstance(info, dict) else None,
        "model_stats": {
            "api_calls": model_stats.get("api_calls") if isinstance(model_stats, dict) else None,
            "instance_cost": model_stats.get("instance_cost") if isinstance(model_stats, dict) else None,
        },
        "scores": {"resolved": resolved},
        "trajectory_format": trajectory_format,
        "config": {
            "agent_type": info_config.get("agent_type") if isinstance(info_config, dict) else None,
            "agent": {
                "action_observation_template": agent_config.get("action_observation_template"),
                "action_regex": agent_config.get("action_regex"),
                "cost_limit": agent_config.get("cost_limit"),
                "format_error_template": agent_config.get("format_error_template"),
                "instance_template": agent_config.get("instance_template"),
                "step_limit": agent_config.get("step_limit"),
                "system_template": agent_config.get("system_template"),
                "timeout_template": agent_config.get("timeout_template"),
            },
            "environment_type": info_config.get("environment_type") if isinstance(info_config, dict) else None,
            "environment": {
                "container_timeout": env_config.get("container_timeout"),
                "cwd": env_config.get("cwd"),
                "env": {"LESS": env_env.get("LESS")},
            },
        },
    }

    transcript = Transcript(messages=parsed_messages, metadata=metadata)
    return AgentRun(transcripts=[transcript], metadata=metadata)


def upload_docent(
    path: Path,
    *,
    collection_name: str | None,
    collection_id: str | None,
    collection_description: str | None = None,
    api_key: str | None = None,
    server_url: str | None = None,
    web_url: str | None = None,
    evaluation_result_path: Path | None = None,
    dry_run: bool = False,
    print_fct: Callable[[str], None] = print,
) -> DocentUploadResult:
    Docent, AgentRun, Transcript, ToolCall, parse_chat_message = _import_docent()

    trajectory_files = _resolve_trajectory_files(path)
    if not trajectory_files:
        raise ValueError(f"No .traj.json files found under {path}")

    resolved_lookup = _load_resolved_lookup(evaluation_result_path) if evaluation_result_path else None

    agent_runs = [
        _build_agent_run(
            traj_path,
            resolved_lookup=resolved_lookup,
            AgentRun=AgentRun,
            Transcript=Transcript,
            ToolCall=ToolCall,
            parse_chat_message=parse_chat_message,
        )
        for traj_path in trajectory_files
    ]

    if dry_run:
        print_fct(f"Parsed {len(agent_runs)} trajectories (dry run; no upload).")
        return DocentUploadResult(
            collection_id=collection_id,
            created_collection=False,
            n_runs=len(agent_runs),
            trajectory_files=trajectory_files,
        )

    if collection_id is None:
        if not collection_name:
            raise ValueError("Provide --collection-id or --collection-name to upload.")
        client = Docent(api_key=api_key, server_url=server_url, web_url=web_url)
        collection_id = client.create_collection(name=collection_name, description=collection_description or "")
        created_collection = True
    else:
        client = Docent(api_key=api_key, server_url=server_url, web_url=web_url)
        created_collection = False

    client.add_agent_runs(collection_id, agent_runs)
    print_fct(f"Uploaded {len(agent_runs)} trajectories to Docent collection {collection_id}.")

    return DocentUploadResult(
        collection_id=collection_id,
        created_collection=created_collection,
        n_runs=len(agent_runs),
        trajectory_files=trajectory_files,
    )
