#!/usr/bin/env python3

"""Upload verifier-evaluation JSONL rows to Docent as one AgentRun per row."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from docent import Docent
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import ToolCall, parse_chat_message
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Install docent-python first: pip install docent-python") from exc

from minisweagent.agents.default import VerifierConfig
from minisweagent.utils.verifier_action_evaluation import (
    _align_prompt_name,
    _load_resolved_config,
)
from minisweagent.verifiers.prompt_loader import apply_prompt_overrides


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Path to verifier_eval_*.jsonl")
    parser.add_argument(
        "--collection-id",
        help="Existing Docent collection ID. If omitted, a new collection is created.",
    )
    parser.add_argument(
        "--collection-name",
        help="Name for a new collection. Defaults to <input_stem>_<UTC timestamp>.",
    )
    parser.add_argument("--api-key", help="Docent API key. Defaults to DOCENT_API_KEY.")
    parser.add_argument("--domain", help="Optional Docent domain, e.g. docent.transluce.org.")
    parser.add_argument("--server-url", help="Optional Docent server URL.")
    parser.add_argument("--web-url", help="Optional Docent web URL.")
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        dest="config_specs",
        help="Optional config spec(s) used to render verifier prompts. Defaults to the SWE-bench benchmark config.",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        help="Optional merged verifier-action JSONL used to recover the previous message.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Runs per upload batch.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Default timeout for Docent HTTP requests.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        dest="wait",
        default=True,
        help="Wait for Docent server-side processing after each upload batch.",
    )
    parser.add_argument(
        "--retry-split-on-failure",
        action=argparse.BooleanOptionalAction,
        dest="retry_split_on_failure",
        default=True,
        help="If a batch is canceled server-side, retry by splitting into smaller batches.",
    )
    parser.add_argument(
        "--disable-proxy-env",
        action="store_true",
        help="Unset HTTP(S)_PROXY and ALL_PROXY before connecting.",
    )
    return parser.parse_args()


def disable_proxy_env() -> None:
    for key in _PROXY_ENV_VARS:
        os.environ.pop(key, None)


def install_default_request_timeout(timeout_seconds: float) -> None:
    original_request = requests.sessions.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_with_timeout


def default_collection_name(path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"{path.stem}_{timestamp}"


def default_source_jsonl(path: Path) -> Path | None:
    candidate = path.parent / "merged_grouped_latest.jsonl"
    if candidate.is_file():
        return candidate
    return None


def int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def row_lookup_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row.get("instance_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("trajectory_relpath") or ""),
        int_or_default(row.get("step_index"), -1),
        int_or_default(row.get("message_index"), -1),
    )


def load_source_lookup(path: Path) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in source rows on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected source row {line_no} to be a JSON object.")
            lookup[row_lookup_key(row)] = row
    return lookup


def augmented_instance_id(row: dict[str, Any]) -> str:
    original = str(row.get("instance_id") or "unknown")
    step_index = int_or_default(row.get("step_index"), -1)
    selected_is_gold = bool(row.get("selected_is_gold"))
    return f"{original}__step_{step_index}__selected_gold_{int(selected_is_gold)}"


def trajectory_name(row: dict[str, Any]) -> str:
    original = str(row.get("instance_id") or "unknown")
    step_index = int_or_default(row.get("step_index"), -1)
    return f"{original}__step_{step_index}"


@lru_cache(maxsize=None)
def get_verifier_config(config_specs: tuple[str, ...], verifier_type: str) -> VerifierConfig:
    resolved_config, _ = _load_resolved_config(list(config_specs) or None)
    agent_config = resolved_config.get("agent") or {}
    if not isinstance(agent_config, dict):
        raise ValueError("Invalid config: 'agent' must be a mapping.")
    verifier_payload = copy.deepcopy(agent_config.get("verifier") or {})
    if not isinstance(verifier_payload, dict):
        raise ValueError("Invalid config: 'agent.verifier' must be a mapping.")

    verifier_payload["enabled"] = True
    verifier_payload["verifier_type"] = verifier_type
    verifier_config = VerifierConfig(**verifier_payload)
    verifier_config.prompt_name = _align_prompt_name(verifier_config.prompt_name, verifier_type)  # type: ignore[arg-type]
    return apply_prompt_overrides(verifier_config)


def build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "instance_id": augmented_instance_id(row),
        "gold_index": row.get("gold_index"),
        "selected_label": row.get("selected_label"),
    }
    return metadata


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    if arguments is None:
        return {}
    return {"value": arguments}


def normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    message_data: dict[str, Any] = {
        "role": role,
        "content": msg.get("content", ""),
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
            arguments = normalize_tool_arguments(function.get("arguments", {}))
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


def source_history_messages(source_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(source_row, dict):
        return []
    history = source_row.get("history_trajectory")
    if not isinstance(history, list):
        return []
    return [message for message in history if isinstance(message, dict) and isinstance(message.get("role"), str)]


def build_candidate_actions_content(source_row: dict[str, Any] | None) -> str | None:
    if not isinstance(source_row, dict):
        return None

    actions = source_row.get("actions")
    if not isinstance(actions, list) or not actions:
        return None

    lines = ["Candidate actions:"]
    for idx, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            continue
        label = str(action.get("label") or f"candidate_{idx}")
        command = str(action.get("command") or "").strip()
        is_gold = bool(action.get("is_gold"))
        suffix = " [gold]" if is_gold else ""
        if command:
            lines.append(f"{idx}. {label}{suffix}: {command}")
        else:
            lines.append(f"{idx}. {label}{suffix}")

    if len(lines) == 1:
        return None
    return "\n".join(lines)


def strip_recent_steps_block(selection_template: str) -> str:
    pattern = re.compile(
        r"\nRecent steps.*?\n(?:\{%.*?\n)*Candidates:\n",
        re.DOTALL,
    )
    stripped = pattern.sub("\nCandidates:\n", selection_template, count=1)
    return stripped.strip()


def build_verifier_prompt_messages(
    row: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    config_specs: tuple[str, ...],
) -> list[Any]:
    verifier_type = str(row.get("verifier_type") or "llm")
    verifier_config = get_verifier_config(config_specs, verifier_type)

    prompt_messages: list[Any] = []
    if verifier_type == "reward_model":
        system_prompt = verifier_config.reward_system_template.strip()
        user_prompt = verifier_config.reward_prompt_template.strip()
    else:
        system_prompt = verifier_config.system_template.strip()
        user_prompt = strip_recent_steps_block(verifier_config.selection_template)

    prompt_messages.append(parse_chat_message({"role": "system", "content": system_prompt}))
    prompt_messages.append(parse_chat_message({"role": "user", "content": user_prompt}))
    return prompt_messages


def build_transcript_messages(
    row: dict[str, Any],
    source_row: dict[str, Any] | None,
    *,
    config_specs: tuple[str, ...],
) -> list[Any]:
    messages: list[Any] = []
    for message in source_history_messages(source_row):
        messages.append(parse_chat_message(normalize_message(message)))

    candidate_actions_content = build_candidate_actions_content(source_row)
    if candidate_actions_content is not None:
        messages.append(parse_chat_message({"role": "user", "content": candidate_actions_content}))

    messages.extend(build_verifier_prompt_messages(row, source_row, config_specs=config_specs))

    verifier_output = row.get("verifier_output") or {}
    if isinstance(verifier_output, dict):
        content = verifier_output.get("raw_output") or json.dumps(verifier_output, ensure_ascii=False)
    else:
        content = json.dumps(verifier_output, ensure_ascii=False)

    messages.append(parse_chat_message({"role": "assistant", "content": content}))
    return messages


def row_to_agent_run(
    row: dict[str, Any],
    line_no: int,
    source_row: dict[str, Any] | None,
    *,
    config_specs: tuple[str, ...],
) -> AgentRun:
    metadata = build_metadata(row)
    messages = build_transcript_messages(row, source_row, config_specs=config_specs)
    name = trajectory_name(row)
    transcript = Transcript(name=name, messages=messages, metadata=metadata)
    return AgentRun(name=name, transcripts=[transcript], metadata=metadata)


def iter_agent_runs(
    path: Path,
    source_lookup: dict[tuple[str, str, str, int, int], dict[str, Any]] | None = None,
    *,
    config_specs: tuple[str, ...],
) -> list[AgentRun]:
    runs: list[AgentRun] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_no}.")
            source_row = None if source_lookup is None else source_lookup.get(row_lookup_key(row))
            runs.append(row_to_agent_run(row, line_no, source_row, config_specs=config_specs))
    return runs


def resolve_collection_id(client: Docent, collection_ref: str) -> str:
    if client.collection_exists(collection_ref):
        return collection_ref

    matches: list[tuple[str, str]] = []
    for collection in client.list_collections():
        if not isinstance(collection, dict):
            continue

        resolved_id = collection.get("collection_id") or collection.get("id")
        resolved_name = collection.get("name") or collection.get("display_name") or collection.get("title")
        if not isinstance(resolved_id, str) or not isinstance(resolved_name, str):
            continue

        if collection_ref == resolved_name:
            matches.append((resolved_id, resolved_name))

    if len(matches) == 1:
        resolved_id, resolved_name = matches[0]
        print(f"Resolved collection name '{resolved_name}' to id '{resolved_id}'", flush=True)
        return resolved_id

    if len(matches) > 1:
        raise SystemExit(
            f"Multiple collections are named '{collection_ref}'. Pass the real collection ID instead."
        )

    raise SystemExit(
        f"Collection '{collection_ref}' was not found as an ID or name. "
        "Pass a real collection ID, or omit --collection-id to create a new collection."
    )


def upload_batch(
    client: Docent,
    collection_id: str,
    batch: list[AgentRun],
    *,
    wait: bool,
    retry_split_on_failure: bool,
) -> tuple[int, list[str]]:
    try:
        result = client.add_agent_runs(collection_id, batch, wait=wait)
        return len(batch), list(result.get("job_ids") or [])
    except RuntimeError as exc:
        if not retry_split_on_failure or len(batch) <= 1:
            failed_names = [run.name or "<unnamed>" for run in batch]
            raise RuntimeError(
                f"Failed to upload batch with {len(batch)} run(s): {failed_names}. Original error: {exc}"
            ) from exc

        midpoint = len(batch) // 2
        left = batch[:midpoint]
        right = batch[midpoint:]
        print(
            f"Batch of {len(batch)} runs failed server-side; retrying as {len(left)} + {len(right)}",
            flush=True,
        )
        left_uploaded, left_job_ids = upload_batch(
            client,
            collection_id,
            left,
            wait=wait,
            retry_split_on_failure=retry_split_on_failure,
        )
        right_uploaded, right_job_ids = upload_batch(
            client,
            collection_id,
            right,
            wait=wait,
            retry_split_on_failure=retry_split_on_failure,
        )
        return left_uploaded + right_uploaded, left_job_ids + right_job_ids


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.is_file():
        raise SystemExit(f"Input file not found: {args.input_jsonl}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be greater than 0")

    if args.disable_proxy_env:
        disable_proxy_env()

    install_default_request_timeout(args.timeout_seconds)

    source_jsonl = args.source_jsonl or default_source_jsonl(args.input_jsonl)
    source_lookup = None
    if source_jsonl is not None:
        if not source_jsonl.is_file():
            raise SystemExit(f"Source JSONL not found: {source_jsonl}")
        source_lookup = load_source_lookup(source_jsonl)
        print(f"Using source rows: {source_jsonl}", flush=True)
    else:
        print("No source rows provided; uploaded transcripts will not include the previous message.", flush=True)

    client_kwargs = {
        "api_key": args.api_key,
        "domain": args.domain,
        "server_url": args.server_url,
        "web_url": args.web_url,
    }

    print("Connecting to Docent...", flush=True)
    try:
        client = Docent(**client_kwargs)
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            "Failed to connect to Docent. "
            f"Try --timeout-seconds 10 to fail faster, or --disable-proxy-env if proxy settings are broken. "
            f"Original error: {exc}"
        ) from exc

    collection_id = args.collection_id
    if collection_id is None:
        collection_name = args.collection_name or default_collection_name(args.input_jsonl)
        collection_id = client.create_collection(
            name=collection_name,
            description=f"Uploaded from {args.input_jsonl}",
        )
        print(f"Created collection: {collection_id} ({collection_name})", flush=True)
    else:
        collection_id = resolve_collection_id(client, collection_id)
        print(f"Using collection: {collection_id}", flush=True)

    config_specs = tuple(args.config_specs or [])
    runs = iter_agent_runs(args.input_jsonl, source_lookup=source_lookup, config_specs=config_specs)
    print(f"Prepared {len(runs)} runs from {args.input_jsonl}", flush=True)

    uploaded = 0
    job_ids: list[str] = []
    for start in range(0, len(runs), args.batch_size):
        batch = runs[start : start + args.batch_size]
        batch_uploaded, batch_job_ids = upload_batch(
            client,
            collection_id,
            batch,
            wait=args.wait,
            retry_split_on_failure=args.retry_split_on_failure,
        )
        uploaded += batch_uploaded
        job_ids.extend(batch_job_ids)
        print(f"Uploaded {uploaded}/{len(runs)}", flush=True)

    if not args.wait and job_ids:
        print(f"Enqueued job_ids={','.join(job_ids)}", flush=True)

    print(f"Done. collection_id={collection_id} uploaded_runs={uploaded}", flush=True)


if __name__ == "__main__":
    main()
