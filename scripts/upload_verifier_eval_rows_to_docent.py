#!/usr/bin/env python3

"""Upload verifier-evaluation JSONL rows to Docent as one AgentRun per row."""

from __future__ import annotations

import argparse
import json
import os
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


def build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    selected_is_gold = bool(row.get("selected_is_gold"))
    metadata: dict[str, Any] = {
        "instance_id": augmented_instance_id(row),
        "original_instance_id": row.get("instance_id"),
        "is_selected": selected_is_gold,
        "selected_is_gold": selected_is_gold,
        "step_index": int_or_default(row.get("step_index"), -1),
        "message_index": int_or_default(row.get("message_index"), -1),
    }
    for key, value in row.items():
        if key in metadata or key == "instance_id":
            continue
        metadata[key] = value
    verifier_output = metadata.get("verifier_output")
    if isinstance(verifier_output, dict):
        verifier_output = dict(verifier_output)
        verifier_output.pop("response", None)
        metadata["verifier_output"] = verifier_output
    return metadata


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


def source_history_messages(source_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(source_row, dict):
        return []
    history = source_row.get("history_trajectory")
    if not isinstance(history, list):
        return []
    return [message for message in history if isinstance(message, dict) and isinstance(message.get("role"), str)]


def build_transcript_messages(row: dict[str, Any], source_row: dict[str, Any] | None) -> list[Any]:
    messages: list[Any] = []
    for message in source_history_messages(source_row):
        messages.append(parse_chat_message(normalize_message(message)))

    metadata = build_metadata(row)
    verifier_output = metadata.get("verifier_output") or {}
    if isinstance(verifier_output, dict):
        content = verifier_output.get("raw_output") or json.dumps(verifier_output, ensure_ascii=False)
    else:
        content = json.dumps(verifier_output, ensure_ascii=False)

    messages.append(parse_chat_message({"role": "assistant", "content": content}))
    return messages


def row_to_agent_run(row: dict[str, Any], line_no: int, source_row: dict[str, Any] | None) -> AgentRun:
    metadata = build_metadata(row)
    messages = build_transcript_messages(row, source_row)
    name = trajectory_name(row)
    transcript = Transcript(name=name, messages=messages, metadata=metadata)
    return AgentRun(name=name, transcripts=[transcript], metadata=metadata)


def iter_agent_runs(
    path: Path,
    source_lookup: dict[tuple[str, str, str, int, int], dict[str, Any]] | None = None,
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
            runs.append(row_to_agent_run(row, line_no, source_row))
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

    runs = iter_agent_runs(args.input_jsonl, source_lookup=source_lookup)
    print(f"Prepared {len(runs)} runs from {args.input_jsonl}", flush=True)

    uploaded = 0
    for start in range(0, len(runs), args.batch_size):
        batch = runs[start : start + args.batch_size]
        client.add_agent_runs(collection_id, batch)
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(runs)}", flush=True)

    print(f"Done. collection_id={collection_id} uploaded_runs={uploaded}", flush=True)


if __name__ == "__main__":
    main()
