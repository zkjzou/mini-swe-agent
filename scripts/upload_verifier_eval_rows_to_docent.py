#!/usr/bin/env python3

"""Upload verifier-evaluation JSONL rows to Docent as one AgentRun per row."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from docent import Docent
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import parse_chat_message
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
    parser.add_argument("--server-url", help="Optional Docent server URL.")
    parser.add_argument("--web-url", help="Optional Docent web URL.")
    parser.add_argument("--batch-size", type=int, default=100, help="Runs per upload batch.")
    return parser.parse_args()


def disable_proxy_env() -> None:
    for key in _PROXY_ENV_VARS:
        os.environ.pop(key, None)


def default_collection_name(path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"{path.stem}_{timestamp}"


def build_metadata(row: dict) -> dict:
    metadata = dict(row)
    verifier_output = metadata.get("verifier_output")
    if isinstance(verifier_output, dict):
        verifier_output = dict(verifier_output)
        verifier_output.pop("response", None)
        metadata["verifier_output"] = verifier_output
    return metadata


def row_to_agent_run(row: dict, line_no: int) -> AgentRun:
    metadata = build_metadata(row)
    verifier_output = metadata.get("verifier_output") or {}
    if isinstance(verifier_output, dict):
        content = verifier_output.get("raw_output") or json.dumps(verifier_output, ensure_ascii=False)
    else:
        content = json.dumps(verifier_output, ensure_ascii=False)

    message = parse_chat_message({"role": "assistant", "content": content})
    name = f"{row.get('instance_id', 'unknown')}:{row.get('row_index', line_no - 1)}"
    transcript = Transcript(messages=[message], metadata=metadata)
    return AgentRun(name=name, transcripts=[transcript], metadata=metadata)


def iter_agent_runs(path: Path) -> list[AgentRun]:
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
            runs.append(row_to_agent_run(row, line_no))
    return runs


def main() -> None:
    args = parse_args()
    disable_proxy_env()

    if not args.input_jsonl.is_file():
        raise SystemExit(f"Input file not found: {args.input_jsonl}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    client = Docent(api_key=args.api_key, server_url=args.server_url, web_url=args.web_url)
    collection_id = args.collection_id
    if collection_id is None:
        collection_name = args.collection_name or default_collection_name(args.input_jsonl)
        collection_id = client.create_collection(
            name=collection_name,
            description=f"Uploaded from {args.input_jsonl}",
        )
        print(f"Created collection: {collection_id} ({collection_name})", flush=True)
    else:
        print(f"Using collection: {collection_id}", flush=True)

    runs = iter_agent_runs(args.input_jsonl)
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
