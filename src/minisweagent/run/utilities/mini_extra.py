#!/usr/bin/env python3

"""This is the central entry point to the mini-extra script. Use subcommands
to invoke other command line utilities like running on benchmarks, editing config,
inspecting trajectories, etc.
"""

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    ("minisweagent.run.utilities.config", ["config"], "Manage the global config file"),
    ("minisweagent.run.utilities.inspector", ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    ("minisweagent.run.benchmarks.swebench", ["swebench"], "Evaluate on SWE-bench (batch mode)"),
    ("minisweagent.run.benchmarks.swebench_single", ["swebench-single"], "Evaluate on SWE-bench (single instance)"),
    (
        "minisweagent.run.extra.monte_carlo",
        ["monte-carlo-rollout", "monte-carlo", "mc-rollout"],
        "Replay merged verifier rows and branch-roll out each candidate action",
    ),
    ("minisweagent.run.utilities.upload_docent", ["upload-docent"], "Upload trajectories to Docent"),
    (
        "minisweagent.run.utilities.sample_verifier_actions",
        ["sample-verifier-actions"],
        "Sample additional verifier action candidates from successful trajectories",
    ),
    (
        "minisweagent.run.utilities.merge_verifier_actions",
        ["merge-verifier-actions"],
        "Merge verifier action-sampling datasets from multiple sources",
    ),
    (
        "minisweagent.run.utilities.evaluate_verifier_actions",
        ["evaluate-verifier-actions", "eval-verifier-actions"],
        "Evaluate verifier gold-action selection on merged candidate rows",
    ),
    (
        "minisweagent.run.utilities.reanalyze_verifier_predictions",
        ["reanalyze-verifier-predictions"],
        "Reanalyze existing verifier prediction rows and write combined label distributions",
    ),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from mini-swe-agent.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            return import_module(module_path).app(args[1:], prog_name=f"mini-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
