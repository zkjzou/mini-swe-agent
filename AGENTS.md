# Repository Guidelines

# The MOST IMPORTANT RULE YOU NEED TO FOLLOW: ONLY EDIT FILES IN /home/zkjzou/SWE-PRM/mini-swe-agent
After each action you take, you need to output you didn't edit files outside the current work directory. Otherwise, I will assume you disobey this rule. 

## Project Structure & Module Organization
- `src/minisweagent/` is the main package. Core areas: `agents/`, `models/`, `environments/`, `run/`, and `config/`.
- `tests/` holds pytest suites, typically named `test_*.py` and grouped by area (e.g., `tests/models/`).
- `docs/` contains MkDocs content and assets; `mkdocs.yml` is the site config.
- `tests/test_data/` stores fixture data (excluded from linting).

Example layout:
```
src/minisweagent/
  agents/  models/  environments/  run/  config/
tests/
docs/
```

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `pip install -e '.[dev]'` installs dev dependencies (pytest, ruff, mkdocs, pre-commit).
- `mini` or `mini -v` runs the CLI; `mini-extra` runs extra utilities.
- `python src/minisweagent/run/hello_world.py` runs a minimal script directly.
- `pytest -n auto` runs the test suite in parallel (recommended).
- `pre-commit install` and `pre-commit run --all-files` apply lint/format hooks.
- `mkdocs serve` previews docs locally.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; 120-char line length (see `pyproject.toml`).
- Use Ruff for linting and formatting (`ruff` + `ruff-format` via pre-commit).
- Prefer `snake_case` for modules/functions and `CamelCase` for classes.
- Keep components minimal; add new variants in `extra/` folders instead of expanding core logic.

## Testing Guidelines
- Tests use `pytest` with `pytest-xdist` for parallel runs.
- Place new tests under `tests/` near related domains (e.g., `tests/environments/`).
- Keep tests concise and readable; avoid verbose fixtures unless needed.

## Commit & Pull Request Guidelines
- Recent commits use short prefixes like `Feat:`, `Fix:`, `Doc:`, `CI:`, or `chore:` (often with PR numbers).
- PRs should include: a brief summary, linked issues, and tests run (or note if not run).
- For UI/textual changes, include a screenshot or short GIF when practical.

## Design & Architecture Notes
- The project aims to stay minimal, hackable, and high-quality.
- Prefer adding new component versions over complicating existing ones; keep shared helpers in `utils/`.

# ExecPlans
When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.

# Project Overview
The high-level research goals and conceptual framework for this project are outlined in PROJECT.md. This research builds upon the existing mini-swe-agent repository, which should be used as the primary reference for environment utilities and structural context. PROJECT.md is provided for high-level context and research alignment only. It defines the "what" and "why" of the SWE-WorldPRM framework. It is not a step-by-step implementation guide. For detailed coding instructions, logic flow, and granular implementation guidance, rely on the existing codebase patterns and specific task directives rather than the research document.