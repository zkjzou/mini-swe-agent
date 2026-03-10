# Feature Map Template

Use this file as the starting point for a living feature tracker. It is designed to keep implemented features,
planned features, ownership, and implementation touchpoints in one place.

## How To Use

1. Copy this file to `FEATURE_MAP.md` for the active tracker.
2. Add one row per feature or feature slice.
3. Keep `Status`, `Priority`, and `Next Step` current.
4. Link the main code, tests, configs, docs, and any ExecPlan when the feature becomes non-trivial.
5. Prefer adding new rows over rewriting history. Move completed planned work into the existing-features table.

## Status Legend

- `Live`: implemented and expected to work
- `In Progress`: currently being built
- `Planned`: accepted but not started
- `Needs Design`: idea exists but scope is not settled
- `Blocked`: cannot proceed yet
- `Experimental`: available but still unstable or research-grade
- `Deprecated`: still present but should not be extended

## Priority Legend

- `P0`: critical
- `P1`: important next
- `P2`: useful but not urgent
- `P3`: optional or exploratory

## Project Snapshot

| Field | Value |
| --- | --- |
| Project | |
| Maintainer | |
| Last reviewed | YYYY-MM-DD |
| Primary branch | |
| Notes | |

## Existing Features

| Feature ID | Area | Feature | Status | User Value | Main Entry Points | Tests/Docs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FEAT-001 | | | Live | | | | |

## Planned Features

| Feature ID | Area | Feature | Status | Priority | Why Now | Proposed Entry Points | Dependencies | Next Step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLAN-001 | | | Planned | P2 | | | | |

## Cross-Cutting Gaps

| Gap ID | Type | Description | Affects | Suggested Fix | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| GAP-001 | Docs | | | | | Planned |

## Decision Log

| Date | Decision | Reason | Related Features |
| --- | --- | --- | --- |
| YYYY-MM-DD | | | |

## Release Checklist

- [ ] Existing feature rows reflect current behavior
- [ ] Planned feature rows have owners or next actions
- [ ] Tests/docs links are updated for changed features
- [ ] Deprecated features are marked clearly

