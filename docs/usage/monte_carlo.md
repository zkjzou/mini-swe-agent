# Monte Carlo rollouts

Use `mini-extra monte-carlo` to replay a saved trajectory up to a chosen step and then sample rollouts from that exact environment state.

## Basic usage

```
mini-extra monte-carlo path/to/run.traj.json --step 3 --rollouts 5 --rollout-steps 1
```

This replays the first three action steps from `run.traj.json`, then executes five rollouts (one step each) from that state. Outputs are written under `monte_carlo_rollouts/<trajectory-name>/` and a summary file is stored at `monte_carlo_rollouts/rollouts.jsonl`.

## Include or exclude assistant thoughts

By default, the rollout prompt history includes assistant messages from the trajectory. To exclude them:

```
mini-extra monte-carlo path/to/run.traj.json --step 3 --exclude-thoughts
```

This keeps the system and user messages but drops assistant messages from the prompt history.

## Fixed actions

To execute a specific action instead of sampling from a model, pass `--rollout-action`:

```
mini-extra monte-carlo path/to/run.traj.json --step 2 --rollout-steps 1 --rollout-action "echo hello"
```

You can also pass a JSON list of actions with `--rollout-actions-json` (one action per rollout step).

## Notes

- `--step` is 1-based and refers to action steps (assistant actions). Use `--step 0` to skip replay.
- If the trajectory does not include config info, supply a fallback config with `--config`.
- Use `--model-kwargs-json` to control sampling parameters such as temperature or top_p.
