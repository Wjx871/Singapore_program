# Sealed Independent Test Executor

The dedicated executor preserves the ordinary Test guard and has two modes.

Preflight (never predicts Independent Test probability):

```bash
.venv/bin/python -m scripts.run_sealed_evaluation \
  --config configs/experiment.yaml \
  --preflight-only
```

Authorized execution requires the frozen executor commit SHA, a unique run ID,
the exact CLI authorization phrase, a clean worktree, a non-CI environment,
and the exact interactive confirmation phrase. It must be run only once after
the executor commit has been pushed and frozen.
