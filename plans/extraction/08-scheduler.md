# Phase 8 — Scheduler / Background Jobs

## Status: SKIPPED

### Reason

No background jobs. Compaction runs inline within `AgentEngine::run` when
history exceeds the configured limit.

### Notes

- This phase was evaluated and intentionally skipped.
