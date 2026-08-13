# Phase 2 — Storage Layer

## Status: SKIPPED

### Reason

senno is a library crate with no persistence. Session state (`AgentSession`) is
a plain serializable struct the host application stores wherever it likes.

### Notes

- This phase was evaluated and intentionally skipped.
- If senno ever grows session-store adapters (redis/postgres), revisit here.
