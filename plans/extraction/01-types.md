# Phase 1 — Types

## Objective

Port the canonical LLM types and define senno's own error type, replacing every
`arche::error::AppError` use.

## Tasks

- [ ] `src/error.rs`: `senno::Error` (thiserror) with variants:
  - `Provider { provider, detail, retryable }` — upstream LLM/auth failures
    (replaces `dependency_failed` / `dependency_failed_permanent`)
  - `BadResponse(String)` — unusable model output (no tool call, bad args)
  - `Config(String)` — local misconfiguration (replaces most `internal_error`)
  - `Tool { name, detail }` — for `AgentFlow` implementations to return
  - `Internal(String)` — everything else
  - Helpers: `provider()`, `provider_permanent()`, `bad_response()`, `config()`,
    `tool()`, `internal()`, `is_provider_error()`, `is_bad_response()`,
    `is_retryable()`
- [ ] `src/types.rs`: port `arche/src/llm/types.rs` verbatim; `parse()` maps to
  `Error::BadResponse`
- [ ] `src/provider.rs`: `LlmProvider` trait + `LlmStream`, error type swapped
- [ ] `src/one_shot.rs`: trivial port
- [ ] `src/agent/types.rs`, `src/agent/config.rs`: port; `AgentConfigBuilder::build`
  returns `Error::Config` for empty model

## Outputs

`src/error.rs`, `src/types.rs`, `src/provider.rs`, `src/one_shot.rs`,
`src/agent/types.rs`, `src/agent/config.rs`

## Validation

Unit tests ported alongside each file pass; error Display strings keep the
detail (engine and token retry logs rely on it).
