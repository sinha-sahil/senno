# senno — Extraction Overview

## Goal

Extract arche's LLM stack (`src/llm`, `src/agent`, `src/gcp/vertex`, plus the GCP
token machinery it depends on) into `senno`: a standalone crate publishing to
crates.io, with its own error type, feature flags, CI, and tests.

**senno** (Italian: wits, judgment) — provider-agnostic LLM client and streaming
tool-calling agent engine, Vertex AI first (Gemini + Claude), with built-in
history compaction.

## Scope

**In:**

- Canonical LLM types (`Message`, `ContentPart`, `GenerateRequest`,
  `GenerateResponse`, `StreamChunk`, `ToolDefinition`, `ParameterSchema`) and the
  `LlmProvider` trait + `one_shot`.
- Agent engine: streaming tool-call loop, `AgentFlow`, sessions,
  `HistoryCompactor` + `LlmSummaryCompactor`, `AgentConfig`.
- Vertex AI provider: Gemini + Anthropic-on-Vertex, API-key and service-account
  auth, SSE stream parsing, thought-signature handling.
- GCP token machinery (`ServiceAccountKey`, `TokenSource`) copied in under the
  `vertex` feature; metadata-server auth kept and made public API.
- axum SSE adapter (`to_sse_event`) behind an `axum` feature.
- Integration tests (`tests/llm.rs`), CI release workflow, cargo-husky hooks,
  pinned toolchain.

**Out (explicitly):**

- No behavior changes beyond the error-type swap. New features (embeddings,
  observability hooks, subagents) come after 0.1.0.
- arche keeps its copy of the code until the already-planned 5.0.0 breaking
  release deletes `src/llm`, `src/agent`, `src/gcp/vertex` there.
- GCS signed-URL signing (`sign_blob`, `signer_email`, raw RSA parsing) stays in
  arche — it is a GCS concern, not a Vertex one. This drops the `rsa`/`sha2`
  deps entirely.

## Success Criteria

- `cargo check`, `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test --all-features`, `cargo fmt --check` all pass in the new repo.
- Default features build with no HTTP/axum dependencies (core = types + agent).
- Public API surface: `senno::{Error, Message, GenerateRequest, …, LlmProvider,
  one_shot}`, `senno::agent::*`, `senno::vertex::*` (feature-gated),
  `senno::agent::to_sse_event` (feature-gated).
- Ready to publish 0.1.0 (`cargo publish --dry-run` clean).

## Dependencies

- Source of truth read from arche at 4.15.2 (commit 12b3da5).
- Error convention: senno owns a `thiserror` enum `senno::Error`; the
  one-AppError rule stays arche-internal. Apps write
  `impl From<senno::Error> for AppError` at the boundary.
- arche-side deletion rides the 5.0.0 breaking release (with arche-extensions).
- User actions to finish claiming: `cargo publish` (needs cargo login),
  register senno.rs, create the GitHub repo and push.
