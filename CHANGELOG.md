# Changelog

All notable changes to senno are documented here.


## [0.3.0] - 2026-08-19

### Features

- Add support for embedding models in Vertex AI (656a92d)


## [0.2.0] - 2026-08-13

### Features

- Initial extraction of llm, agent and vertex modules from arche (f3a2582)

## [0.1.0] - 2026-08-14

Initial release — extracted from [arche](https://github.com/sinha-sahil/arche)
4.15.2, where this stack ran in production as `arche::llm`, `arche::agent`, and
`arche::gcp::vertex`.

### Added

- Canonical LLM types (`Message`, `ContentPart`, `GenerateRequest`,
  `GenerateResponse`, `StreamChunk`, `ToolDefinition`, `ParameterSchema`),
  the `LlmProvider` trait, and `one_shot`.
- Agent engine: streaming tool-call loop over an `AgentFlow`, session state,
  and history compaction (`LlmSummaryCompactor` with truncation fallback).
- `vertex` feature: Gemini + Anthropic (Claude) on Vertex AI — API-key and
  service-account auth, cached token source (metadata-server auth included),
  SSE stream parsing, Gemini thinking-model `thoughtSignature` handling.
- `axum` feature: `agent::to_sse_event` adapter.
- `senno::Error` — provider/bad-response/config/tool taxonomy with
  `is_retryable()`.

### Changed (vs. the arche modules)

- Errors are `senno::Error` instead of arche's `AppError`; convert at the
  application boundary.
- Vertex lives at `senno::providers::vertex` (vendor code under `providers/`).
- GCS signed-URL signing removed from the token module (dropped the `rsa` and
  `sha2` dependencies); `TokenSource::metadata` and `CompactionResult` are now
  public API.
