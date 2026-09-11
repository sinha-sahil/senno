# Changelog

All notable changes to senno are documented here.


## [0.6.0] - 2026-09-11

### Features

- Add mcp tool sources to the agent flow (f395caa)


## [0.5.1] - 2026-09-10

### Bug Fixes

- Stop leaking provider and tool errors over the wire (ece5b1e)


## [0.5.0] - 2026-08-21

### Features

- Added config-driven construction for Vertex provider (6d0527d)


## [0.4.1] - 2026-08-20

### Bug Fixes

- Added missing predict api type in gemini (b24c419)

### Miscellaneous

- Fix the version reader for bump (826c697)


## [0.4.0] - 2026-08-19

### Features

- Add token usage parameters to agent and providers (0c67cd8)


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
