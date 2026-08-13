# Phase 7 — Remote Calls (Vertex AI)

## Objective

Port the Vertex AI provider: HTTP client, auth resolution, both model backends,
SSE stream parsing.

## Tasks

- [x] `src/providers/vertex/client.rs`: `VertexClient` (+ `VertexProvider` enum),
  `authorize` (API key header vs Bearer token), `send` with status handling,
  `LlmProvider` impl dispatching to backends
- [x] `src/providers/vertex/config.rs`: `VertexConfig` builder + `resolve_auth`
  (VERTEX_API_KEY / GEMINI_API_KEY → ApiKey; VERTEX_PROJECT_ID + SA key →
  ServiceAccount with region default `asia-south1`)
- [x] `src/providers/vertex/gemini.rs`: generateContent/streamGenerateContent,
  generativelanguage vs aiplatform endpoints, thought-signature buffering,
  url_context web-fetch tool, SSE frame parsing
- [x] `src/providers/vertex/anthropic.rs`: rawPredict/streamRawPredict,
  SA-auth-only guard, content-block mapping, input_json_delta accumulation
- [x] `src/providers/vertex/mod.rs`: `get_vertex_client` entry point; re-export
  public surface (`VertexClient`, `VertexProvider`, `VertexConfig`,
  `ServiceAccountKey`, `TokenSource`); public path is
  `senno::providers::vertex` — vendor code lives under `src/providers/`, one
  directory per provider, backends flat inside
- [x] All `AppError::dependency_failed("vertex-ai", …)` → `Error::provider`

## Outputs

`src/providers/mod.rs`, `src/providers/vertex/{mod,client,config,env,token,gemini,anthropic}.rs`

## Validation

Provider unit tests (endpoint construction, wire serialization, SSE frame
boundary, tool-result mapping) pass under `--features vertex`.
