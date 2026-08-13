# senno

> *senno* — Italian for **wits, judgment, good sense**. In *Orlando Furioso*,
> when Orlando loses his senno, it is found stored in a jar on the Moon.
> Yours lives in this crate. And *il senno di poi* — hindsight — is exactly
> what an agent engine should give you about every response it produced.

**Wits for your backend**: a provider-agnostic LLM client and streaming
tool-calling agent engine for Rust services — Vertex AI first (Gemini and
Claude on Vertex), with built-in conversation history compaction and an
axum-ready SSE adapter.

Extracted from [arche](https://github.com/sinha-sahil/arche), where it powered
production agents; arche ≥ 5.0 depends on senno being your LLM layer instead of
shipping its own.

## Features

| Feature flag | What you get |
|---|---|
| *(default)* | Canonical LLM types, `LlmProvider` trait, `one_shot`, the full agent engine + compaction. No HTTP client in the tree. |
| `vertex` | `VertexClient`: Gemini + Anthropic (Claude) on Vertex AI. API-key auth (Gemini) or service-account auth (both), token caching, streaming SSE parsing, Gemini thinking-model `thoughtSignature` handling. |
| `axum` | `agent::to_sse_event` — map agent events straight into `axum::response::sse`. |

## Quick start

### One-shot prompt (Gemini via API key)

```rust
use senno::providers::vertex::{VertexProvider, get_vertex_client};
use senno::{GenerateRequest, one_shot};

// Auth resolved from VERTEX_API_KEY / GEMINI_API_KEY env.
let client = get_vertex_client(VertexProvider::Gemini, None).await?;
let resp = one_shot(
    &client,
    &GenerateRequest::one_shot("gemini-2.5-flash", "You are terse.", "Why is the sky blue?"),
)
.await?;
println!("{}", resp.text().unwrap_or_default());
```

### Claude on Vertex (service account)

```rust
use senno::providers::vertex::{VertexConfig, VertexProvider, get_vertex_client};

let client = get_vertex_client(
    VertexProvider::Anthropic,
    Some(
        VertexConfig::default()
            .with_project_id("my-project")
            .with_region("asia-south1")
            .with_service_account_key_path("/path/to/sa.json"),
    ),
)
.await?;
```

### Typed output via tool calling

```rust
use senno::{GenerateRequest, ParameterSchema, ToolDefinition};

#[derive(serde::Deserialize)]
struct Verdict { answer: String }

let tool = ToolDefinition::new("report", "Report the answer").with_parameters(
    ParameterSchema::object()
        .with_property("answer", ParameterSchema::string("the answer"))
        .with_required(["answer"]),
);
let req = GenerateRequest::one_shot("gemini-2.5-flash", "sys", "prompt").with_tools(vec![tool]);
let verdict: Verdict = senno::one_shot(&client, &req).await?.parse()?;
```

### A streaming agent with compaction

```rust
use senno::agent::{AgentConfig, get_agent_engine};

let engine = get_agent_engine(client, AgentConfig::builder("gemini-2.5-flash").build()?)
    .with_default_summarizer("gemini-2.5-flash-lite");

// In an axum handler (feature = "axum"):
// let stream = engine.run(&flow, &mut session, &user_message);
// Sse::new(stream.map(|e| e.map(senno::agent::to_sse_event)))
```

Implement `senno::agent::AgentFlow` to define your system prompt, tool
definitions, and tool execution. The engine handles the streaming loop, tool
rounds, session state, and — when history exceeds your configured limit —
compacts older turns into an LLM-written summary at a safe message boundary
(tool-call/result pairs are never split; on compactor failure it degrades to
plain truncation).

## Errors

Everything returns `senno::Error` — a small enum (`Provider`, `BadResponse`,
`Config`, `Tool`, `Internal`) with `is_retryable()` for transient upstream
failures. Convert at your application boundary, e.g.:

```rust
impl From<senno::Error> for AppError {
    fn from(e: senno::Error) -> Self {
        AppError::dependency_failed("senno", e.to_string())
    }
}
```

## Vertex auth resolution

| Method | Models | Configuration |
|---|---|---|
| API key | Gemini only | `VertexConfig::with_api_key(...)` or `VERTEX_API_KEY` / `GEMINI_API_KEY` env |
| Service account | Gemini + Anthropic | `with_service_account_key(ServiceAccountKey)` or `with_service_account_key_path(...)`, plus `VERTEX_PROJECT_ID` (+ optional `VERTEX_REGION`, default `asia-south1`) |

senno does **not** auto-resolve `GOOGLE_APPLICATION_CREDENTIALS`; pass the key
explicitly. `TokenSource::metadata` is available for GCE/Cloud Run
metadata-server auth if you wire it yourself.

## Toolchain

Pinned via `rust-toolchain.toml` (Rust 1.97.0, edition 2024).

## License

MIT
