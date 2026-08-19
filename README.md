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
| *(default)* | Canonical LLM types, the `LlmProvider`, `EmbeddingProvider`, `BatchGenerationProvider` and `BatchEmbeddingProvider` traits, `one_shot`, the full agent engine + compaction. No HTTP client in the tree. |
| `vertex` | `VertexClient`: Gemini + Anthropic (Claude) on Vertex AI. API-key auth (Gemini) or service-account auth (both), token caching, streaming SSE parsing, Gemini thinking-model `thoughtSignature` handling, Gemini text embeddings via `EmbeddingProvider`, async bulk generation and bulk embedding at half price via `BatchGenerationProvider` / `BatchEmbeddingProvider`. |
| `batch` | `run_batch` / `run_embedding_batch` — submit a batch job and poll it to completion. Enabled by `vertex`. |
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

### Text embeddings (Gemini)

```rust
use senno::{EmbedRequest, EmbedTaskType, EmbeddingProvider};

let resp = client
    .embed(
        &EmbedRequest::new("gemini-embedding-001", ["red shoes", "blue shoes"])
            .with_task_type(EmbedTaskType::RetrievalDocument)
            .with_output_dimensionality(768),
    )
    .await?;

// One entry per input text, in request order.
let vectors: Vec<Vec<f32>> = resp
    .into_embeddings()?
    .into_iter()
    .map(|e| e.values)
    .collect();
```

Works with either auth mode — batched `batchEmbedContents` calls on API-key
auth, per-text `:predict` calls on service-account auth (`gemini-embedding-001`
accepts exactly one text per predict call). Both fan out with bounded
concurrency, retry rate limits and server faults with backoff, and preserve
input order. Requires `VertexProvider::Gemini`; the Anthropic client returns a
config error.

Bulk embedding is billed per text, so a call that fails partway keeps what it
already paid for. `into_embeddings()` above is the strict reading — it consumes
the response and fails on the first error. The lenient alternative keeps the
successes and tells you which inputs to retry:

```rust
for (index, error) in resp.failures() {
    eprintln!("text {index} still needs an embedding: {error}");
}
let vector = resp.embeddings[0].as_ref().map(|e| &e.values);
```

`resp.total_token_count` carries the billed input tokens where the provider
reports them: per text on the `:predict` path, per call on `batchEmbedContents`
(which also reports no truncation flag, so `Embedding::truncated` is only ever
populated under service-account auth).

> **Normalization:** `gemini-embedding-001` only normalizes its full 3072-dim
> output. If you set `with_output_dimensionality` to anything smaller (768,
> 1536), the vectors come back **unnormalized** and you must L2-normalize them
> yourself before any cosine-similarity comparison. senno returns the values as
> the API gives them. `gemini-embedding-2` normalizes truncated dimensions for
> you.

`with_title` is only accepted alongside `EmbedTaskType::RetrievalDocument` —
Google rejects every other pairing, so senno rejects it before the round trip.
Model ids may be written either way: `gemini-embedding-001` or
`models/gemini-embedding-001`.

### Bulk jobs (Gemini Batch API)

Asynchronous batches at half the interactive price, with results typically
within 24 hours. API-key auth only — the service-account Vertex batch API
works through GCS/BigQuery, which senno doesn't manage. All requests in one
batch must target the same model.

`BatchJob<T>` is shared by both kinds: `BatchJob<GenerateResponse>` for
generation, `BatchJob<Embedding>` for embeddings. Same states, same polling,
same cancel endpoint.

```rust
use senno::providers::vertex::{VertexProvider, get_vertex_client};
use senno::GenerateRequest;

let client = get_vertex_client(VertexProvider::Gemini, None).await?;
let done = senno::run_batch(
    &client,
    &[
        GenerateRequest::one_shot("gemini-2.5-flash", "You are terse.", "Why is the sky blue?"),
        GenerateRequest::one_shot("gemini-2.5-flash", "You are terse.", "Why is the sea salty?"),
    ],
)
.await?;

for response in done.responses {
    println!("{:?}", response?.text());
}
```

`run_batch` submits and polls to completion — 30s to start, doubling to a 5
minute ceiling — so the job name never has to reach your code.

`done.responses` come back in request order; a per-item failure is an `Err`
entry, so one bad request doesn't hide the other results. `cancel_batch(&name)`
stops a running job.

### Bulk embeddings (Gemini Batch API)

Embedding a corpus is the cheapest thing to move off the interactive path:
$0.075 per 1M input tokens against $0.15. Submit once, poll, collect.

```rust
use senno::{EmbedRequest, EmbedTaskType};

let done = senno::run_embedding_batch(
    &client,
    &EmbedRequest::new("gemini-embedding-001", corpus)
        .with_task_type(EmbedTaskType::RetrievalDocument),
)
.await?;

for embedding in done.responses {
    println!("{:?}", embedding?.values);
}
```

`responses` is one entry per input text in submission order, with per-item
errors as `Err`. `cancel_embedding_batch(&name)` stops a running job.

For a job that outlives the process, keep the name from `create_embedding_batch`
and resume with `wait_for_embedding_batch(&client, &name, polling)`.
`BatchPolling` sets the interval, the ceiling, and an optional timeout — on
timeout the job keeps running and the error names it, so you can poll again.
`wait_for_batch` is the generation equivalent.

Use `embed()` instead when you need vectors now — a search query can't wait 24
hours. The usual split is batch for the corpus, interactive for queries.

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
| Metadata server | Gemini + Anthropic | `VertexConfig::with_metadata_server()` — GCE/Cloud Run workload identity, no key to ship |
| Service account | Gemini + Anthropic | `with_service_account_key(ServiceAccountKey)` or `with_service_account_key_path(...)`, plus `VERTEX_PROJECT_ID` (+ optional `VERTEX_REGION`, default `asia-south1`) |

senno does **not** auto-resolve `GOOGLE_APPLICATION_CREDENTIALS`; pass the key
explicitly, or use `with_metadata_server()` on Google infrastructure.

Explicit `VertexConfig` values win over environment variables, so a stray
`GEMINI_API_KEY` can't silently redirect a configured service account.
`VertexConfig::with_base_url(...)` replaces the Google origin for proxies and
egress gateways.

## Toolchain

Pinned via `rust-toolchain.toml` (Rust 1.97.0, edition 2024).

## License

MIT
