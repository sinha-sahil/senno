# Extending

Five plug points, in order of likelihood:

| You want to | Plug into |
|---|---|
| Define agent behaviour (prompts, tools) | `impl AgentFlow` |
| Add tools you did not write (MCP servers, a plugin registry) | `impl ToolSource`, returned from `AgentFlow::tool_sources` |
| Swap the LLM backend (OpenAI, Bedrock, Ollama, local) | `impl LlmProvider` |
| Replace default summarization (vector recall, server-side memory) | `impl HistoryCompactor` |
| Surface custom UI events to the client | Return `ToolOutput::text(..).data(type, payload)` |

Nothing in `senno::agent` changes when you extend any of these.

## Getting started

Minimum viable agent with the built-in Vertex + Gemini. Needs the `vertex` (Vertex client) and `axum` (`to_sse_event`) features:

```toml
senno = { version = "0.1", features = ["vertex", "axum"] }
```

```rust
use senno::agent::{AgentConfig, AgentFlow, AgentSession, ToolOutput, get_agent_engine, to_sse_event};
use senno::providers::vertex::{VertexProvider, get_vertex_client};
use senno::{Error, ToolDefinition};

struct MyFlow;

impl AgentFlow for MyFlow {
    fn system_prompt(&self) -> String {
        "You are a helpful assistant.".into()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }

    fn execute_tool<'a>(
        &'a self,
        _name: &'a str,
        _args: &'a serde_json::Value,
        _session: &'a AgentSession,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move { Ok(ToolOutput::text("")) })
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let client = get_vertex_client(VertexProvider::Gemini, None).await?;
    let config = AgentConfig::builder("gemini-2.5-flash").build()?;
    let engine = get_agent_engine(client, config);

    let mut session = AgentSession::new("sess-1", "demo");
    let stream = engine.run(&MyFlow, &mut session, "Hi!");

    use futures::StreamExt;
    futures::pin_mut!(stream);
    while let Some(event) = stream.next().await {
        if let Ok(e) = event {
            let sse = to_sse_event(e);
            println!("{sse:?}");
        }
    }
    Ok(())
}
```

## Custom flow and tools

`AgentFlow` captures your domain: system prompt, tool schemas, tool executors. The flow can hold DB pools, API clients, cached data — anything. It has to be `Send + Sync`.

```rust
use senno::agent::{AgentFlow, AgentSession, ToolOutput};
use senno::{Error, ParameterSchema, ToolDefinition};
use std::future::Future;
use std::pin::Pin;

struct ShoppingFlow {
    db: sqlx::PgPool,
}

impl AgentFlow for ShoppingFlow {
    fn system_prompt(&self) -> String {
        "You help shoppers find products. Use search_catalog when they ask for something.".into()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "search_catalog",
                "Search the product catalog by natural-language query.",
            )
            .with_parameters(
                ParameterSchema::object()
                    .with_property(
                        "query",
                        ParameterSchema::string("Natural-language search query"),
                    )
                    .with_property(
                        "limit",
                        ParameterSchema::integer("Max results (default 5)"),
                    )
                    .with_required(["query"]),
            ),
        ]
    }

    fn execute_tool<'a>(
        &'a self,
        name: &'a str,
        args: &'a serde_json::Value,
        _session: &'a AgentSession,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            match name {
                "search_catalog" => {
                    let query = args["query"].as_str().unwrap_or("");
                    // ... run sqlx query ...
                    let products = serde_json::json!([{"id": "p1", "name": "Red shoes"}]);

                    Ok(ToolOutput::text(format!("Found 1 match for {query}."))
                        .data("product_list", products))
                }
                _ => Ok(ToolOutput::text(format!("Unknown tool: {name}"))),
            }
        })
    }
}
```

Key points:

- **`ToolDefinition` is typed** — use the `ParameterSchema` builders; don't hand-write JSON Schema.
- **`ToolOutput::text(..).data(..)` is dual-output** — `text` is what the LLM sees and reasons over; `data` reaches the client directly via `SseEvent::Data` for custom UI rendering. Use text-only for "facts for the model" and add data for "render this card in the UI."
- **Session is read-only to tools.** Tools see `&AgentSession`, can read metadata/history for context, but mutation goes through `ToolOutput.session_metadata` which the engine merges.

## Tools you did not write

`tool_definitions` is for tools you implement. For tools that arrive from
somewhere else — an MCP server, a plugin registry, a remote catalogue — implement
`ToolSource` and name it in `tool_sources`. That method is defaulted to empty, so
a flow that has none stays exactly as written above.

```rust
pub trait ToolSource: Send + Sync {
    fn definitions(&self) -> &[ToolDefinition];
    fn handles(&self, name: &str) -> bool;
    fn invoke<'a>(&'a self, name: &'a str, args: &'a Value)
        -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>>;
}
```

`McpToolset` (feature `mcp`) is one, so registering MCP servers is one method:

```rust
impl AgentFlow for ShoppingFlow {
    fn tool_sources(&self) -> Vec<&dyn ToolSource> {
        vec![&self.mcp]
    }
    // system_prompt, tool_definitions and execute_tool unchanged
}
```

The engine does the rest:

- **It offers your tools first**, then each source's, dropping any name you
  already serve and warning about it. Model providers reject two tools of one
  name, and your implementation wins the clash.
- **It routes by ownership.** A name you did not declare in `tool_definitions`
  goes to the first source whose `handles` claims it. Everything else goes to
  `execute_tool`, so a source's tool never reaches your match arms and there is
  no fallthrough to write.

A source is a plain trait, so nothing about `senno::agent` depends on MCP — the
`mcp` feature supplies an implementation, not a special case.

## Custom LLM backend

Any type implementing `senno::LlmProvider` drops into `get_agent_engine`. Example: OpenAI Chat Completions.

```rust
use senno::{Error, GenerateRequest, GenerateResponse, LlmProvider, LlmStream, StreamChunk};
use std::future::Future;
use std::pin::Pin;

pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
}

impl LlmProvider for OpenAiClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>> {
        Box::pin(async move {
            // 1. Convert `request` → OpenAI's JSON body
            // 2. POST https://api.openai.com/v1/chat/completions
            // 3. Convert response → canonical GenerateResponse
            todo!()
        })
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(async move {
            // 1. Convert `request` with stream=true
            // 2. POST, parse OpenAI SSE
            // 3. Yield StreamChunk::{Text, ToolCall, Done} as frames arrive
            todo!()
        })
    }
}
```

From the engine's perspective this is identical to `VertexClient`:

```rust
let engine = get_agent_engine(OpenAiClient { .. }, config);
```

## Custom history compactor

The default `LlmSummaryCompactor` makes an extra LLM call every time history overflows. If that's too expensive, or you want semantic recall, implement `HistoryCompactor` yourself.

```rust
use senno::Error;
use senno::agent::{ChatMessage, HistoryCompactor};
use std::future::Future;
use std::pin::Pin;

struct VectorMemoryCompactor {
    // e.g. a qdrant client
}

impl HistoryCompactor for VectorMemoryCompactor {
    fn compact<'a>(
        &'a self,
        messages: &'a [ChatMessage],
    ) -> Pin<Box<dyn Future<Output = Result<ChatMessage, Error>> + Send + 'a>> {
        Box::pin(async move {
            // 1. Embed messages, upsert to your vector store keyed by session id.
            // 2. Return a stub Assistant message so the model knows prior history exists.
            Ok(ChatMessage::Assistant {
                content: "[prior turns stored in long-term memory]".into(),
            })
        })
    }
}
```

Attach it to the engine:

```rust
let engine = get_agent_engine(client, config)
    .with_compactor(VectorMemoryCompactor { /* ... */ });
```

Or use the built-in summarizer without writing your own:

```rust
let engine = get_agent_engine(client, config)
    .with_default_summarizer("gemini-2.5-flash-lite"); // cheaper model for summaries
```

## Surfacing custom client events

Anything interesting a tool produces — UI cards, inline citations, product thumbnails, progress indicators — can reach the client without going through the LLM's text channel.

```rust
Ok(ToolOutput::text("Found 3 matches.")
    .data("product_card_list", serde_json::json!([
        { "id": "p1", "name": "Red shoes", "image": "..." },
        { "id": "p2", "name": "Blue shoes", "image": "..." },
        { "id": "p3", "name": "Green shoes", "image": "..." },
    ])))
```

Engine emits `SseEvent::Data { type: "product_card_list", payload: [...] }`. Client JS dispatches on `event.type` and renders the appropriate component.

The LLM only sees `"Found 3 matches."` — it doesn't have to hallucinate JSON to describe the cards.
