# MCP client

`feature = "mcp"`. Connect to a foreign [Model Context Protocol][mcp] server,
adopt its tools as senno `ToolDefinition`s, and let an agent call them.

senno owns the transport, the listing, the schema parse and the call. The host
owns which server, which credential, and when to connect.

MCP is a tool *source*, not a kind of agent. Register the servers once, then add
their tools to whatever `AgentFlow` you already have — a flow keeps its own
prompt, its own native tools, and gains the foreign ones.

```rust
use senno::mcp::{McpServer, McpToolset};

let mcp = McpToolset::connect(vec![
    McpServer::new("visit", "https://example.test/mcp")
        .with_header("X-Service-Token", token)
        .with_tool_prefix("visit__"),
    McpServer::new("ledger", "https://ledger.test/mcp"),
])
.await?;
```

```rust
impl AgentFlow for BookingFlow {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        my_native_tools()
    }

    fn tool_sources(&self) -> Vec<&dyn ToolSource> {
        vec![&self.mcp]
    }

    fn execute_tool<'a>(&'a self, name: &'a str, args: &'a Value, session: &'a AgentSession)
        -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>>
    {
        Box::pin(async move {
            match name {
                "search_catalog" => self.search_catalog(args).await,
                _ => Err(Error::tool(name, "this flow serves no such tool")),
            }
        })
    }
}
```

`tool_sources` is the whole integration. `McpToolset` implements
[`ToolSource`](../../src/agent/types.rs), and the engine does the rest: it offers
your own tools followed by every registered tool, and sends a call to whichever
owns the name. A foreign tool never reaches `execute_tool`, so there is no
fallthrough arm to write and no dispatch order to remember.

Your tools win a name clash — a source's copy of a name you already serve is
dropped with a warning, so the model never sees two of the same name and a
foreign server cannot shadow your `search_catalog`. Between two registered
servers, the first to claim a name keeps it. `mcp.close()` terminates every
registered session.

A registration is all-or-nothing: one unreachable server fails
`McpToolset::connect`. A silently partial toolset leaves a prompt promising
tools that are not there, which is worse than a failed build.

## Connecting

`McpToolset::connect` connects each server in turn; `McpClient::connect` is the
single-server form underneath it. Either performs the handshake and the listing
in one call:
`initialize`, the `notifications/initialized` notification, then `tools/list`
followed to the end of its cursor. The tool list is in hand when `connect`
returns, which is what makes `AgentFlow::tool_definitions` — a **sync** trait
method — free of I/O.

Connect once per configuration and share the toolset. It carries no per-caller
state, so one registration can serve every conversation on that configuration —
build it beside the flow rather than per request.

A `Mcp-Session-Id` returned by `initialize` is captured and echoed on every
later request. A server that issues one is stateful, so a client bound to that
session must not be shared across users.

When such a server answers 404 the session is gone, and the spec requires the
client to open a new one. `request` does that: it re-runs the handshake without
a session id and replays the call once. A second 404 in a row is an error rather
than a loop, and a 404 from a server that never issued a session id is treated
as a plain missing endpoint. **The tool list is not re-fetched** — a toolset
that changed underneath an open conversation is worse than a stale one, so it
stays pinned to what `connect` saw.

`close()` sends the `DELETE` the spec asks for, with the session id and the
configured headers, and forgets the session. A server answering 405 keeps its
own sessions, which is fine. It is a no-op when no session was ever issued, so
calling it on a stateless server costs nothing.

`timeout` bounds one request, not the whole of `connect`, which makes at least
three. A host that connects while holding a shared lock should put its own
deadline around the call.

Redirects are **not followed**. The configured headers are credentials, and
only well-known ones get stripped on a cross-origin hop, so following a 302
would hand `X-Service-Token` to whatever host the endpoint named. A redirect
surfaces as its own status error; point the config at the final URL.

A response is read with an 8 MiB cap and refused past it. The server is foreign,
and an unbounded read is an out-of-memory waiting to happen in whatever process
is holding the connection.

## Transport

One POST per JSON-RPC call, `Accept: application/json, text/event-stream`. A
reply framed as SSE is read as SSE; anything else is parsed as JSON.

A stream can carry more than the answer — a progress notification, or a
server-initiated request. The response is the frame with no `method`; every
other frame is skipped. Taking the first frame that merely *parses* would swallow
a `notifications/progress` and report the call as unreadable.

The protocol version is negotiated, not assumed. `2025-06-18` is offered on
`initialize`; the version the server answers with is what the
`MCP-Protocol-Version` header carries from then on. That header is deliberately
**absent from the `initialize` request** — the spec scopes it to subsequent
requests, and a server must reject an unsupported version header, which would
pre-empt the negotiation it comes from.

senno speaks `2025-06-18` and `2025-03-26`. A server answering with anything
else is refused rather than humoured: `2024-11-05` in particular is the
deprecated HTTP+SSE transport, so adopting it would mean speaking this
transport at a server that does not implement it.

`tools` must appear in the server's declared capabilities. Calling `tools/list`
without it would use a capability that was never negotiated, so a server that
declares no tools fails `connect` with that message instead of an opaque
method error.

The 10s timeout is overridable and bounds one request. When it passes, the
client sends `notifications/cancelled` for that request id, because a server is
otherwise entitled to keep working after we stop waiting. The `initialize`
request is never cancelled — the spec forbids it.

**Read-only calls are retried; a tool call never is.** `initialize` and
`tools/list` retry up to three times on a transient failure (5xx, 408, 429, or a
transport error) with a 500ms backoff doubling to a cap. `tools/call` is sent
exactly once: MCP carries no idempotency key, so replaying a timed-out
`commit_booking` is a second booking.

Authentication is whatever static headers you pass. OAuth — dynamic client
registration and authorization-server discovery — is not implemented. Header
values never reach `Debug` output; only header names do.

## What survives the schema parse

senno's `ParameterSchema` is a narrow subset of JSON Schema: `type`,
`description`, `properties`, `items`, `required`, and string `enum`. A foreign
`inputSchema` is downcast into it, and the parse never fails — it drops what it
cannot express:

| In the foreign schema | Result |
| --- | --- |
| `type` missing, `properties` or `items` present | inferred as object / array |
| `type: ["string", "null"]` | first non-null member |
| `anyOf` / `oneOf` / `allOf`, no usable `type` | first branch that converts |
| `format`, `additionalProperties`, `$schema`, `minimum`, `default`, `const` | dropped |
| `$ref` with no sibling `type` | the property is dropped |
| a non-string `enum` | the enum is dropped, the type is kept |

`required` keeps only the names whose property survived, so the model is never
told to send a field the advertised schema no longer declares. A tool whose
top-level schema is not an object is dropped from the listing with a warning
rather than advertised in a shape the provider would reject.

Both live providers ignore the dropped keywords anyway: `parameters` goes
straight to Gemini `functionDeclarations` and Anthropic `input_schema`.

## Calling a tool

The display name follows the spec's order — `title`, then `annotations.title`,
then `name`.

**Behaviour hints are dropped unless you ask for them.** `destructiveHint` and
`readOnlyHint` are the server's claims about itself, and the spec requires a
client to treat them as untrusted; a hostile server labels a destructive tool
read-only. They reach the model only for a server built with
`with_trusted_annotations()`. `annotations.title` is exempt: it is display text,
and the spec names it in the precedence above.

When a tool publishes an `outputSchema`, `structuredContent` is checked against
it and a mismatch is logged with the offending path — `$.lines[1].sku` — rather
than failing the call, since the model can usually still use the answer. The
check covers what `ParameterSchema` can express: types, required fields, nested
objects and arrays, string enums. `null` never counts as a mismatch, because the
downcast dropped nullability and flagging it would be reporting a violation of
senno's own making.

`content[]` text blocks are joined into `ToolOutput::content`. Blocks with no
text — images, embedded resources — are counted in a `debug!` and skipped;
`ToolOutput` has nowhere to put them. `structuredContent` becomes
`ToolOutput::data("structured", …)`, and stands in as the text when there are
no text blocks, so the model still sees the result.

`isError: true` returns `Ok`. It is the tool answering the model — "no
availability for those dates" — not a fault, and surfacing it as an error would
put a `tool_error` frame in front of the end user for an ordinary answer.
Transport failures, non-2xx statuses and JSON-RPC `error` objects are
`Error::Provider`: retryable for 5xx, 408 and 429, permanent otherwise.

## Names

Two servers can name a tool the same thing, and model providers reject a
duplicate. `McpServer::with_tool_prefix` renames a server's tools on the way
in; the client keeps the mapping and calls the server under its own name.
Within a registration the first server to claim a name keeps it and the rest are
warned about, and a server that lists the same tool twice — or hands back a
cursor that never advances — has its repeats dropped the same way.

## Not implemented

Whole features, deliberately out of scope: `resources/*`, `prompts/*`,
`completion/*`, stdio transport, and OAuth (dynamic client registration and
authorization-server discovery). Static headers are the only credential.

The client also never opens the optional `GET` SSE stream, so server-initiated
requests and `listChanged` notifications never arrive, and stream resumability
(`Last-Event-ID`) has nothing to resume. A tool list is fetched at `connect` and
never refreshed; reconnect to pick up a change. That is a consequence of the
same decision: one POST per call, nothing long-lived to manage.

[mcp]: https://modelcontextprotocol.io
