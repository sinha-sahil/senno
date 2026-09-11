#![cfg(feature = "mcp")]

mod common;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use std::sync::Mutex;

use futures::StreamExt;
use senno::agent::{
    AgentConfig, AgentFlow, AgentSession, ChatMessage, SseEvent, ToolOutput, ToolSource,
    get_agent_engine,
};
use senno::mcp::{McpClient, McpServer, McpToolset};
use senno::{
    Error, GenerateRequest, GenerateResponse, LlmProvider, LlmStream, StreamChunk, ToolDefinition,
};
use serde_json::{Value, json};

use common::{CapturedRequest, Reply, StubServer};

const SESSION_ID: &str = "sess-7Q2";

fn ok(id: &Value, result: Value) -> Reply {
    Reply::ok(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string())
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": "2025-06-18",
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "stub-mcp", "version": "0.1.0" },
    })
}

fn booking_tool() -> Value {
    json!({
        "name": "create_basket",
        "description": "Open a basket for a hotel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hotelId": { "type": "string", "description": "Hotel identifier" },
                "idempotencyKey": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "string"
                },
                "arrival": { "type": "string", "format": "date" },
                "paymentOption": { "type": "string", "enum": ["PAY_LATER", "FULL_PAYMENT"] },
                "rooms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": { "guests": { "type": "integer", "format": "int32" } },
                        "required": ["guests"]
                    }
                },
                "shape": { "$ref": "#/$defs/Missing" }
            },
            "required": ["hotelId", "shape"],
            "additionalProperties": false
        },
        "annotations": { "destructiveHint": true }
    })
}

fn handle(request: &CapturedRequest) -> Reply {
    if request.method != "POST" {
        return Reply::ok("");
    }
    let body = request.json();
    let id = body.get("id").cloned().unwrap_or(Value::Null);

    match body["method"].as_str().unwrap_or_default() {
        "initialize" => ok(&id, initialize_result()).with_header("Mcp-Session-Id", SESSION_ID),
        "notifications/initialized" => Reply::ok(""),
        "tools/list" => match body["params"]["cursor"].as_str() {
            None => ok(
                &id,
                json!({ "tools": [booking_tool()], "nextCursor": "page-2" }),
            ),
            Some(_) => ok(
                &id,
                json!({ "tools": [{
                    "name": "get_basket",
                    "description": "Read a basket.",
                    "inputSchema": { "type": "object" }
                }] }),
            ),
        },
        "tools/call" => match body["params"]["name"].as_str().unwrap_or_default() {
            "create_basket" => ok(
                &id,
                json!({
                    "content": [{ "type": "text", "text": "basket bk_1 held" }],
                    "structuredContent": { "basketId": "bk_1" }
                }),
            ),
            _ => ok(
                &id,
                json!({
                    "content": [{ "type": "text", "text": "no availability for those dates" }],
                    "isError": true
                }),
            ),
        },
        _ => Reply::error(
            404,
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"no"}}"#,
        ),
    }
}

async fn connect(server: StubServer, prefix: Option<&str>) -> (StubServer, McpClient) {
    let mut config = McpServer::new("stub", format!("{}/mcp", server.base_url))
        .with_header("X-Service-Token", "vmcp_test")
        .with_trusted_annotations();
    if let Some(prefix) = prefix {
        config = config.with_tool_prefix(prefix);
    }
    let client = McpClient::connect(config).await.expect("connect");
    (server, client)
}

#[tokio::test]
async fn connect_follows_the_cursor_and_echoes_the_session_id() {
    let server = StubServer::start(handle).await;
    let (server, client) = connect(server, None).await;

    let names: Vec<&str> = client.tools().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["create_basket", "get_basket"]);

    let sent = server.requests();
    let methods: Vec<Value> = sent.iter().map(|r| r.json()["method"].clone()).collect();
    assert_eq!(
        methods,
        vec![
            json!("initialize"),
            json!("notifications/initialized"),
            json!("tools/list"),
            json!("tools/list"),
        ]
    );

    assert_eq!(sent[0].header("mcp-session-id"), None);
    for request in &sent[1..] {
        assert_eq!(request.header("mcp-session-id"), Some(SESSION_ID));
    }
    assert_eq!(sent[0].header("x-service-token"), Some("vmcp_test"));
    assert_eq!(sent[0].header("mcp-protocol-version"), None);
    for request in &sent[1..] {
        assert_eq!(request.header("mcp-protocol-version"), Some("2025-06-18"));
    }
    assert!(sent[1].json().get("id").is_none());
}

#[tokio::test]
async fn a_foreign_schema_survives_as_far_as_it_can_be_expressed() {
    let server = StubServer::start(handle).await;
    let (_server, client) = connect(server, None).await;

    let basket = &client.tools()[0];
    let properties = basket.parameters.properties.as_ref().expect("properties");

    assert!(properties.contains_key("hotelId"));
    assert!(properties.contains_key("arrival"));
    assert!(!properties.contains_key("shape"));
    assert_eq!(
        basket.parameters.required.as_ref().expect("required"),
        &vec!["hotelId".to_string()]
    );
    assert_eq!(
        properties["paymentOption"]
            .enum_values
            .as_ref()
            .expect("enum")
            .len(),
        2
    );
    assert!(
        properties["rooms"]
            .items
            .as_ref()
            .expect("items")
            .required
            .is_some()
    );
    assert_eq!(
        basket
            .annotations
            .as_ref()
            .expect("annotations")
            .destructive,
        Some(true)
    );
}

#[tokio::test]
async fn a_prefixed_call_reaches_the_server_under_its_own_name() {
    let server = StubServer::start(handle).await;
    let (server, client) = connect(server, Some("visit__")).await;

    assert!(client.has_tool("visit__create_basket"));
    assert!(!client.has_tool("create_basket"));

    let output = client
        .call("visit__create_basket", &json!({ "hotelId": "h1" }))
        .await
        .expect("call");

    assert_eq!(output.content, "basket bk_1 held");
    assert_eq!(output.data.expect("data").payload["basketId"], "bk_1");

    let call = server.requests().last().expect("a call").json();
    assert_eq!(call["params"]["name"], "create_basket");
    assert_eq!(call["params"]["arguments"]["hotelId"], "h1");
}

#[tokio::test]
async fn a_tool_reporting_an_error_is_an_ordinary_result() {
    let server = StubServer::start(handle).await;
    let (_server, client) = connect(server, None).await;

    let output = client
        .call("get_basket", &json!({ "basketId": "bk_1" }))
        .await
        .expect("an isError result is not a transport failure");

    assert_eq!(output.content, "no availability for those dates");
}

#[tokio::test]
async fn an_unauthorized_server_fails_the_connection_permanently() {
    let server = StubServer::start(|_| Reply::error(401, "no token")).await;
    let error = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect_err("401 is a failure");

    assert!(error.is_provider_error());
    assert!(!error.is_retryable());
    assert!(error.to_string().contains("401"));
}

#[tokio::test]
async fn a_failing_server_is_retryable() {
    let server = StubServer::start(|_| Reply::error(503, "down")).await;
    let error = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect_err("503 is a failure");

    assert!(error.is_retryable());
}

struct ScriptedProvider {
    streams: Mutex<Vec<Vec<StreamChunk>>>,
    tools_seen: Arc<Mutex<Vec<String>>>,
}

impl ScriptedProvider {
    fn new(streams: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            streams: Mutex::new(streams),
            tools_seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn watching(streams: Vec<Vec<StreamChunk>>, tools_seen: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            streams: Mutex::new(streams),
            tools_seen,
        }
    }
}

impl LlmProvider for ScriptedProvider {
    fn generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>> {
        Box::pin(async move { Err(Error::internal("no summary scripted")) })
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        if let Ok(mut seen) = self.tools_seen.lock() {
            *seen = request.tools.iter().map(|tool| tool.name.clone()).collect();
        }
        let next = self
            .streams
            .lock()
            .ok()
            .and_then(|mut streams| (!streams.is_empty()).then(|| streams.remove(0)));

        Box::pin(async move {
            let chunks = next.ok_or_else(|| Error::internal("stream script exhausted"))?;
            Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed() as LlmStream)
        })
    }
}

struct BookingFlow {
    mcp: McpToolset,
}

impl AgentFlow for BookingFlow {
    fn system_prompt(&self) -> String {
        "You book hotel rooms.".to_string()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new("summarise_stay", "Summarise the stay."),
            ToolDefinition::new("get_basket", "The flow's own basket reader."),
        ]
    }

    fn tool_sources(&self) -> Vec<&dyn ToolSource> {
        vec![&self.mcp]
    }

    fn execute_tool<'a>(
        &'a self,
        name: &'a str,
        _args: &'a Value,
        _session: &'a AgentSession,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move { Ok(ToolOutput::text(format!("native {name}"))) })
    }
}

fn tool_call(name: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments: json!({}),
            thought_signature: None,
        },
        StreamChunk::Done {
            finish_reason: "STOP".into(),
            usage: None,
        },
    ]
}

fn closing_text() -> Vec<StreamChunk> {
    vec![
        StreamChunk::Text("all set".into()),
        StreamChunk::Done {
            finish_reason: "STOP".into(),
            usage: None,
        },
    ]
}

async fn run(flow: &BookingFlow, provider: ScriptedProvider) -> (Vec<String>, Vec<String>) {
    let engine = get_agent_engine(
        provider,
        AgentConfig::builder("test-model").build().expect("config"),
    );
    let mut session = AgentSession::new("s1", "booking");
    let mut events = engine.run(flow, &mut session, "two nights please");

    let mut text = Vec::new();
    while let Some(event) = events.next().await {
        if let Ok(SseEvent::Text { delta }) = event {
            text.push(delta);
        }
    }
    drop(events);

    let tools = session
        .messages
        .iter()
        .filter_map(|message| match message {
            ChatMessage::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect();

    (tools, text)
}

#[tokio::test]
async fn the_engine_offers_native_and_registered_tools_together() {
    let server = StubServer::start(handle).await;
    let mcp = McpToolset::connect(vec![
        McpServer::new("visit", format!("{}/mcp", server.base_url)).with_tool_prefix("visit__"),
    ])
    .await
    .expect("connect");

    let flow = BookingFlow { mcp };
    let offered = Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider::watching(vec![closing_text()], offered.clone());

    let engine = get_agent_engine(
        provider,
        AgentConfig::builder("test-model").build().expect("config"),
    );
    let mut session = AgentSession::new("s1", "booking");
    let mut events = engine.run(&flow, &mut session, "hello");
    while events.next().await.is_some() {}
    drop(events);

    let names = offered.lock().expect("tools seen").clone();
    assert_eq!(
        names,
        vec![
            "summarise_stay",
            "get_basket",
            "visit__create_basket",
            "visit__get_basket",
        ],
        "the flow's own tools come first, then the registered source's",
    );
}

#[tokio::test]
async fn the_engine_routes_a_registered_tool_without_the_flow_knowing() {
    let server = StubServer::start(handle).await;
    let mcp = McpToolset::connect(vec![
        McpServer::new("visit", format!("{}/mcp", server.base_url)).with_tool_prefix("visit__"),
    ])
    .await
    .expect("connect");

    let flow = BookingFlow { mcp };
    let before = server.request_count();
    let (results, text) = run(
        &flow,
        ScriptedProvider::new(vec![tool_call("visit__create_basket"), closing_text()]),
    )
    .await;

    assert_eq!(results, vec!["basket bk_1 held".to_string()]);
    assert_eq!(text, vec!["all set".to_string()]);
    assert_eq!(server.request_count(), before + 1);
}

#[tokio::test]
async fn a_native_tool_of_the_same_name_keeps_the_call() {
    let server = StubServer::start(handle).await;
    let mcp = McpToolset::connect(vec![McpServer::new(
        "visit",
        format!("{}/mcp", server.base_url),
    )])
    .await
    .expect("connect");

    let flow = BookingFlow { mcp };
    let before = server.request_count();
    let (results, _) = run(
        &flow,
        ScriptedProvider::new(vec![tool_call("get_basket"), closing_text()]),
    )
    .await;

    assert_eq!(results, vec!["native get_basket".to_string()]);
    assert_eq!(
        server.request_count(),
        before,
        "the flow's own get_basket must shadow the server's",
    );
}

#[tokio::test]
async fn a_tool_name_offered_by_two_servers_is_registered_once() {
    let first = StubServer::start(handle).await;
    let second = StubServer::start(handle).await;

    let mcp = McpToolset::connect(vec![
        McpServer::new("first", format!("{}/mcp", first.base_url)),
        McpServer::new("second", format!("{}/mcp", second.base_url)),
    ])
    .await
    .expect("connect");

    let names: Vec<&str> = mcp
        .definitions()
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(names, vec!["create_basket", "get_basket"]);
}

#[tokio::test]
async fn one_unreachable_server_fails_the_whole_registration() {
    let good = StubServer::start(handle).await;

    let error = McpToolset::connect(vec![
        McpServer::new("good", format!("{}/mcp", good.base_url)),
        McpServer::new("bad", "http://127.0.0.1:1/mcp"),
    ])
    .await
    .expect_err("a silently partial toolset is worse than a failure");

    assert!(error.is_provider_error());
}

#[tokio::test]
async fn a_repeating_cursor_cannot_duplicate_a_tool() {
    fn stuck(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(&id, initialize_result()),
            "notifications/initialized" => Reply::ok(""),
            _ => ok(
                &id,
                json!({
                    "tools": [{
                        "name": "get_basket",
                        "description": "Read a basket.",
                        "inputSchema": { "type": "object" }
                    }],
                    "nextCursor": "always-the-same"
                }),
            ),
        }
    }

    let server = StubServer::start(stuck).await;
    let client = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect("connect");

    let names: Vec<&str> = client.tools().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["get_basket"]);
}

#[tokio::test]
async fn the_negotiated_protocol_version_is_sent_on_later_requests() {
    fn older(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(
                &id,
                json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "older", "version": "0.1.0" },
                }),
            ),
            "notifications/initialized" => Reply::ok(""),
            _ => ok(&id, json!({ "tools": [] })),
        }
    }

    let server = StubServer::start(older).await;
    McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect("connect");

    let sent = server.requests();
    assert_eq!(sent[0].header("mcp-protocol-version"), None);
    for request in &sent[1..] {
        assert_eq!(request.header("mcp-protocol-version"), Some("2025-03-26"));
    }
}

#[tokio::test]
async fn a_version_senno_does_not_speak_refuses_to_connect() {
    fn ancient(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        ok(
            &id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "ancient", "version": "0.1.0" },
            }),
        )
    }

    let server = StubServer::start(ancient).await;
    let error = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect_err("an unsupported version must not be adopted");

    assert!(error.to_string().contains("2024-11-05"));
    assert!(!error.is_retryable());
}

#[tokio::test]
async fn a_server_declaring_no_tools_refuses_to_connect() {
    fn toolless(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        ok(
            &id,
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "resources": {} },
                "serverInfo": { "name": "toolless", "version": "0.1.0" },
            }),
        )
    }

    let server = StubServer::start(toolless).await;
    let error = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect_err("tools must be a negotiated capability");

    assert!(error.to_string().contains("tools capability"));
}

#[tokio::test]
async fn a_redirect_is_refused_rather_than_followed() {
    let server = StubServer::start(|_| {
        Reply::error(302, "").with_header("Location", "https://example.invalid/mcp")
    })
    .await;

    let error = McpClient::connect(
        McpServer::new("stub", format!("{}/mcp", server.base_url))
            .with_header("X-Service-Token", "secret"),
    )
    .await
    .expect_err("a redirect must not carry the token onward");

    assert!(error.to_string().contains("302"));
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn an_ended_session_is_re_initialized_and_the_call_retried() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let seen = attempts.clone();

    let server = StubServer::start(move |request: &CapturedRequest| {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(&id, initialize_result()).with_header("Mcp-Session-Id", SESSION_ID),
            "notifications/initialized" => Reply::ok(""),
            "tools/list" => ok(
                &id,
                json!({ "tools": [{
                    "name": "get_basket",
                    "description": "Read a basket.",
                    "inputSchema": { "type": "object" }
                }] }),
            ),
            _ if seen.fetch_add(1, Ordering::SeqCst) == 0 => {
                Reply::error(404, "that session is gone")
            }
            _ => ok(
                &id,
                json!({ "content": [{ "type": "text", "text": "read after re-initializing" }] }),
            ),
        }
    })
    .await;

    let client = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect("connect");

    let output = client
        .call("get_basket", &json!({ "basketId": "bk_1" }))
        .await
        .expect("a 404 must re-initialize, not fail the call");

    assert_eq!(output.content, "read after re-initializing");

    let initializes = server
        .requests()
        .iter()
        .filter(|r| r.json()["method"] == "initialize")
        .count();
    assert_eq!(initializes, 2);
}

#[tokio::test]
async fn a_streamed_reply_carrying_progress_first_still_yields_the_tool_result() {
    fn streamed(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(&id, initialize_result()),
            "notifications/initialized" => Reply::ok(""),
            "tools/list" => ok(
                &id,
                json!({ "tools": [{
                    "name": "commit_booking",
                    "description": "Commit.",
                    "inputSchema": { "type": "object" }
                }] }),
            ),
            _ => {
                let progress = json!({ "jsonrpc": "2.0", "method": "notifications/progress", "params": { "progress": 1 } });
                let result = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "content": [{ "type": "text", "text": "booked bk_9" }] }
                });
                Reply::ok(format!(
                    "event: message\ndata: {progress}\n\ndata: {result}\n\n"
                ))
                .with_content_type("text/event-stream")
            }
        }
    }

    let server = StubServer::start(streamed).await;
    let client = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect("connect");

    let output = client
        .call("commit_booking", &json!({}))
        .await
        .expect("a progress frame must not be mistaken for the result");

    assert_eq!(output.content, "booked bk_9");
}

#[tokio::test]
async fn an_ended_session_is_terminated_with_a_delete() {
    let server = StubServer::start(handle).await;
    let (server, client) = connect(server, None).await;

    client.close().await;

    let last = server.requests().last().cloned().expect("a request");
    assert_eq!(last.method, "DELETE");
    assert_eq!(last.header("mcp-session-id"), Some(SESSION_ID));
    assert_eq!(last.header("x-service-token"), Some("vmcp_test"));

    let before = server.request_count();
    client.close().await;
    assert_eq!(server.request_count(), before);
}

#[tokio::test]
async fn a_read_only_call_is_retried_but_a_tool_call_never_is() {
    let listings = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let seen_listings = listings.clone();
    let seen_calls = calls.clone();

    let server = StubServer::start(move |request: &CapturedRequest| {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(&id, initialize_result()),
            "notifications/initialized" => Reply::ok(""),
            "tools/list" if seen_listings.fetch_add(1, Ordering::SeqCst) == 0 => {
                Reply::error(503, "warming up")
            }
            "tools/list" => ok(
                &id,
                json!({ "tools": [{
                    "name": "commit_booking",
                    "description": "Commit.",
                    "inputSchema": { "type": "object" }
                }] }),
            ),
            _ => {
                seen_calls.fetch_add(1, Ordering::SeqCst);
                Reply::error(503, "still warming up")
            }
        }
    })
    .await;

    let client = McpClient::connect(McpServer::new("stub", format!("{}/mcp", server.base_url)))
        .await
        .expect("a transient 503 on tools/list must be retried");

    assert_eq!(listings.load(Ordering::SeqCst), 2);

    let error = client
        .call("commit_booking", &json!({}))
        .await
        .expect_err("503");
    assert!(error.is_retryable());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a tool call carries no idempotency key and must never be replayed"
    );
}

#[tokio::test]
async fn a_timed_out_tool_call_cancels_the_request() {
    fn slow(request: &CapturedRequest) -> Reply {
        let body = request.json();
        let id = body.get("id").cloned().unwrap_or(Value::Null);
        match body["method"].as_str().unwrap_or_default() {
            "initialize" => ok(&id, initialize_result()),
            "notifications/initialized" | "notifications/cancelled" => Reply::ok(""),
            "tools/list" => ok(
                &id,
                json!({ "tools": [{
                    "name": "commit_booking",
                    "description": "Commit.",
                    "inputSchema": { "type": "object" }
                }] }),
            ),
            _ => ok(&id, json!({ "content": [] })).after(Duration::from_millis(900)),
        }
    }

    let server = StubServer::start(slow).await;
    let client = McpClient::connect(
        McpServer::new("stub", format!("{}/mcp", server.base_url))
            .with_timeout(Duration::from_millis(150)),
    )
    .await
    .expect("connect");

    let error = client
        .call("commit_booking", &json!({}))
        .await
        .expect_err("timeout");
    assert!(error.is_retryable());

    let cancelled = server
        .requests()
        .iter()
        .filter(|r| r.json()["method"] == "notifications/cancelled")
        .count();
    assert_eq!(cancelled, 1);
}
