use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock};
use std::time::Duration;

use futures::StreamExt;
use reqwest::StatusCode;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::{Value, json};

use crate::agent::ToolOutput;
use crate::error::Error;
use crate::types::{ParameterSchema, SchemaType, ToolAnnotations, ToolDefinition};

use super::schema::{mismatch, to_parameter_schema};
use super::types::{
    ContentBlock, InitializeResult, JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpAnnotations,
    McpTool, ToolCallResult, ToolsListResult,
};

const JSON_RPC_VERSION: &str = "2.0";
const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_TOOL_PAGES: usize = 32;
const PROTOCOL_HEADER: &str = "mcp-protocol-version";
const SESSION_HEADER: &str = "mcp-session-id";
const EVENT_STREAM: &str = "text/event-stream";
const EMPTY_RESULT: &str = "The tool returned no content.";
const SUPPORTED_PROTOCOL_VERSIONS: [&str; 2] = ["2025-06-18", "2025-03-26"];
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(8);

#[derive(Clone)]
pub struct McpServer {
    name: String,
    url: String,
    headers: Vec<(String, String)>,
    timeout: Duration,
    protocol_version: String,
    tool_prefix: Option<String>,
    trust_annotations: bool,
}

impl McpServer {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            protocol_version: DEFAULT_PROTOCOL_VERSION.to_string(),
            tool_prefix: None,
            trust_annotations: false,
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_tool_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    pub fn with_trusted_annotations(mut self) -> Self {
        self.trust_annotations = true;
        self
    }
}

impl fmt::Debug for McpServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names: Vec<&str> = self.headers.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("McpServer")
            .field("name", &self.name)
            .field("url", &self.url)
            .field("headers", &header_names)
            .field("timeout", &self.timeout)
            .field("protocol_version", &self.protocol_version)
            .field("tool_prefix", &self.tool_prefix)
            .field("trust_annotations", &self.trust_annotations)
            .finish()
    }
}

pub struct McpClient {
    server: McpServer,
    http: reqwest::Client,
    negotiated: RwLock<Negotiated>,
    tools: Vec<ToolDefinition>,
    remote_names: HashMap<String, String>,
    output_schemas: HashMap<String, ParameterSchema>,
    next_id: AtomicU64,
}

impl fmt::Debug for McpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (session_id, protocol_version) = self.negotiated();
        f.debug_struct("McpClient")
            .field("server", &self.server)
            .field("tools", &self.tools.len())
            .field("protocol", &protocol_version)
            .field("session", &session_id.is_some())
            .finish()
    }
}

struct Negotiated {
    session_id: Option<String>,
    protocol_version: String,
}

#[derive(Clone, Copy)]
enum Phase {
    Initialize,
    Operation,
}

#[derive(Clone, Copy, PartialEq)]
enum Retry {
    Safe,
    Never,
}

enum Exchange {
    Result {
        value: Value,
        session_id: Option<String>,
    },
    SessionExpired,
}

impl McpClient {
    pub async fn connect(server: McpServer) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .timeout(server.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| {
                Error::config(format!("mcp: no HTTP client for '{}': {e}", server.name))
            })?;

        let protocol_version = server.protocol_version.clone();
        let mut client = Self {
            server,
            http,
            negotiated: RwLock::new(Negotiated {
                session_id: None,
                protocol_version,
            }),
            tools: Vec::new(),
            remote_names: HashMap::new(),
            output_schemas: HashMap::new(),
            next_id: AtomicU64::new(1),
        };

        client.handshake().await?;
        let listing = client.fetch_tools().await?;
        client.tools = listing.tools;
        client.remote_names = listing.remote_names;
        client.output_schemas = listing.output_schemas;

        let (session_id, protocol_version) = client.negotiated();
        tracing::info!(
            server = %client.server.name,
            tools = client.tools.len(),
            protocol = %protocol_version,
            session = session_id.is_some(),
            "mcp: connected",
        );

        Ok(client)
    }

    pub fn name(&self) -> &str {
        &self.server.name
    }

    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.remote_names.contains_key(name)
    }

    pub async fn close(&self) {
        let (session_id, protocol_version) = self.negotiated();
        let Some(session) = session_id else {
            return;
        };

        let request = self.static_headers(
            self.http
                .delete(&self.server.url)
                .header(PROTOCOL_HEADER, protocol_version)
                .header(SESSION_HEADER, session.as_str()),
        );

        match request.send().await {
            Ok(response) if response.status() == StatusCode::METHOD_NOT_ALLOWED => tracing::debug!(
                server = %self.server.name,
                "mcp: the server keeps its own sessions",
            ),
            Ok(response) => tracing::debug!(
                server = %self.server.name,
                status = %response.status(),
                "mcp: session terminated",
            ),
            Err(e) => tracing::debug!(
                server = %self.server.name,
                error = %e,
                "mcp: session not terminated",
            ),
        }

        let mut guard = self
            .negotiated
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        guard.session_id = None;
    }

    pub async fn call(&self, name: &str, arguments: &Value) -> Result<ToolOutput, Error> {
        let remote = self.remote_names.get(name).ok_or_else(|| {
            Error::tool(
                name,
                format!("'{}' does not serve this tool", self.server.name),
            )
        })?;

        let params = json!({ "name": remote, "arguments": arguments_object(arguments) });
        let value = self
            .request("tools/call", Some(params), Retry::Never)
            .await?;

        let result: ToolCallResult = serde_json::from_value(value).map_err(|e| {
            self.permanent(format!(
                "tools/call '{remote}' returned an unreadable result: {e}"
            ))
        })?;

        if result.is_error.unwrap_or(false) {
            tracing::warn!(server = %self.server.name, tool = %remote, "mcp: tool reported an error");
        }

        if let (Some(schema), Some(structured)) = (
            self.output_schemas.get(name),
            result.structured_content.as_ref(),
        ) && let Some(problem) = mismatch(schema, structured)
        {
            tracing::warn!(
                server = %self.server.name,
                tool = %remote,
                problem = %problem,
                "mcp: structured output does not match the tool's own schema",
            );
        }

        Ok(to_output(result))
    }

    async fn handshake(&self) -> Result<(), Error> {
        let params = json!({
            "protocolVersion": self.server.protocol_version,
            "capabilities": {},
            "clientInfo": { "name": "senno", "version": env!("CARGO_PKG_VERSION") },
        });

        let Exchange::Result { value, session_id } = self
            .attempt("initialize", Some(params), Phase::Initialize, Retry::Safe)
            .await?
        else {
            return Err(self.permanent("initialize: the server has no session".into()));
        };

        let InitializeResult {
            protocol_version,
            capabilities,
            server_info,
        } = serde_json::from_value(value)
            .map_err(|e| self.permanent(format!("initialize is unreadable: {e}")))?;

        let protocol_version = match present(protocol_version.as_deref()) {
            Some(version) if SUPPORTED_PROTOCOL_VERSIONS.contains(&version) => version.to_string(),
            Some(version) => {
                return Err(self.permanent(format!(
                    "the server speaks MCP {version}, senno speaks {}",
                    SUPPORTED_PROTOCOL_VERSIONS.join(" or ")
                )));
            }
            None => self.server.protocol_version.clone(),
        };

        if capabilities.and_then(|declared| declared.tools).is_none() {
            return Err(self.permanent("the server declares no tools capability".into()));
        }

        {
            let mut guard = self
                .negotiated
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            guard.session_id = session_id;
            guard.protocol_version = protocol_version.clone();
        }

        let (remote, version) = server_info
            .map(|info| {
                (
                    info.name.unwrap_or_default(),
                    info.version.unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        tracing::debug!(
            server = %self.server.name,
            remote,
            version,
            protocol = %protocol_version,
            "mcp: initialized",
        );

        self.notify("notifications/initialized", None).await;
        Ok(())
    }

    async fn fetch_tools(&self) -> Result<Listing, Error> {
        let mut listing = Listing::default();
        let mut cursor: Option<String> = None;

        for _ in 0..MAX_TOOL_PAGES {
            let params = cursor.map(|cursor| json!({ "cursor": cursor }));
            let value = self.request("tools/list", params, Retry::Safe).await?;

            let page: ToolsListResult = serde_json::from_value(value)
                .map_err(|e| self.permanent(format!("tools/list is unreadable: {e}")))?;

            for tool in page.tools {
                let Some(definition) = definition_from(&self.server, &tool) else {
                    tracing::warn!(
                        server = %self.server.name,
                        tool = %tool.name,
                        "mcp: tool dropped, its input schema is not expressible",
                    );
                    continue;
                };
                if listing.remote_names.contains_key(&definition.name) {
                    tracing::warn!(
                        server = %self.server.name,
                        tool = %definition.name,
                        "mcp: tool listed twice, the first is kept",
                    );
                    continue;
                }
                if let Some(schema) = tool
                    .output_schema
                    .as_ref()
                    .and_then(to_parameter_schema)
                    .filter(|schema| matches!(schema.schema_type, SchemaType::Object))
                {
                    listing
                        .output_schemas
                        .insert(definition.name.clone(), schema);
                }
                listing
                    .remote_names
                    .insert(definition.name.clone(), tool.name);
                listing.tools.push(definition);
            }

            cursor = page.next_cursor;
            if cursor.is_none() {
                return Ok(listing);
            }
        }

        tracing::warn!(
            server = %self.server.name,
            pages = MAX_TOOL_PAGES,
            "mcp: tool listing truncated",
        );
        Ok(listing)
    }

    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        retry: Retry,
    ) -> Result<Value, Error> {
        if let Exchange::Result { value, .. } = self
            .attempt(method, params.clone(), Phase::Operation, retry)
            .await?
        {
            return Ok(value);
        }

        tracing::info!(
            server = %self.server.name,
            method,
            "mcp: the server ended the session, re-initializing",
        );
        self.handshake().await?;

        match self
            .attempt(method, params, Phase::Operation, retry)
            .await?
        {
            Exchange::Result { value, .. } => Ok(value),
            Exchange::SessionExpired => {
                Err(self.permanent(format!("{method}: the session ended twice running")))
            }
        }
    }

    async fn attempt(
        &self,
        method: &str,
        params: Option<Value>,
        phase: Phase,
        retry: Retry,
    ) -> Result<Exchange, Error> {
        if retry == Retry::Never {
            return self.exchange(method, params, phase).await;
        }

        for attempt in 1..MAX_ATTEMPTS {
            match self.exchange(method, params.clone(), phase).await {
                Err(e) if e.is_retryable() => {
                    let delay = backoff(attempt);
                    tracing::warn!(
                        server = %self.server.name,
                        method,
                        attempt,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "mcp: transient failure, retrying",
                    );
                    tokio::time::sleep(delay).await;
                }
                outcome => return outcome,
            }
        }

        self.exchange(method, params, phase).await
    }

    async fn exchange(
        &self,
        method: &str,
        params: Option<Value>,
        phase: Phase,
    ) -> Result<Exchange, Error> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION,
            id: Some(id),
            method,
            params,
        };

        let response = match self.builder(&body, phase).send().await {
            Ok(response) => response,
            Err(e) => {
                if e.is_timeout() && matches!(phase, Phase::Operation) {
                    self.cancel(id, method).await;
                }
                return Err(self.send_error(method, e));
            }
        };

        let status = response.status();
        let session_id = header_value(response.headers(), SESSION_HEADER);
        let content_type = header_value(response.headers(), CONTENT_TYPE.as_str());

        if status == StatusCode::NOT_FOUND
            && matches!(phase, Phase::Operation)
            && self.negotiated().0.is_some()
        {
            return Ok(Exchange::SessionExpired);
        }

        let (body, truncated) = match read_capped(response).await {
            Ok(read) => read,
            Err(e) if status.is_success() => return Err(self.send_error(method, e)),
            Err(_) => (String::new(), false),
        };

        if !status.is_success() {
            return Err(self.status_error(method, status, &body));
        }
        if truncated {
            return Err(self.permanent(format!(
                "{method}: the response is larger than {MAX_RESPONSE_BYTES} bytes"
            )));
        }

        let envelope = envelope(content_type.as_deref().unwrap_or_default(), &body)
            .ok_or_else(|| self.permanent(format!("{method}: response is not JSON-RPC")))?;

        if let Some(error) = envelope.error {
            return Err(self.rpc_error(method, error));
        }

        Ok(Exchange::Result {
            value: envelope.result.unwrap_or(Value::Null),
            session_id,
        })
    }

    async fn cancel(&self, id: u64, method: &str) {
        let params = json!({ "requestId": id, "reason": "the client stopped waiting" });
        tracing::warn!(
            server = %self.server.name,
            method,
            request = id,
            "mcp: timed out, cancelling",
        );
        self.notify("notifications/cancelled", Some(params)).await;
    }

    async fn notify(&self, method: &str, params: Option<Value>) {
        let body = JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION,
            id: None,
            method,
            params,
        };

        match self.builder(&body, Phase::Operation).send().await {
            Err(e) => tracing::debug!(
                server = %self.server.name,
                method,
                error = %e,
                "mcp: notification not delivered",
            ),
            Ok(response) if !response.status().is_success() => tracing::debug!(
                server = %self.server.name,
                method,
                status = %response.status(),
                "mcp: notification refused",
            ),
            Ok(_) => {}
        }
    }

    fn builder(&self, body: &JsonRpcRequest<'_>, phase: Phase) -> reqwest::RequestBuilder {
        let mut builder = self.static_headers(
            self.http
                .post(&self.server.url)
                .header(CONTENT_TYPE, "application/json")
                .header(ACCEPT, "application/json, text/event-stream")
                .json(body),
        );

        if matches!(phase, Phase::Initialize) {
            return builder;
        }

        let (session_id, protocol_version) = self.negotiated();
        builder = builder.header(PROTOCOL_HEADER, protocol_version);
        match session_id {
            Some(session) => builder.header(SESSION_HEADER, session),
            None => builder,
        }
    }

    fn static_headers(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (name, value) in &self.server.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
    }

    fn negotiated(&self) -> (Option<String>, String) {
        let guard = self
            .negotiated
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        (guard.session_id.clone(), guard.protocol_version.clone())
    }

    fn send_error(&self, method: &str, error: reqwest::Error) -> Error {
        if error.is_builder() {
            return Error::config(format!("mcp: '{}' {method}: {error}", self.server.name));
        }
        Error::provider(self.server.name.clone(), format!("{method}: {error}"))
    }

    fn status_error(&self, method: &str, status: StatusCode, body: &str) -> Error {
        let detail = format!("{method} failed ({status}): {}", truncate(body));
        if is_transient(status) {
            Error::provider(self.server.name.clone(), detail)
        } else {
            Error::provider_permanent(self.server.name.clone(), detail)
        }
    }

    fn rpc_error(&self, method: &str, error: JsonRpcError) -> Error {
        let data = error
            .data
            .map(|data| format!(" ({})", truncate(&data.to_string())))
            .unwrap_or_default();
        self.permanent(format!(
            "{method} rejected [{}]: {}{data}",
            error.code, error.message
        ))
    }

    fn permanent(&self, detail: String) -> Error {
        Error::provider_permanent(self.server.name.clone(), detail)
    }
}

#[derive(Default)]
struct Listing {
    tools: Vec<ToolDefinition>,
    remote_names: HashMap<String, String>,
    output_schemas: HashMap<String, ParameterSchema>,
}

fn definition_from(server: &McpServer, tool: &McpTool) -> Option<ToolDefinition> {
    let parameters = match tool.input_schema.as_ref() {
        None => ParameterSchema::object(),
        Some(schema) => match to_parameter_schema(schema) {
            Some(parameters) if matches!(parameters.schema_type, SchemaType::Object) => parameters,
            _ => return None,
        },
    };

    let description = present(tool.description.as_deref())
        .or_else(|| display_title(tool))
        .unwrap_or(tool.name.as_str())
        .to_string();

    let name = match &server.tool_prefix {
        Some(prefix) => format!("{prefix}{}", tool.name),
        None => tool.name.clone(),
    };

    let mut definition = ToolDefinition::new(name, description).with_parameters(parameters);

    if let Some(title) = display_title(tool) {
        definition = definition.with_title(title.to_string());
    }
    match tool.annotations.as_ref() {
        Some(annotations) if server.trust_annotations => {
            definition = definition.with_annotations(to_annotations(annotations));
        }
        Some(_) => tracing::debug!(
            server = %server.name,
            tool = %tool.name,
            "mcp: behaviour hints dropped, the server is not marked trusted",
        ),
        None => {}
    }

    Some(definition)
}

async fn read_capped(response: reqwest::Response) -> Result<(String, bool), reqwest::Error> {
    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut truncated = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if buffer.len() + chunk.len() > MAX_RESPONSE_BYTES {
            truncated = true;
            break;
        }
        buffer.extend_from_slice(&chunk);
    }

    Ok((String::from_utf8_lossy(&buffer).into_owned(), truncated))
}

fn display_title(tool: &McpTool) -> Option<&str> {
    present(tool.title.as_deref()).or_else(|| {
        tool.annotations
            .as_ref()
            .and_then(|annotations| present(annotations.title.as_deref()))
    })
}

fn present(text: Option<&str>) -> Option<&str> {
    text.filter(|value| !value.trim().is_empty())
}

fn to_annotations(annotations: &McpAnnotations) -> ToolAnnotations {
    ToolAnnotations {
        read_only: annotations.read_only_hint,
        destructive: annotations.destructive_hint,
        idempotent: annotations.idempotent_hint,
        open_world: annotations.open_world_hint,
    }
}

fn to_output(result: ToolCallResult) -> ToolOutput {
    let (text, skipped) = readable_text(&result.content);
    if skipped > 0 {
        tracing::debug!(skipped, "mcp: content blocks with no text were dropped");
    }

    let content = text
        .or_else(|| {
            result
                .structured_content
                .as_ref()
                .and_then(|data| serde_json::to_string(data).ok())
        })
        .unwrap_or_else(|| EMPTY_RESULT.to_string());

    let output = ToolOutput::text(content);
    match result.structured_content {
        Some(data) => output.data("structured", data),
        None => output,
    }
}

fn readable_text(blocks: &[ContentBlock]) -> (Option<String>, usize) {
    let mut text: Vec<&str> = Vec::new();
    let mut skipped = 0;

    for block in blocks {
        match block.text.as_deref() {
            Some(value) if block.kind == "text" => text.push(value),
            _ => skipped += 1,
        }
    }

    ((!text.is_empty()).then(|| text.join("\n")), skipped)
}

fn arguments_object(arguments: &Value) -> Value {
    match arguments {
        Value::Object(_) => arguments.clone(),
        _ => json!({}),
    }
}

fn envelope(content_type: &str, body: &str) -> Option<JsonRpcResponse> {
    if content_type.contains(EVENT_STREAM) {
        return sse_envelope(body);
    }
    response_frame(body)
}

fn response_frame(payload: &str) -> Option<JsonRpcResponse> {
    let frame: JsonRpcResponse = serde_json::from_str(payload).ok()?;
    frame.method.is_none().then_some(frame)
}

fn sse_envelope(body: &str) -> Option<JsonRpcResponse> {
    let mut payload = String::new();

    for line in body.lines() {
        if line.trim().is_empty() {
            if let Some(frame) = response_frame(&payload) {
                return Some(frame);
            }
            payload.clear();
            continue;
        }

        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        if !payload.is_empty() {
            payload.push('\n');
        }
        payload.push_str(data.trim_start());
    }

    response_frame(&payload)
}

fn header_value(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(String::from)
}

fn backoff(attempt: u32) -> Duration {
    RETRY_BASE_DELAY
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(RETRY_MAX_DELAY)
}

fn is_transient(status: StatusCode) -> bool {
    status.is_server_error()
        || status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
}

fn truncate(body: &str) -> String {
    const LIMIT: usize = 500;
    match body.char_indices().nth(LIMIT) {
        Some((end, _)) => format!("{}…", &body[..end]),
        None => body.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(input_schema: Value) -> McpTool {
        serde_json::from_value(json!({
            "name": "get_basket",
            "description": "Read a basket",
            "inputSchema": input_schema,
        }))
        .expect("tool")
    }

    #[test]
    fn a_prefix_renames_the_tool_and_the_remote_name_is_kept_separately() {
        let server =
            McpServer::new("visit", "https://example.test/mcp").with_tool_prefix("visit__");
        let definition =
            definition_from(&server, &tool(json!({ "type": "object" }))).expect("definition");

        assert_eq!(definition.name, "visit__get_basket");
    }

    #[test]
    fn a_tool_whose_schema_is_not_an_object_is_dropped() {
        let server = McpServer::new("visit", "https://example.test/mcp");
        assert!(definition_from(&server, &tool(json!({ "type": "string" }))).is_none());
        assert!(definition_from(&server, &tool(json!({ "$ref": "#/x" }))).is_none());
    }

    #[test]
    fn a_tool_with_no_schema_advertises_an_empty_object() {
        let server = McpServer::new("visit", "https://example.test/mcp");
        let bare: McpTool =
            serde_json::from_value(json!({ "name": "ping", "description": "" })).expect("tool");
        let definition = definition_from(&server, &bare).expect("definition");

        assert!(matches!(
            definition.parameters.schema_type,
            SchemaType::Object
        ));
        assert_eq!(definition.description, "ping");
    }

    #[test]
    fn the_title_field_outranks_the_annotation_title() {
        let server = McpServer::new("visit", "https://example.test/mcp");

        let both: McpTool = serde_json::from_value(json!({
            "name": "create_basket",
            "title": "Open a Basket",
            "description": "Open one.",
            "inputSchema": { "type": "object" },
            "annotations": { "title": "Legacy Name" },
        }))
        .expect("tool");
        assert_eq!(
            definition_from(&server, &both).expect("definition").title(),
            Some("Open a Basket")
        );

        let annotated_only: McpTool = serde_json::from_value(json!({
            "name": "create_basket",
            "description": "Open one.",
            "inputSchema": { "type": "object" },
            "annotations": { "title": "Legacy Name" },
        }))
        .expect("tool");
        assert_eq!(
            definition_from(&server, &annotated_only)
                .expect("definition")
                .title(),
            Some("Legacy Name")
        );
    }

    #[test]
    fn behaviour_hints_are_dropped_unless_the_server_is_trusted() {
        let annotated: McpTool = serde_json::from_value(json!({
            "name": "commit_booking",
            "description": "Commit",
            "inputSchema": { "type": "object" },
            "annotations": { "destructiveHint": true, "readOnlyHint": false },
        }))
        .expect("tool");

        let untrusted = McpServer::new("visit", "https://example.test/mcp");
        assert!(
            definition_from(&untrusted, &annotated)
                .expect("definition")
                .annotations
                .is_none()
        );

        let trusted = untrusted.clone().with_trusted_annotations();
        let annotations = definition_from(&trusted, &annotated)
            .expect("definition")
            .annotations
            .expect("annotations");
        assert_eq!(annotations.destructive, Some(true));
        assert_eq!(annotations.read_only, Some(false));
    }

    #[test]
    fn an_annotation_title_still_sets_the_display_name_on_an_untrusted_server() {
        let server = McpServer::new("visit", "https://example.test/mcp");
        let annotated: McpTool = serde_json::from_value(json!({
            "name": "commit_booking",
            "description": "Commit",
            "inputSchema": { "type": "object" },
            "annotations": { "title": "Confirm Booking" },
        }))
        .expect("tool");

        assert_eq!(
            definition_from(&server, &annotated)
                .expect("definition")
                .title(),
            Some("Confirm Booking")
        );
    }

    #[test]
    fn a_json_response_is_read_as_an_envelope() {
        let parsed = envelope(
            "application/json",
            r#"{"jsonrpc":"2.0","id":1,"result":{"ok":1}}"#,
        )
        .expect("envelope");
        assert_eq!(parsed.result.expect("result")["ok"], 1);
    }

    #[test]
    fn an_sse_framed_response_is_read_as_an_envelope() {
        let body =
            "event: message\r\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":2}}\r\n\r\n";
        let parsed = envelope("text/event-stream; charset=utf-8", body).expect("envelope");
        assert_eq!(parsed.result.expect("result")["ok"], 2);
    }

    #[test]
    fn a_progress_notification_on_the_stream_is_not_taken_for_the_result() {
        let body = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"progress\":1}}\n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":9}}\n",
            "\n",
        );

        let parsed = envelope("text/event-stream", body).expect("envelope");
        assert_eq!(parsed.result.expect("result")["ok"], 9);
    }

    #[test]
    fn a_server_initiated_request_on_the_stream_is_skipped_too() {
        let body = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"id\":99,\"method\":\"sampling/createMessage\",\"params\":{}}\n",
            "\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":10}}\n",
            "\n",
        );

        let parsed = envelope("text/event-stream", body).expect("envelope");
        assert_eq!(parsed.result.expect("result")["ok"], 10);
    }

    #[test]
    fn a_lone_notification_is_not_a_response_at_all() {
        let body = r#"{"jsonrpc":"2.0","method":"notifications/message","params":{}}"#;
        assert!(envelope("application/json", body).is_none());
    }

    #[test]
    fn a_multi_line_sse_payload_is_rejoined_before_parsing() {
        let body = "data: {\"jsonrpc\":\"2.0\",\ndata: \"id\":1,\"result\":{\"ok\":3}}\n\n";
        let parsed = envelope("text/event-stream", body).expect("envelope");
        assert_eq!(parsed.result.expect("result")["ok"], 3);
    }

    #[test]
    fn text_blocks_are_joined_and_untyped_blocks_are_counted() {
        let result: ToolCallResult = serde_json::from_value(json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "image", "data": "…" },
                { "type": "text", "text": "second" }
            ]
        }))
        .expect("result");

        let (text, skipped) = readable_text(&result.content);
        assert_eq!(text.expect("text"), "first\nsecond");
        assert_eq!(skipped, 1);
    }

    #[test]
    fn structured_content_reaches_the_model_when_there_is_no_text() {
        let result: ToolCallResult = serde_json::from_value(
            json!({ "content": [], "structuredContent": { "total": 4200 } }),
        )
        .expect("result");

        let output = to_output(result);
        assert_eq!(output.content, r#"{"total":4200}"#);
        assert_eq!(output.data.expect("data").r#type, "structured");
    }

    #[test]
    fn an_empty_result_still_says_something() {
        let result: ToolCallResult =
            serde_json::from_value(json!({ "content": [] })).expect("result");
        assert_eq!(to_output(result).content, EMPTY_RESULT);
    }

    #[test]
    fn non_object_arguments_are_replaced_with_an_empty_object() {
        assert_eq!(arguments_object(&Value::Null), json!({}));
        assert_eq!(arguments_object(&json!("hotel")), json!({}));
        assert_eq!(
            arguments_object(&json!({ "hotelId": "h1" })),
            json!({ "hotelId": "h1" })
        );
    }

    #[test]
    fn server_faults_and_throttling_are_the_only_transient_statuses() {
        assert!(is_transient(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_transient(StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_transient(StatusCode::UNAUTHORIZED));
        assert!(!is_transient(StatusCode::BAD_REQUEST));
    }
}
