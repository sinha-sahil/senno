use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub(super) struct JsonRpcRequest<'a> {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    pub method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct JsonRpcResponse {
    pub method: Option<String>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
pub(super) struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct InitializeResult {
    pub protocol_version: Option<String>,
    pub capabilities: Option<ServerCapabilities>,
    pub server_info: Option<ServerInfo>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ServerCapabilities {
    pub tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ServerInfo {
    pub name: Option<String>,
    pub version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolsListResult {
    #[serde(default)]
    pub tools: Vec<McpTool>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct McpTool {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
    pub output_schema: Option<Value>,
    pub annotations: Option<McpAnnotations>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct McpAnnotations {
    pub title: Option<String>,
    pub read_only_hint: Option<bool>,
    pub destructive_hint: Option<bool>,
    pub idempotent_hint: Option<bool>,
    pub open_world_hint: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ToolCallResult {
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub structured_content: Option<Value>,
    pub is_error: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlock {
    #[serde(rename = "type", default)]
    pub kind: String,
    pub text: Option<String>,
}
