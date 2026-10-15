use crate::error::Error;
use crate::provider::LlmStream;
use crate::types::{ContentPart, GenerateRequest, GenerateResponse, Role, StreamChunk, Usage};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::client::LiteLlmClient;

pub(crate) async fn generate(
    client: &LiteLlmClient,
    request: &GenerateRequest,
) -> Result<GenerateResponse, Error> {
    let wire = to_chat_request(request, false);
    let resp = client
        .send(
            client
                .http
                .post(client.endpoint("chat/completions"))
                .json(&wire),
        )
        .await?;
    let wire: ChatResponse = resp
        .json()
        .await
        .map_err(|e| Error::provider("litellm", format!("Parse failed: {e}")))?;
    Ok(from_chat_response(wire))
}

pub(crate) async fn stream_generate(
    client: &LiteLlmClient,
    request: &GenerateRequest,
) -> Result<LlmStream, Error> {
    let wire = to_chat_request(request, true);
    let resp = client
        .send(
            client
                .http
                .post(client.endpoint("chat/completions"))
                .json(&wire),
        )
        .await?;
    Ok(parse_sse(resp))
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Deserialize)]
struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: WireFunctionCall,
}

#[derive(Serialize, Deserialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolDef,
}

#[derive(Serialize)]
struct WireToolDef {
    name: String,
    description: String,
    parameters: crate::types::ParameterSchema,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Option<WireMessage>,
    delta: Option<WireDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireDelta {
    content: Option<String>,
    tool_calls: Option<Vec<WireToolDelta>>,
}

#[derive(Deserialize)]
struct WireToolDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<WireFunctionDelta>,
}

#[derive(Deserialize)]
struct WireFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct WireUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    pub(super) total_tokens: Option<u32>,
}

impl From<WireUsage> for Usage {
    fn from(u: WireUsage) -> Self {
        Usage {
            input_tokens: u.prompt_tokens,
            cached_input_tokens: None,
            output_tokens: u.completion_tokens,
            reasoning_tokens: None,
            total_tokens: u.total_tokens,
        }
    }
}

fn to_chat_request(req: &GenerateRequest, stream: bool) -> ChatRequest {
    let mut messages = Vec::new();
    if let Some(system) = &req.system {
        messages.push(WireMessage {
            role: "system".into(),
            content: Some(system.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }
    for message in &req.messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let text = message
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        let tool_calls: Vec<_> = message
            .content
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } => Some(WireToolCall {
                    id: id.clone(),
                    kind: "function".into(),
                    function: WireFunctionCall {
                        name: name.clone(),
                        arguments: arguments.to_string(),
                    },
                }),
                _ => None,
            })
            .collect();
        if !text.is_empty() || !tool_calls.is_empty() {
            messages.push(WireMessage {
                role: role.into(),
                content: (!text.is_empty()).then_some(text),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                tool_call_id: None,
                name: None,
            });
        }
        for part in &message.content {
            if let ContentPart::ToolResult {
                tool_call_id,
                name,
                content,
            } = part
            {
                messages.push(WireMessage {
                    role: "tool".into(),
                    content: Some(content.to_string()),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.clone()),
                    name: Some(name.clone()),
                });
            }
        }
    }
    ChatRequest {
        model: req.model.clone(),
        messages,
        tools: (!req.tools.is_empty()).then(|| {
            req.tools
                .iter()
                .map(|t| WireTool {
                    kind: "function",
                    function: WireToolDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect()
        }),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stream,
        stream_options: stream.then_some(StreamOptions {
            include_usage: true,
        }),
    }
}

fn from_chat_response(resp: ChatResponse) -> GenerateResponse {
    let mut content = Vec::new();
    let mut stop_reason = None;
    if let Some(choice) = resp.choices.into_iter().next() {
        stop_reason = choice.finish_reason;
        if let Some(message) = choice.message {
            if let Some(text) = message.content.filter(|t| !t.is_empty()) {
                content.push(ContentPart::Text(text));
            }
            if let Some(calls) = message.tool_calls {
                for call in calls {
                    content.push(ContentPart::ToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments: parse_args(&call.function.arguments),
                        thought_signature: None,
                    });
                }
            }
        }
    }
    GenerateResponse {
        content,
        stop_reason,
        usage: resp.usage.map(Usage::from),
    }
}

fn parse_sse(resp: reqwest::Response) -> LlmStream {
    let stream = async_stream::stream! {
        let mut bytes = resp.bytes_stream().fuse();
        let mut buffer = Vec::new();
        let mut usage = None;
        let mut tools: Vec<PendingTool> = Vec::new();
        while let Some(chunk) = bytes.next().await {
            buffer.extend_from_slice(&chunk.map_err(|e| Error::provider("litellm", format!("Stream read error: {e}")))?);
            while let Some((pos, sep)) = find_frame_boundary(&buffer) {
                let frame = buffer.drain(..pos + sep).collect::<Vec<u8>>();
                let Ok(event) = std::str::from_utf8(&frame[..frame.len() - sep]) else { continue };
                for line in event.lines().filter_map(|l| l.strip_prefix("data: ")) {
                    if line.trim() == "[DONE]" { continue; }
                    let Ok(resp) = serde_json::from_str::<ChatResponse>(line) else { continue; };
                    usage = resp.usage.map(Usage::from).or(usage);
                    for choice in resp.choices {
                        if let Some(delta) = choice.delta {
                            if let Some(text) = delta.content.filter(|t| !t.is_empty()) {
                                yield Ok(StreamChunk::Text(text));
                            }
                            if let Some(calls) = delta.tool_calls {
                                for call in calls { merge_tool_delta(&mut tools, call); }
                            }
                        }
                        if let Some(reason) = choice.finish_reason {
                            for tool in tools.drain(..) {
                                yield Ok(StreamChunk::ToolCall { id: tool.id, name: tool.name, arguments: parse_args(&tool.arguments), thought_signature: None });
                            }
                            yield Ok(StreamChunk::Done { finish_reason: reason, usage: usage.take() });
                        }
                    }
                }
            }
        }
    };
    Box::pin(stream)
}

#[derive(Default)]
struct PendingTool {
    id: String,
    name: String,
    arguments: String,
}

fn merge_tool_delta(tools: &mut Vec<PendingTool>, delta: WireToolDelta) {
    let index = delta.index.unwrap_or(tools.len());
    while tools.len() <= index {
        tools.push(PendingTool::default());
    }
    let tool = &mut tools[index];
    if let Some(id) = delta.id {
        tool.id = id;
    }
    if let Some(function) = delta.function {
        if let Some(name) = function.name {
            tool.name = name;
        }
        if let Some(arguments) = function.arguments {
            tool.arguments.push_str(&arguments);
        }
    }
}

fn parse_args(args: &str) -> serde_json::Value {
    serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({}))
}

fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    let lf = buf.windows(2).position(|w| w == b"\n\n");
    match (crlf, lf) {
        (Some(c), Some(l)) if c <= l => Some((c, 4)),
        (Some(_), Some(l)) => Some((l, 2)),
        (Some(c), None) => Some((c, 4)),
        (None, Some(l)) => Some((l, 2)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ParameterSchema, ToolDefinition};

    #[test]
    fn chat_request_maps_system_tools_and_params() {
        let req = GenerateRequest::one_shot("gpt-4o-mini", "sys", "hi")
            .with_max_tokens(10)
            .with_tools(vec![
                ToolDefinition::new("search", "Search").with_parameters(ParameterSchema::object()),
            ]);
        let json = serde_json::to_value(to_chat_request(&req, false)).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["tools"][0]["function"]["name"], "search");
        assert_eq!(json["max_tokens"], 10);
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn tool_calls_round_trip_from_response() {
        let wire: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}]}}]
        })).unwrap();
        let resp = from_chat_response(wire);
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_calls"));
        assert!(matches!(&resp.content[0], ContentPart::ToolCall { name, .. } if name == "search"));
    }

    #[test]
    fn tool_results_become_tool_messages() {
        let req = GenerateRequest::new(
            "m",
            vec![Message::tool_result(
                "call_1",
                "search",
                serde_json::json!({"ok": true}),
            )],
        );
        let json = serde_json::to_value(to_chat_request(&req, false)).unwrap();
        assert_eq!(json["messages"][0]["role"], "tool");
        assert_eq!(json["messages"][0]["tool_call_id"], "call_1");
    }
}
