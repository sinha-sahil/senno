//! senno requests ⇄ the Bedrock Converse API.

use aws_sdk_bedrockruntime::types as bt;
use aws_smithy_types::Document;

use crate::error::Error;
use crate::provider::LlmStream;
use crate::types::{ContentPart, GenerateRequest, GenerateResponse, Role, Usage};

use super::client::BedrockClient;
use super::document::{to_document, to_value};
use super::error::{PROVIDER, from_sdk};
use super::stream;

/// Prefix for a leading assistant message (a compaction summary) that has to
/// be re-sent as a user message: Converse requires the first message to come
/// from the user.
const LEADING_ASSISTANT_PREFIX: &str = "Summary of the earlier conversation:\n";

/// Everything a Converse or ConverseStream call needs, built once from a senno
/// request.
pub(super) struct ConverseParts {
    pub system: Vec<bt::SystemContentBlock>,
    pub messages: Vec<bt::Message>,
    pub inference: Option<bt::InferenceConfiguration>,
    pub tools: Option<bt::ToolConfiguration>,
    pub additional_fields: Option<Document>,
}

pub(super) fn to_converse(request: &GenerateRequest) -> Result<ConverseParts, Error> {
    if request.web_fetch {
        tracing::warn!("web_fetch is unavailable on Bedrock; the model cannot fetch pages");
    }
    if request.thinking_budget.is_some() {
        // Extended thinking returns reasoning blocks that must be sent back
        // verbatim on the next tool round, and senno's history does not keep
        // them yet. Ignoring the budget is safer than a broken second round.
        tracing::warn!("thinking_budget is not supported on Bedrock yet and is ignored");
    }

    let system = request
        .system
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| vec![bt::SystemContentBlock::Text(s.to_string())])
        .unwrap_or_default();

    Ok(ConverseParts {
        system,
        messages: to_messages(&request.messages)?,
        inference: inference_config(request),
        tools: tool_config(request)?,
        additional_fields: additional_fields(request),
    })
}

fn to_messages(messages: &[crate::types::Message]) -> Result<Vec<bt::Message>, Error> {
    // Converse wants alternating roles and a user message first. senno's agent
    // history can hold two assistant messages in a row (text, then a tool call
    // in the next round) and start with an assistant summary after compaction,
    // so merge runs of one role and re-role a leading assistant summary.
    let mut merged: Vec<(Role, Vec<bt::ContentBlock>)> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let mut role = message.role;
        let mut blocks = to_blocks(&message.content)?;
        if index == 0 && role == Role::Assistant {
            let text_only = blocks
                .iter()
                .all(|b| matches!(b, bt::ContentBlock::Text(_)));
            if text_only {
                role = Role::User;
                for block in &mut blocks {
                    if let bt::ContentBlock::Text(text) = block {
                        *text = format!("{LEADING_ASSISTANT_PREFIX}{text}");
                        break;
                    }
                }
            }
        }
        if blocks.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some((last_role, last_blocks)) if *last_role == role => last_blocks.extend(blocks),
            _ => merged.push((role, blocks)),
        }
    }

    merged
        .into_iter()
        .map(|(role, blocks)| {
            bt::Message::builder()
                .role(match role {
                    Role::User => bt::ConversationRole::User,
                    Role::Assistant => bt::ConversationRole::Assistant,
                })
                .set_content(Some(blocks))
                .build()
                .map_err(|e| Error::internal(format!("bedrock message build failed: {e}")))
        })
        .collect()
}

fn to_blocks(parts: &[ContentPart]) -> Result<Vec<bt::ContentBlock>, Error> {
    let mut blocks = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            // Converse rejects empty text blocks.
            ContentPart::Text(text) if text.is_empty() => {}
            ContentPart::Text(text) => blocks.push(bt::ContentBlock::Text(text.clone())),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
                ..
            } => blocks.push(bt::ContentBlock::ToolUse(
                bt::ToolUseBlock::builder()
                    .tool_use_id(id)
                    .name(name)
                    .input(to_document(arguments))
                    .build()
                    .map_err(|e| Error::internal(format!("bedrock tool_use build failed: {e}")))?,
            )),
            ContentPart::ToolResult {
                tool_call_id,
                content,
                ..
            } => blocks.push(bt::ContentBlock::ToolResult(
                bt::ToolResultBlock::builder()
                    .tool_use_id(tool_call_id)
                    .content(bt::ToolResultContentBlock::Text(tool_result_text(content)))
                    .build()
                    .map_err(|e| {
                        Error::internal(format!("bedrock tool_result build failed: {e}"))
                    })?,
            )),
        }
    }
    Ok(blocks)
}

/// A JSON string is sent as its text; anything else as serialized JSON. An
/// empty result becomes a placeholder, since Converse rejects empty text.
fn tool_result_text(content: &serde_json::Value) -> String {
    let text = match content {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if text.is_empty() {
        "(empty result)".to_string()
    } else {
        text
    }
}

fn inference_config(request: &GenerateRequest) -> Option<bt::InferenceConfiguration> {
    if request.max_tokens.is_none() && request.temperature.is_none() && request.top_p.is_none() {
        return None;
    }
    Some(
        bt::InferenceConfiguration::builder()
            .set_max_tokens(
                request
                    .max_tokens
                    .map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
            )
            .set_temperature(request.temperature)
            .set_top_p(request.top_p)
            .build(),
    )
}

fn tool_config(request: &GenerateRequest) -> Result<Option<bt::ToolConfiguration>, Error> {
    if request.tools.is_empty() {
        return Ok(None);
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            let schema = serde_json::to_value(&tool.parameters).map_err(|e| {
                Error::internal(format!("tool {} schema did not serialize: {e}", tool.name))
            })?;
            let mut spec = bt::ToolSpecification::builder()
                .name(&tool.name)
                .input_schema(bt::ToolInputSchema::Json(to_document(&schema)));
            if !tool.description.is_empty() {
                spec = spec.description(&tool.description);
            }
            spec.build()
                .map(bt::Tool::ToolSpec)
                .map_err(|e| Error::internal(format!("bedrock tool spec build failed: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    bt::ToolConfiguration::builder()
        .set_tools(Some(tools))
        .build()
        .map(Some)
        .map_err(|e| Error::internal(format!("bedrock tool config build failed: {e}")))
}

/// Model-specific fields Converse has no slot for. `top_k` is passed through
/// for models that accept it (Anthropic); a model that does not rejects the
/// request with a validation error naming the field.
fn additional_fields(request: &GenerateRequest) -> Option<Document> {
    let top_k = request.top_k?;
    Some(to_document(&serde_json::json!({ "top_k": top_k })))
}

fn trace(enabled: bool) -> bt::GuardrailTrace {
    if enabled {
        bt::GuardrailTrace::Enabled
    } else {
        bt::GuardrailTrace::Disabled
    }
}

pub(super) fn guardrail(client: &BedrockClient) -> Option<bt::GuardrailConfiguration> {
    client.guardrail.as_ref().map(|g| {
        bt::GuardrailConfiguration::builder()
            .guardrail_identifier(&g.id)
            .guardrail_version(&g.version)
            .trace(trace(g.trace))
            .build()
    })
}

pub(super) fn stream_guardrail(client: &BedrockClient) -> Option<bt::GuardrailStreamConfiguration> {
    client.guardrail.as_ref().map(|g| {
        bt::GuardrailStreamConfiguration::builder()
            .guardrail_identifier(&g.id)
            .guardrail_version(&g.version)
            .trace(trace(g.trace))
            .build()
    })
}

pub(super) fn to_usage(usage: &bt::TokenUsage) -> Usage {
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let cache_read = usage.cache_read_input_tokens().map(to_u32);
    let cache_write = usage.cache_write_input_tokens().map(to_u32);
    // senno's input_tokens counts every prompt token, cached ones included;
    // Bedrock reports cache reads and writes separately.
    let input = to_u32(usage.input_tokens())
        .saturating_add(cache_read.unwrap_or(0))
        .saturating_add(cache_write.unwrap_or(0));
    let output = to_u32(usage.output_tokens());
    Usage {
        input_tokens: Some(input),
        cached_input_tokens: cache_read,
        cache_write_input_tokens: cache_write,
        output_tokens: Some(output),
        reasoning_tokens: None,
        total_tokens: Some(to_u32(usage.total_tokens()).max(input.saturating_add(output))),
    }
}

pub(super) fn from_content(blocks: &[bt::ContentBlock]) -> Vec<ContentPart> {
    blocks
        .iter()
        .filter_map(|block| match block {
            bt::ContentBlock::Text(text) => Some(ContentPart::Text(text.clone())),
            bt::ContentBlock::ToolUse(tool) => Some(ContentPart::ToolCall {
                id: tool.tool_use_id().to_string(),
                name: tool.name().to_string(),
                arguments: to_value(tool.input()),
                thought_signature: None,
            }),
            _ => None,
        })
        .collect()
}

pub(super) async fn generate(
    client: &BedrockClient,
    request: &GenerateRequest,
) -> Result<GenerateResponse, Error> {
    let parts = to_converse(request)?;
    let output = client
        .inner
        .converse()
        .model_id(&request.model)
        .set_system(Some(parts.system).filter(|s| !s.is_empty()))
        .set_messages(Some(parts.messages))
        .set_inference_config(parts.inference)
        .set_tool_config(parts.tools)
        .set_additional_model_request_fields(parts.additional_fields)
        .set_guardrail_config(guardrail(client))
        .send()
        .await
        .map_err(|e| from_sdk("Converse", e))?;

    let content = match output.output() {
        Some(bt::ConverseOutput::Message(message)) => from_content(message.content()),
        _ => {
            return Err(Error::provider_permanent(
                PROVIDER,
                "Converse returned no message",
            ));
        }
    };

    Ok(GenerateResponse {
        content,
        stop_reason: Some(output.stop_reason().as_str().to_string()),
        usage: output.usage().map(to_usage),
    })
}

pub(super) async fn stream_generate(
    client: &BedrockClient,
    request: &GenerateRequest,
) -> Result<LlmStream, Error> {
    let parts = to_converse(request)?;
    let output = client
        .inner
        .converse_stream()
        .model_id(&request.model)
        .set_system(Some(parts.system).filter(|s| !s.is_empty()))
        .set_messages(Some(parts.messages))
        .set_inference_config(parts.inference)
        .set_tool_config(parts.tools)
        .set_additional_model_request_fields(parts.additional_fields)
        .set_guardrail_config(stream_guardrail(client))
        .send()
        .await
        .map_err(|e| from_sdk("ConverseStream", e))?;

    Ok(stream::into_chunks(output.stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ParameterSchema, ToolDefinition};
    use serde_json::json;

    fn roles(messages: &[bt::Message]) -> Vec<bt::ConversationRole> {
        messages.iter().map(|m| m.role().clone()).collect()
    }

    #[test]
    fn consecutive_messages_of_one_role_are_merged() {
        let history = vec![
            Message::user("find shoes"),
            Message::tool_call("t1", "search", json!({"q": "shoes"})),
            Message::tool_result("t1", "search", json!("3 results")),
            Message::assistant("Here are three."),
            Message::tool_call("t2", "recommend", json!({"ids": ["a"]})),
        ];
        let messages = to_messages(&history).unwrap();
        assert_eq!(
            roles(&messages),
            vec![
                bt::ConversationRole::User,
                bt::ConversationRole::Assistant,
                bt::ConversationRole::User,
                bt::ConversationRole::Assistant,
            ]
        );
        // The last assistant message holds the text and the second tool call.
        let last = messages.last().unwrap().content();
        assert!(matches!(&last[0], bt::ContentBlock::Text(t) if t == "Here are three."));
        assert!(matches!(&last[1], bt::ContentBlock::ToolUse(t) if t.name() == "recommend"));
    }

    #[test]
    fn leading_assistant_summary_is_sent_as_user() {
        let history = vec![
            Message::assistant("They wanted shoes."),
            Message::user("cheaper?"),
        ];
        let messages = to_messages(&history).unwrap();
        assert_eq!(roles(&messages), vec![bt::ConversationRole::User]);
        let blocks = messages[0].content();
        assert!(matches!(&blocks[0], bt::ContentBlock::Text(t)
            if t == "Summary of the earlier conversation:\nThey wanted shoes."));
        assert!(matches!(&blocks[1], bt::ContentBlock::Text(t) if t == "cheaper?"));
    }

    #[test]
    fn empty_text_is_dropped_and_empty_tool_result_gets_placeholder() {
        let history = vec![
            Message::user(""),
            Message::user("hi"),
            Message::tool_result("t1", "noop", json!("")),
        ];
        let messages = to_messages(&history).unwrap();
        let blocks = messages[0].content();
        assert_eq!(blocks.len(), 2);
        match &blocks[1] {
            bt::ContentBlock::ToolResult(r) => {
                assert!(
                    matches!(&r.content()[0], bt::ToolResultContentBlock::Text(t) if t == "(empty result)")
                )
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn tool_arguments_and_json_results_survive() {
        let history = vec![
            Message::tool_call("t1", "search", json!({"q": "shoes", "limit": 5})),
            Message::tool_result("t1", "search", json!({"count": 2})),
        ];
        let messages = to_messages(&history).unwrap();
        match &messages[0].content()[0] {
            bt::ContentBlock::ToolUse(t) => {
                assert_eq!(t.tool_use_id(), "t1");
                assert_eq!(to_value(t.input()), json!({"q": "shoes", "limit": 5}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }
        match &messages[1].content()[0] {
            bt::ContentBlock::ToolResult(r) => {
                assert!(
                    matches!(&r.content()[0], bt::ToolResultContentBlock::Text(t) if t == r#"{"count":2}"#)
                )
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn tools_carry_their_json_schema() {
        let tool = ToolDefinition::new("search", "Search the catalog").with_parameters(
            ParameterSchema::object()
                .with_property("q", ParameterSchema::string("query"))
                .with_required(["q"]),
        );
        let request = GenerateRequest::new("m", vec![Message::user("hi")]).with_tools(vec![tool]);
        let config = tool_config(&request).unwrap().unwrap();
        match &config.tools()[0] {
            bt::Tool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "search");
                assert_eq!(spec.description(), Some("Search the catalog"));
                let Some(bt::ToolInputSchema::Json(schema)) = spec.input_schema() else {
                    panic!("schema missing");
                };
                assert_eq!(
                    to_value(schema),
                    json!({
                        "type": "object",
                        "properties": {"q": {"type": "string", "description": "query"}},
                        "required": ["q"],
                    })
                );
            }
            other => panic!("expected tool spec, got {other:?}"),
        }
    }

    #[test]
    fn inference_settings_map_and_top_k_rides_additional_fields() {
        let request = GenerateRequest::new("m", vec![Message::user("hi")])
            .with_max_tokens(512)
            .with_temperature(0.2)
            .with_top_k(40);
        let inference = inference_config(&request).unwrap();
        assert_eq!(inference.max_tokens(), Some(512));
        assert_eq!(inference.temperature(), Some(0.2));
        assert_eq!(
            to_value(&additional_fields(&request).unwrap()),
            json!({"top_k": 40})
        );
        assert!(inference_config(&GenerateRequest::new("m", vec![])).is_none());
    }

    #[test]
    fn usage_counts_cache_reads_and_writes_as_input() {
        let usage = bt::TokenUsage::builder()
            .input_tokens(100)
            .output_tokens(20)
            .total_tokens(170)
            .cache_read_input_tokens(40)
            .cache_write_input_tokens(10)
            .build()
            .unwrap();
        let u = to_usage(&usage);
        assert_eq!(u.input_tokens, Some(150));
        assert_eq!(u.cached_input_tokens, Some(40));
        assert_eq!(u.cache_write_input_tokens, Some(10));
        assert_eq!(u.output_tokens, Some(20));
        assert_eq!(u.total_tokens, Some(170));
    }
}
