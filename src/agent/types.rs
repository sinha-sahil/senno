use crate::error::Error;
use crate::types::{ToolDefinition, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

const UNAVAILABLE_MESSAGE: &str = "The assistant is temporarily unavailable. Please try again.";
const INTERRUPTED_MESSAGE: &str = "The reply was interrupted. Please try again.";
const MAX_TOOL_ROUNDS_MESSAGE: &str = "Maximum tool calling rounds exceeded";

pub trait AgentFlow: Send + Sync {
    fn system_prompt(&self) -> String;

    fn tool_definitions(&self) -> Vec<ToolDefinition>;

    fn execute_tool<'a>(
        &'a self,
        name: &'a str,
        args: &'a Value,
        session: &'a AgentSession,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>>;

    fn tool_sources(&self) -> Vec<&dyn ToolSource> {
        Vec::new()
    }
}

pub trait ToolSource: Send + Sync {
    fn definitions(&self) -> &[ToolDefinition];

    fn handles(&self, name: &str) -> bool;

    fn invoke<'a>(
        &'a self,
        name: &'a str,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>>;
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub data: Option<DataEvent>,
    pub session_metadata: Option<serde_json::Value>,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            data: None,
            session_metadata: None,
        }
    }

    pub fn data(mut self, data_type: impl Into<String>, payload: serde_json::Value) -> Self {
        self.data = Some(DataEvent {
            r#type: data_type.into(),
            payload,
        });
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.session_metadata = Some(metadata);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEvent {
    pub r#type: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum SseEvent {
    Text {
        delta: String,
    },
    ToolStatus {
        tool: String,
        status: ToolCallStatus,
        label: Option<String>,
    },
    Data {
        r#type: String,
        payload: serde_json::Value,
    },
    Error {
        code: String,
        message: String,
    },
    Done {
        session_id: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub calls: u32,
    pub compactions: u32,
    pub unmetered_calls: u32,
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsage {
    fn add(&mut self, usage: Option<&Usage>, compaction: bool) {
        self.calls = self.calls.saturating_add(1);
        if compaction {
            self.compactions = self.compactions.saturating_add(1);
        }
        let Some(u) = usage else {
            self.unmetered_calls = self.unmetered_calls.saturating_add(1);
            return;
        };

        let input = u64::from(u.input_tokens.unwrap_or(0));
        let output = u64::from(u.output_tokens.unwrap_or(0));
        let reasoning = u.reasoning_tokens.map(u64::from).unwrap_or_else(|| {
            u.total_tokens
                .map_or(0, |t| u64::from(t).saturating_sub(input + output))
        });
        let total = u
            .total_tokens
            .map_or(input + output + reasoning, u64::from)
            .max(input + output + reasoning);

        self.input_tokens = self.input_tokens.saturating_add(input);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(u64::from(u.cached_input_tokens.unwrap_or(0)));
        self.output_tokens = self.output_tokens.saturating_add(output);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(reasoning);
        self.total_tokens = self.total_tokens.saturating_add(total);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionUsage {
    pub turns: u32,
    pub totals: TokenUsage,
    pub by_model: BTreeMap<String, TokenUsage>,
}

impl SessionUsage {
    pub fn record_call(&mut self, model: &str, usage: Option<&Usage>) {
        self.entry(model, usage, false);
    }

    pub fn record_compaction(&mut self, model: &str, usage: Option<&Usage>) {
        self.entry(model, usage, true);
    }

    pub fn record_turn(&mut self) {
        self.turns = self.turns.saturating_add(1);
    }

    fn entry(&mut self, model: &str, usage: Option<&Usage>, compaction: bool) {
        self.totals.add(usage, compaction);
        self.by_model
            .entry(model.to_string())
            .or_default()
            .add(usage, compaction);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Calling,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub flow: String,
    pub messages: Vec<ChatMessage>,
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub usage: SessionUsage,
    pub created_at: String,
    pub last_active: String,
}

impl AgentSession {
    pub fn new(id: impl Into<String>, flow: impl Into<String>) -> Self {
        let now = now_rfc3339();
        Self {
            id: id.into(),
            flow: flow.into(),
            messages: Vec::new(),
            metadata: serde_json::json!({}),
            usage: SessionUsage::default(),
            created_at: now.clone(),
            last_active: now,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatMessage {
    User {
        content: String,
    },
    Assistant {
        content: String,
    },
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
        /// Used by Gemini 2.5+ thinking models. Other providers serialize this as absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        name: String,
        content: String,
    },
}

pub(crate) fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "RFC3339 format unexpectedly failed; using epoch");
            "1970-01-01T00:00:00Z".into()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_enum_serializes_tagged() {
        let msg = ChatMessage::User {
            content: "hi".into(),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["kind"], "user");
        assert_eq!(json["content"], "hi");

        let tool = ChatMessage::ToolCall {
            id: "t1".into(),
            name: "search".into(),
            args: serde_json::json!({"q": "x"}),
            thought_signature: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["kind"], "tool_call");
        assert_eq!(json["id"], "t1");
    }

    #[test]
    fn chat_message_round_trips() {
        let original = ChatMessage::ToolResult {
            tool_call_id: "t1".into(),
            name: "search".into(),
            content: "ok".into(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        match back {
            ChatMessage::ToolResult {
                tool_call_id,
                name,
                content,
            } => {
                assert_eq!(tool_call_id, "t1");
                assert_eq!(name, "search");
                assert_eq!(content, "ok");
            }
            _ => panic!("wrong variant"),
        }
    }

    fn gemini_usage(input: u32, output: u32, thoughts: Option<u32>, total: Option<u32>) -> Usage {
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            reasoning_tokens: thoughts,
            total_tokens: total,
            cached_input_tokens: None,
        }
    }

    #[test]
    fn reported_thinking_tokens_are_kept_out_of_output() {
        let mut u = SessionUsage::default();
        u.record_call(
            "gemini-2.5-flash",
            Some(&gemini_usage(100, 50, Some(250), Some(400))),
        );
        assert_eq!(u.totals.input_tokens, 100);
        assert_eq!(u.totals.output_tokens, 50);
        assert_eq!(u.totals.reasoning_tokens, 250);
        assert_eq!(u.totals.total_tokens, 400);
    }

    #[test]
    fn thinking_tokens_are_derived_when_the_provider_omits_them() {
        let mut u = SessionUsage::default();
        u.record_call(
            "gemini-2.5-flash",
            Some(&gemini_usage(100, 50, None, Some(400))),
        );
        assert_eq!(u.totals.reasoning_tokens, 250);
        assert_eq!(u.totals.total_tokens, 400);
    }

    #[test]
    fn total_falls_back_to_the_sum_and_never_shrinks_below_it() {
        let mut u = SessionUsage::default();
        u.record_call("m", Some(&gemini_usage(100, 50, None, None)));
        assert_eq!(u.totals.reasoning_tokens, 0);
        assert_eq!(u.totals.total_tokens, 150);

        let mut low = SessionUsage::default();
        low.record_call("m", Some(&gemini_usage(100, 50, Some(10), Some(9))));
        assert_eq!(low.totals.total_tokens, 160);
    }

    #[test]
    fn a_call_without_usage_is_counted_and_flagged() {
        let mut u = SessionUsage::default();
        u.record_call("m", None);
        assert_eq!(u.totals.calls, 1);
        assert_eq!(u.totals.unmetered_calls, 1);
        assert_eq!(u.totals.total_tokens, 0);
    }

    #[test]
    fn compaction_is_counted_separately_but_still_billed() {
        let mut u = SessionUsage::default();
        u.record_call("m", Some(&gemini_usage(10, 5, None, None)));
        u.record_compaction("m", Some(&gemini_usage(200, 40, None, None)));
        assert_eq!(u.totals.calls, 2);
        assert_eq!(u.totals.compactions, 1);
        assert_eq!(u.totals.total_tokens, 255);
    }

    #[test]
    fn usage_splits_by_model() {
        let mut u = SessionUsage::default();
        u.record_call("gemini-2.5-flash", Some(&gemini_usage(10, 5, None, None)));
        u.record_call("gemini-2.5-pro", Some(&gemini_usage(20, 10, None, None)));
        assert_eq!(u.by_model["gemini-2.5-flash"].total_tokens, 15);
        assert_eq!(u.by_model["gemini-2.5-pro"].total_tokens, 30);
        assert_eq!(u.totals.total_tokens, 45);
    }

    #[test]
    fn serialized_session_keys_stay_snake_case() {
        let mut session = AgentSession::new("s1", "commerce");
        session
            .usage
            .record_call("gemini-2.5-flash", Some(&Usage::default()));
        let json = serde_json::to_value(&session).unwrap();

        assert!(json.get("created_at").is_some());
        assert!(json.get("last_active").is_some());
        let totals = &json["usage"]["totals"];
        for key in [
            "unmetered_calls",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ] {
            assert!(totals.get(key).is_some(), "missing {key}");
        }
        assert!(json["usage"].get("by_model").is_some());
    }

    #[test]
    fn session_without_usage_field_still_deserializes() {
        let stored = r#"{"id":"s1","flow":"commerce","messages":[],"metadata":{},
            "created_at":"2026-08-20T00:00:00Z","last_active":"2026-08-20T00:00:00Z"}"#;
        let session: AgentSession = serde_json::from_str(stored).unwrap();
        assert_eq!(session.usage, SessionUsage::default());
    }

    #[test]
    fn agent_session_new_has_valid_timestamps_and_object_metadata() {
        let s = AgentSession::new("abc", "shopping");
        assert_eq!(s.id, "abc");
        assert!(!s.created_at.is_empty());
        assert_eq!(s.created_at, s.last_active);
        assert!(s.metadata.is_object());
    }
}

impl SseEvent {
    pub fn from_result(event: Result<SseEvent, Error>) -> SseEvent {
        match event {
            Ok(event) => event,
            Err(error) => {
                tracing::error!(error = %error, "agent stream failed");
                Self::client_error("internal", UNAVAILABLE_MESSAGE)
            }
        }
    }

    pub(crate) fn llm_error() -> Self {
        Self::client_error("llm_error", UNAVAILABLE_MESSAGE)
    }

    pub(crate) fn stream_error() -> Self {
        Self::client_error("stream_error", INTERRUPTED_MESSAGE)
    }

    pub(crate) fn tool_error(tool: &str) -> Self {
        Self::client_error("tool_error", format!("The {tool} tool failed."))
    }

    pub(crate) fn max_tool_rounds() -> Self {
        Self::client_error("max_tool_rounds", MAX_TOOL_ROUNDS_MESSAGE)
    }

    fn client_error(code: &str, message: impl Into<String>) -> Self {
        Self::Error {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Default)]
pub struct StreamStats {
    pub text: u32,
    pub tool_status: u32,
    pub data: u32,
    pub errors: u32,
    pub done: u32,
    pub saw_event: bool,
    pub client_dropped: bool,
}

impl StreamStats {
    pub fn record(&mut self, event: &SseEvent) {
        self.saw_event = true;
        match event {
            SseEvent::Text { .. } => self.text += 1,
            SseEvent::ToolStatus { .. } => self.tool_status += 1,
            SseEvent::Data { .. } => self.data += 1,
            SseEvent::Error { .. } => self.errors += 1,
            SseEvent::Done { .. } => self.done += 1,
        }
    }
}
