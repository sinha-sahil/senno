use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use futures::StreamExt;
use senno::agent::{
    AgentConfig, AgentFlow, AgentSession, ChatMessage, SseEvent, ToolOutput, get_agent_engine,
};
use senno::{
    Error, GenerateRequest, GenerateResponse, LlmProvider, LlmStream, StreamChunk, ToolDefinition,
    Usage,
};

const MODEL: &str = "gemini-3.1-pro-preview";
const UPSTREAM_DETAIL: &str = "API error (404): projects/acme/locations/asia-south1/publishers/google/models/gemini-3.1-pro-preview not found";
const TOOL_DETAIL: &str = "dependency_failed: opensearch index products-acme unreachable";
const LEAKS: [&str; 6] = [
    "vertex",
    "projects/",
    "gemini",
    "404",
    "opensearch",
    "dependency_failed",
];

enum Round {
    Refuse,
    Chunks(Vec<Result<StreamChunk, Error>>),
}

struct Scripted {
    rounds: Mutex<Vec<Round>>,
}

impl Scripted {
    fn new(rounds: Vec<Round>) -> Self {
        Self {
            rounds: Mutex::new(rounds),
        }
    }
}

impl LlmProvider for Scripted {
    fn generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>> {
        Box::pin(async { Err(Error::internal("generate is not scripted")) })
    }

    fn stream_generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        let next = {
            let mut rounds = self.rounds.lock().unwrap();
            if rounds.is_empty() {
                None
            } else {
                Some(rounds.remove(0))
            }
        };
        Box::pin(async move {
            match next {
                Some(Round::Refuse) => Err(upstream()),
                Some(Round::Chunks(chunks)) => {
                    Ok(futures::stream::iter(chunks).boxed() as LlmStream)
                }
                None => Err(Error::internal("script exhausted")),
            }
        })
    }
}

struct FailingTool;

impl AgentFlow for FailingTool {
    fn system_prompt(&self) -> String {
        "system".into()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new("lookup", "look something up")]
    }

    fn execute_tool<'a>(
        &'a self,
        name: &'a str,
        _args: &'a serde_json::Value,
        _session: &'a AgentSession,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move { Err(Error::tool(name, TOOL_DETAIL)) })
    }
}

fn upstream() -> Error {
    Error::provider("vertex-ai", UPSTREAM_DETAIL)
}

fn text(delta: &str) -> Result<StreamChunk, Error> {
    Ok(StreamChunk::Text(delta.into()))
}

fn tool_call() -> Result<StreamChunk, Error> {
    Ok(StreamChunk::ToolCall {
        id: "call-1".into(),
        name: "lookup".into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
    })
}

fn done(usage: Option<Usage>) -> Result<StreamChunk, Error> {
    Ok(StreamChunk::Done {
        finish_reason: "STOP".into(),
        usage,
    })
}

async fn run(provider: Scripted, session: &mut AgentSession) -> Vec<SseEvent> {
    let config = AgentConfig::builder(MODEL)
        .max_tool_rounds(3)
        .build()
        .unwrap();
    let engine = get_agent_engine(provider, config);
    let mut events = Vec::new();
    let mut stream = engine.run(&FailingTool, session, "hello");
    while let Some(event) = stream.next().await {
        events.push(event.expect("engine yields Ok events"));
    }
    events
}

fn error_frame(events: &[SseEvent]) -> (&str, &str) {
    events
        .iter()
        .find_map(|event| match event {
            SseEvent::Error { code, message } => Some((code.as_str(), message.as_str())),
            _ => None,
        })
        .expect("an error frame")
}

fn assert_clean(message: &str) {
    let lower = message.to_lowercase();
    for leak in LEAKS {
        assert!(
            !lower.contains(leak),
            "client message must not mention '{leak}': {message}"
        );
    }
}

#[tokio::test]
async fn a_refused_request_yields_a_fixed_message_then_done() {
    let mut session = AgentSession::new("s1", "commerce");
    let events = run(Scripted::new(vec![Round::Refuse]), &mut session).await;

    assert_eq!(events.len(), 2);
    let (code, message) = error_frame(&events);
    assert_eq!(code, "llm_error");
    assert_clean(message);
    assert!(matches!(events.last(), Some(SseEvent::Done { .. })));
}

#[tokio::test]
async fn a_stream_that_fails_midway_keeps_the_text_and_scrubs_the_error() {
    let mut session = AgentSession::new("s1", "commerce");
    let rounds = vec![Round::Chunks(vec![text("partial"), Err(upstream())])];
    let events = run(Scripted::new(rounds), &mut session).await;

    assert!(matches!(&events[0], SseEvent::Text { delta } if delta == "partial"));
    let (code, message) = error_frame(&events);
    assert_eq!(code, "stream_error");
    assert_clean(message);
    assert!(matches!(events.last(), Some(SseEvent::Done { .. })));
}

#[tokio::test]
async fn a_tool_failure_reaches_the_model_but_not_the_client() {
    let mut session = AgentSession::new("s1", "commerce");
    let rounds = vec![
        Round::Chunks(vec![tool_call(), done(None)]),
        Round::Chunks(vec![text("recovered"), done(None)]),
    ];
    let events = run(Scripted::new(rounds), &mut session).await;

    let (code, message) = error_frame(&events);
    assert_eq!(code, "tool_error");
    assert_clean(message);

    let fed_back = session.messages.iter().any(|message| {
        matches!(message, ChatMessage::ToolResult { content, .. } if content.contains(TOOL_DETAIL))
    });
    assert!(fed_back, "the model must still see why the tool failed");
}

#[tokio::test]
async fn a_metered_turn_bills_the_session_and_the_done_frame_names_only_the_session() {
    let mut session = AgentSession::new("s1", "commerce");
    let usage = Usage {
        input_tokens: Some(100),
        cached_input_tokens: None,
        output_tokens: Some(20),
        reasoning_tokens: Some(30),
        total_tokens: Some(150),
    };
    let rounds = vec![Round::Chunks(vec![text("hi"), done(Some(usage))])];
    let events = run(Scripted::new(rounds), &mut session).await;

    let Some(SseEvent::Done { session_id }) = events.last() else {
        panic!("last event must be Done");
    };
    assert_eq!(session_id.as_str(), "s1");
    assert_eq!(session.usage.by_model[MODEL].total_tokens, 150);
}

#[test]
fn from_result_never_forwards_the_error_text() {
    let SseEvent::Error { code, message } = SseEvent::from_result(Err(upstream())) else {
        panic!("an Err must become an error frame");
    };
    assert_eq!(code, "internal");
    assert_clean(&message);
}
