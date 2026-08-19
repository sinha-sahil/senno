use std::pin::Pin;
use std::sync::Mutex;

use futures::StreamExt;
use senno::agent::{AgentConfig, AgentFlow, AgentSession, SseEvent, ToolOutput, get_agent_engine};
use senno::{
    ContentPart, Error, GenerateRequest, GenerateResponse, LlmProvider, LlmStream, StreamChunk,
    ToolDefinition, Usage,
};

const MODEL: &str = "test-model";

struct ScriptedProvider {
    streams: Mutex<Vec<Vec<StreamChunk>>>,
    summary: Mutex<Option<GenerateResponse>>,
}

impl ScriptedProvider {
    fn new(streams: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            streams: Mutex::new(streams),
            summary: Mutex::new(None),
        }
    }

    fn with_summary(self, usage: Option<Usage>) -> Self {
        *self.summary.lock().unwrap() = Some(GenerateResponse {
            content: vec![ContentPart::Text("prior turns summarized".into())],
            stop_reason: Some("STOP".into()),
            usage,
        });
        self
    }
}

impl LlmProvider for ScriptedProvider {
    fn generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>
    {
        let resp = self.summary.lock().unwrap().clone();
        Box::pin(async move { resp.ok_or_else(|| Error::internal("no summary scripted")) })
    }

    fn stream_generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        let next = {
            let mut streams = self.streams.lock().unwrap();
            if streams.is_empty() {
                None
            } else {
                Some(streams.remove(0))
            }
        };
        Box::pin(async move {
            let chunks = next.ok_or_else(|| Error::internal("stream script exhausted"))?;
            Ok(futures::stream::iter(chunks.into_iter().map(Ok)).boxed() as LlmStream)
        })
    }
}

struct OneToolFlow;

impl AgentFlow for OneToolFlow {
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
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move { Ok(ToolOutput::text(format!("{name} result"))) })
    }
}

fn usage(input: u32, output: u32, reasoning: u32, total: u32) -> Usage {
    Usage {
        input_tokens: Some(input),
        cached_input_tokens: None,
        output_tokens: Some(output),
        reasoning_tokens: Some(reasoning),
        total_tokens: Some(total),
    }
}

fn done(usage: Option<Usage>) -> StreamChunk {
    StreamChunk::Done {
        finish_reason: "STOP".into(),
        usage,
    }
}

fn tool_call() -> StreamChunk {
    StreamChunk::ToolCall {
        id: "call-1".into(),
        name: "lookup".into(),
        arguments: serde_json::json!({}),
        thought_signature: None,
    }
}

fn config(max_history: usize) -> AgentConfig {
    AgentConfig::builder(MODEL)
        .max_tool_rounds(5)
        .max_history_messages(max_history)
        .build()
        .unwrap()
}

async fn drive(
    provider: ScriptedProvider,
    session: &mut AgentSession,
    summarize: bool,
) -> Vec<SseEvent> {
    let engine = if summarize {
        get_agent_engine(provider, config(2)).with_default_summarizer(MODEL)
    } else {
        get_agent_engine(provider, config(80))
    };
    let mut events = Vec::new();
    let mut stream = engine.run(&OneToolFlow, session, "find me something");
    while let Some(event) = stream.next().await {
        events.push(event.expect("engine yields Ok events"));
    }
    events
}

#[tokio::test]
async fn a_two_round_turn_accumulates_every_call_into_the_session() {
    let provider = ScriptedProvider::new(vec![
        vec![tool_call(), done(Some(usage(100, 20, 30, 150)))],
        vec![
            StreamChunk::Text("here you go".into()),
            done(Some(usage(200, 40, 10, 250))),
        ],
    ]);

    let mut session = AgentSession::new("s1", "commerce");
    let events = drive(provider, &mut session, false).await;

    assert_eq!(session.usage.turns, 1);
    assert_eq!(session.usage.totals.calls, 2);
    assert_eq!(session.usage.totals.unmetered_calls, 0);
    assert_eq!(session.usage.totals.input_tokens, 300);
    assert_eq!(session.usage.totals.output_tokens, 60);
    assert_eq!(session.usage.totals.reasoning_tokens, 40);
    assert_eq!(session.usage.totals.total_tokens, 400);
    assert_eq!(session.usage.by_model[MODEL].total_tokens, 400);

    let Some(SseEvent::Done { usage, .. }) = events.last() else {
        panic!("last event must be Done");
    };
    assert_eq!(usage.totals.total_tokens, 400);
    assert_eq!(usage.turns, 1);
}

#[tokio::test]
async fn a_second_turn_adds_to_the_first_rather_than_replacing_it() {
    let mut session = AgentSession::new("s1", "commerce");

    let first = ScriptedProvider::new(vec![vec![
        StreamChunk::Text("one".into()),
        done(Some(usage(10, 5, 0, 15))),
    ]]);
    drive(first, &mut session, false).await;

    let second = ScriptedProvider::new(vec![vec![
        StreamChunk::Text("two".into()),
        done(Some(usage(20, 10, 0, 30))),
    ]]);
    drive(second, &mut session, false).await;

    assert_eq!(session.usage.turns, 2);
    assert_eq!(session.usage.totals.calls, 2);
    assert_eq!(session.usage.totals.total_tokens, 45);
}

#[tokio::test]
async fn a_done_without_usage_is_counted_and_flagged_not_dropped() {
    let provider = ScriptedProvider::new(vec![vec![
        StreamChunk::Text("no numbers".into()),
        done(None),
    ]]);

    let mut session = AgentSession::new("s1", "commerce");
    drive(provider, &mut session, false).await;

    assert_eq!(session.usage.totals.calls, 1);
    assert_eq!(session.usage.totals.unmetered_calls, 1);
    assert_eq!(session.usage.totals.total_tokens, 0);
}

#[tokio::test]
async fn a_stream_that_never_reports_done_still_counts_the_call() {
    let provider = ScriptedProvider::new(vec![vec![StreamChunk::Text("truncated".into())]]);

    let mut session = AgentSession::new("s1", "commerce");
    drive(provider, &mut session, false).await;

    assert_eq!(session.usage.totals.calls, 1);
    assert_eq!(session.usage.totals.unmetered_calls, 1);
}

#[tokio::test]
async fn compaction_tokens_are_billed_to_the_session_that_triggered_them() {
    let provider = ScriptedProvider::new(vec![vec![
        StreamChunk::Text("after compaction".into()),
        done(Some(usage(10, 5, 0, 15))),
    ]])
    .with_summary(Some(usage(500, 60, 0, 560)));

    let mut session = AgentSession::new("s1", "commerce");
    session.messages = vec![
        senno::agent::ChatMessage::User {
            content: "a".into(),
        },
        senno::agent::ChatMessage::Assistant {
            content: "b".into(),
        },
        senno::agent::ChatMessage::User {
            content: "c".into(),
        },
        senno::agent::ChatMessage::Assistant {
            content: "d".into(),
        },
    ];

    drive(provider, &mut session, true).await;

    assert_eq!(session.usage.totals.compactions, 1);
    assert_eq!(session.usage.totals.calls, 2);
    assert_eq!(session.usage.totals.input_tokens, 510);
    assert_eq!(session.usage.totals.total_tokens, 575);
}

#[tokio::test]
async fn usage_survives_the_redis_round_trip_leonardo_does() {
    let provider = ScriptedProvider::new(vec![vec![
        StreamChunk::Text("hello".into()),
        done(Some(usage(100, 20, 30, 150))),
    ]]);

    let mut session = AgentSession::new("s1", "commerce");
    drive(provider, &mut session, false).await;

    let stored = serde_json::to_string(&session).unwrap();
    let loaded: AgentSession = serde_json::from_str(&stored).unwrap();

    assert_eq!(loaded.usage, session.usage);
    assert_eq!(loaded.usage.totals.reasoning_tokens, 30);
}
