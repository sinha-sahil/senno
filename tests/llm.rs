use std::pin::Pin;

use senno::{
    BatchEmbeddingProvider, BatchGenerationProvider, BatchJob, BatchState, ContentPart,
    EmbedRequest, EmbedResponse, Embedding, EmbeddingProvider, Error, GenerateRequest,
    GenerateResponse, LlmProvider, LlmStream, ParameterSchema, Role, ToolDefinition, Usage,
    one_shot,
};

struct StubProvider {
    resp: GenerateResponse,
}

impl LlmProvider for StubProvider {
    fn generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>
    {
        let resp = self.resp.clone();
        Box::pin(async move { Ok(resp) })
    }

    fn stream_generate<'a>(
        &'a self,
        _request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(async move { Err(Error::internal("stub has no stream")) })
    }
}

struct StubEmbedder;

impl EmbeddingProvider for StubEmbedder {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EmbedResponse, Error>> + Send + 'a>> {
        Box::pin(async move {
            Ok(EmbedResponse {
                embeddings: request
                    .texts
                    .iter()
                    .map(|t| {
                        // An empty text stands in for one the provider refused.
                        if t.is_empty() {
                            return Err(Error::provider("stub", "empty text"));
                        }
                        Ok(Embedding {
                            values: vec![t.len() as f32],
                            token_count: Some(t.len() as u32),
                            truncated: None,
                        })
                    })
                    .collect(),
                total_token_count: Some(request.texts.iter().map(|t| t.len() as u32).sum()),
            })
        })
    }
}

struct StubBatcher;

impl BatchGenerationProvider for StubBatcher {
    fn create_batch<'a>(
        &'a self,
        requests: &'a [GenerateRequest],
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a,
        >,
    > {
        let name = format!("batches/{}", requests.len());
        Box::pin(async move {
            Ok(BatchJob {
                name,
                state: BatchState::Pending,
                responses: vec![],
            })
        })
    }

    fn get_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            Ok(BatchJob {
                name: name.into(),
                state: BatchState::Succeeded,
                responses: vec![],
            })
        })
    }

    fn cancel_batch<'a>(
        &'a self,
        _name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move { Ok(()) })
    }
}

fn stub(content: Vec<ContentPart>, usage: Option<Usage>) -> StubProvider {
    StubProvider {
        resp: GenerateResponse {
            content,
            stop_reason: Some("STOP".into()),
            usage,
        },
    }
}

fn usage() -> Usage {
    Usage {
        input_tokens: Some(10),
        output_tokens: Some(5),
        total_tokens: Some(15),
        ..Usage::default()
    }
}

fn tool_call(name: &str, args: serde_json::Value) -> ContentPart {
    ContentPart::ToolCall {
        id: format!("{name}-1"),
        name: name.to_string(),
        arguments: args,
        thought_signature: None,
    }
}

fn report_tool() -> ToolDefinition {
    ToolDefinition::new("report", "reports").with_parameters(
        ParameterSchema::object().with_property("answer", ParameterSchema::string("the answer")),
    )
}

#[derive(Debug, serde::Deserialize)]
struct Out {
    answer: String,
}

#[tokio::test]
async fn one_shot_request_is_a_single_user_turn_at_zero_temperature() {
    let request = GenerateRequest::one_shot("m", "sys", "hi");

    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].role, Role::User);
    assert_eq!(request.system.as_deref(), Some("sys"));
    assert_eq!(request.temperature, Some(0.0));
    assert!(request.tools.is_empty());
    assert!(!request.web_fetch);
}

#[tokio::test]
async fn every_option_is_a_request_builder() {
    let request = GenerateRequest::one_shot("m", "sys", "hi")
        .with_web_fetch(true)
        .with_thinking_budget(0)
        .with_max_tokens(512)
        .with_tools(vec![report_tool()]);

    assert!(request.web_fetch);
    assert_eq!(request.thinking_budget, Some(0));
    assert_eq!(request.max_tokens, Some(512));
    assert_eq!(request.tools.len(), 1);
}

#[tokio::test]
async fn text_joins_parts_and_usage_survives() {
    let provider = stub(
        vec![
            ContentPart::Text("part one ".into()),
            ContentPart::Text("part two".into()),
        ],
        Some(usage()),
    );
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();

    assert_eq!(done.text().unwrap(), "part one part two");
    assert_eq!(done.stop_reason(), Some("STOP"));
    assert_eq!(done.usage.as_ref().unwrap().total_tokens, Some(15));
}

#[tokio::test]
async fn usage_is_readable_even_when_the_model_returned_nothing() {
    let provider = stub(vec![], Some(usage()));
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();

    assert!(done.text().is_none());
    assert_eq!(done.usage.as_ref().unwrap().input_tokens, Some(10));
}

#[tokio::test]
async fn parse_deserializes_the_tool_arguments() {
    let provider = stub(
        vec![tool_call("report", serde_json::json!({"answer": "42"}))],
        Some(usage()),
    );
    let request = GenerateRequest::one_shot("m", "sys", "hi").with_tools(vec![report_tool()]);

    let done = one_shot(&provider, &request).await.unwrap();
    assert_eq!(done.parse::<Out>().unwrap().answer, "42");
}

#[tokio::test]
async fn a_response_carrying_text_and_a_tool_call_exposes_both() {
    let provider = stub(
        vec![
            ContentPart::Text("here you go".into()),
            tool_call("report", serde_json::json!({"answer": "42"})),
        ],
        None,
    );
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();

    assert_eq!(done.text().unwrap(), "here you go");
    assert_eq!(done.parse::<Out>().unwrap().answer, "42");
}

#[tokio::test]
async fn parse_takes_the_first_tool_call_and_the_rest_stay_inspectable() {
    let provider = stub(
        vec![
            tool_call("report", serde_json::json!({"answer": "first"})),
            tool_call("other", serde_json::json!({"answer": "second"})),
        ],
        None,
    );
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();

    assert_eq!(done.parse::<Out>().unwrap().answer, "first");
    assert_eq!(done.tool_calls().len(), 2);
    let names: Vec<&str> = done
        .tool_calls()
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["report", "other"]);
}

#[tokio::test]
async fn embedding_provider_is_object_safe_and_keeps_input_order() {
    let provider: std::sync::Arc<dyn EmbeddingProvider> = std::sync::Arc::new(StubEmbedder);
    let resp = provider
        .embed(&EmbedRequest::new("m", ["a", "bbb"]))
        .await
        .unwrap();

    assert!(resp.is_complete());
    assert_eq!(resp.total_token_count, Some(4));

    let embeddings = resp.into_embeddings().unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].values, vec![1.0]);
    assert_eq!(embeddings[1].values, vec![3.0]);
}

#[tokio::test]
async fn a_partial_embed_failure_keeps_the_texts_that_succeeded() {
    let provider: std::sync::Arc<dyn EmbeddingProvider> = std::sync::Arc::new(StubEmbedder);
    let resp = provider
        .embed(&EmbedRequest::new("m", ["a", "", "ccc"]))
        .await
        .unwrap();

    assert!(!resp.is_complete());
    assert_eq!(resp.embeddings.len(), 3);

    let failures: Vec<usize> = resp.failures().map(|(index, _)| index).collect();
    assert_eq!(failures, vec![1]);

    assert_eq!(resp.embeddings[0].as_ref().unwrap().values, vec![1.0]);
    assert_eq!(resp.embeddings[2].as_ref().unwrap().values, vec![3.0]);

    assert!(resp.into_embeddings().is_err());
}

impl BatchEmbeddingProvider for StubBatcher {
    fn create_embedding_batch<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>>
    {
        let name = format!("batches/embed-{}", request.texts.len());
        Box::pin(async move {
            Ok(BatchJob {
                name,
                state: BatchState::Pending,
                responses: vec![],
            })
        })
    }

    fn get_embedding_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>>
    {
        Box::pin(async move {
            Ok(BatchJob {
                name: name.into(),
                state: BatchState::Succeeded,
                responses: vec![Ok(Embedding {
                    values: vec![0.5],
                    token_count: Some(3),
                    truncated: None,
                })],
            })
        })
    }

    fn cancel_embedding_batch<'a>(
        &'a self,
        _name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move { Ok(()) })
    }
}

#[tokio::test]
async fn batch_generation_provider_is_object_safe_across_its_lifecycle() {
    let provider: std::sync::Arc<dyn BatchGenerationProvider> = std::sync::Arc::new(StubBatcher);

    let job = provider
        .create_batch(&[GenerateRequest::one_shot("m", "sys", "hi")])
        .await
        .unwrap();
    assert_eq!(job.name, "batches/1");
    assert!(!job.state.is_terminal());

    let job = provider.get_batch(&job.name).await.unwrap();
    assert_eq!(job.state, BatchState::Succeeded);
    assert!(job.state.is_terminal());

    provider.cancel_batch(&job.name).await.unwrap();
}

#[tokio::test]
async fn batch_embedding_provider_shares_the_job_shape_with_generation() {
    let provider: std::sync::Arc<dyn BatchEmbeddingProvider> = std::sync::Arc::new(StubBatcher);

    let job = provider
        .create_embedding_batch(&EmbedRequest::new("gemini-embedding-001", ["a", "b"]))
        .await
        .unwrap();
    assert_eq!(job.name, "batches/embed-2");
    assert!(!job.state.is_terminal());

    let job = provider.get_embedding_batch(&job.name).await.unwrap();
    assert_eq!(job.state, BatchState::Succeeded);
    assert_eq!(job.responses[0].as_ref().unwrap().values, vec![0.5]);

    provider.cancel_embedding_batch(&job.name).await.unwrap();
}

#[tokio::test]
async fn parse_errors_without_a_tool_call_and_on_mismatched_arguments() {
    let provider = stub(vec![ContentPart::Text("prose instead".into())], None);
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();
    assert!(
        done.parse::<serde_json::Value>()
            .unwrap_err()
            .is_bad_response()
    );

    #[derive(Debug, serde::Deserialize)]
    struct Typed {
        #[allow(dead_code)]
        answer: u32,
    }
    let provider = stub(
        vec![tool_call(
            "report",
            serde_json::json!({"answer": "not a number"}),
        )],
        None,
    );
    let done = one_shot(&provider, &GenerateRequest::one_shot("m", "sys", "hi"))
        .await
        .unwrap();
    assert!(done.parse::<Typed>().unwrap_err().is_bad_response());
}
