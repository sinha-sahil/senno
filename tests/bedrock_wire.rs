//! End-to-end tests for the Bedrock wire paths (Converse, InvokeModel),
//! against a stub HTTP server. ConverseStream's binary event framing is
//! covered by the provider's unit tests on the decoded events.

#![cfg(feature = "bedrock")]

mod common;

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_bedrockruntime::config::{Credentials, SharedCredentialsProvider};
use common::{Reply, StubServer};
use senno::providers::bedrock::{
    BedrockClient, BedrockConfig, BedrockGuardrail, get_bedrock_client,
};
use senno::{
    EmbedImage, EmbedRequest, EmbeddingProvider, GenerateRequest, LlmProvider, Message,
    ParameterSchema, ToolDefinition,
};
use serde_json::json;

fn sdk_config() -> SdkConfig {
    SdkConfig::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("eu-central-1"))
        .credentials_provider(SharedCredentialsProvider::new(Credentials::new(
            "AKIDTEST", "secret", None, None, "test",
        )))
        .build()
}

async fn client(server: &StubServer, config: BedrockConfig) -> BedrockClient {
    get_bedrock_client(Some(
        config
            .with_sdk_config(sdk_config())
            .with_endpoint_url(&server.base_url)
            .with_max_attempts(1),
    ))
    .await
    .expect("bedrock client")
}

#[tokio::test]
async fn converse_sends_history_tools_and_guardrail_and_parses_tool_call() {
    let server = StubServer::start(|_| {
        Reply::ok(
            json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"text": "Searching."},
                    {"toolUse": {"toolUseId": "tu-1", "name": "search_catalog",
                                 "input": {"catalog": {"query": "red shoes"}}}}
                ]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 900, "outputTokens": 40, "totalTokens": 1040,
                          "cacheReadInputTokens": 100},
                "metrics": {"latencyMs": 12}
            })
            .to_string(),
        )
    })
    .await;
    let client = client(
        &server,
        BedrockConfig::default().with_guardrail(BedrockGuardrail::new("gr-123", "2")),
    )
    .await;

    let tool = ToolDefinition::new("search_catalog", "Search").with_parameters(
        ParameterSchema::object().with_property("query", ParameterSchema::string("q")),
    );
    let request = GenerateRequest::new(
        "anthropic.claude-sonnet-5",
        vec![
            Message::user("red shoes please"),
            Message::assistant("Sure."),
            Message::assistant("One moment."),
        ],
    )
    .with_system("You are a shop assistant.")
    .with_max_tokens(256)
    .with_tools(vec![tool]);

    let response = client.generate(&request).await.expect("converse");

    // What went out.
    let sent = &server.requests()[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/model/anthropic.claude-sonnet-5/converse");
    assert!(
        sent.header("authorization")
            .is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256")),
        "request must be SigV4-signed"
    );
    let body = sent.json();
    assert_eq!(
        body["system"],
        json!([{"text": "You are a shop assistant."}])
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 256);
    assert_eq!(body["guardrailConfig"]["guardrailIdentifier"], "gr-123");
    assert_eq!(body["guardrailConfig"]["guardrailVersion"], "2");
    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "search_catalog"
    );
    // Two assistant messages in a row are merged into one.
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);

    // What came back.
    assert_eq!(response.text().as_deref(), Some("Searching."));
    assert_eq!(response.stop_reason(), Some("tool_use"));
    let call: serde_json::Value = response.parse().unwrap();
    assert_eq!(call, json!({"catalog": {"query": "red shoes"}}));
    let usage = response.usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(1000));
    assert_eq!(usage.cached_input_tokens, Some(100));
    assert_eq!(usage.output_tokens, Some(40));
}

#[tokio::test]
async fn throttling_maps_to_a_retryable_error() {
    let server = StubServer::start(|_| {
        Reply::error(429, r#"{"message":"Too many tokens, please wait"}"#)
            .with_header("x-amzn-ErrorType", "ThrottlingException")
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let err = client
        .generate(&GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap_err();
    assert!(err.is_retryable(), "{err}");
    assert!(err.to_string().contains("ThrottlingException"), "{err}");
}

#[tokio::test]
async fn access_denied_is_permanent() {
    let server = StubServer::start(|_| {
        Reply::error(403, r#"{"message":"not authorized to invoke"}"#)
            .with_header("x-amzn-ErrorType", "AccessDeniedException")
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let err = client
        .generate(&GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap_err();
    assert!(err.is_provider_error());
    assert!(!err.is_retryable(), "{err}");
}

#[tokio::test]
async fn titan_multimodal_embeds_text_and_image_per_item() {
    let server = StubServer::start(|request| {
        let body = request.json();
        // Echo back which inputs arrived, as the vector.
        let has_text = body.get("inputText").is_some() as u8 as f32;
        let has_image = body.get("inputImage").is_some() as u8 as f32;
        Reply::ok(json!({"embedding": [has_text, has_image], "inputTextTokenCount": 3}).to_string())
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let request = EmbedRequest::new("amazon.titan-embed-image-v1", ["red shoe", "blue bag"])
        .with_images([EmbedImage::new(vec![1, 2, 3]), EmbedImage::new(vec![4])])
        .with_output_dimensionality(1024);
    let response = client.embed(&request).await.expect("embed");

    let vectors: Vec<Vec<f32>> = response
        .into_embeddings()
        .unwrap()
        .into_iter()
        .map(|e| e.values)
        .collect();
    assert_eq!(vectors, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);

    let sent = server.requests();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].path, "/model/amazon.titan-embed-image-v1/invoke");
    let first = sent
        .iter()
        .map(|r| r.json())
        .find(|b| b["inputText"] == "red shoe")
        .expect("first item sent");
    assert_eq!(first["inputImage"], "AQID");
    assert_eq!(first["embeddingConfig"]["outputEmbeddingLength"], 1024);
}

#[tokio::test]
async fn one_failed_embedding_keeps_the_others() {
    let server = StubServer::start(|request| {
        if request.json()["inputText"] == "bad" {
            Reply::error(400, r#"{"message":"input too long"}"#)
                .with_header("x-amzn-ErrorType", "ValidationException")
        } else {
            Reply::ok(json!({"embedding": [0.5], "inputTextTokenCount": 1}).to_string())
        }
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let response = client
        .embed(&EmbedRequest::new(
            "amazon.titan-embed-text-v2:0",
            ["ok", "bad", "ok"],
        ))
        .await
        .expect("embed");
    assert_eq!(
        response.failures().map(|(i, _)| i).collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(response.total_token_count, Some(2));
}

#[tokio::test]
async fn missing_region_is_a_config_error() {
    let no_region = SdkConfig::builder()
        .behavior_version(BehaviorVersion::latest())
        .build();
    // SAFETY: test-only; no other test in this binary reads BEDROCK_REGION.
    unsafe { std::env::remove_var("BEDROCK_REGION") };
    let err = get_bedrock_client(Some(BedrockConfig::default().with_sdk_config(no_region)))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("region"), "{err}");
}
