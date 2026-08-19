//! End-to-end tests for the Vertex wire paths, against a stub HTTP server.

#![cfg(feature = "vertex")]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{CapturedRequest, Reply, StubServer, TOKEN_BODY};
use senno::providers::vertex::{VertexClient, VertexConfig, VertexProvider, get_vertex_client};
use senno::{
    BatchEmbeddingProvider, BatchGenerationProvider, BatchState, EmbedRequest, EmbedTaskType,
    EmbeddingProvider, Error, GenerateRequest,
};

const MAX_TEXTS_PER_CALL: usize = 100;
const MAX_CHARS_PER_CALL: usize = 80_000;

async fn api_key_client(base_url: &str) -> VertexClient {
    let config = VertexConfig::default()
        .with_api_key("test-key")
        .with_base_url(base_url);
    get_vertex_client(VertexProvider::Gemini, Some(config))
        .await
        .expect("api-key client")
}

async fn service_account_client(base_url: &str) -> VertexClient {
    let config = VertexConfig::default()
        .with_project_id("proj")
        .with_region("asia-south1")
        // Metadata auth, so the same stub can mint the token.
        .with_metadata_server_url(base_url)
        .with_base_url(base_url);
    get_vertex_client(VertexProvider::Gemini, Some(config))
        .await
        .expect("service-account client")
}

fn predict_body(value: f32, token_count: u32) -> String {
    format!(
        r#"{{"predictions":[{{"embeddings":{{"values":[{value:?}],"statistics":{{"token_count":{token_count}.0,"truncated":false}}}}}}]}}"#
    )
}

fn instance_text(request: &CapturedRequest) -> String {
    request.json()["instances"][0]["content"]
        .as_str()
        .expect("instance content")
        .to_string()
}

fn batch_texts(request: &CapturedRequest) -> Vec<String> {
    request.json()["requests"]
        .as_array()
        .expect("requests array")
        .iter()
        .map(|item| {
            item["content"]["parts"][0]["text"]
                .as_str()
                .expect("text part")
                .to_string()
        })
        .collect()
}

/// Echoes each text back as its own vector. Texts must be numeric.
fn echo_batch(request: &CapturedRequest) -> Reply {
    let items = batch_texts(request);
    let embeddings: Vec<String> = items
        .iter()
        .map(|text| format!(r#"{{"values":[{text}]}}"#))
        .collect();
    Reply::ok(format!(
        r#"{{"embeddings":[{}],"usageMetadata":{{"promptTokenCount":{}}}}}"#,
        embeddings.join(","),
        items.len()
    ))
}

fn numbered(count: usize) -> Vec<String> {
    (0..count).map(|i| i.to_string()).collect()
}

fn assert_no_index_drift(embeddings: &[Result<senno::Embedding, Error>]) {
    for (index, result) in embeddings.iter().enumerate() {
        if let Ok(embedding) = result {
            assert_eq!(
                embedding.values,
                vec![index as f32],
                "index {index} drifted"
            );
        }
    }
}

#[tokio::test]
async fn an_empty_request_never_reaches_the_network() {
    let stub = StubServer::start(|_| Reply::ok("{}")).await;
    let client = api_key_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::new("m", Vec::<String>::new()))
        .await
        .unwrap();

    assert!(resp.embeddings.is_empty());
    assert_eq!(resp.total_token_count, None);
    assert_eq!(stub.request_count(), 0);
}

#[tokio::test]
async fn a_bad_title_pairing_is_caught_before_the_network() {
    let stub = StubServer::start(|_| Reply::ok("{}")).await;
    let client = api_key_client(&stub.base_url).await;

    let request = EmbedRequest::new("m", ["a"])
        .with_title("Doc")
        .with_task_type(EmbedTaskType::SemanticSimilarity);
    let err = client.embed(&request).await.unwrap_err();

    assert!(matches!(err, Error::Config(_)), "got {err:?}");
    assert_eq!(stub.request_count(), 0);
}

#[tokio::test]
async fn predict_sends_the_documented_wire_shape_with_a_bearer_token() {
    let stub = StubServer::start(|req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        Reply::ok(predict_body(1.0, 4))
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let request = EmbedRequest::single("gemini-embedding-001", "hello")
        .with_task_type(EmbedTaskType::RetrievalQuery)
        .with_output_dimensionality(768);
    client.embed(&request).await.unwrap();

    let calls = stub.api_requests();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];

    assert_eq!(call.header("authorization"), Some("Bearer stub-token"));
    assert!(
        call.path.ends_with(
            "/v1/projects/proj/locations/asia-south1/publishers/google/models/gemini-embedding-001:predict"
        ),
        "unexpected path: {}",
        call.path
    );

    let body = call.json();
    assert_eq!(body["instances"][0]["content"], "hello");
    assert_eq!(body["instances"][0]["task_type"], "RETRIEVAL_QUERY");
    assert_eq!(body["parameters"]["outputDimensionality"], 768);
}

#[tokio::test]
async fn predict_fans_out_per_text_and_keeps_input_order() {
    let stub = StubServer::start(|req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        // The first text answers last: completion order is the reverse.
        let (value, delay) = match instance_text(req).as_str() {
            "first" => (1.0, 180),
            "second" => (2.0, 90),
            _ => (3.0, 0),
        };
        Reply::ok(predict_body(value, 5)).after(Duration::from_millis(delay))
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let request = EmbedRequest::new("m", ["first", "second", "third"]);
    let resp = client.embed(&request).await.unwrap();

    assert_eq!(stub.api_requests().len(), 3, "one call per text");
    assert_eq!(resp.total_token_count, Some(15));

    let values: Vec<f32> = resp
        .into_embeddings()
        .unwrap()
        .into_iter()
        .map(|e| e.values[0])
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[tokio::test]
async fn predict_keeps_the_texts_that_succeeded_when_one_is_rejected() {
    let stub = StubServer::start(|req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        if instance_text(req) == "bad" {
            return Reply::error(400, r#"{"error":{"message":"invalid argument"}}"#);
        }
        Reply::ok(predict_body(7.0, 3))
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let request = EmbedRequest::new("m", ["good", "bad", "also good"]);
    let resp = client.embed(&request).await.unwrap();

    assert!(!resp.is_complete());
    assert_eq!(resp.failures().map(|(i, _)| i).collect::<Vec<_>>(), vec![1]);
    assert_eq!(resp.embeddings[0].as_ref().unwrap().values, vec![7.0]);
    assert_eq!(resp.embeddings[2].as_ref().unwrap().values, vec![7.0]);
    assert_eq!(resp.total_token_count, Some(6), "only what came back");
    assert_eq!(stub.api_requests().len(), 3, "a 400 is not retried");
}

#[tokio::test]
async fn a_failed_text_does_not_stop_the_ones_still_queued_behind_it() {
    let stub = StubServer::start(|req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        let text = instance_text(req);
        // Text 1 fails immediately while 0, 2 and 3 are still in flight and
        // 4..19 have not been dispatched yet — only 4 run at a time.
        if text == "1" {
            return Reply::error(400, r#"{"error":{"message":"invalid argument"}}"#);
        }
        Reply::ok(predict_body(text.parse::<f32>().unwrap(), 1)).after(Duration::from_millis(50))
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::new("m", numbered(20)))
        .await
        .unwrap();

    assert_eq!(
        stub.api_requests().len(),
        20,
        "every text must still be attempted"
    );
    assert_eq!(resp.embeddings.len(), 20);
    assert_eq!(resp.failures().map(|(i, _)| i).collect::<Vec<_>>(), vec![1]);
    assert_eq!(resp.embeddings.iter().filter(|r| r.is_ok()).count(), 19);
    assert_no_index_drift(&resp.embeddings);
}

#[tokio::test]
async fn a_rate_limit_is_retried_and_then_succeeds() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = seen.clone();
    let stub = StubServer::start(move |req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
            return Reply::error(429, r#"{"error":{"message":"quota exceeded"}}"#);
        }
        Reply::ok(predict_body(4.0, 2))
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::single("m", "hello"))
        .await
        .unwrap();

    assert_eq!(resp.into_embeddings().unwrap()[0].values, vec![4.0]);
    assert_eq!(stub.api_requests().len(), 2, "the 429, then the retry");
}

#[tokio::test]
async fn retries_stop_at_the_attempt_budget() {
    let stub = StubServer::start(|req| {
        if req.is_token_fetch() {
            return Reply::ok(TOKEN_BODY);
        }
        Reply::error(503, r#"{"error":{"message":"unavailable"}}"#)
    })
    .await;
    let client = service_account_client(&stub.base_url).await;

    let err = client
        .embed(&EmbedRequest::single("m", "hello"))
        .await
        .unwrap_err();

    assert!(err.is_retryable(), "got {err:?}");
    assert_eq!(stub.api_requests().len(), 3, "bounded, not unbounded");
}

#[tokio::test]
async fn batch_sends_the_documented_wire_shape_with_the_api_key() {
    let stub = StubServer::start(echo_batch).await;
    let client = api_key_client(&stub.base_url).await;

    let request = EmbedRequest::new("gemini-embedding-001", ["1", "2"])
        .with_task_type(EmbedTaskType::RetrievalDocument)
        .with_title("Doc");
    client.embed(&request).await.unwrap();

    let calls = stub.api_requests();
    assert_eq!(calls.len(), 1);
    let call = &calls[0];

    assert_eq!(call.header("x-goog-api-key"), Some("test-key"));
    assert!(
        call.path
            .ends_with("/v1beta/models/gemini-embedding-001:batchEmbedContents"),
        "unexpected path: {}",
        call.path
    );

    let body = call.json();
    assert_eq!(body["requests"][0]["model"], "models/gemini-embedding-001");
    assert_eq!(body["requests"][0]["content"]["parts"][0]["text"], "1");
    assert_eq!(body["requests"][0]["taskType"], "RETRIEVAL_DOCUMENT");
    assert_eq!(body["requests"][0]["title"], "Doc");
}

#[tokio::test]
async fn batch_splits_past_the_call_limit_and_reassembles_in_input_order() {
    let stub = StubServer::start(echo_batch).await;
    let client = api_key_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::new("m", numbered(250)))
        .await
        .unwrap();

    let calls = stub.api_requests();
    assert_eq!(calls.len(), 3, "250 texts at 100 per call");
    let sizes: Vec<usize> = calls.iter().map(|c| batch_texts(c).len()).collect();
    assert_eq!(sizes.iter().sum::<usize>(), 250);
    assert!(
        sizes.iter().all(|s| *s <= MAX_TEXTS_PER_CALL),
        "no call may exceed the limit: {sizes:?}"
    );

    let values: Vec<f32> = resp
        .into_embeddings()
        .unwrap()
        .into_iter()
        .map(|e| e.values[0])
        .collect();
    assert_eq!(values, (0..250).map(|i| i as f32).collect::<Vec<f32>>());
}

#[tokio::test]
async fn batch_splits_on_the_character_budget_too() {
    let stub = StubServer::start(|req| {
        let count = batch_texts(req).len();
        let embeddings = vec![r#"{"values":[1.0]}"#; count].join(",");
        Reply::ok(format!(r#"{{"embeddings":[{embeddings}]}}"#))
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    // Well under the 100-text limit, well over the character budget.
    let texts = vec!["a".repeat(MAX_CHARS_PER_CALL / 2 + 1); 4];
    let resp = client.embed(&EmbedRequest::new("m", texts)).await.unwrap();

    assert_eq!(
        stub.api_requests().len(),
        4,
        "the count limit alone would send 1"
    );
    assert!(resp.is_complete());
    assert_eq!(resp.embeddings.len(), 4);
}

#[tokio::test]
async fn a_failed_batch_call_only_fails_the_texts_it_carried() {
    let stub = StubServer::start(|req| {
        // Refuse whichever call carries text "120" — the middle chunk.
        if batch_texts(req).iter().any(|t| t == "120") {
            return Reply::error(400, r#"{"error":{"message":"invalid argument"}}"#);
        }
        echo_batch(req)
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::new("m", numbered(250)))
        .await
        .unwrap();

    assert!(!resp.is_complete());
    let failed: Vec<usize> = resp.failures().map(|(index, _)| index).collect();
    assert_eq!(failed, (100..200).collect::<Vec<usize>>());
    assert_eq!(
        resp.total_token_count,
        Some(150),
        "billed for what returned"
    );
    assert_no_index_drift(&resp.embeddings);
}

#[tokio::test]
async fn ten_thousand_texts_failing_near_the_end_keep_everything_before_it() {
    let stub = StubServer::start(|req| {
        if batch_texts(req).iter().any(|t| t == "9900") {
            return Reply::error(400, r#"{"error":{"message":"invalid argument"}}"#);
        }
        echo_batch(req)
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    let resp = client
        .embed(&EmbedRequest::new("m", numbered(10_000)))
        .await
        .unwrap();

    assert_eq!(resp.embeddings.len(), 10_000);
    assert_eq!(resp.embeddings.iter().filter(|r| r.is_ok()).count(), 9_900);
    assert_eq!(
        resp.failures().map(|(index, _)| index).collect::<Vec<_>>(),
        (9_900..10_000).collect::<Vec<usize>>()
    );
    assert_no_index_drift(&resp.embeddings);
}

#[tokio::test]
async fn batch_rejects_a_response_that_does_not_cover_every_text() {
    let stub = StubServer::start(|_| Reply::ok(r#"{"embeddings":[{"values":[1.0]}]}"#)).await;
    let client = api_key_client(&stub.base_url).await;

    let err = client
        .embed(&EmbedRequest::new("m", ["a", "b"]))
        .await
        .unwrap_err();

    assert!(err.is_bad_response(), "got {err:?}");
    assert!(
        err.to_string().contains("expected 2 embeddings, got 1"),
        "got {err}"
    );
}

#[tokio::test]
async fn a_call_where_nothing_succeeded_surfaces_as_an_error() {
    let stub = StubServer::start(|_| Reply::error(403, r#"{"error":{"message":"denied"}}"#)).await;
    let client = api_key_client(&stub.base_url).await;

    let err = client
        .embed(&EmbedRequest::new("m", ["a", "b"]))
        .await
        .unwrap_err();

    assert!(!err.is_retryable(), "403 is permanent: {err:?}");
    assert!(err.to_string().contains("403"), "got {err}");
}

#[tokio::test]
async fn creating_an_embedding_batch_posts_the_async_method() {
    let stub = StubServer::start(|_| {
        Reply::ok(r#"{"name":"batches/xyz","metadata":{"state":"JOB_STATE_PENDING"}}"#)
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    let request = EmbedRequest::new("gemini-embedding-001", ["alpha", "beta"])
        .with_task_type(EmbedTaskType::RetrievalDocument);
    let job = client.create_embedding_batch(&request).await.unwrap();

    assert_eq!(job.name, "batches/xyz");
    assert_eq!(job.state, BatchState::Pending);

    let calls = stub.api_requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].header("x-goog-api-key"), Some("test-key"));
    assert!(
        calls[0]
            .path
            .ends_with("/v1beta/models/gemini-embedding-001:asyncBatchEmbedContent"),
        "unexpected path: {}",
        calls[0].path
    );
    assert_eq!(
        calls[0].json()["batch"]["inputConfig"]["requests"]["requests"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn polling_a_finished_embedding_batch_returns_the_vectors() {
    let stub = StubServer::start(|_| {
        Reply::ok(
            r#"{"name":"batches/xyz","metadata":{"state":"JOB_STATE_SUCCEEDED"},
                "response":{"inlinedEmbedContentResponses":{"inlinedResponses":[
                    {"response":{"embedding":{"values":[0.1]},"tokenCount":4}},
                    {"error":{"code":3,"message":"bad item"}}
                ]}}}"#,
        )
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    let job = client.get_embedding_batch("batches/xyz").await.unwrap();

    assert!(job.state.is_terminal());
    assert_eq!(job.responses.len(), 2);
    assert_eq!(job.responses[0].as_ref().unwrap().values, vec![0.1]);
    assert_eq!(job.responses[0].as_ref().unwrap().token_count, Some(4));
    assert!(job.responses[1].is_err());
    assert!(stub.api_requests()[0].path.ends_with("/v1beta/batches/xyz"));
}

#[tokio::test]
async fn generation_and_embedding_jobs_cancel_through_the_same_endpoint() {
    let stub = StubServer::start(|_| Reply::ok("{}")).await;
    let client = api_key_client(&stub.base_url).await;

    client.cancel_embedding_batch("batches/xyz").await.unwrap();

    let calls = stub.api_requests();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].path.ends_with("/v1beta/batches/xyz:cancel"));
}

#[tokio::test]
async fn generation_batches_still_post_to_batch_generate_content() {
    let stub = StubServer::start(|_| {
        Reply::ok(r#"{"name":"batches/gen","metadata":{"state":"BATCH_STATE_RUNNING"}}"#)
    })
    .await;
    let client = api_key_client(&stub.base_url).await;

    let job = client
        .create_batch(&[GenerateRequest::one_shot("gemini-2.5-flash", "sys", "hi")])
        .await
        .unwrap();

    assert_eq!(job.state, BatchState::Running);
    assert!(
        stub.api_requests()[0]
            .path
            .ends_with("/v1beta/models/gemini-2.5-flash:batchGenerateContent")
    );
}
