use crate::error::Error;
use crate::types::{EmbedRequest, EmbedResponse, EmbedTaskType, Embedding};

use super::client::VertexClient;
use super::config::ResolvedAuth;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

const MAX_TEXTS_PER_BATCH_CALL: usize = 100;
/// Both APIs also cap tokens per call. No local tokenizer, so this is a
/// conservative ~4-chars-per-token stand-in for that budget.
const MAX_CHARS_PER_BATCH_CALL: usize = 80_000;
const REQUEST_CONCURRENCY: usize = 4;

pub(crate) async fn embed(
    client: &VertexClient,
    request: &EmbedRequest,
) -> Result<EmbedResponse, Error> {
    validate(request)?;
    if request.texts.is_empty() {
        return Ok(EmbedResponse {
            embeddings: Vec::new(),
            total_token_count: None,
        });
    }

    let response = match &client.auth {
        ResolvedAuth::ApiKey { .. } => embed_batch(client, request).await,
        ResolvedAuth::ServiceAccount {
            project_id, region, ..
        } => {
            let url = predict_endpoint(
                &client.vertex_base(region),
                project_id,
                region,
                &request.model,
            );
            embed_predict(client, request, &url).await
        }
    };
    or_total_failure(response)
}

fn or_total_failure(response: EmbedResponse) -> Result<EmbedResponse, Error> {
    if response.embeddings.iter().any(Result::is_ok) {
        return Ok(response);
    }
    match response.embeddings.iter().find_map(|r| r.as_ref().err()) {
        Some(error) => Err(error.clone()),
        None => Ok(response),
    }
}

pub(super) fn validate(request: &EmbedRequest) -> Result<(), Error> {
    if request.title.is_some() && request.task_type != Some(EmbedTaskType::RetrievalDocument) {
        return Err(Error::config(
            "EmbedRequest::with_title is only accepted with EmbedTaskType::RetrievalDocument",
        ));
    }
    Ok(())
}

pub(super) fn model_id(model: &str) -> &str {
    model.strip_prefix("models/").unwrap_or(model)
}

fn chunks(texts: &[String]) -> Vec<(usize, &[String])> {
    let mut calls = Vec::new();
    let mut start = 0;
    let mut chars = 0;

    for (index, text) in texts.iter().enumerate() {
        let len = text.chars().count();
        let full =
            index - start >= MAX_TEXTS_PER_BATCH_CALL || chars + len > MAX_CHARS_PER_BATCH_CALL;
        if index > start && full {
            calls.push((start, &texts[start..index]));
            start = index;
            chars = 0;
        }
        chars += len;
    }
    if start < texts.len() {
        calls.push((start, &texts[start..]));
    }
    calls
}

fn add_tokens(total: Option<u32>, count: Option<u32>) -> Option<u32> {
    match (total, count) {
        (None, None) => None,
        (total, count) => Some(total.unwrap_or(0).saturating_add(count.unwrap_or(0))),
    }
}

#[derive(Serialize)]
struct BatchRequest {
    requests: Vec<BatchItem>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchItem {
    model: String,
    content: TextContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

#[derive(Serialize)]
pub(super) struct TextContent {
    parts: Vec<TextPart>,
}

impl TextContent {
    pub(super) fn new(text: &str) -> Self {
        Self {
            parts: vec![TextPart { text: text.into() }],
        }
    }
}

#[derive(Serialize)]
struct TextPart {
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchResponse {
    embeddings: Vec<BatchEmbedding>,
    usage_metadata: Option<BatchUsage>,
}

#[derive(Deserialize)]
struct BatchEmbedding {
    values: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchUsage {
    prompt_token_count: Option<u32>,
}

async fn embed_batch(client: &VertexClient, request: &EmbedRequest) -> EmbedResponse {
    let url = batch_endpoint(client.gemini_api_base(), &request.model);
    // Built by loop rather than `map`: closures returning borrowing futures
    // trip rustc's higher-ranked lifetime inference here.
    let mut calls = Vec::new();
    for (start, chunk) in chunks(&request.texts) {
        calls.push(embed_one_batch(client, &url, request, start, chunk));
    }

    let results: Vec<_> = futures::stream::iter(calls)
        .buffered(REQUEST_CONCURRENCY)
        .collect()
        .await;

    let mut embeddings = Vec::with_capacity(request.texts.len());
    let mut total_token_count = None;
    for (chunk_embeddings, tokens) in results {
        embeddings.extend(chunk_embeddings);
        total_token_count = add_tokens(total_token_count, tokens);
    }

    EmbedResponse {
        embeddings,
        total_token_count,
    }
}

async fn embed_one_batch(
    client: &VertexClient,
    url: &str,
    request: &EmbedRequest,
    start: usize,
    chunk: &[String],
) -> (Vec<Result<Embedding, Error>>, Option<u32>) {
    match try_embed_batch(client, url, request, start, chunk).await {
        Ok((embeddings, tokens)) => (embeddings.into_iter().map(Ok).collect(), tokens),
        Err(e) => (failed_chunk(chunk.len(), &e), None),
    }
}

fn failed_chunk(len: usize, error: &Error) -> Vec<Result<Embedding, Error>> {
    (0..len).map(|_| Err(error.clone())).collect()
}

async fn try_embed_batch(
    client: &VertexClient,
    url: &str,
    request: &EmbedRequest,
    start: usize,
    chunk: &[String],
) -> Result<(Vec<Embedding>, Option<u32>), Error> {
    let wire = BatchRequest {
        requests: chunk.iter().map(|t| batch_item(request, t)).collect(),
    };

    let resp = client
        .send_with_retry(|| client.http.post(url).json(&wire), start)
        .await?;
    let wire: BatchResponse = resp
        .json()
        .await
        .map_err(|e| Error::provider("vertex-ai", format!("Parse failed: {e}")))?;

    if wire.embeddings.len() != chunk.len() {
        return Err(Error::bad_response(format!(
            "expected {} embeddings, got {}",
            chunk.len(),
            wire.embeddings.len()
        )));
    }

    let embeddings = wire
        .embeddings
        .into_iter()
        .map(|e| Embedding {
            values: e.values,
            // batchEmbedContents reports neither per-text tokens nor truncation.
            token_count: None,
            truncated: None,
        })
        .collect();

    Ok((
        embeddings,
        wire.usage_metadata.and_then(|u| u.prompt_token_count),
    ))
}

fn batch_endpoint(base: &str, model: &str) -> String {
    let model = model_id(model);
    format!("{base}/v1beta/models/{model}:batchEmbedContents")
}

fn batch_item(request: &EmbedRequest, text: &str) -> BatchItem {
    BatchItem {
        model: format!("models/{}", model_id(&request.model)),
        content: TextContent::new(text),
        task_type: request.task_type.map(EmbedTaskType::as_str),
        title: request.title.clone(),
        output_dimensionality: request.output_dimensionality,
    }
}

#[derive(Serialize)]
struct PredictRequest {
    instances: Vec<Instance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<PredictParams>,
}

#[derive(Serialize)]
struct Instance {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PredictParams {
    output_dimensionality: u32,
}

#[derive(Deserialize)]
struct PredictResponse {
    predictions: Vec<Prediction>,
}

#[derive(Deserialize)]
struct Prediction {
    embeddings: PredictionEmbeddings,
}

#[derive(Deserialize)]
struct PredictionEmbeddings {
    values: Vec<f32>,
    statistics: Option<Statistics>,
}

#[derive(Deserialize)]
struct Statistics {
    token_count: Option<f64>,
    truncated: Option<bool>,
}

impl From<Prediction> for Embedding {
    fn from(p: Prediction) -> Self {
        let stats = p.embeddings.statistics;
        Embedding {
            values: p.embeddings.values,
            token_count: stats.as_ref().and_then(|s| s.token_count).map(|t| t as u32),
            truncated: stats.as_ref().and_then(|s| s.truncated),
        }
    }
}

/// `gemini-embedding-001` accepts exactly one text per `:predict` call, so bulk
/// work is a bounded fan-out.
async fn embed_predict(client: &VertexClient, request: &EmbedRequest, url: &str) -> EmbedResponse {
    let mut calls = Vec::new();
    for (index, text) in request.texts.iter().enumerate() {
        calls.push(embed_single_instance(client, url, request, index, text));
    }

    let embeddings: Vec<Result<Embedding, Error>> = futures::stream::iter(calls)
        .buffered(REQUEST_CONCURRENCY)
        .collect()
        .await;

    let total_token_count = embeddings
        .iter()
        .flatten()
        .fold(None, |total, e| add_tokens(total, e.token_count));

    EmbedResponse {
        embeddings,
        total_token_count,
    }
}

async fn embed_single_instance(
    client: &VertexClient,
    url: &str,
    request: &EmbedRequest,
    index: usize,
    text: &str,
) -> Result<Embedding, Error> {
    let wire = PredictRequest {
        instances: vec![Instance {
            content: text.to_string(),
            task_type: request.task_type.map(EmbedTaskType::as_str),
            title: request.title.clone(),
        }],
        parameters: request
            .output_dimensionality
            .map(|output_dimensionality| PredictParams {
                output_dimensionality,
            }),
    };

    let resp = client
        .send_with_retry(|| client.http.post(url).json(&wire), index)
        .await?;
    let wire: PredictResponse = resp
        .json()
        .await
        .map_err(|e| Error::provider("vertex-ai", format!("Parse failed: {e}")))?;

    wire.predictions
        .into_iter()
        .next()
        .map(Embedding::from)
        .ok_or_else(|| Error::bad_response("predict returned no embedding"))
}

fn predict_endpoint(base: &str, project_id: &str, region: &str, model: &str) -> String {
    let model = model_id(model);
    format!(
        "{base}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:predict"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> EmbedRequest {
        EmbedRequest::new("gemini-embedding-001", ["hello"])
    }

    fn texts(specs: &[usize]) -> Vec<String> {
        specs.iter().map(|len| "a".repeat(*len)).collect()
    }

    const GEMINI_API: &str = "https://generativelanguage.googleapis.com";
    const VERTEX_API: &str = "https://asia-south1-aiplatform.googleapis.com";

    #[test]
    fn batch_endpoint_targets_the_gemini_api() {
        assert_eq!(
            batch_endpoint(GEMINI_API, "gemini-embedding-001"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
        );
    }

    #[test]
    fn predict_endpoint_hangs_off_the_publisher_model_path() {
        assert_eq!(
            predict_endpoint(VERTEX_API, "proj", "asia-south1", "gemini-embedding-001"),
            "https://asia-south1-aiplatform.googleapis.com/v1/projects/proj/locations/asia-south1/publishers/google/models/gemini-embedding-001:predict"
        );
    }

    #[test]
    fn a_models_prefix_from_the_docs_is_not_doubled_up() {
        assert_eq!(
            batch_endpoint(GEMINI_API, "models/gemini-embedding-001"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
        );
        assert!(
            predict_endpoint(VERTEX_API, "proj", "global", "models/gemini-embedding-001")
                .ends_with("/publishers/google/models/gemini-embedding-001:predict")
        );

        let req = EmbedRequest::new("models/gemini-embedding-001", ["hello"]);
        let json = serde_json::to_value(batch_item(&req, "hello")).unwrap();
        assert_eq!(json["model"], "models/gemini-embedding-001");
    }

    #[test]
    fn batch_item_prefixes_model_and_maps_options() {
        let req = request()
            .with_task_type(crate::types::EmbedTaskType::RetrievalDocument)
            .with_title("Doc")
            .with_output_dimensionality(256);
        let json = serde_json::to_value(batch_item(&req, "hello")).unwrap();

        assert_eq!(json["model"], "models/gemini-embedding-001");
        assert_eq!(json["content"]["parts"][0]["text"], "hello");
        assert_eq!(json["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(json["title"], "Doc");
        assert_eq!(json["outputDimensionality"], 256);
    }

    #[test]
    fn batch_item_omits_unset_options() {
        let json = serde_json::to_value(batch_item(&request(), "hello")).unwrap();
        assert!(json.get("taskType").is_none());
        assert!(json.get("title").is_none());
        assert!(json.get("outputDimensionality").is_none());
    }

    #[test]
    fn predict_request_mixes_snake_case_instances_and_camel_case_parameters() {
        let wire = PredictRequest {
            instances: vec![Instance {
                content: "hello".into(),
                task_type: Some("RETRIEVAL_QUERY"),
                title: None,
            }],
            parameters: Some(PredictParams {
                output_dimensionality: 768,
            }),
        };
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["instances"][0]["content"], "hello");
        assert_eq!(json["instances"][0]["task_type"], "RETRIEVAL_QUERY");
        assert!(json["instances"][0].get("title").is_none());
        assert_eq!(json["parameters"]["outputDimensionality"], 768);
    }

    #[test]
    fn predict_request_omits_parameters_without_dimensionality() {
        let wire = PredictRequest {
            instances: vec![],
            parameters: None,
        };
        let json = serde_json::to_value(&wire).unwrap();
        assert!(json.get("parameters").is_none());
    }

    #[test]
    fn prediction_parses_protobuf_float_statistics() {
        let resp: PredictResponse = serde_json::from_str(
            r#"{"predictions":[{"embeddings":{"values":[0.1,0.2],"statistics":{"token_count":7.0,"truncated":false}}}]}"#,
        )
        .unwrap();
        let embedding = Embedding::from(resp.predictions.into_iter().next().unwrap());

        assert_eq!(embedding.values, vec![0.1, 0.2]);
        assert_eq!(embedding.token_count, Some(7));
        assert_eq!(embedding.truncated, Some(false));
    }

    #[test]
    fn prediction_without_statistics_still_parses() {
        let resp: PredictResponse =
            serde_json::from_str(r#"{"predictions":[{"embeddings":{"values":[1.0]}}]}"#).unwrap();
        let embedding = Embedding::from(resp.predictions.into_iter().next().unwrap());

        assert_eq!(embedding.values, vec![1.0]);
        assert_eq!(embedding.token_count, None);
        assert_eq!(embedding.truncated, None);
    }

    #[test]
    fn batch_response_parses_values_in_order() {
        let resp: BatchResponse =
            serde_json::from_str(r#"{"embeddings":[{"values":[0.1]},{"values":[0.2]}]}"#).unwrap();
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[1].values, vec![0.2]);
        assert!(resp.usage_metadata.is_none());
    }

    #[test]
    fn batch_response_picks_up_the_billed_token_count() {
        let resp: BatchResponse = serde_json::from_str(
            r#"{"embeddings":[{"values":[0.1]}],"usageMetadata":{"promptTokenCount":42}}"#,
        )
        .unwrap();
        assert_eq!(
            resp.usage_metadata.and_then(|u| u.prompt_token_count),
            Some(42)
        );
    }

    #[test]
    fn title_without_retrieval_document_is_rejected_before_any_request() {
        let err = validate(&request().with_title("Doc")).unwrap_err();
        assert!(matches!(err, Error::Config(_)), "got {err:?}");

        validate(
            &request()
                .with_title("Doc")
                .with_task_type(EmbedTaskType::RetrievalDocument),
        )
        .unwrap();
        validate(&request().with_task_type(EmbedTaskType::RetrievalQuery)).unwrap();
        validate(&request()).unwrap();
    }

    #[test]
    fn chunks_split_on_the_text_count() {
        let texts = texts(&[1; 250]);
        let calls = chunks(&texts);

        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, 0);
        assert_eq!(calls[0].1.len(), MAX_TEXTS_PER_BATCH_CALL);
        assert_eq!(calls[1].0, MAX_TEXTS_PER_BATCH_CALL);
        assert_eq!(calls[2].0, 2 * MAX_TEXTS_PER_BATCH_CALL);
        assert_eq!(calls[2].1.len(), 50);
    }

    #[test]
    fn chunks_split_on_the_character_budget_before_the_count() {
        let texts = texts(&[MAX_CHARS_PER_BATCH_CALL / 2 + 1; 4]);
        let calls = chunks(&texts);

        assert_eq!(calls.len(), 4, "each pair blows the budget");
        assert_eq!(calls.iter().map(|(_, c)| c.len()).sum::<usize>(), 4);
    }

    #[test]
    fn chunks_keep_one_oversized_text_in_a_call_of_its_own() {
        let texts = texts(&[1, MAX_CHARS_PER_BATCH_CALL + 10, 1]);
        let calls = chunks(&texts);

        assert_eq!(calls.len(), 3);
        assert_eq!(calls[1].0, 1);
        assert_eq!(calls[1].1.len(), 1);
    }

    #[test]
    fn chunks_cover_every_text_exactly_once_and_in_order() {
        let texts = texts(&[1; 101]);
        let calls = chunks(&texts);
        let covered: usize = calls.iter().map(|(_, c)| c.len()).sum();

        assert_eq!(covered, 101);
        let mut expected = 0;
        for (start, chunk) in calls {
            assert_eq!(start, expected);
            expected += chunk.len();
        }
    }

    #[test]
    fn a_failed_call_yields_one_error_per_text_it_covered() {
        let error = Error::provider("vertex-ai", "API error (429): slow down");
        let results = failed_chunk(3, &error);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_err()));
        assert_eq!(
            results[2].as_ref().unwrap_err().to_string(),
            "vertex-ai: API error (429): slow down"
        );
    }

    #[test]
    fn a_response_where_nothing_succeeded_is_an_error_not_an_empty_success() {
        let error = Error::provider_permanent("vertex-ai", "API error (403): denied");

        let all_failed = EmbedResponse {
            embeddings: failed_chunk(2, &error),
            total_token_count: None,
        };
        assert!(or_total_failure(all_failed).is_err());

        let partial = EmbedResponse {
            embeddings: vec![
                Ok(Embedding {
                    values: vec![1.0],
                    token_count: None,
                    truncated: None,
                }),
                Err(error),
            ],
            total_token_count: None,
        };
        let partial = or_total_failure(partial).expect("a partial result survives");
        assert_eq!(partial.failures().count(), 1);

        let empty = EmbedResponse {
            embeddings: Vec::new(),
            total_token_count: None,
        };
        assert!(
            or_total_failure(empty).is_ok(),
            "an empty result is not a failure"
        );
    }

    #[test]
    fn token_counts_add_up_but_stay_none_when_nothing_was_reported() {
        assert_eq!(add_tokens(None, None), None);
        assert_eq!(add_tokens(None, Some(7)), Some(7));
        assert_eq!(add_tokens(Some(7), None), Some(7));
        assert_eq!(add_tokens(Some(7), Some(5)), Some(12));
        assert_eq!(add_tokens(Some(u32::MAX), Some(5)), Some(u32::MAX));
    }
}
