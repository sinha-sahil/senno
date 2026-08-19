use crate::error::Error;
use crate::types::{
    BatchJob, BatchState, EmbedRequest, Embedding, GenerateRequest, GenerateResponse,
};

use super::client::{VertexClient, VertexProvider};
use super::config::ResolvedAuth;
use super::embed::{TextContent, model_id, validate};
use super::gemini;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub(crate) async fn create(
    client: &VertexClient,
    requests: &[GenerateRequest],
) -> Result<BatchJob<GenerateResponse>, Error> {
    ensure_supported(client)?;
    let model = shared_model(requests)?;
    let url = create_endpoint(client.gemini_api_base(), model);
    let wire = create_wire(requests);

    operation_job::<GenerateOutput, _>(
        client,
        client.http.post(&url).json(&wire),
        gemini::from_wire,
    )
    .await
}

pub(crate) async fn get(
    client: &VertexClient,
    name: &str,
) -> Result<BatchJob<GenerateResponse>, Error> {
    ensure_supported(client)?;
    let url = job_endpoint(client.gemini_api_base(), name);

    operation_job::<GenerateOutput, _>(client, client.http.get(&url), gemini::from_wire).await
}

pub(crate) async fn create_embeddings(
    client: &VertexClient,
    request: &EmbedRequest,
) -> Result<BatchJob<Embedding>, Error> {
    ensure_supported(client)?;
    validate(request)?;
    if request.texts.is_empty() {
        return Err(Error::config(
            "create_embedding_batch requires at least one text",
        ));
    }
    let url = embed_create_endpoint(client.gemini_api_base(), &request.model);
    let wire = create_embed_wire(request);

    operation_job::<EmbedOutput, _>(client, client.http.post(&url).json(&wire), Embedding::from)
        .await
}

pub(crate) async fn get_embeddings(
    client: &VertexClient,
    name: &str,
) -> Result<BatchJob<Embedding>, Error> {
    ensure_supported(client)?;
    let url = job_endpoint(client.gemini_api_base(), name);

    operation_job::<EmbedOutput, _>(client, client.http.get(&url), Embedding::from).await
}

pub(crate) async fn cancel(client: &VertexClient, name: &str) -> Result<(), Error> {
    ensure_supported(client)?;
    let url = cancel_endpoint(client.gemini_api_base(), name);

    let req = client.authorize(client.http.post(&url)).await?;
    client.send(req).await?;
    Ok(())
}

fn ensure_supported(client: &VertexClient) -> Result<(), Error> {
    match (client.provider, &client.auth) {
        (VertexProvider::Anthropic, _) => {
            Err(Error::config("Batch jobs require VertexProvider::Gemini"))
        }
        (_, ResolvedAuth::ServiceAccount { .. }) => Err(Error::config(
            "Batch jobs use the Gemini Batch API and need API-key auth; Vertex service-account batch prediction requires GCS/BigQuery I/O that senno does not manage",
        )),
        (VertexProvider::Gemini, ResolvedAuth::ApiKey { .. }) => Ok(()),
    }
}

fn shared_model(requests: &[GenerateRequest]) -> Result<&str, Error> {
    let first = requests
        .first()
        .ok_or_else(|| Error::config("create_batch requires at least one request"))?;
    if requests.iter().any(|r| r.model != first.model) {
        return Err(Error::config(
            "all requests in a batch must target the same model",
        ));
    }
    Ok(&first.model)
}

#[derive(Serialize)]
struct CreateBatchWire<C> {
    batch: BatchWire<C>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchWire<C> {
    display_name: String,
    input_config: C,
}

#[derive(Serialize)]
struct InputConfig<R> {
    requests: InlinedRequests<R>,
}

#[derive(Serialize)]
struct InlinedRequests<R> {
    requests: Vec<R>,
}

#[derive(Serialize)]
struct InlinedRequest {
    request: gemini::Request,
    metadata: RequestKey,
}

#[derive(Serialize, Deserialize)]
struct RequestKey {
    key: String,
}

fn batch_wire<C>(input_config: C) -> CreateBatchWire<C> {
    CreateBatchWire {
        batch: BatchWire {
            display_name: "senno".into(),
            input_config,
        },
    }
}

fn create_wire(requests: &[GenerateRequest]) -> CreateBatchWire<InputConfig<InlinedRequest>> {
    batch_wire(InputConfig {
        requests: InlinedRequests {
            requests: requests
                .iter()
                .enumerate()
                .map(|(index, request)| InlinedRequest {
                    request: gemini::to_wire(request),
                    metadata: RequestKey {
                        key: index.to_string(),
                    },
                })
                .collect(),
        },
    })
}

#[derive(Serialize)]
struct EmbedInlinedRequest {
    request: EmbedContentWire,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentWire {
    content: TextContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

// The embeddings batch carries no per-request metadata key — the Gemini SDK
// doesn't send one, so results are matched back by position.
fn create_embed_wire(request: &EmbedRequest) -> CreateBatchWire<InputConfig<EmbedInlinedRequest>> {
    batch_wire(InputConfig {
        requests: InlinedRequests {
            requests: request
                .texts
                .iter()
                .map(|text| EmbedInlinedRequest {
                    request: EmbedContentWire {
                        content: TextContent::new(text),
                        task_type: request.task_type.map(crate::types::EmbedTaskType::as_str),
                        title: request.title.clone(),
                        output_dimensionality: request.output_dimensionality,
                    },
                })
                .collect(),
        },
    })
}

#[derive(Deserialize)]
struct Operation<O> {
    name: String,
    metadata: Option<OperationMetadata>,
    response: Option<O>,
    error: Option<Status>,
}

#[derive(Deserialize)]
struct OperationMetadata {
    state: Option<String>,
}

/// The two batch kinds differ only in the JSON key their results arrive under.
trait BatchOutput: DeserializeOwned {
    type Item;

    fn into_items(self) -> Vec<InlinedResponse<Self::Item>>;
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Inlined<R> {
    inlined_responses: Vec<InlinedResponse<R>>,
}

#[derive(Deserialize)]
struct InlinedResponse<R> {
    metadata: Option<RequestKey>,
    response: Option<R>,
    error: Option<Status>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateOutput {
    inlined_responses: Option<Inlined<gemini::Response>>,
}

impl BatchOutput for GenerateOutput {
    type Item = gemini::Response;

    fn into_items(self) -> Vec<InlinedResponse<Self::Item>> {
        self.inlined_responses
            .map(|i| i.inlined_responses)
            .unwrap_or_default()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedOutput {
    inlined_embed_content_responses: Option<Inlined<EmbedItem>>,
}

impl BatchOutput for EmbedOutput {
    type Item = EmbedItem;

    fn into_items(self) -> Vec<InlinedResponse<Self::Item>> {
        self.inlined_embed_content_responses
            .map(|i| i.inlined_responses)
            .unwrap_or_default()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedItem {
    embedding: Option<ContentEmbedding>,
    token_count: Option<u32>,
}

#[derive(Deserialize)]
struct ContentEmbedding {
    values: Vec<f32>,
}

impl From<EmbedItem> for Embedding {
    fn from(item: EmbedItem) -> Self {
        Embedding {
            values: item.embedding.map(|e| e.values).unwrap_or_default(),
            token_count: item.token_count,
            truncated: None,
        }
    }
}

#[derive(Deserialize)]
struct Status {
    code: Option<i64>,
    message: Option<String>,
}

async fn operation_job<O, T>(
    client: &VertexClient,
    request: reqwest::RequestBuilder,
    convert: impl Fn(O::Item) -> T,
) -> Result<BatchJob<T>, Error>
where
    O: BatchOutput,
{
    let req = client.authorize(request).await?;
    let resp = client.send(req).await?;
    let operation: Operation<O> = resp
        .json()
        .await
        .map_err(|e| Error::provider("vertex-ai", format!("Parse failed: {e}")))?;

    Ok(job_from_operation(operation, convert))
}

fn job_from_operation<O, T>(operation: Operation<O>, convert: impl Fn(O::Item) -> T) -> BatchJob<T>
where
    O: BatchOutput,
{
    let state = if operation.error.is_some() {
        BatchState::Failed
    } else {
        operation
            .metadata
            .as_ref()
            .and_then(|m| m.state.as_deref())
            .map(parse_state)
            .unwrap_or(BatchState::Unknown)
    };
    let responses = operation
        .response
        .map(|output| responses_in_request_order(output.into_items(), convert))
        .unwrap_or_default();

    BatchJob {
        name: operation.name,
        state,
        responses,
    }
}

fn parse_state(state: &str) -> BatchState {
    match state.rsplit('_').next().unwrap_or_default() {
        "PENDING" => BatchState::Pending,
        "RUNNING" => BatchState::Running,
        "SUCCEEDED" => BatchState::Succeeded,
        "FAILED" => BatchState::Failed,
        "CANCELLED" => BatchState::Cancelled,
        "EXPIRED" => BatchState::Expired,
        _ => BatchState::Unknown,
    }
}

fn responses_in_request_order<R, T>(
    items: Vec<InlinedResponse<R>>,
    convert: impl Fn(R) -> T,
) -> Vec<Result<T, Error>> {
    let mut indexed: Vec<(usize, InlinedResponse<R>)> = items.into_iter().enumerate().collect();
    indexed.sort_by_key(|(position, item)| {
        item.metadata
            .as_ref()
            .and_then(|m| m.key.parse::<usize>().ok())
            .unwrap_or(*position)
    });
    indexed
        .into_iter()
        .map(|(_, item)| item_result(item, &convert))
        .collect()
}

fn item_result<R, T>(item: InlinedResponse<R>, convert: impl Fn(R) -> T) -> Result<T, Error> {
    if let Some(error) = item.error {
        return Err(Error::provider_permanent(
            "vertex-ai",
            format!(
                "batch item failed ({}): {}",
                error.code.unwrap_or_default(),
                error.message.unwrap_or_default()
            ),
        ));
    }
    item.response
        .map(convert)
        .ok_or_else(|| Error::bad_response("batch item carried neither response nor error"))
}

fn create_endpoint(base: &str, model: &str) -> String {
    format!(
        "{base}/v1beta/models/{}:batchGenerateContent",
        model_id(model)
    )
}

fn embed_create_endpoint(base: &str, model: &str) -> String {
    format!(
        "{base}/v1beta/models/{}:asyncBatchEmbedContent",
        model_id(model)
    )
}

fn job_endpoint(base: &str, name: &str) -> String {
    format!("{base}/v1beta/{name}")
}

fn cancel_endpoint(base: &str, name: &str) -> String {
    format!("{base}/v1beta/{name}:cancel")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EmbedTaskType;

    const API: &str = "https://generativelanguage.googleapis.com";

    fn requests(models: &[&str]) -> Vec<GenerateRequest> {
        models
            .iter()
            .map(|m| GenerateRequest::one_shot(*m, "sys", "hi"))
            .collect()
    }

    fn api_key_client(provider: VertexProvider) -> VertexClient {
        VertexClient::new(
            reqwest::Client::new(),
            ResolvedAuth::ApiKey {
                api_key: "k".into(),
            },
            provider,
            None,
        )
    }

    fn generation_job(json: &str) -> BatchJob<GenerateResponse> {
        job_from_operation(
            serde_json::from_str::<Operation<GenerateOutput>>(json).unwrap(),
            gemini::from_wire,
        )
    }

    fn embedding_job(json: &str) -> BatchJob<Embedding> {
        job_from_operation(
            serde_json::from_str::<Operation<EmbedOutput>>(json).unwrap(),
            Embedding::from,
        )
    }

    #[test]
    fn endpoints_follow_the_batches_resource_names() {
        assert_eq!(
            create_endpoint(API, "gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent"
        );
        assert_eq!(
            embed_create_endpoint(API, "gemini-embedding-001"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:asyncBatchEmbedContent"
        );
        assert_eq!(
            job_endpoint(API, "batches/abc123"),
            "https://generativelanguage.googleapis.com/v1beta/batches/abc123"
        );
        assert_eq!(
            cancel_endpoint(API, "batches/abc123"),
            "https://generativelanguage.googleapis.com/v1beta/batches/abc123:cancel"
        );
    }

    #[test]
    fn both_batch_kinds_are_polled_and_cancelled_at_the_same_urls() {
        assert_eq!(
            job_endpoint(API, "batches/abc"),
            job_endpoint(API, "batches/abc")
        );
        assert!(job_endpoint(API, "batches/abc").ends_with("/v1beta/batches/abc"));
        assert!(cancel_endpoint(API, "batches/abc").ends_with(":cancel"));
    }

    #[test]
    fn shared_model_rejects_empty_and_mixed_batches() {
        assert!(matches!(shared_model(&[]), Err(Error::Config(_))));
        assert!(matches!(
            shared_model(&requests(&["a", "b"])),
            Err(Error::Config(_))
        ));
        assert_eq!(shared_model(&requests(&["a", "a"])).unwrap(), "a");
    }

    #[test]
    fn create_wire_nests_requests_with_index_keys() {
        let json = serde_json::to_value(create_wire(&requests(&["m", "m"]))).unwrap();
        let items = json["batch"]["inputConfig"]["requests"]["requests"]
            .as_array()
            .unwrap();

        assert_eq!(json["batch"]["displayName"], "senno");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["metadata"]["key"], "0");
        assert_eq!(items[1]["metadata"]["key"], "1");
        assert_eq!(items[0]["request"]["contents"][0]["role"], "user");
        assert_eq!(
            items[0]["request"]["systemInstruction"]["parts"][0]["text"],
            "sys"
        );
        assert!(items[0]["request"].get("model").is_none());
    }

    #[test]
    fn embed_wire_nests_one_request_per_text() {
        let request = EmbedRequest::new("gemini-embedding-001", ["alpha", "beta"])
            .with_task_type(EmbedTaskType::RetrievalDocument)
            .with_title("Doc")
            .with_output_dimensionality(768);
        let json = serde_json::to_value(create_embed_wire(&request)).unwrap();
        let items = json["batch"]["inputConfig"]["requests"]["requests"]
            .as_array()
            .unwrap();

        assert_eq!(json["batch"]["displayName"], "senno");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["request"]["content"]["parts"][0]["text"], "alpha");
        assert_eq!(items[1]["request"]["content"]["parts"][0]["text"], "beta");
        assert_eq!(items[0]["request"]["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(items[0]["request"]["title"], "Doc");
        assert_eq!(items[0]["request"]["outputDimensionality"], 768);
        // The model rides in the URL, not the item.
        assert!(items[0]["request"].get("model").is_none());
    }

    #[test]
    fn embed_wire_omits_unset_options() {
        let json =
            serde_json::to_value(create_embed_wire(&EmbedRequest::single("m", "x"))).unwrap();
        let item = &json["batch"]["inputConfig"]["requests"]["requests"][0]["request"];

        assert!(item.get("taskType").is_none());
        assert!(item.get("title").is_none());
        assert!(item.get("outputDimensionality").is_none());
    }

    #[test]
    fn pending_operation_maps_to_pending_job_without_responses() {
        let job =
            generation_job(r#"{"name":"batches/abc","metadata":{"state":"BATCH_STATE_PENDING"}}"#);

        assert_eq!(job.name, "batches/abc");
        assert_eq!(job.state, BatchState::Pending);
        assert!(job.responses.is_empty());
    }

    #[test]
    fn parse_state_accepts_batch_and_job_prefixes_and_unknowns() {
        assert_eq!(parse_state("BATCH_STATE_SUCCEEDED"), BatchState::Succeeded);
        assert_eq!(parse_state("JOB_STATE_SUCCEEDED"), BatchState::Succeeded);
        assert_eq!(parse_state("BATCH_STATE_CANCELLED"), BatchState::Cancelled);
        assert_eq!(parse_state("BATCH_STATE_UNSPECIFIED"), BatchState::Unknown);
        assert_eq!(parse_state("something-new"), BatchState::Unknown);
    }

    #[test]
    fn operation_level_error_maps_to_failed() {
        let job = generation_job(r#"{"name":"batches/abc","error":{"code":13,"message":"boom"}}"#);
        assert_eq!(job.state, BatchState::Failed);
    }

    #[test]
    fn finished_operation_orders_responses_by_key_and_keeps_item_errors() {
        let job = generation_job(
            r#"{
                "name": "batches/abc",
                "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
                "response": {"inlinedResponses": {"inlinedResponses": [
                    {"metadata": {"key": "1"}, "error": {"code": 3, "message": "bad item"}},
                    {"metadata": {"key": "0"}, "response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}]}}
                ]}}
            }"#,
        );

        assert_eq!(job.state, BatchState::Succeeded);
        assert_eq!(job.responses.len(), 2);
        assert_eq!(job.responses[0].as_ref().unwrap().text().unwrap(), "hello");
        let err = job.responses[1].as_ref().unwrap_err();
        assert!(err.is_provider_error());
        assert!(!err.is_retryable());
    }

    #[test]
    fn finished_embedding_operation_reads_the_embed_content_envelope() {
        let job = embedding_job(
            r#"{
                "name": "batches/xyz",
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {"inlinedEmbedContentResponses": {"inlinedResponses": [
                    {"response": {"embedding": {"values": [0.1, 0.2]}, "tokenCount": 5}},
                    {"error": {"code": 3, "message": "bad item"}},
                    {"response": {"embedding": {"values": [0.3]}}}
                ]}}
            }"#,
        );

        assert_eq!(job.name, "batches/xyz");
        assert_eq!(job.state, BatchState::Succeeded);
        assert_eq!(job.responses.len(), 3);

        let first = job.responses[0].as_ref().unwrap();
        assert_eq!(first.values, vec![0.1, 0.2]);
        assert_eq!(first.token_count, Some(5));
        assert_eq!(first.truncated, None);

        assert!(job.responses[1].is_err());
        assert_eq!(job.responses[2].as_ref().unwrap().values, vec![0.3]);
    }

    #[test]
    fn an_embedding_batch_without_keys_stays_in_submission_order() {
        let job = embedding_job(
            r#"{
                "name": "batches/xyz",
                "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                "response": {"inlinedEmbedContentResponses": {"inlinedResponses": [
                    {"response": {"embedding": {"values": [0.0]}}},
                    {"response": {"embedding": {"values": [1.0]}}},
                    {"response": {"embedding": {"values": [2.0]}}}
                ]}}
            }"#,
        );

        let values: Vec<f32> = job
            .responses
            .iter()
            .map(|r| r.as_ref().unwrap().values[0])
            .collect();
        assert_eq!(values, vec![0.0, 1.0, 2.0]);
    }

    #[tokio::test]
    async fn anthropic_client_is_rejected_before_any_request() {
        let client = api_key_client(VertexProvider::Anthropic);

        assert!(matches!(
            create(&client, &requests(&["m"])).await.unwrap_err(),
            Error::Config(_)
        ));
        assert!(matches!(
            create_embeddings(&client, &EmbedRequest::single("m", "x"))
                .await
                .unwrap_err(),
            Error::Config(_)
        ));
    }

    #[tokio::test]
    async fn mixed_models_are_rejected_before_any_request() {
        let client = api_key_client(VertexProvider::Gemini);
        let err = create(&client, &requests(&["a", "b"])).await.unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[tokio::test]
    async fn an_embedding_batch_needs_texts_and_a_valid_title_pairing() {
        let client = api_key_client(VertexProvider::Gemini);

        let empty = EmbedRequest::new("m", Vec::<String>::new());
        assert!(matches!(
            create_embeddings(&client, &empty).await.unwrap_err(),
            Error::Config(_)
        ));

        let bad_title = EmbedRequest::single("m", "x")
            .with_title("Doc")
            .with_task_type(EmbedTaskType::Clustering);
        assert!(matches!(
            create_embeddings(&client, &bad_title).await.unwrap_err(),
            Error::Config(_)
        ));
    }
}
