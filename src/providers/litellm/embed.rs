use crate::error::Error;
use crate::types::{EmbedRequest, EmbedResponse, Embedding};
use serde::{Deserialize, Serialize};

use super::chat::WireUsage;
use super::client::LiteLlmClient;

pub(crate) async fn embed(
    client: &LiteLlmClient,
    request: &EmbedRequest,
) -> Result<EmbedResponse, Error> {
    let wire = EmbeddingRequest {
        model: request.model.clone(),
        input: request.texts.clone(),
        dimensions: request.output_dimensionality,
    };
    let resp = client
        .send(client.http.post(client.endpoint("embeddings")).json(&wire))
        .await?;
    let wire: EmbeddingResponse = resp
        .json()
        .await
        .map_err(|e| Error::provider("litellm", format!("Parse failed: {e}")))?;
    Ok(EmbedResponse {
        embeddings: wire
            .data
            .into_iter()
            .map(|d| {
                Ok(Embedding {
                    values: d.embedding,
                    token_count: None,
                    truncated: None,
                })
            })
            .collect(),
        total_token_count: wire.usage.and_then(|u| u.total_tokens),
    })
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}
