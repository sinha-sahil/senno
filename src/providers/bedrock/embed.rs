//! Titan embeddings through InvokeModel: Titan Text Embeddings (v1, v2) and
//! Titan Multimodal Embeddings G1 (text, image, or both per item).
//!
//! Titan takes one input per call, so a request fans out to one InvokeModel
//! per item with bounded concurrency. Embedding models are throttled by
//! requests per minute, which the SDK's retry handles with backoff.

use base64::Engine as _;
use futures::StreamExt;
use serde::Deserialize;

use crate::error::Error;
use crate::types::{EmbedImage, EmbedRequest, EmbedResponse, Embedding};

use super::client::BedrockClient;
use super::error::{PROVIDER, from_sdk};

const REQUEST_CONCURRENCY: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TitanKind {
    /// `amazon.titan-embed-text-v1`: text only, fixed 1,536 dimensions.
    TextV1,
    /// `amazon.titan-embed-text-v2:0`: text only, 256 / 512 / 1,024 dimensions.
    TextV2,
    /// `amazon.titan-embed-image-v1`: text and/or image, 256 / 384 / 1,024.
    Multimodal,
}

pub(super) fn titan_kind(model: &str) -> Result<TitanKind, Error> {
    if model.contains("titan-embed-image") {
        Ok(TitanKind::Multimodal)
    } else if model.contains("titan-embed-text-v2") {
        Ok(TitanKind::TextV2)
    } else if model.contains("titan-embed-text") {
        Ok(TitanKind::TextV1)
    } else {
        Err(Error::config(format!(
            "Bedrock embeddings support Amazon Titan models (titan-embed-text-v1, titan-embed-text-v2, titan-embed-image-v1); got '{model}'"
        )))
    }
}

pub(super) fn validate(request: &EmbedRequest) -> Result<TitanKind, Error> {
    let kind = titan_kind(&request.model)?;
    if request.title.is_some() {
        return Err(Error::config("Titan embeddings do not accept a title"));
    }
    if !request.images.is_empty() && kind != TitanKind::Multimodal {
        return Err(Error::config(format!(
            "'{}' embeds text only; use amazon.titan-embed-image-v1 for images",
            request.model
        )));
    }
    if !request.texts.is_empty()
        && !request.images.is_empty()
        && request.texts.len() != request.images.len()
    {
        return Err(Error::config(format!(
            "texts ({}) and images ({}) pair up item by item and must be the same length",
            request.texts.len(),
            request.images.len()
        )));
    }
    if kind == TitanKind::TextV1 && request.output_dimensionality.is_some() {
        return Err(Error::config(
            "amazon.titan-embed-text-v1 has a fixed dimension; output_dimensionality needs titan-embed-text-v2 or titan-embed-image-v1",
        ));
    }
    Ok(kind)
}

pub(super) fn body(
    kind: TitanKind,
    text: Option<&str>,
    image: Option<&EmbedImage>,
    dimensions: Option<u32>,
) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    if let Some(text) = text {
        body.insert("inputText".into(), text.into());
    }
    match kind {
        TitanKind::TextV1 => {}
        TitanKind::TextV2 => {
            if let Some(d) = dimensions {
                body.insert("dimensions".into(), d.into());
            }
            body.insert("normalize".into(), true.into());
        }
        TitanKind::Multimodal => {
            if let Some(image) = image {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&image.data);
                body.insert("inputImage".into(), encoded.into());
            }
            if let Some(d) = dimensions {
                body.insert(
                    "embeddingConfig".into(),
                    serde_json::json!({ "outputEmbeddingLength": d }),
                );
            }
        }
    }
    serde_json::Value::Object(body)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TitanResponse {
    embedding: Vec<f32>,
    #[serde(default)]
    input_text_token_count: Option<u32>,
}

pub(super) async fn embed(
    client: &BedrockClient,
    request: &EmbedRequest,
) -> Result<EmbedResponse, Error> {
    let kind = validate(request)?;
    let count = request.item_count();

    let embeddings: Vec<Result<Embedding, Error>> = futures::stream::iter(0..count)
        .map(|i| {
            let text = request.texts.get(i).map(String::as_str);
            let image = request.images.get(i);
            embed_one(
                client,
                &request.model,
                kind,
                text,
                image,
                request.output_dimensionality,
            )
        })
        .buffered(REQUEST_CONCURRENCY)
        .collect()
        .await;

    let counted: Vec<u32> = embeddings
        .iter()
        .filter_map(|r| r.as_ref().ok().and_then(|e| e.token_count))
        .collect();
    let total_token_count = (!counted.is_empty()).then(|| counted.iter().sum());

    Ok(EmbedResponse {
        embeddings,
        total_token_count,
    })
}

async fn embed_one(
    client: &BedrockClient,
    model: &str,
    kind: TitanKind,
    text: Option<&str>,
    image: Option<&EmbedImage>,
    dimensions: Option<u32>,
) -> Result<Embedding, Error> {
    let payload = serde_json::to_vec(&body(kind, text, image, dimensions))
        .map_err(|e| Error::internal(format!("titan request did not serialize: {e}")))?;

    let output = client
        .inner
        .invoke_model()
        .model_id(model)
        .content_type("application/json")
        .accept("application/json")
        .body(aws_sdk_bedrockruntime::primitives::Blob::new(payload))
        .send()
        .await
        .map_err(|e| from_sdk("InvokeModel", e))?;

    let parsed: TitanResponse = serde_json::from_slice(output.body().as_ref()).map_err(|e| {
        Error::provider_permanent(PROVIDER, format!("Titan response was not readable: {e}"))
    })?;
    if parsed.embedding.is_empty() {
        return Err(Error::provider_permanent(
            PROVIDER,
            "Titan returned an empty embedding",
        ));
    }
    Ok(Embedding {
        values: parsed.embedding,
        token_count: parsed.input_text_token_count,
        truncated: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_ids_map_to_titan_kinds() {
        assert_eq!(
            titan_kind("amazon.titan-embed-image-v1").unwrap(),
            TitanKind::Multimodal
        );
        assert_eq!(
            titan_kind("amazon.titan-embed-text-v2:0").unwrap(),
            TitanKind::TextV2
        );
        assert_eq!(
            titan_kind("amazon.titan-embed-text-v1").unwrap(),
            TitanKind::TextV1
        );
        assert!(titan_kind("cohere.embed-english-v3").is_err());
    }

    #[test]
    fn multimodal_body_carries_text_image_and_length() {
        let image = EmbedImage::new(vec![1u8, 2, 3]);
        let b = body(
            TitanKind::Multimodal,
            Some("red shoe"),
            Some(&image),
            Some(1024),
        );
        assert_eq!(
            b,
            json!({
                "inputText": "red shoe",
                "inputImage": "AQID",
                "embeddingConfig": {"outputEmbeddingLength": 1024},
            })
        );
    }

    #[test]
    fn text_v2_body_normalizes_and_sets_dimensions() {
        assert_eq!(
            body(TitanKind::TextV2, Some("hi"), None, Some(512)),
            json!({"inputText": "hi", "dimensions": 512, "normalize": true})
        );
    }

    #[test]
    fn images_need_the_multimodal_model() {
        let request = EmbedRequest::new("amazon.titan-embed-text-v2:0", ["a"])
            .with_images([EmbedImage::new(vec![1])]);
        assert!(validate(&request).is_err());
    }

    #[test]
    fn texts_and_images_must_pair_up() {
        let request = EmbedRequest::new("amazon.titan-embed-image-v1", ["a", "b"])
            .with_images([EmbedImage::new(vec![1])]);
        assert!(validate(&request).is_err());

        let images_only =
            EmbedRequest::from_images("amazon.titan-embed-image-v1", [EmbedImage::new(vec![1])]);
        assert_eq!(validate(&images_only).unwrap(), TitanKind::Multimodal);
        assert_eq!(images_only.item_count(), 1);
    }
}
