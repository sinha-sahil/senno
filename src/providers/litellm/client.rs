use crate::error::Error;
use crate::provider::{EmbeddingProvider, LlmProvider, LlmStream};
use crate::types::{EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse};
use std::pin::Pin;
use std::time::Duration;

use super::config::{LiteLlmConfig, resolve_api_key, resolve_base_url};
use super::{chat, embed};

pub struct LiteLlmClient {
    pub(crate) http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
}

impl LiteLlmClient {
    pub fn new(config: Option<LiteLlmConfig>) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::internal(format!("Failed to build LiteLLM HTTP client: {e}")))?;
        Ok(Self::with_http(http, config))
    }

    pub fn with_http(http: reqwest::Client, config: Option<LiteLlmConfig>) -> Self {
        let config = config.unwrap_or_default();
        Self {
            http,
            api_key: resolve_api_key(config.api_key),
            base_url: resolve_base_url(config.base_url),
        }
    }

    pub(crate) fn endpoint(&self, path: &str) -> String {
        format!("{}/v1/{path}", self.base_url)
    }

    pub(crate) async fn send(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, Error> {
        let req = if let Some(key) = &self.api_key {
            req.bearer_auth(key)
        } else {
            req
        };
        let resp = req
            .send()
            .await
            .map_err(|e| Error::provider("litellm", format!("Request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let transient = status.is_server_error()
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status == reqwest::StatusCode::REQUEST_TIMEOUT;
            let body = resp.text().await.unwrap_or_default();
            let detail = format!("API error ({status}): {body}");
            return Err(if transient {
                Error::provider("litellm", detail)
            } else {
                Error::provider_permanent("litellm", detail)
            });
        }
        Ok(resp)
    }
}

impl LlmProvider for LiteLlmClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>
    {
        Box::pin(chat::generate(self, request))
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(chat::stream_generate(self, request))
    }
}

impl EmbeddingProvider for LiteLlmClient {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EmbedResponse, Error>> + Send + 'a>> {
        Box::pin(embed::embed(self, request))
    }
}
