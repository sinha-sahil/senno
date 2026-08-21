use crate::error::Error;
use crate::provider::{
    BatchEmbeddingProvider, BatchGenerationProvider, EmbeddingProvider, LlmProvider, LlmStream,
};
use crate::types::{
    BatchJob, EmbedRequest, EmbedResponse, Embedding, GenerateRequest, GenerateResponse,
};
use std::pin::Pin;
use std::str::FromStr;
use std::time::Duration;

use super::config::{ResolvedAuth, regional_host};
use super::{anthropic, batch, embed, gemini};

const HTTP_MAX_ATTEMPTS: u32 = 3;
const HTTP_RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
const HTTP_RETRY_MAX_DELAY: Duration = Duration::from_secs(8);
const HTTP_RETRY_JITTER: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexProvider {
    Gemini,
    Anthropic,
}

pub struct VertexClient {
    pub(crate) http: reqwest::Client,
    pub(crate) auth: ResolvedAuth,
    pub(crate) provider: VertexProvider,
    base_url: Option<String>,
}

impl VertexClient {
    pub(crate) fn new(
        http: reqwest::Client,
        auth: ResolvedAuth,
        provider: VertexProvider,
        base_url: Option<String>,
    ) -> Self {
        Self {
            http,
            auth,
            provider,
            base_url,
        }
    }

    pub(crate) fn gemini_api_base(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com")
    }

    pub(crate) fn vertex_base(&self, region: &str) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| format!("https://{}", regional_host(region)))
    }

    pub(crate) async fn authorize(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, Error> {
        match &self.auth {
            ResolvedAuth::ApiKey { api_key } => Ok(req.header("x-goog-api-key", api_key.as_str())),
            ResolvedAuth::ServiceAccount { token_source, .. } => {
                let bearer = token_source
                    .access_token(&["https://www.googleapis.com/auth/cloud-platform"])
                    .await?;
                Ok(req.header("Authorization", format!("Bearer {bearer}")))
            }
        }
    }

    pub(crate) async fn send(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, Error> {
        let resp = req
            .send()
            .await
            .map_err(|e| Error::provider("vertex-ai", format!("Request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let transient = is_transient_status(status);
            let body = resp.text().await.unwrap_or_default();
            let detail = format!("API error ({status}): {body}");
            return Err(if transient {
                Error::provider("vertex-ai", detail)
            } else {
                Error::provider_permanent("vertex-ai", detail)
            });
        }

        Ok(resp)
    }

    /// Pass the item's index as `jitter_seed` so concurrent callers don't wake
    /// into the same retry.
    pub(crate) async fn send_with_retry<F>(
        &self,
        build: F,
        jitter_seed: usize,
    ) -> Result<reqwest::Response, Error>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        for attempt in 1..HTTP_MAX_ATTEMPTS {
            let req = self.authorize(build()).await?;
            match self.send(req).await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.is_retryable() => {
                    let delay = backoff_delay(attempt, jitter_seed);
                    tracing::warn!(
                        attempt,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "Transient Vertex AI error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }

        let req = self.authorize(build()).await?;
        self.send(req).await
    }
}

fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

fn backoff_delay(attempt: u32, jitter_seed: usize) -> Duration {
    let backoff = HTTP_RETRY_BASE_DELAY
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(HTTP_RETRY_MAX_DELAY);
    let jitter = (HTTP_RETRY_JITTER.as_millis() as u64).max(1);
    backoff.saturating_add(Duration::from_millis(jitter_seed as u64 % jitter))
}

impl LlmProvider for VertexClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<GenerateResponse, Error>> + Send + 'a>>
    {
        Box::pin(async move {
            match self.provider {
                VertexProvider::Gemini => gemini::generate(self, request).await,
                VertexProvider::Anthropic => anthropic::generate(self, request).await,
            }
        })
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(async move {
            match self.provider {
                VertexProvider::Gemini => gemini::stream_generate(self, request).await,
                VertexProvider::Anthropic => anthropic::stream_generate(self, request).await,
            }
        })
    }
}

impl EmbeddingProvider for VertexClient {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EmbedResponse, Error>> + Send + 'a>> {
        Box::pin(async move {
            match self.provider {
                VertexProvider::Gemini => embed::embed(self, request).await,
                VertexProvider::Anthropic => Err(Error::config(
                    "Embeddings require VertexProvider::Gemini; Anthropic on Vertex exposes no embeddings API",
                )),
            }
        })
    }
}

impl BatchGenerationProvider for VertexClient {
    fn create_batch<'a>(
        &'a self,
        requests: &'a [GenerateRequest],
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a,
        >,
    > {
        Box::pin(batch::create(self, requests))
    }

    fn get_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<
        Box<
            dyn std::future::Future<Output = Result<BatchJob<GenerateResponse>, Error>> + Send + 'a,
        >,
    > {
        Box::pin(batch::get(self, name))
    }

    fn cancel_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(batch::cancel(self, name))
    }
}

impl BatchEmbeddingProvider for VertexClient {
    fn create_embedding_batch<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>>
    {
        Box::pin(batch::create_embeddings(self, request))
    }

    fn get_embedding_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<BatchJob<Embedding>, Error>> + Send + 'a>>
    {
        Box::pin(batch::get_embeddings(self, name))
    }

    fn cancel_embedding_batch<'a>(
        &'a self,
        name: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(batch::cancel(self, name))
    }
}

impl VertexProvider {
    pub fn all() -> &'static [VertexProvider] {
        &[Self::Gemini, Self::Anthropic]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Anthropic => "anthropic",
        }
    }
}

impl FromStr for VertexProvider {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::all()
            .iter()
            .copied()
            .find(|variant| variant.as_str() == value)
            .ok_or_else(|| {
                Error::Config(format!(
                    "unknown vertex variant '{value}' — expected one of: {}",
                    Self::all()
                        .iter()
                        .map(|variant| variant.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;

    #[test]
    fn origins_are_googles_own_until_a_test_overrides_them() {
        let client = VertexClient::new(
            reqwest::Client::new(),
            ResolvedAuth::ApiKey {
                api_key: "k".into(),
            },
            VertexProvider::Gemini,
            None,
        );

        assert_eq!(
            client.gemini_api_base(),
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(
            client.vertex_base("asia-south1"),
            "https://asia-south1-aiplatform.googleapis.com"
        );
        assert_eq!(
            client.vertex_base("global"),
            "https://aiplatform.googleapis.com",
            "global answers on the bare host"
        );

        let client = VertexClient::new(
            reqwest::Client::new(),
            ResolvedAuth::ApiKey {
                api_key: "k".into(),
            },
            VertexProvider::Gemini,
            Some("http://127.0.0.1:9".into()),
        );
        assert_eq!(client.gemini_api_base(), "http://127.0.0.1:9");
        assert_eq!(client.vertex_base("asia-south1"), "http://127.0.0.1:9");
    }

    #[test]
    fn only_rate_limits_timeouts_and_server_faults_are_transient() {
        assert!(is_transient_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_transient_status(StatusCode::REQUEST_TIMEOUT));
        assert!(is_transient_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_transient_status(StatusCode::SERVICE_UNAVAILABLE));

        assert!(!is_transient_status(StatusCode::BAD_REQUEST));
        assert!(!is_transient_status(StatusCode::UNAUTHORIZED));
        assert!(!is_transient_status(StatusCode::FORBIDDEN));
        assert!(!is_transient_status(StatusCode::NOT_FOUND));
    }

    #[test]
    fn backoff_doubles_per_attempt_and_stays_capped() {
        assert_eq!(backoff_delay(1, 0), Duration::from_millis(500));
        assert_eq!(backoff_delay(2, 0), Duration::from_secs(1));
        assert_eq!(backoff_delay(3, 0), Duration::from_secs(2));
        assert_eq!(backoff_delay(9, 0), HTTP_RETRY_MAX_DELAY);
        assert_eq!(backoff_delay(0, 0), HTTP_RETRY_BASE_DELAY, "no underflow");
        assert_eq!(backoff_delay(u32::MAX, 0), HTTP_RETRY_MAX_DELAY);
    }

    #[test]
    fn jitter_separates_concurrent_callers_within_a_bounded_window() {
        let a = backoff_delay(1, 7);
        let b = backoff_delay(1, 8);
        assert_ne!(a, b);
        assert!(a >= HTTP_RETRY_BASE_DELAY);
        assert!(b < HTTP_RETRY_BASE_DELAY + HTTP_RETRY_JITTER);
    }

    #[test]
    fn vertex_provider_round_trips_through_str() {
        for variant in VertexProvider::all() {
            assert_eq!(
                variant.as_str().parse::<VertexProvider>().unwrap(),
                *variant
            );
        }
    }

    #[test]
    fn an_unknown_vertex_variant_lists_the_supported_ones() {
        let err = "llama".parse::<VertexProvider>().unwrap_err().to_string();
        assert!(err.contains("llama"));
        assert!(err.contains("gemini"));
    }
}
