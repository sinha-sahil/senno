use std::future::Future;
use std::pin::Pin;

use crate::error::Error;
use crate::provider::{EmbeddingProvider, LlmProvider, LlmStream};
use crate::types::{EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse};

use super::config::BedrockGuardrail;
use super::{converse, embed};

/// Amazon Bedrock client: chat through Converse, embeddings through Titan.
#[derive(Clone)]
pub struct BedrockClient {
    pub(super) inner: aws_sdk_bedrockruntime::Client,
    pub(super) guardrail: Option<BedrockGuardrail>,
}

impl std::fmt::Debug for BedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockClient")
            .field("region", &self.inner.config().region())
            .field("guardrail", &self.guardrail)
            .finish_non_exhaustive()
    }
}

impl BedrockClient {
    pub(super) fn new(
        config: aws_sdk_bedrockruntime::Config,
        guardrail: Option<BedrockGuardrail>,
    ) -> Self {
        Self {
            inner: aws_sdk_bedrockruntime::Client::from_conf(config),
            guardrail,
        }
    }

    /// The region calls go to.
    pub fn region(&self) -> Option<&str> {
        self.inner.config().region().map(|r| r.as_ref())
    }

    /// The underlying AWS SDK client, for Bedrock features senno does not wrap.
    pub fn sdk_client(&self) -> &aws_sdk_bedrockruntime::Client {
        &self.inner
    }
}

impl LlmProvider for BedrockClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>> {
        Box::pin(converse::generate(self, request))
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(converse::stream_generate(self, request))
    }
}

impl EmbeddingProvider for BedrockClient {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbedResponse, Error>> + Send + 'a>> {
        Box::pin(embed::embed(self, request))
    }
}
