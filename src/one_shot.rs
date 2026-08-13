use crate::error::Error;
use crate::provider::LlmProvider;
use crate::types::{GenerateRequest, GenerateResponse};

pub async fn one_shot(
    provider: &dyn LlmProvider,
    request: &GenerateRequest,
) -> Result<GenerateResponse, Error> {
    provider.generate(request).await
}
