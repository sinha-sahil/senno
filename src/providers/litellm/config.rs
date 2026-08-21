pub const DEFAULT_BASE_URL: &str = "http://localhost:4000";

#[derive(Debug, Clone, Default)]
pub struct LiteLlmConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl LiteLlmConfig {
    pub fn with_api_key(mut self, val: impl Into<String>) -> Self {
        self.api_key = Some(val.into());
        self
    }

    pub fn with_base_url(mut self, val: impl Into<String>) -> Self {
        self.base_url = Some(val.into());
        self
    }
}

pub(crate) fn resolve_api_key(configured: Option<String>) -> Option<String> {
    configured.or_else(|| std::env::var("LITELLM_API_KEY").ok())
}

pub(crate) fn resolve_base_url(configured: Option<String>) -> String {
    configured
        .or_else(|| std::env::var("LITELLM_BASE_URL").ok())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string()
}
