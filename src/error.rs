/// The one error type senno returns. Hosts convert at the boundary
/// (e.g. `impl From<senno::Error> for AppError`).
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// An upstream service (model API, token endpoint) failed.
    #[error("{provider}: {detail}")]
    Provider {
        provider: String,
        detail: String,
        retryable: bool,
    },

    /// The model's output could not be used (no tool call, malformed args).
    #[error("bad response: {0}")]
    BadResponse(String),

    /// Local misconfiguration (missing key, bad env, invalid builder input).
    #[error("config: {0}")]
    Config(String),

    /// A tool invoked by an agent flow failed.
    #[error("tool {name}: {detail}")]
    Tool { name: String, detail: String },

    /// Anything else.
    #[error("{0}")]
    Internal(String),
}

impl Error {
    pub fn provider(provider: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            detail: detail.into(),
            retryable: true,
        }
    }

    pub fn provider_permanent(provider: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            detail: detail.into(),
            retryable: false,
        }
    }

    pub fn bad_response(detail: impl Into<String>) -> Self {
        Self::BadResponse(detail.into())
    }

    pub fn config(detail: impl Into<String>) -> Self {
        Self::Config(detail.into())
    }

    pub fn tool(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Tool {
            name: name.into(),
            detail: detail.into(),
        }
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::Internal(detail.into())
    }

    pub fn is_provider_error(&self) -> bool {
        matches!(self, Self::Provider { .. })
    }

    pub fn is_bad_response(&self) -> bool {
        matches!(self, Self::BadResponse(_))
    }

    /// Whether retrying the same call may succeed. Only transient upstream
    /// failures qualify.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Provider {
                retryable: true,
                ..
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_display_keeps_upstream_and_detail() {
        let e = Error::provider("vertex-ai", "API error (500): boom");
        assert_eq!(e.to_string(), "vertex-ai: API error (500): boom");
        assert!(e.is_provider_error());
        assert!(e.is_retryable());
    }

    #[test]
    fn permanent_provider_error_is_not_retryable() {
        let e = Error::provider_permanent("gcp-oauth2", "invalid_grant");
        assert!(e.is_provider_error());
        assert!(!e.is_retryable());
    }

    #[test]
    fn bad_response_is_flagged_and_not_retryable() {
        let e = Error::bad_response("model returned no tool call");
        assert!(e.is_bad_response());
        assert!(!e.is_retryable());
    }
}
