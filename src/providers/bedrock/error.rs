use aws_sdk_bedrockruntime::error::{DisplayErrorContext, ProvideErrorMetadata, SdkError};

use crate::error::Error;

pub(super) const PROVIDER: &str = "bedrock";

/// Error codes worth retrying: capacity and transient server faults. The SDK
/// has already retried these with backoff by the time senno sees them, so a
/// caller retrying again is a deliberate, slower second attempt.
const RETRYABLE_CODES: &[&str] = &[
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
];

/// Map a Bedrock error code to a senno error. Unknown codes are permanent:
/// validation, access-denied and missing-model errors do not heal on retry.
pub(super) fn from_code(operation: &str, code: Option<&str>, message: Option<&str>) -> Error {
    let code_str = code.unwrap_or("Unknown");
    let detail = match message {
        Some(m) if !m.is_empty() => format!("{operation} {code_str}: {m}"),
        _ => format!("{operation} {code_str}"),
    };
    if code.is_some_and(|c| RETRYABLE_CODES.contains(&c)) {
        Error::provider(PROVIDER, detail)
    } else {
        Error::provider_permanent(PROVIDER, detail)
    }
}

pub(super) fn from_sdk<E, R>(operation: &str, err: SdkError<E, R>) -> Error
where
    E: ProvideErrorMetadata + std::error::Error + Send + Sync + 'static,
    R: std::fmt::Debug,
{
    match &err {
        SdkError::ServiceError(service) => {
            let e = service.err();
            from_code(operation, e.code(), e.message())
        }
        SdkError::ConstructionFailure(_) => Error::config(format!(
            "{operation}: request could not be built: {}",
            DisplayErrorContext(&err)
        )),
        // Timeouts, connection failures and unreadable responses are transient.
        _ => Error::provider(
            PROVIDER,
            format!("{operation}: {}", DisplayErrorContext(&err)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttling_is_retryable() {
        let e = from_code(
            "Converse",
            Some("ThrottlingException"),
            Some("Too many tokens"),
        );
        assert!(e.is_retryable());
        assert_eq!(
            e.to_string(),
            "bedrock: Converse ThrottlingException: Too many tokens"
        );
    }

    #[test]
    fn validation_and_access_denied_are_permanent() {
        for code in [
            "ValidationException",
            "AccessDeniedException",
            "ResourceNotFoundException",
        ] {
            let e = from_code("Converse", Some(code), None);
            assert!(e.is_provider_error());
            assert!(!e.is_retryable(), "{code} must not be retryable");
        }
    }

    #[test]
    fn missing_code_is_permanent() {
        assert!(!from_code("InvokeModel", None, None).is_retryable());
    }
}
