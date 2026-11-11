use aws_config::retry::RetryConfig;
use aws_config::{BehaviorVersion, Region, SdkConfig};

use crate::error::Error;

/// Region env var consulted before the AWS default chain, so a host can point
/// Bedrock at a different region than its other AWS clients.
const REGION_ENV: &str = "BEDROCK_REGION";

/// Attempts per call, the first included. The SDK's standard retry mode backs
/// off on throttling and transient server errors.
const DEFAULT_MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Default)]
pub struct BedrockConfig {
    /// AWS region, e.g. `eu-central-1`. Falls back to `BEDROCK_REGION`, then
    /// the AWS default chain (`AWS_REGION`, profile).
    pub region: Option<String>,
    /// Replace the regional endpoint, e.g. for a VPC endpoint without private
    /// DNS or a test stub.
    pub endpoint_url: Option<String>,
    /// Attempts per call including the first; default 3.
    pub max_attempts: Option<u32>,
    /// Apply this Bedrock Guardrail to every chat call.
    pub guardrail: Option<BedrockGuardrail>,
    /// A fully built AWS config to use instead of the default chain (custom
    /// credentials provider, HTTP client, ...). `region`, `endpoint_url` and
    /// `max_attempts` still override it.
    pub sdk_config: Option<SdkConfig>,
}

/// A Bedrock Guardrail applied to every Converse call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BedrockGuardrail {
    /// Guardrail id or ARN.
    pub id: String,
    /// Published version number, or `DRAFT`.
    pub version: String,
    /// Ask Bedrock to return the guardrail trace (for debugging).
    pub trace: bool,
}

impl BedrockGuardrail {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            trace: false,
        }
    }

    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }
}

impl BedrockConfig {
    pub fn with_region(mut self, val: impl Into<String>) -> Self {
        self.region = Some(val.into());
        self
    }

    pub fn with_endpoint_url(mut self, val: impl Into<String>) -> Self {
        self.endpoint_url = Some(val.into());
        self
    }

    pub fn with_max_attempts(mut self, val: u32) -> Self {
        self.max_attempts = Some(val);
        self
    }

    pub fn with_guardrail(mut self, guardrail: BedrockGuardrail) -> Self {
        self.guardrail = Some(guardrail);
        self
    }

    pub fn with_sdk_config(mut self, sdk_config: SdkConfig) -> Self {
        self.sdk_config = Some(sdk_config);
        self
    }
}

pub(super) async fn resolve(
    config: &BedrockConfig,
) -> Result<aws_sdk_bedrockruntime::Config, Error> {
    let explicit_region = config.region.clone().or_else(|| {
        std::env::var(REGION_ENV)
            .ok()
            .filter(|r| !r.trim().is_empty())
    });

    let sdk = match &config.sdk_config {
        Some(sdk) => sdk.clone(),
        None => {
            let mut loader = aws_config::defaults(BehaviorVersion::latest());
            if let Some(region) = &explicit_region {
                loader = loader.region(Region::new(region.clone()));
            }
            loader.load().await
        }
    };

    let has_region = explicit_region.is_some() || sdk.region().is_some();
    let mut builder = aws_sdk_bedrockruntime::config::Builder::from(&sdk);
    if let Some(region) = explicit_region {
        builder = builder.region(Region::new(region));
    }
    if !has_region {
        return Err(Error::config(
            "Bedrock needs a region: set BedrockConfig::with_region, BEDROCK_REGION or AWS_REGION",
        ));
    }
    if let Some(url) = &config.endpoint_url {
        builder = builder.endpoint_url(url.trim_end_matches('/'));
    }
    let attempts = config.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS).max(1);
    builder = builder.retry_config(RetryConfig::standard().with_max_attempts(attempts));

    Ok(builder.build())
}
