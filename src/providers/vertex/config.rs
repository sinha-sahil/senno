use super::env::{resolve_optional, resolve_with_default};
use super::token::{ServiceAccountKey, TokenSource};
use crate::error::Error;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct VertexConfig {
    pub api_key: Option<String>,
    pub project_id: Option<String>,
    pub region: Option<String>,
    pub service_account_key: Option<ServiceAccountKey>,
    pub service_account_key_path: Option<String>,
    pub metadata_server: Option<String>,
    pub base_url: Option<String>,
}

impl VertexConfig {
    pub fn with_api_key(mut self, val: impl Into<String>) -> Self {
        self.api_key = Some(val.into());
        self
    }

    pub fn with_project_id(mut self, val: impl Into<String>) -> Self {
        self.project_id = Some(val.into());
        self
    }

    pub fn with_region(mut self, val: impl Into<String>) -> Self {
        self.region = Some(val.into());
        self
    }

    pub fn with_service_account_key(mut self, key: ServiceAccountKey) -> Self {
        self.service_account_key = Some(key);
        self
    }

    pub fn with_service_account_key_path(mut self, val: impl Into<String>) -> Self {
        self.service_account_key_path = Some(val.into());
        self
    }

    pub fn with_metadata_server(mut self) -> Self {
        self.metadata_server = Some(super::token::DEFAULT_METADATA_BASE_URL.to_string());
        self
    }

    pub fn with_metadata_server_url(mut self, val: impl Into<String>) -> Self {
        self.metadata_server = Some(val.into());
        self
    }

    pub fn with_base_url(mut self, val: impl Into<String>) -> Self {
        self.base_url = Some(val.into());
        self
    }
}

pub(crate) enum ResolvedAuth {
    ApiKey {
        api_key: String,
    },
    ServiceAccount {
        project_id: String,
        region: String,
        token_source: Arc<TokenSource>,
    },
}

pub(crate) async fn resolve_auth(
    config: Option<VertexConfig>,
    http: reqwest::Client,
) -> Result<ResolvedAuth, Error> {
    let config = config.unwrap_or_default();

    let configured_service_account = config.project_id.is_some()
        && (config.service_account_key.is_some()
            || config.service_account_key_path.is_some()
            || config.metadata_server.is_some());

    if let Some(api_key) = config.api_key {
        return Ok(ResolvedAuth::ApiKey { api_key });
    }

    if !configured_service_account
        && let Some(api_key) = resolve_optional::<String>(None, "VERTEX_API_KEY")
            .or_else(|| resolve_optional::<String>(None, "GEMINI_API_KEY"))
    {
        return Ok(ResolvedAuth::ApiKey { api_key });
    }

    let project_id = resolve_optional(config.project_id, "VERTEX_PROJECT_ID");

    if let Some(project_id) = project_id {
        let region =
            resolve_with_default(config.region, "VERTEX_REGION", "asia-south1".to_string());

        let token_source = match (
            config.service_account_key,
            config.service_account_key_path,
            config.metadata_server,
        ) {
            (Some(key), _, _) => TokenSource::new(http, key),
            (None, Some(path), _) => {
                TokenSource::new(http, ServiceAccountKey::from_path(&path).await?)
            }
            (None, None, Some(base_url)) => TokenSource::metadata(http, base_url),
            (None, None, None) => {
                return Err(Error::config(
                    "Vertex AI service-account auth requires service_account_key, service_account_key_path, or metadata_server",
                ));
            }
        };
        let token_source = Arc::new(token_source);

        return Ok(ResolvedAuth::ServiceAccount {
            project_id,
            region,
            token_source,
        });
    }

    Err(Error::config(
        "Vertex AI requires either VERTEX_API_KEY/GEMINI_API_KEY, or VERTEX_PROJECT_ID + service_account_key/service_account_key_path/metadata_server",
    ))
}

pub(crate) fn regional_host(region: &str) -> String {
    if region == "global" {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{region}-aiplatform.googleapis.com")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regional_host_prefixes_every_region_but_global() {
        assert_eq!(
            regional_host("asia-south1"),
            "asia-south1-aiplatform.googleapis.com"
        );
        assert_eq!(
            regional_host("us-central1"),
            "us-central1-aiplatform.googleapis.com"
        );
        assert_eq!(regional_host("global"), "aiplatform.googleapis.com");
    }
}
