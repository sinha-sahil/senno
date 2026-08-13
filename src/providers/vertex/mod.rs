mod anthropic;
mod client;
mod config;
mod env;
mod gemini;
mod token;

pub use client::{VertexClient, VertexProvider};
pub use config::VertexConfig;
pub use token::{DEFAULT_METADATA_BASE_URL, ServiceAccountKey, TokenSource};

use std::time::Duration;

use crate::error::Error;

pub async fn get_vertex_client(
    provider: VertexProvider,
    config: Option<VertexConfig>,
) -> Result<VertexClient, Error> {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| Error::internal(format!("Failed to build Vertex HTTP client: {e}")))?;
    let auth = config::resolve_auth(config, http.clone()).await?;
    Ok(VertexClient::new(http, auth, provider))
}
