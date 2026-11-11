//! Amazon Bedrock: chat through the Converse API (Claude, GPT and every other
//! Converse model) and Titan embeddings through InvokeModel.
//!
//! Credentials come from the AWS default chain, so on EKS the pod's IRSA role
//! is picked up with no key to ship. Calls go to the regional
//! `bedrock-runtime` endpoint, which resolves to a VPC endpoint when one with
//! private DNS exists.

mod client;
mod config;
mod converse;
mod document;
mod embed;
mod error;
mod stream;

pub use client::BedrockClient;
pub use config::{BedrockConfig, BedrockGuardrail};

use crate::error::Error;

/// Build a Bedrock client from `config`, falling back to the AWS default
/// chain (`AWS_REGION`, web identity / IRSA, instance role, ...) for anything
/// not set.
pub async fn get_bedrock_client(config: Option<BedrockConfig>) -> Result<BedrockClient, Error> {
    let config = config.unwrap_or_default();
    let sdk = config::resolve(&config).await?;
    Ok(BedrockClient::new(sdk, config.guardrail))
}
