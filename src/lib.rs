//! # senno
//!
//! *senno* — a provider-agnostic LLM client and
//! streaming tool-calling agent engine, Vertex AI first, with built-in history
//! compaction.
//!
//! - Core (no default features): canonical LLM types, the [`LlmProvider`],
//!   [`EmbeddingProvider`], [`BatchGenerationProvider`], and
//!   [`BatchEmbeddingProvider`] traits, [`one_shot`], and the [`agent`] engine.
//! - `vertex` feature: `providers::vertex` — Gemini and Claude on Vertex AI
//!   with API-key or service-account auth, plus Gemini text embeddings and
//!   Gemini Batch API bulk generation and bulk embedding at half price.
//! - `batch` feature (implied by `vertex`): [`run_batch`] and
//!   [`run_embedding_batch`] — submit a batch job and poll it to completion.
//! - `axum` feature: `agent::to_sse_event` — adapt agent events to
//!   `axum::response::sse`.

pub mod agent;
#[cfg(feature = "batch")]
mod batch;
pub mod error;
mod one_shot;
mod provider;
pub mod providers;
mod types;

#[cfg(feature = "batch")]
pub use batch::{
    BatchPolling, run_batch, run_embedding_batch, wait_for_batch, wait_for_embedding_batch,
};
pub use error::Error;
pub use one_shot::one_shot;
pub use provider::{
    BatchEmbeddingProvider, BatchGenerationProvider, EmbeddingProvider, LlmProvider, LlmStream,
};
pub use types::*;
