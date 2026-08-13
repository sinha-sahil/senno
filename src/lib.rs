//! # senno
//!
//! *senno* (Italian: wits, judgment) — a provider-agnostic LLM client and
//! streaming tool-calling agent engine, Vertex AI first, with built-in history
//! compaction.
//!
//! - Core (no default features): canonical LLM types, the [`LlmProvider`]
//!   trait, [`one_shot`], and the [`agent`] engine.
//! - `vertex` feature: `providers::vertex` — Gemini and Claude on Vertex AI
//!   with API-key or service-account auth.
//! - `axum` feature: `agent::to_sse_event` — adapt agent events to
//!   `axum::response::sse`.

pub mod agent;
pub mod error;
mod one_shot;
mod provider;
pub mod providers;
mod types;

pub use error::Error;
pub use one_shot::one_shot;
pub use provider::{LlmProvider, LlmStream};
pub use types::*;
