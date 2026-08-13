mod compactor;
mod config;
mod engine;
#[cfg(feature = "axum")]
mod stream;
mod types;

pub use compactor::{HistoryCompactor, LlmSummaryCompactor};
pub use config::{AgentConfig, AgentConfigBuilder};
pub use engine::{AgentEngine, CompactionResult};
#[cfg(feature = "axum")]
pub use stream::to_sse_event;
pub use types::*;

use crate::provider::LlmProvider;
use std::sync::Arc;

pub fn get_agent_engine(provider: impl LlmProvider + 'static, config: AgentConfig) -> AgentEngine {
    AgentEngine::new(Arc::new(provider), config)
}
