use std::future::Future;
use std::pin::Pin;

use futures::StreamExt;
use senno::agent::{
    AgentConfig, AgentFlow, AgentSession, ToolOutput, ToolSource, get_agent_engine,
};
use senno::mcp::{McpServer, McpToolset};
use senno::providers::vertex::{VertexProvider, get_vertex_client};
use senno::{Error, ParameterSchema, ToolDefinition};
use serde_json::Value;

struct BookingFlow {
    mcp: McpToolset,
}

impl AgentFlow for BookingFlow {
    fn system_prompt(&self) -> String {
        "You book hotel rooms. Create a basket before adding anything to it.".to_string()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new("summarise_stay", "Summarise the booking for the guest.")
                .with_parameters(
                    ParameterSchema::object()
                        .with_property("basketId", ParameterSchema::string("Basket identifier"))
                        .with_required(["basketId"]),
                ),
        ]
    }

    fn tool_sources(&self) -> Vec<&dyn ToolSource> {
        vec![&self.mcp]
    }

    fn execute_tool<'a>(
        &'a self,
        name: &'a str,
        args: &'a Value,
        _session: &'a AgentSession,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            match name {
                "summarise_stay" => {
                    let basket = args["basketId"].as_str().unwrap_or_default();
                    Ok(ToolOutput::text(format!("Basket {basket}: 2 nights.")))
                }
                _ => Err(Error::tool(name, "this flow serves no such tool")),
            }
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mcp = McpToolset::connect(vec![
        McpServer::new("visit", "https://visit.example/mcp")
            .with_header("X-Service-Token", "vmcp_…")
            .with_tool_prefix("visit__"),
        McpServer::new("ledger", "https://ledger.example/mcp"),
    ])
    .await?;

    let flow = BookingFlow { mcp };

    let provider = get_vertex_client(VertexProvider::Gemini, None).await?;
    let engine = get_agent_engine(provider, AgentConfig::builder("gemini-2.5-flash").build()?);

    let mut session = AgentSession::new("session-1", "booking");
    let mut events = engine.run(&flow, &mut session, "Two nights in Gothenburg in May");
    while let Some(event) = events.next().await {
        println!("{event:?}");
    }
    drop(events);

    flow.mcp.close().await;
    Ok(())
}
