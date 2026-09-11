use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::{ToolOutput, ToolSource};
use crate::error::Error;
use crate::types::ToolDefinition;

use super::client::{McpClient, McpServer};

pub struct McpToolset {
    clients: Vec<Arc<McpClient>>,
    tools: Vec<ToolDefinition>,
}

impl fmt::Debug for McpToolset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let servers: Vec<&str> = self.clients.iter().map(|client| client.name()).collect();
        f.debug_struct("McpToolset")
            .field("servers", &servers)
            .field("tools", &self.tools.len())
            .finish()
    }
}

impl McpToolset {
    pub async fn connect(servers: Vec<McpServer>) -> Result<Self, Error> {
        let mut clients = Vec::with_capacity(servers.len());
        for server in servers {
            clients.push(Arc::new(McpClient::connect(server).await?));
        }
        Ok(Self::over(clients))
    }

    fn over(clients: Vec<Arc<McpClient>>) -> Self {
        let mut tools: Vec<ToolDefinition> = Vec::new();
        let mut listed: HashSet<&str> = HashSet::new();

        for client in &clients {
            for tool in client.tools() {
                if !listed.insert(tool.name.as_str()) {
                    tracing::warn!(
                        server = client.name(),
                        tool = %tool.name,
                        "mcp: duplicate tool name dropped",
                    );
                    continue;
                }
                tools.push(tool.clone());
            }
        }

        Self { clients, tools }
    }

    pub async fn close(&self) {
        for client in &self.clients {
            client.close().await;
        }
    }
}

impl ToolSource for McpToolset {
    fn definitions(&self) -> &[ToolDefinition] {
        &self.tools
    }

    fn handles(&self, name: &str) -> bool {
        self.clients.iter().any(|client| client.has_tool(name))
    }

    fn invoke<'a>(
        &'a self,
        name: &'a str,
        args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            let client = self
                .clients
                .iter()
                .find(|client| client.has_tool(name))
                .ok_or_else(|| Error::tool(name, "no registered MCP server serves this tool"))?;

            client.call(name, args).await
        })
    }
}
