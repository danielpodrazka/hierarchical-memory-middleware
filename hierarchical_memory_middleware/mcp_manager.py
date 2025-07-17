"""Simple manager for external MCP servers."""

import asyncio
import logging
import subprocess
import os
from typing import Dict, List, Optional
from pydantic_ai.mcp import MCPServerStreamableHTTP
from .config import ExternalMCPServer


class SimpleMCPManager:
    """Simple manager for external MCP servers."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.clients: Dict[str, MCPServerStreamableHTTP] = {}
        self.logger = logging.getLogger(__name__)

    async def start_server(self, name: str, config: ExternalMCPServer) -> Optional[MCPServerStreamableHTTP]:
        """Start an MCP server and return the client."""
        if not config.enabled:
            return None

        try:
            # Start the server process
            env = {**os.environ, **config.env}
            process = subprocess.Popen(
                [config.command] + config.args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.processes[name] = process

            # Wait a moment for server to start
            await asyncio.sleep(2)

            # Create the pydantic-ai client
            client = MCPServerStreamableHTTP(
                url=f"http://127.0.0.1:{config.port}/mcp",
                tool_prefix=config.tool_prefix
            )

            self.clients[name] = client
            self.logger.info(f"Started MCP server '{name}' on port {config.port}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to start server '{name}': {e}")
            return None

    def stop_all(self):
        """Stop all MCP servers."""
        for name, process in self.processes.items():
            try:
                process.terminate()
                self.logger.info(f"Stopped MCP server '{name}'")
            except Exception as e:
                self.logger.error(f"Error stopping server '{name}': {e}")

        self.processes.clear()
        self.clients.clear()

    def get_clients(self) -> List[MCPServerStreamableHTTP]:
        """Get all active MCP clients."""
        return list(self.clients.values())
