"""MCP server for hierarchical memory browsing tools."""

import logging
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

from ..config import Config
from ..storage import DuckDBStorage
from ..middleware.conversation_manager import HierarchicalConversationManager


logger = logging.getLogger(__name__)


class MemoryMCPServer:
    """MCP server that provides read-only access to hierarchical memory system."""

    def __init__(self, config: Config):
        """Initialize the MCP server with storage and conversation manager."""
        self.config = config
        self.storage = DuckDBStorage(config.db_path)

        # Create FastMCP server instance
        self.mcp = FastMCP("hierarchical-memory-server")

        # Register MCP tools
        self._register_tools()

        # Start server on background port for internal use
        self.server_url = f"http://127.0.0.1:{config.mcp_port}"

        # Initialize conversation manager WITHOUT MCP tools (tools are provided by this server)
        self.conversation_manager = HierarchicalConversationManager(
            config, self.storage  # No MCP server URL - avoid circular dependency
        )

        logger.info(f"MemoryMCPServer initialized on {self.server_url}")

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.mcp.tool()
        async def expand_node(node_id: int, conversation_id: str) -> Dict[str, Any]:
            """Retrieve full content of a conversation node by composite ID.

            This tool allows expanding compressed/summarized nodes to see their
            full original content, including all details that may have been
            compressed away in the hierarchical memory system.

            Args:
                node_id: The node identifier within the conversation
                conversation_id: The conversation identifier

            Returns:
                Dictionary containing the full node details including content,
                metadata, timestamps, and any AI components if it's an AI node.
            """
            try:
                logger.info(f"Expanding node {node_id} in conversation {conversation_id}")
                result = await self.conversation_manager.get_node_details(node_id, conversation_id)

                if result is None:
                    return {
                        "error": f"Node {node_id} not found in conversation {conversation_id}",
                        "node_id": node_id,
                        "conversation_id": conversation_id
                    }

                return {
                    "success": True,
                    "node_id": result["node_id"],
                    "conversation_id": result["conversation_id"],
                    "node_type": result["node_type"],
                    "content": result["content"],
                    "summary": result["summary"],
                    "timestamp": result["timestamp"],
                    "sequence_number": result["sequence_number"],
                    "line_count": result["line_count"],
                    "compression_level": result["level"],
                    "tokens_used": result["tokens_used"],
                    "topics": result["topics"],
                    "ai_components": result["ai_components"],
                }

            except Exception as e:
                logger.error(f"Error expanding node {node_id} in conversation {conversation_id}: {str(e)}", exc_info=True)
                return {
                    "error": f"Failed to expand node {node_id} in conversation {conversation_id}: {str(e)}",
                    "node_id": node_id,
                    "conversation_id": conversation_id
                }

        @self.mcp.tool()
        async def search_memory(
            query: str, limit: int = 10
        ) -> Dict[str, Any]:
            """Search across conversation history for relevant nodes.

            This tool allows searching through all conversation nodes (both
            compressed and full) to find content relevant to a query.

            Args:
                query: The search query string
                limit: Maximum number of results to return (default: 10)

            Returns:
                Dictionary containing search results with relevance scores
                and node summaries.
            """
            try:
                logger.info(f"Searching memory for: {query[:50]}...")
                results = await self.conversation_manager.search_memory(query, limit)

                return {
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "results": results
                }

            except Exception as e:
                logger.error(f"Error searching memory: {str(e)}", exc_info=True)
                return {
                    "error": f"Failed to search memory: {str(e)}",
                    "query": query
                }

        @self.mcp.tool()
        async def get_conversation_stats() -> Dict[str, Any]:
            """Get overview statistics of the current conversation memory state.

            This tool provides a high-level view of the conversation's
            hierarchical memory state, including compression statistics
            and node counts at different levels.

            Returns:
                Dictionary containing conversation statistics and compression info.
            """
            try:
                logger.info("Getting conversation statistics")
                stats = await self.conversation_manager.get_conversation_summary()

                return {
                    "success": True,
                    **stats
                }

            except Exception as e:
                logger.error(f"Error getting conversation stats: {str(e)}", exc_info=True)
                return {
                    "error": f"Failed to get conversation stats: {str(e)}"
                }

        @self.mcp.tool()
        async def get_recent_nodes(limit: int = 10) -> Dict[str, Any]:
            """Get the most recent conversation nodes.

            This tool retrieves the latest nodes from the current conversation,
            useful for getting context about recent interactions.

            Args:
                limit: Maximum number of recent nodes to return (default: 10)

            Returns:
                Dictionary containing the most recent conversation nodes.
            """
            try:
                logger.info(f"Getting {limit} recent nodes")

                if not self.conversation_manager.conversation_id:
                    return {
                        "error": "No active conversation",
                        "recent_nodes": []
                    }

                # Get recent nodes from storage
                recent_nodes = await self.storage.get_recent_nodes(
                    conversation_id=self.conversation_manager.conversation_id,
                    limit=limit
                )

                nodes_data = []
                for node in recent_nodes:
                    nodes_data.append({
                        "node_id": node.node_id,
                        "conversation_id": node.conversation_id,
                        "node_type": node.node_type.value,
                        "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                        "summary": node.summary,
                        "timestamp": node.timestamp.isoformat(),
                        "sequence_number": node.sequence_number,
                        "compression_level": node.level.name,
                        "line_count": node.line_count
                    })

                return {
                    "success": True,
                    "conversation_id": self.conversation_manager.conversation_id,
                    "nodes_count": len(nodes_data),
                    "recent_nodes": nodes_data
                }

            except Exception as e:
                logger.error(f"Error getting recent nodes: {str(e)}", exc_info=True)
                return {
                    "error": f"Failed to get recent nodes: {str(e)}"
                }

    async def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Start or resume a conversation for the MCP server."""
        return await self.conversation_manager.start_conversation(conversation_id)

    async def start_background_server(self):
        """Start MCP server in background for internal agent use."""
        import asyncio
        import aiohttp

        try:
            # Start server in background task with streamable-http transport
            logger.info(f"Starting background MCP server on port {self.config.mcp_port}")
            task = asyncio.create_task(
                self.mcp.run_async(transport="streamable-http", host="127.0.0.1", port=self.config.mcp_port)
            )

            logger.info(f"Background MCP server started on {self.server_url}")
            return task

        except Exception as e:
            logger.error(f"Failed to start background MCP server: {e}")
            raise RuntimeError(f"Failed to start MCP server: {e}")

    def run(self, transport: str = "streamable-http", host: str = "127.0.0.1", port: int = 8000):
        """Run the MCP server with specified transport."""
        logger.info(f"Starting MCP server on {transport}://{host}:{port}")
        self.mcp.run(transport=transport, host=host, port=port)

    async def run_async(self, transport: str = "streamable-http", host: str = "127.0.0.1", port: int = 8000):
        """Run the MCP server asynchronously with specified transport."""
        logger.info(f"Starting MCP server on {transport}://{host}:{port}")
        await self.mcp.run_async(transport=transport, host=host, port=port)


# Create convenience function for quick server creation
async def create_memory_server(config: Config) -> MemoryMCPServer:
    """Create and initialize a memory MCP server."""
    server = MemoryMCPServer(config)
    return server
