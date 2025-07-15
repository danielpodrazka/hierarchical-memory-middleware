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

        # State to track current conversation_id for tools
        self.current_conversation_id: Optional[str] = None

        # Create FastMCP server instance
        self.mcp = FastMCP("hierarchical-memory-server")

        # Register MCP tools
        self._register_tools()

        # Start server on background port for internal use
        self.server_url = f"http://127.0.0.1:{config.mcp_port}"

        # Initialize conversation manager WITHOUT MCP tools (tools are provided by this server)
        self.conversation_manager = HierarchicalConversationManager(
            config,
            self.storage,  # No MCP server URL - avoid circular dependency
        )

        logger.info(f"MemoryMCPServer initialized on {self.server_url}")

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.mcp.tool()
        async def set_conversation_id(conversation_id: str) -> Dict[str, Any]:
            """Set the conversation ID that will be used by other tools.

            This tool must be called first to establish the conversation context
            before using other tools like expand_node etc.

            Args:
                conversation_id: The conversation identifier to use for subsequent tool calls

            Returns:
                Dictionary confirming the conversation ID has been set.
            """
            try:
                logger.info(f"Setting conversation ID to: {conversation_id}")
                self.current_conversation_id = conversation_id

                # Also set the conversation_id on the conversation manager
                # so that its methods work with the correct conversation
                self.conversation_manager.conversation_id = conversation_id

                return {
                    "success": True,
                    "message": f"Conversation ID set to: {conversation_id}",
                    "conversation_id": conversation_id,
                }

            except Exception as e:
                logger.error(f"Error setting conversation ID: {str(e)}", exc_info=True)
                return {"error": f"Failed to set conversation ID: {str(e)}"}

        @self.mcp.tool()
        async def expand_node(node_id: int) -> Dict[str, Any]:
            """Retrieve full content of a conversation node by composite ID.

            This tool allows expanding compressed/summarized nodes to see their
            full original content, including all details that may have been
            compressed away in the hierarchical memory system.

            Note: The conversation_id must be set first using set_conversation_id tool.

            Args:
                node_id: The node identifier within the conversation

            Returns:
                Dictionary containing the full node details including content,
                metadata, timestamps, and any AI components if it's an AI node.
            """
            try:
                # Check if conversation_id has been set
                if self.current_conversation_id is None:
                    return {
                        "error": "No conversation ID set. Please call set_conversation_id first.",
                        "node_id": node_id,
                    }

                logger.info(
                    f"Expanding node {node_id} in conversation {self.current_conversation_id}"
                )
                result = await self.conversation_manager.get_node_details(
                    node_id, self.current_conversation_id
                )

                if result is None:
                    return {
                        "error": f"Node {node_id} not found in conversation {self.current_conversation_id}",
                        "node_id": node_id,
                        "conversation_id": self.current_conversation_id,
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
                logger.error(
                    f"Error expanding node {node_id} in conversation {self.current_conversation_id}: {str(e)}",
                    exc_info=True,
                )
                return {
                    "error": f"Failed to expand node {node_id} in conversation {self.current_conversation_id}: {str(e)}",
                    "node_id": node_id,
                    "conversation_id": self.current_conversation_id,
                }

        @self.mcp.tool()
        async def find(query: str, limit: int = 10, regex: bool = False) -> Dict[str, Any]:
            """Full text search for exact matches or regex matches.

            Note: The conversation_id must be set first using set_conversation_id tool.

            Args:
                query: The search query string
                limit: Maximum number of results to return (default: 10)
                regex: Whether to treat query as regex pattern (default: False)

            Returns:
                Dictionary containing search results with relevance scores
                and node summaries.
            """
            try:
                if self.current_conversation_id is None:
                    return {
                        "error": "No conversation ID set. Please call set_conversation_id first.",
                        "query": query,
                    }

                search_type = "regex" if regex else "exact"
                logger.info(f"Searching memory ({search_type}) for: {query[:50]}...")
                results = await self.conversation_manager.find(query, limit, regex)

                return {
                    "success": True,
                    "query": query,
                    "search_type": search_type,
                    "results_count": len(results),
                    "results": results,
                }

            except Exception as e:
                logger.error(f"Error searching memory: {str(e)}", exc_info=True)
                return {"error": f"Failed to search memory: {str(e)}", "query": query}

        @self.mcp.tool()
        async def get_conversation_stats() -> Dict[str, Any]:
            """Get overview statistics of the current conversation memory state.

            This tool provides a high-level view of the conversation's
            hierarchical memory state, including compression statistics
            and node counts at different levels.

            Note: The conversation_id must be set first using set_conversation_id tool.

            Returns:
                Dictionary containing conversation statistics and compression info.
            """
            try:
                if self.current_conversation_id is None:
                    return {
                        "error": "No conversation ID set. Please call set_conversation_id first."
                    }

                logger.info("Getting conversation statistics")
                stats = await self.conversation_manager.get_conversation_summary()

                return {"success": True, **stats}

            except Exception as e:
                logger.error(
                    f"Error getting conversation stats: {str(e)}", exc_info=True
                )
                return {"error": f"Failed to get conversation stats: {str(e)}"}

    async def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Start or resume a conversation for the MCP server.

        This method automatically sets the conversation_id state, so tools
        can be used immediately after calling this method.
        """
        # Start/resume the conversation
        result_conversation_id = await self.conversation_manager.start_conversation(
            conversation_id
        )

        # Automatically set the conversation_id state for tools
        self.current_conversation_id = result_conversation_id
        self.conversation_manager.conversation_id = result_conversation_id

        logger.info(
            f"Started conversation and set conversation_id to: {result_conversation_id}"
        )

        return result_conversation_id

    def run(
        self,
        transport: str = "streamable-http",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the MCP server with specified transport."""
        logger.info(f"Starting MCP server on {transport}://{host}:{port}")
        self.mcp.run(transport=transport, host=host, port=port)

    async def run_async(
        self,
        transport: str = "streamable-http",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """Run the MCP server asynchronously with specified transport."""
        logger.info(f"Starting MCP server on {transport}://{host}:{port}")
        await self.mcp.run_async(transport=transport, host=host, port=port)


# Create convenience function for quick server creation
async def create_memory_server(config: Config) -> MemoryMCPServer:
    """Create and initialize a memory MCP server."""
    server = MemoryMCPServer(config)
    return server
