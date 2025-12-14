#!/usr/bin/env python3
"""Minimal stdio MCP server for Claude Agent SDK integration.

This server is designed to be spawned as a subprocess by the Claude Agent SDK
and provides memory tools via stdio transport.

Usage:
    python -m hierarchical_memory_middleware.mcp_server.stdio_memory_server \
        --conversation-id <id> --db-path <path>
"""

import argparse
import logging
import sys
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP

# Suppress logging to stderr since stdio is used for communication
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/hmm_stdio_server.log")],
)
logger = logging.getLogger(__name__)


def create_memory_server(conversation_id: str, db_path: str) -> FastMCP:
    """Create a FastMCP server with memory tools.

    Args:
        conversation_id: The conversation ID to use for all operations
        db_path: Path to the DuckDB database

    Returns:
        Configured FastMCP server instance
    """
    # Import storage lazily to avoid startup overhead
    from ..storage import DuckDBStorage
    from ..models import CompressionLevel, NodeType

    # Initialize storage
    storage = DuckDBStorage(db_path)

    # Create FastMCP server
    mcp = FastMCP("memory-tools")

    @mcp.tool()
    async def expand_node(node_id: int) -> Dict[str, Any]:
        """Retrieve full content of a conversation node by its ID.

        Use this tool to see the complete original content of a compressed
        or summarized message from the conversation history.

        Args:
            node_id: The numeric node identifier

        Returns:
            Full node content including text, metadata, and timestamps
        """
        try:
            node = await storage.get_node(node_id, conversation_id)
            if not node:
                return {"error": f"Node {node_id} not found in this conversation"}

            return {
                "success": True,
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "content": node.content,
                "summary": node.summary,
                "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                "compression_level": node.level.name,
                "topics": node.topics or [],
                "line_count": node.line_count,
            }
        except Exception as e:
            logger.error(f"Error expanding node {node_id}: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def search_memory(query: str, limit: int = 10) -> Dict[str, Any]:
        """Search conversation history for specific content.

        Use this tool to find past messages or topics discussed earlier
        in the conversation.

        Args:
            query: Text to search for in the conversation history
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of matching messages with relevance scores
        """
        try:
            results = await storage.search_nodes(
                conversation_id=conversation_id,
                query=query,
                limit=limit,
            )

            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "node_id": r.node.node_id,
                        "content_preview": r.node.content[:300] if r.node.content else "",
                        "summary": r.node.summary,
                        "relevance": r.relevance_score,
                        "match_type": r.match_type,
                        "compression_level": r.node.level.name,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_memory_stats() -> Dict[str, Any]:
        """Get statistics about the conversation memory.

        Use this tool to understand how much conversation history is stored
        and how it's organized across compression levels.

        Returns:
            Statistics including message counts and compression distribution
        """
        try:
            nodes = await storage.get_conversation_nodes(conversation_id)

            # Count by compression level
            level_counts = {}
            for level in CompressionLevel:
                count = len([n for n in nodes if n.level == level])
                if count > 0:
                    level_counts[level.name] = count

            # Count by type
            user_count = len([n for n in nodes if n.node_type == NodeType.USER])
            ai_count = len([n for n in nodes if n.node_type == NodeType.AI])

            return {
                "success": True,
                "conversation_id": conversation_id,
                "total_messages": len(nodes),
                "user_messages": user_count,
                "ai_messages": ai_count,
                "compression_levels": level_counts,
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_recent_nodes(count: int = 10) -> Dict[str, Any]:
        """Get the most recent conversation messages.

        Use this tool to see recent messages in full detail.

        Args:
            count: Number of recent messages to retrieve (default: 10)

        Returns:
            List of recent messages with full content
        """
        try:
            nodes = await storage.get_conversation_nodes(conversation_id)

            # Filter to FULL level nodes and get most recent
            full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]
            recent = full_nodes[-count:] if len(full_nodes) > count else full_nodes

            return {
                "success": True,
                "count": len(recent),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type.value,
                        "content": n.content,
                        "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                    }
                    for n in recent
                ],
            }
        except Exception as e:
            logger.error(f"Error getting recent nodes: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_system_prompt() -> Dict[str, Any]:
        """Get the current system prompt / scratchpad for this conversation.

        Use this tool to read your persistent notes and behavioral preferences
        that you've saved for this conversation session.

        Returns:
            The current system prompt content, or empty if not set
        """
        try:
            content = await storage.get_system_prompt(conversation_id)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": content or "",
                "has_content": content is not None and len(content) > 0,
            }
        except Exception as e:
            logger.error(f"Error getting system prompt: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def set_system_prompt(content: str) -> Dict[str, Any]:
        """Set or replace the system prompt / scratchpad for this conversation.

        Use this tool to save persistent notes, behavioral preferences,
        or context that should persist across the conversation. This acts
        as a scratchpad where you can write notes to yourself.

        Examples of what to store:
        - User preferences: "User prefers concise responses"
        - Project context: "Working on TypeScript strict mode project"
        - Behavioral notes: "Remember: don't explain basics, user is senior dev"
        - Task tracking: "Currently debugging auth flow in middleware"

        Args:
            content: The new system prompt content (replaces existing)

        Returns:
            Success status and the new content
        """
        try:
            await storage.set_system_prompt(conversation_id, content)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": content,
                "action": "replaced",
            }
        except Exception as e:
            logger.error(f"Error setting system prompt: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def append_to_system_prompt(content: str) -> Dict[str, Any]:
        """Append content to the system prompt / scratchpad.

        Use this tool to add notes without replacing existing content.
        Useful for incrementally building up context or adding new
        observations throughout the conversation.

        Args:
            content: The content to append (will be added on a new line)

        Returns:
            Success status and the updated full content
        """
        try:
            updated = await storage.append_system_prompt(conversation_id, content)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "content": updated,
                "action": "appended",
            }
        except Exception as e:
            logger.error(f"Error appending to system prompt: {e}")
            return {"error": str(e)}

    return mcp


def main():
    """Main entry point for the stdio memory server."""
    parser = argparse.ArgumentParser(description="Stdio MCP memory server")
    parser.add_argument("--conversation-id", required=True, help="Conversation ID")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    logger.info(f"Starting stdio memory server for conversation: {args.conversation_id}")

    try:
        mcp = create_memory_server(args.conversation_id, args.db_path)
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
