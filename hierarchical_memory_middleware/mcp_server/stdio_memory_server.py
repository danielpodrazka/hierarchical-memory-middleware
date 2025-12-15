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
from typing import Dict, Any

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
    async def search_memory(
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """Search conversation history for specific content.

        Use this tool to find past messages or topics discussed earlier
        in the conversation.

        Args:
            query: Text to search for in the conversation history
            limit: Maximum number of results to return (default: 10)
            mode: Search mode - "keyword" (exact match), "semantic" (meaning-based),
                  or "hybrid" (combines both, default)

        Returns:
            List of matching messages with relevance scores
        """
        try:
            # Choose search method based on mode
            if mode == "semantic":
                results = await storage.search_nodes_semantic(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )
            elif mode == "hybrid":
                results = await storage.search_nodes_hybrid(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )
            else:  # keyword (default fallback)
                results = await storage.search_nodes(
                    conversation_id=conversation_id,
                    query=query,
                    limit=limit,
                )

            return {
                "success": True,
                "query": query,
                "mode": mode,
                "count": len(results),
                "results": [
                    {
                        "node_id": r.node.node_id,
                        "content_preview": r.node.content[:300]
                        if r.node.content
                        else "",
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

    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token for Claude models)."""
        if not text:
            return 0
        return len(text) // 4

    @mcp.tool()
    async def get_memory_stats() -> Dict[str, Any]:
        """Get statistics about the conversation memory.

        Use this tool to understand how much conversation history is stored
        and how it's organized across compression levels.

        Returns:
            Statistics including message counts, compression distribution, and token counts
        """
        try:
            nodes = await storage.get_conversation_nodes(conversation_id)

            # Group nodes by compression level
            nodes_by_level = {level: [] for level in CompressionLevel}
            for n in nodes:
                nodes_by_level[n.level].append(n)

            # Calculate token stats for each level
            def get_context_text(node):
                """Get the text that would actually be used in context."""
                if node.level == CompressionLevel.FULL:
                    return node.content
                elif node.summary:
                    return node.summary
                else:
                    return node.content

            level_counts = {}
            token_stats_by_level = {}
            total_current_tokens = 0
            total_original_tokens = 0

            for level in CompressionLevel:
                level_nodes = nodes_by_level[level]
                count = len(level_nodes)
                if count > 0:
                    level_counts[level.name] = count

                    current_tokens = sum(_estimate_tokens(get_context_text(n)) for n in level_nodes)
                    original_tokens = sum(_estimate_tokens(n.content) for n in level_nodes)

                    token_stats_by_level[level.name] = {
                        "count": count,
                        "current_tokens": current_tokens,
                        "original_tokens": original_tokens,
                        "compression_ratio": round(original_tokens / current_tokens, 2) if current_tokens > 0 else 0,
                    }
                    total_current_tokens += current_tokens
                    total_original_tokens += original_tokens

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
                "token_stats": {
                    "by_level": token_stats_by_level,
                    "total_current_tokens": total_current_tokens,
                    "total_original_tokens": total_original_tokens,
                    "overall_compression_ratio": round(total_original_tokens / total_current_tokens, 2) if total_current_tokens > 0 else 0,
                    "tokens_saved": total_original_tokens - total_current_tokens,
                    "tokens_saved_percent": round((1 - total_current_tokens / total_original_tokens) * 100, 1) if total_original_tokens > 0 else 0,
                },
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

    @mcp.tool()
    async def backfill_embeddings() -> Dict[str, Any]:
        """Generate embeddings for all messages that don't have them yet.

        This enables semantic search by creating vector embeddings for
        conversation history. Run this once to enable semantic search
        on existing conversations.

        Note: Requires the embeddings optional dependency to be installed:
        pip install 'hierarchical-memory-middleware[embeddings]'

        Returns:
            Number of messages updated with embeddings
        """
        try:
            updated_count = await storage.backfill_embeddings(conversation_id)
            return {
                "success": True,
                "conversation_id": conversation_id,
                "embeddings_generated": updated_count,
                "message": f"Generated embeddings for {updated_count} messages",
            }
        except Exception as e:
            logger.error(f"Error backfilling embeddings: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def yield_to_human(reason: str = "Task complete") -> Dict[str, Any]:
        """Signal that you need human input or have completed the current task.

        Use this tool in agentic mode when:
        - You've finished a multi-step task and want human review
        - You need clarification or a decision from the user
        - You're blocked and need additional information
        - You've reached a natural stopping point

        The chat interface will detect this tool call and pause for human input
        instead of auto-continuing.

        Args:
            reason: Brief explanation of why you're yielding (e.g., "Task complete",
                   "Need clarification on X", "Blocked on Y")

        Returns:
            Confirmation that the yield signal was sent
        """
        # This tool is a signal - the actual pausing logic is in the chat loop
        return {
            "success": True,
            "action": "yield_to_human",
            "reason": reason,
            "message": f"Yielding to human: {reason}",
        }

    return mcp


def main():
    """Main entry point for the stdio memory server."""
    parser = argparse.ArgumentParser(description="Stdio MCP memory server")
    parser.add_argument("--conversation-id", required=True, help="Conversation ID")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    logger.info(
        f"Starting stdio memory server for conversation: {args.conversation_id}"
    )

    try:
        mcp = create_memory_server(args.conversation_id, args.db_path)
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
