"""Claude Agent SDK conversation manager with hierarchical memory integration.

This module provides a conversation manager that uses Claude Agent SDK
(which leverages Claude CLI authentication - Pro/Max subscription) instead
of direct API calls, while maintaining the hierarchical memory compression
system.
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from ..config import Config
from ..storage import DuckDBStorage
from ..compression import TfidfCompressor, CompressionManager
from ..advanced_hierarchy import AdvancedCompressionManager
from ..models import CompressionLevel, NodeType, HierarchyThresholds, ModelConfig

logger = logging.getLogger(__name__)

# Suppress verbose logging from Claude Agent SDK internals
logging.getLogger("claude_agent_sdk._internal").setLevel(logging.WARNING)
logging.getLogger("claude_agent_sdk._internal.transport").setLevel(logging.WARNING)


class ClaudeAgentSDKConversationManager:
    """Manages conversations using Claude Agent SDK with hierarchical memory compression.

    This manager uses the Claude Agent SDK which authenticates via Claude CLI,
    allowing users to use their Claude Pro/Max subscription instead of API credits.

    The hierarchical memory system compresses older conversation nodes to enable
    effectively infinite conversations within context limits.
    """

    def __init__(
        self,
        config: Config,
        model_config: ModelConfig,
        storage: Optional[DuckDBStorage] = None,
        allowed_tools: Optional[List[str]] = None,
        permission_mode: str = "default",
        enable_memory_tools: bool = True,
    ):
        """Initialize the Claude Agent SDK conversation manager.

        Args:
            config: Application configuration
            model_config: Model configuration from ModelManager
            storage: Optional storage instance (creates new if not provided)
            allowed_tools: List of allowed tools for the agent (default: memory tools only)
            permission_mode: Permission mode for tool execution
                - "default": Requires approval for each tool use
                - "acceptEdits": Auto-approve file edits
                - "bypassPermissions": No prompts (for automation)
            enable_memory_tools: Whether to enable memory tools via stdio subprocess (default: True)
        """
        self.config = config
        self.model_config = model_config
        self.conversation_id: Optional[str] = None
        self.permission_mode = permission_mode
        self.enable_memory_tools = enable_memory_tools

        config.setup_logging()

        # Initialize storage
        self.storage = storage or DuckDBStorage(config.db_path)

        # Initialize compression system
        self.compressor = TfidfCompressor(max_words=8)

        # Create hierarchy thresholds
        self.hierarchy_thresholds = HierarchyThresholds(
            summary_threshold=config.recent_node_limit,
            meta_threshold=50,
            archive_threshold=200,
            meta_group_size=20,
            meta_group_max=40,
        )

        # Initialize compression managers
        self.compression_manager = AdvancedCompressionManager(
            base_compressor=self.compressor, thresholds=self.hierarchy_thresholds
        )
        self.simple_compression_manager = CompressionManager(
            compressor=self.compressor, recent_node_limit=config.recent_node_limit
        )

        # Default allowed tools - minimal set for conversation
        self.allowed_tools = allowed_tools or []

        # Track tool calls and results
        self._current_tool_calls: List[Dict[str, Any]] = []
        self._current_tool_results: List[Dict[str, Any]] = []

        # Track AI view for debugging
        self._last_ai_view_data = None

        # Lazy import of Claude Agent SDK
        self._sdk_imported = False
        self._query = None
        self._ClaudeAgentOptions = None
        self._AssistantMessage = None
        self._ResultMessage = None
        self._TextBlock = None

        # Use subscription mode (clears API key to force OAuth)
        self.use_subscription = config.agent_use_subscription

        logger.debug(
            f"Initialized ClaudeAgentSDKConversationManager with model: {model_config.model_name}, "
            f"use_subscription: {self.use_subscription}"
        )

    def _build_agent_options(self, system_prompt: str) -> "ClaudeAgentOptions":
        """Build ClaudeAgentOptions with proper configuration.

        Args:
            system_prompt: The system prompt to use

        Returns:
            Configured ClaudeAgentOptions
        """
        self._ensure_sdk_imported()

        # Build environment overrides
        env = {}
        if self.use_subscription:
            # Clear API key to force CLI to use OAuth credentials from ~/.claude/.credentials.json
            # This allows using Claude Pro/Max subscription instead of API credits
            env["ANTHROPIC_API_KEY"] = ""
            logger.debug("Using subscription mode: cleared ANTHROPIC_API_KEY to use OAuth")

        # Build MCP servers configuration for memory tools
        # Use stdio transport with a minimal memory server subprocess
        mcp_servers = {}
        if self.enable_memory_tools and self.conversation_id:
            import sys

            mcp_servers["memory"] = {
                "command": sys.executable,
                "args": [
                    "-m",
                    "hierarchical_memory_middleware.mcp_server.stdio_memory_server",
                    "--conversation-id",
                    self.conversation_id,
                    "--db-path",
                    self.config.db_path,
                ],
                "env": env,  # Pass same env to subprocess
            }
            logger.debug(
                f"Configured memory tools via stdio subprocess for conversation {self.conversation_id}"
            )

        # Build allowed tools list including memory tools if enabled
        allowed_tools = list(self.allowed_tools) if self.allowed_tools else []
        if mcp_servers:
            # Add memory server tools (tool names from stdio_memory_server.py)
            memory_tools = [
                "mcp__memory__expand_node",
                "mcp__memory__search_memory",
                "mcp__memory__get_memory_stats",
                "mcp__memory__get_recent_nodes",
            ]
            allowed_tools.extend(memory_tools)

        return self._ClaudeAgentOptions(
            allowed_tools=allowed_tools if allowed_tools else None,
            permission_mode=self.permission_mode,
            system_prompt=system_prompt,
            model=self.model_config.model_name,
            env=env,
            mcp_servers=mcp_servers if mcp_servers else None,
        )

    def _ensure_sdk_imported(self) -> None:
        """Lazily import Claude Agent SDK modules."""
        if self._sdk_imported:
            return

        try:
            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                AssistantMessage,
                ResultMessage,
                TextBlock,
            )

            self._query = query
            self._ClaudeAgentOptions = ClaudeAgentOptions
            self._AssistantMessage = AssistantMessage
            self._ResultMessage = ResultMessage
            self._TextBlock = TextBlock
            self._sdk_imported = True
            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK is not installed. "
                "Install it with: pip install claude-agent-sdk\n"
                "Also ensure Claude CLI is installed and authenticated."
            ) from e

    async def start_conversation(
        self, conversation_id: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        """Start a new conversation or resume an existing one.

        Args:
            conversation_id: Optional ID to resume existing conversation
            name: Optional human-readable name for new conversations

        Returns:
            The conversation ID
        """
        if conversation_id:
            # Resume existing conversation
            self.conversation_id = conversation_id
            logger.info(f"Resumed conversation: {conversation_id}")
        else:
            # Create new conversation
            import uuid

            self.conversation_id = str(uuid.uuid4())
            # Ensure conversation exists in storage
            await self.storage._ensure_conversation_exists(self.conversation_id)
            if name:
                await self.storage.set_conversation_name(self.conversation_id, name)
            logger.info(f"Started new conversation: {self.conversation_id}")

        return self.conversation_id

    async def _build_memory_context(self) -> str:
        """Build the hierarchical memory context for the conversation.

        Returns:
            A string containing the compressed conversation history
        """
        if not self.conversation_id:
            return ""

        # Get all nodes for this conversation
        nodes = await self.storage.get_conversation_nodes(self.conversation_id)

        if not nodes:
            return ""

        # Build hierarchical view
        context_parts = []

        # Group nodes by compression level
        full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]
        summary_nodes = [n for n in nodes if n.level == CompressionLevel.SUMMARY]
        meta_nodes = [n for n in nodes if n.level == CompressionLevel.META]
        archive_nodes = [n for n in nodes if n.level == CompressionLevel.ARCHIVE]

        # Add archive context (highest level summary)
        if archive_nodes:
            context_parts.append("=== ARCHIVED CONTEXT ===")
            for node in archive_nodes:
                context_parts.append(f"[Archive] {node.summary or node.content[:200]}")

        # Add meta-level summaries
        if meta_nodes:
            context_parts.append("\n=== META SUMMARIES ===")
            for node in meta_nodes:
                topics = ", ".join(node.topics) if node.topics else "general"
                context_parts.append(f"[Meta: {topics}] {node.summary or node.content[:300]}")

        # Add summary-level nodes
        if summary_nodes:
            context_parts.append("\n=== CONVERSATION SUMMARIES ===")
            for node in summary_nodes[-10:]:  # Last 10 summaries
                role = "User" if node.node_type == NodeType.USER else "Assistant"
                content = node.summary or node.content[:200]
                context_parts.append(f"[{role}] {content}")

        # Add full recent nodes
        if full_nodes:
            context_parts.append("\n=== RECENT CONVERSATION ===")
            for node in full_nodes[-self.config.recent_node_limit :]:
                role = "User" if node.node_type == NodeType.USER else "Assistant"
                context_parts.append(f"[{role}] {node.content}")

        return "\n".join(context_parts)

    async def chat(self, user_message: str) -> str:
        """Main conversation interface with hierarchical memory integration.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response text
        """
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        self._ensure_sdk_imported()

        # Clear tool tracking
        self._current_tool_calls = []
        self._current_tool_results = []

        # Build memory context
        memory_context = await self._build_memory_context()

        # Build system prompt with memory context
        system_prompt = self._build_system_prompt(memory_context)

        # Create options for the agent (handles subscription mode)
        options = self._build_agent_options(system_prompt)

        # Collect response
        full_response = ""

        try:
            async for message in self._query(prompt=user_message, options=options):
                if isinstance(message, self._AssistantMessage):
                    for block in message.content:
                        if isinstance(block, self._TextBlock):
                            full_response += block.text
                elif isinstance(message, self._ResultMessage):
                    logger.debug(
                        f"Query completed: duration={message.duration_ms}ms, "
                        f"cost=${message.total_cost_usd}"
                    )
        except Exception as e:
            logger.exception(f"Error during Claude Agent SDK query: {e}")
            raise

        # Save the user message as a node
        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.USER,
            content=user_message,
        )

        # Save the AI response as a node
        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.AI,
            content=full_response,
            ai_components={
                "tool_calls": self._current_tool_calls,
                "tool_results": self._current_tool_results,
            },
        )

        # Check and perform compression if needed
        await self._check_and_compress()

        return full_response

    async def chat_stream(self, user_message: str) -> AsyncIterator[str]:
        """Streaming version of chat with hierarchical memory integration.

        Args:
            user_message: The user's message

        Yields:
            Chunks of the assistant's response as they arrive
        """
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        self._ensure_sdk_imported()

        # Clear tool tracking
        self._current_tool_calls = []
        self._current_tool_results = []

        # Build memory context
        memory_context = await self._build_memory_context()

        # Build system prompt with memory context
        system_prompt = self._build_system_prompt(memory_context)

        # Create options for the agent (handles subscription mode)
        options = self._build_agent_options(system_prompt)

        # Collect full response while streaming
        full_response = ""

        try:
            async for message in self._query(prompt=user_message, options=options):
                if isinstance(message, self._AssistantMessage):
                    for block in message.content:
                        if isinstance(block, self._TextBlock):
                            chunk = block.text
                            full_response += chunk
                            yield chunk
                elif isinstance(message, self._ResultMessage):
                    logger.debug(
                        f"Query completed: duration={message.duration_ms}ms, "
                        f"cost=${message.total_cost_usd}"
                    )
        except Exception as e:
            logger.exception(f"Error during Claude Agent SDK streaming query: {e}")
            raise

        # Save nodes after streaming completes
        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.USER,
            content=user_message,
        )

        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.AI,
            content=full_response,
            ai_components={
                "tool_calls": self._current_tool_calls,
                "tool_results": self._current_tool_results,
            },
        )

        # Check and perform compression
        await self._check_and_compress()

    def _build_system_prompt(self, memory_context: str) -> str:
        """Build the system prompt including memory context.

        Args:
            memory_context: The compressed conversation history

        Returns:
            Complete system prompt
        """
        base_prompt = """You are a helpful AI assistant with access to conversation memory.

You have access to the conversation history below, organized hierarchically:
- ARCHIVED CONTEXT: Very old conversations, highly compressed
- META SUMMARIES: Groups of related conversations, summarized by topic
- CONVERSATION SUMMARIES: Recent conversations in summary form
- RECENT CONVERSATION: The most recent exchanges in full detail

Use this context to maintain continuity and reference past discussions when relevant.
"""

        if memory_context:
            return f"{base_prompt}\n\n{memory_context}"
        return base_prompt

    async def _check_and_compress(self) -> None:
        """Check if compression is needed and perform it."""
        if not self.conversation_id:
            return

        nodes = await self.storage.get_conversation_nodes(self.conversation_id)
        full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]

        # Compress if we exceed the threshold
        if len(full_nodes) > self.hierarchy_thresholds.summary_threshold:
            await self._compress_old_nodes(full_nodes)

    async def _compress_old_nodes(self, full_nodes: list) -> None:
        """Compress older nodes to SUMMARY level.

        Args:
            full_nodes: List of nodes at FULL compression level
        """
        # Keep recent nodes, compress older ones
        nodes_to_compress = full_nodes[: -self.hierarchy_thresholds.summary_threshold]

        for node in nodes_to_compress:
            # Generate summary using TF-IDF compression
            compressed = self.compressor.compress_node(node)

            # Update node in storage using apply_compression_result
            await self.storage.apply_compression_result(
                conversation_id=self.conversation_id,
                compression_result=compressed,
            )

            logger.debug(f"Compressed node {node.node_id} to SUMMARY level")

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation.

        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_id:
            return {}

        nodes = await self.storage.get_conversation_nodes(self.conversation_id)

        stats = {
            "conversation_id": self.conversation_id,
            "total_nodes": len(nodes),
            "compression_levels": {
                "full": len([n for n in nodes if n.level == CompressionLevel.FULL]),
                "summary": len([n for n in nodes if n.level == CompressionLevel.SUMMARY]),
                "meta": len([n for n in nodes if n.level == CompressionLevel.META]),
                "archive": len([n for n in nodes if n.level == CompressionLevel.ARCHIVE]),
            },
            "user_messages": len([n for n in nodes if n.node_type == NodeType.USER]),
            "ai_messages": len([n for n in nodes if n.node_type == NodeType.AI]),
        }

        return stats

    async def search_memory(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search conversation memory.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching nodes with relevance info
        """
        if not self.conversation_id:
            return []

        results = await self.storage.search_nodes(
            conversation_id=self.conversation_id, query=query, limit=limit
        )

        return [
            {
                "node_id": r.node.node_id,
                "content": r.node.content[:500],
                "summary": r.node.summary,
                "relevance": r.relevance_score,
                "match_type": r.match_type,
                "compression_level": r.node.level.name,
            }
            for r in results
        ]

    async def expand_node(self, node_id: int) -> Optional[str]:
        """Expand a compressed node to get its full content.

        Args:
            node_id: ID of the node to expand

        Returns:
            Full content of the node, or None if not found
        """
        node = await self.storage.get_node(node_id)
        if node:
            return node.content
        return None

    async def close(self) -> None:
        """Clean up resources."""
        if self.storage:
            await self.storage.close()

    # ===================================================================
    # CLI COMPATIBILITY METHODS
    # These methods provide compatibility with the CLI interface
    # ===================================================================

    async def set_conversation_name(self, conversation_id: str, name: str) -> bool:
        """Set a name for the conversation.

        Args:
            conversation_id: The conversation ID
            name: The name to set

        Returns:
            True if successful
        """
        return await self.storage.set_conversation_name(conversation_id, name)

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation.

        Returns:
            Dictionary with conversation summary
        """
        if not self.conversation_id:
            return {}

        stats = await self.get_conversation_stats()
        nodes = await self.storage.get_conversation_nodes(self.conversation_id)

        # Get recent nodes for context
        recent_nodes = [n for n in nodes if n.level == CompressionLevel.FULL][-5:]

        return {
            **stats,
            "recent_messages": [
                {
                    "role": "user" if n.node_type == NodeType.USER else "assistant",
                    "content": n.content[:200] + "..." if len(n.content) > 200 else n.content,
                }
                for n in recent_nodes
            ],
        }

    async def find(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversation memory (alias for search_memory).

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching nodes
        """
        return await self.search_memory(query, limit)

    async def get_node_details(
        self, node_id: int, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node.

        Args:
            node_id: The node ID
            conversation_id: The conversation ID

        Returns:
            Node details or None if not found
        """
        node = await self.storage.get_node(node_id)
        if not node or node.conversation_id != conversation_id:
            return None

        return {
            "node_id": node.node_id,
            "conversation_id": node.conversation_id,
            "node_type": node.node_type.value,
            "content": node.content,
            "summary": node.summary,
            "timestamp": node.timestamp.isoformat() if node.timestamp else None,
            "sequence_number": node.sequence_number,
            "compression_level": node.level.name,
            "topics": node.topics,
            "line_count": node.line_count,
            "tokens_used": node.tokens_used,
        }

    async def remove_node(self, node_id: int, conversation_id: str) -> bool:
        """Remove a node from the conversation.

        Args:
            node_id: The node ID to remove
            conversation_id: The conversation ID

        Returns:
            True if successfully removed
        """
        return await self.storage.remove_node(node_id, conversation_id)
