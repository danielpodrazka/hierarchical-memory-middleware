"""Claude Agent SDK conversation manager with hierarchical memory integration.

This module provides a conversation manager that uses Claude Agent SDK
(which leverages Claude CLI authentication - Pro/Max subscription) instead
of direct API calls, while maintaining the hierarchical memory compression
system.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator, Union

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    UserMessage,
)
from claude_agent_sdk.types import StreamEvent


@dataclass
class StreamChunk:
    """A text chunk from the stream."""
    text: str


@dataclass
class ToolCallEvent:
    """A tool call event."""
    tool_id: str
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class ToolResultEvent:
    """A tool result event."""
    tool_id: str
    content: str
    is_error: bool = False


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
        agentic_mode: bool = False,
    ):
        """Initialize the Claude Agent SDK conversation manager.

        Args:
            config: Application configuration
            model_config: Model configuration from ModelManager
            storage: Optional storage instance (creates new if not provided)
            allowed_tools: List of allowed tools for the agent (default: memory tools only)
            permission_mode: Permission mode for tool execution
                - "default": SDK handles permissions (some tools may be blocked)
                - "acceptEdits": Auto-approve file edits
                - "bypassPermissions": No prompts (for automation)
            enable_memory_tools: Whether to enable memory tools via stdio subprocess (default: True)
            agentic_mode: Whether agentic mode is enabled (auto-continue with yield_to_human)
        """
        self.config = config
        self.model_config = model_config
        self.conversation_id: Optional[str] = None
        self.permission_mode = permission_mode
        self.enable_memory_tools = enable_memory_tools
        self.agentic_mode = agentic_mode

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

        # Track current streaming state for partial save on interrupt
        self._current_user_message: Optional[str] = None
        self._stream_saved: bool = False

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
                "mcp__memory__get_system_prompt",
                "mcp__memory__set_system_prompt",
                "mcp__memory__append_to_system_prompt",
                "mcp__memory__yield_to_human",
            ]
            allowed_tools.extend(memory_tools)

        return ClaudeAgentOptions(
            allowed_tools=allowed_tools if allowed_tools else None,
            permission_mode=self.permission_mode,
            system_prompt=system_prompt,
            model=self.model_config.model_name,
            env=env,
            mcp_servers=mcp_servers if mcp_servers else None,
            output_format="stream-json",
            include_partial_messages=True,
        )

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

        # Clear tool tracking
        self._current_tool_calls = []
        self._current_tool_results = []

        # Build memory context
        memory_context = await self._build_memory_context()

        # Build system prompt with memory context and scratchpad
        system_prompt = await self._build_system_prompt(memory_context)

        # Create options for the agent (handles subscription mode)
        options = self._build_agent_options(system_prompt)

        # Collect response
        full_response = ""

        try:
            async for message in query(prompt=user_message, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            full_response += block.text
                elif isinstance(message, ResultMessage):
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

    async def chat_stream(
        self, user_message: str, include_tool_events: bool = False
    ) -> AsyncIterator[Union[str, StreamChunk, ToolCallEvent, ToolResultEvent]]:
        """Streaming version of chat with hierarchical memory integration.

        Args:
            user_message: The user's message
            include_tool_events: If True, yields ToolCallEvent and ToolResultEvent objects.
                                 If False (default), yields only text strings for backwards compatibility.

        Yields:
            Text chunks (str) or event objects (StreamChunk, ToolCallEvent, ToolResultEvent)
        """
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        # Clear tool tracking
        self._current_tool_calls = []
        self._current_tool_results = []

        # Track current user message for potential partial save
        self._current_user_message = user_message
        self._stream_saved = False

        # Build memory context
        memory_context = await self._build_memory_context()

        # Build system prompt with memory context and scratchpad
        system_prompt = await self._build_system_prompt(memory_context)

        # Create options for the agent (handles subscription mode)
        options = self._build_agent_options(system_prompt)

        # Collect full response while streaming
        full_response = ""

        try:
            async for message in query(prompt=user_message, options=options):
                # Handle streaming events (partial messages)
                if isinstance(message, StreamEvent):
                    event = message.event
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            chunk = delta.get("text", "")
                            if chunk:
                                full_response += chunk
                                yield chunk

                # Handle assistant messages with tool use
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Only yield text if we haven't been streaming it
                            if not full_response:
                                full_response += block.text
                                yield block.text
                        elif isinstance(block, ToolUseBlock) and include_tool_events:
                            # Track and yield tool call
                            tool_call = {
                                "tool_id": block.id,
                                "tool_name": block.name,
                                "tool_input": block.input,
                            }
                            self._current_tool_calls.append(tool_call)
                            yield ToolCallEvent(
                                tool_id=block.id,
                                tool_name=block.name,
                                tool_input=block.input,
                            )

                # Handle user messages with tool results
                elif isinstance(message, UserMessage) and include_tool_events:
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            # Track and yield tool result
                            tool_result = {
                                "tool_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                            self._current_tool_results.append(tool_result)
                            yield ToolResultEvent(
                                tool_id=block.tool_use_id,
                                content=block.content if isinstance(block.content, str) else str(block.content),
                                is_error=bool(block.is_error),
                            )

                elif isinstance(message, ResultMessage):
                    logger.debug(
                        f"Query completed: duration={message.duration_ms}ms, "
                        f"cost=${message.total_cost_usd}"
                    )
        except Exception as e:
            # Check if this is a SIGINT (Ctrl+C) interrupt - exit code -2
            error_str = str(e)
            if "exit code -2" in error_str or "exit code: -2" in error_str:
                # This is expected when user interrupts with Ctrl+C, don't log as error
                logger.debug(f"Streaming interrupted by user (SIGINT): {e}")
            else:
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

        # Mark stream as saved to prevent duplicate saves
        self._stream_saved = True

        # Check and perform compression
        await self._check_and_compress()

    async def save_partial_response(self, partial_response: str) -> None:
        """Save a partial response when streaming is interrupted.

        This method is called when the user interrupts a streaming response
        (e.g., with Ctrl+C) to save whatever partial response was collected.

        Args:
            partial_response: The partial response text collected so far
        """
        if self._stream_saved:
            # Already saved (stream completed normally)
            return

        if not self.conversation_id or not self._current_user_message:
            # No conversation or user message to save
            return

        logger.info(f"Saving partial response ({len(partial_response)} chars)")

        # Save the user message
        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.USER,
            content=self._current_user_message,
        )

        # Save the partial AI response (mark it as interrupted)
        partial_content = partial_response
        if partial_content:
            partial_content += "\n\n[Response interrupted by user]"
        else:
            partial_content = "[Response interrupted by user before any output]"

        await self.storage.save_conversation_node(
            conversation_id=self.conversation_id,
            node_type=NodeType.AI,
            content=partial_content,
            ai_components={
                "tool_calls": self._current_tool_calls,
                "tool_results": self._current_tool_results,
                "interrupted": True,
            },
        )

        # Mark as saved
        self._stream_saved = True

        # Clear tracking
        self._current_user_message = None

    async def _build_system_prompt(self, memory_context: str) -> str:
        """Build the system prompt including memory context and user scratchpad.

        Args:
            memory_context: The compressed conversation history

        Returns:
            Complete system prompt
        """
        base_prompt = """You are a helpful AI assistant with access to conversation memory.

You are running within a hierarchical memory system. Only recent messages are shown in full detail - older messages are progressively compressed into summaries, meta-summaries, and archived context. Use the memory tools (search_memory, expand_node, get_recent_nodes) when you need to recall specific details from earlier in the conversation.

The conversation history below is organized hierarchically:
- ARCHIVED CONTEXT: Very old conversations, highly compressed
- META SUMMARIES: Groups of related conversations, summarized by topic
- CONVERSATION SUMMARIES: Recent conversations in summary form
- RECENT CONVERSATION: The most recent exchanges in full detail

Use this context to maintain continuity and reference past discussions when relevant.
"""

        # Build the full system prompt
        parts = [base_prompt]

        # Add agentic mode instructions if enabled
        if self.agentic_mode:
            agentic_instructions = """
=== AGENTIC MODE ===
You are running in agentic mode. The system will automatically send "continue" messages to keep you working until you explicitly signal that you need human input.

**IMPORTANT: You MUST call the `yield_to_human` tool when:**
- You have completed the user's request
- You need clarification or a decision from the user
- You are blocked and need additional information
- You've reached a natural stopping point and want feedback

**Do NOT just say "let me know if you need anything else" - call yield_to_human instead.**

Example: After completing a task, call `yield_to_human(reason="Task complete - implemented the feature as requested")`

If you don't call yield_to_human, the system will automatically prompt you to continue working.
"""
            parts.append(agentic_instructions)

        # Add user's scratchpad/system prompt if set
        if self.conversation_id:
            scratchpad = await self.storage.get_system_prompt(self.conversation_id)
            if scratchpad:
                parts.append(f"\n=== YOUR SCRATCHPAD / NOTES ===\n{scratchpad}\n")

        # Add memory context
        if memory_context:
            parts.append(f"\n{memory_context}")

        return "".join(parts)

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
