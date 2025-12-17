"""Claude Agent SDK conversation manager with hierarchical memory integration.

This module provides a conversation manager that uses Claude Agent SDK
(which leverages Claude CLI authentication - Pro/Max subscription) instead
of direct API calls, while maintaining the hierarchical memory compression
system.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator, Union

# Suppress verbose logging from Claude Agent SDK internals BEFORE importing the SDK
# Using CRITICAL to suppress ERROR messages from SIGINT handling (expected during Ctrl+C)
# This must be done before the SDK import to prevent the loggers from being created with default settings
logging.getLogger("claude_agent_sdk").setLevel(logging.CRITICAL)
logging.getLogger("claude_agent_sdk._internal").setLevel(logging.CRITICAL)
logging.getLogger("claude_agent_sdk._internal.transport").setLevel(logging.CRITICAL)
logging.getLogger("claude_agent_sdk._internal.query").setLevel(logging.CRITICAL)

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


# Tools that typically involve file paths (for context tracking)
FILE_PATH_TOOLS = {
    "Read": ["file_path"],
    "Write": ["file_path"],
    "Edit": ["file_path"],
    "Glob": ["path"],
    "Grep": ["path"],
    "NotebookEdit": ["notebook_path"],
}


def _extract_file_paths_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract file paths from tool calls, grouped by action type.

    Returns a dict like:
    {
        "read": ["/path/to/file1.py", "/path/to/file2.py"],
        "edited": ["/path/to/file3.py"],
        "searched": ["/path/to/dir"],
    }
    """
    paths: Dict[str, List[str]] = {
        "read": [],
        "edited": [],
        "searched": [],
    }

    for call in tool_calls:
        tool_name = call.get("tool_name", "")
        args = call.get("args") or call.get("tool_input") or {}

        if tool_name == "Read":
            file_path = args.get("file_path")
            if file_path and file_path not in paths["read"]:
                paths["read"].append(file_path)
        elif tool_name in ("Write", "Edit", "NotebookEdit"):
            file_path = args.get("file_path") or args.get("notebook_path")
            if file_path and file_path not in paths["edited"]:
                paths["edited"].append(file_path)
        elif tool_name in ("Glob", "Grep"):
            path = args.get("path")
            if path and path not in paths["searched"]:
                paths["searched"].append(path)

    # Remove empty categories
    return {k: v for k, v in paths.items() if v}


@dataclass
class StreamChunk:
    """A text chunk from the stream."""
    text: str


@dataclass
class ToolCallStartEvent:
    """Emitted when a tool call starts being generated (before full input is ready)."""
    tool_id: str
    tool_name: str


@dataclass
class ToolCallEvent:
    """A tool call event (emitted when full tool input is ready)."""
    tool_id: str
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class ToolResultEvent:
    """A tool result event."""
    tool_id: str
    content: str
    is_error: bool = False


@dataclass
class UsageEvent:
    """Token usage event emitted at the end of a query."""
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    total_tokens: int
    cost_usd: float
    duration_ms: int
    model: str


from ..config import Config
from ..storage import DuckDBStorage
from ..compression import TfidfCompressor, CompressionManager
from ..advanced_hierarchy import AdvancedCompressionManager
from ..models import CompressionLevel, NodeType, HierarchyThresholds, ModelConfig, ConversationNode


@dataclass
class SlackMCPConfig:
    """Configuration for Slack MCP tools.

    When provided to the conversation manager, enables Slack-specific
    tools like get_slack_channel_history, get_slack_thread_replies, etc.
    """
    bot_token: str  # Slack bot OAuth token (xoxb-...)
    channel_id: str  # Current channel ID for default queries
    working_dir: Optional[str] = None  # Working directory for file downloads

logger = logging.getLogger(__name__)


class ClaudeAgentSDKConversationManager:
    """Manages conversations using Claude Agent SDK with hierarchical memory compression.

    This manager uses the Claude Agent SDK which authenticates via Claude CLI,
    allowing users to use their Claude Pro/Max subscription instead of API credits.

    The hierarchical memory system compresses older conversation nodes to enable
    effectively infinite conversations within context limits.

    Key Features:
        - Seamless integration with Claude Pro/Max subscriptions
        - Automatic context compression for long conversations
        - Multi-level memory hierarchy (FULL -> SUMMARY -> META -> ARCHIVE)
        - MCP tool support for memory operations
        - Streaming responses with tool call events
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
        custom_instructions: Optional[str] = None,
        slack_mcp_config: Optional[SlackMCPConfig] = None,
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
            custom_instructions: Optional custom instructions to append to the system prompt
            slack_mcp_config: Optional Slack configuration for enabling Slack MCP tools
        """
        self.config = config
        self.model_config = model_config
        self.conversation_id: Optional[str] = None
        self.permission_mode = permission_mode
        self.enable_memory_tools = enable_memory_tools
        self.agentic_mode = agentic_mode
        self.custom_instructions = custom_instructions
        self.slack_mcp_config = slack_mcp_config

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

        # HTML context view export settings
        self._html_export_dir: Optional[str] = None
        self._last_system_prompt: Optional[str] = None

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

            # Check if Slack MCP config is provided - use Slack-enabled server
            if self.slack_mcp_config:
                slack_args = [
                    "-m",
                    "hierarchical_memory_middleware.mcp_server.stdio_slack_memory_server",
                    "--conversation-id",
                    self.conversation_id,
                    "--db-path",
                    self.config.db_path,
                    "--slack-bot-token",
                    self.slack_mcp_config.bot_token,
                    "--slack-channel-id",
                    self.slack_mcp_config.channel_id,
                ]
                # Add working directory if specified
                if self.slack_mcp_config.working_dir:
                    slack_args.extend(["--working-dir", self.slack_mcp_config.working_dir])

                mcp_servers["memory"] = {
                    "command": sys.executable,
                    "args": slack_args,
                    "env": env,  # Pass same env to subprocess
                }
                logger.debug(
                    f"Configured Slack-enabled memory tools for conversation {self.conversation_id}, "
                    f"channel {self.slack_mcp_config.channel_id}"
                )
            else:
                # Standard memory server without Slack tools
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

            # Add Slack tools if Slack config is provided
            if self.slack_mcp_config:
                slack_tools = [
                    "mcp__memory__get_slack_channel_history",
                    "mcp__memory__get_slack_thread_replies",
                    "mcp__memory__search_slack_messages",
                    "mcp__memory__get_slack_user_info",
                ]
                allowed_tools.extend(slack_tools)

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
            # Resume existing conversation (or create if doesn't exist)
            self.conversation_id = conversation_id
            # Ensure conversation exists in storage (for foreign key constraints)
            await self.storage._ensure_conversation_exists(self.conversation_id)
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

    def _format_successful_tool_actions(self, ai_components: Optional[Dict[str, Any]]) -> str:
        """Format successful tool actions from an AI node for context.

        Filters out failed attempts (file not found, 0 results, errors) and
        returns a compact summary of what was successfully accessed/done.

        Args:
            ai_components: The ai_components dict from a ConversationNode

        Returns:
            A compact string summarizing successful tool actions, or empty string
        """
        if not ai_components:
            return ""

        tool_calls = ai_components.get("tool_calls", [])
        tool_results = ai_components.get("tool_results", [])

        if not tool_calls or not tool_results:
            return ""

        # Build a map of tool_call_id -> result
        results_by_id = {r.get("tool_call_id") or r.get("tool_id"): r for r in tool_results}

        # Track actions by type for deduplication
        # For file-based tools (Read, Edit, Write), track unique paths
        # For search tools (Grep, Glob), track unique patterns
        read_files = []
        edited_files = []
        wrote_files = []
        grep_results = []  # List of (pattern, result_summary)
        glob_results = []  # List of (pattern, file_count)
        bash_commands = []
        web_fetches = []
        web_searches = []
        other_tools = set()

        for call in tool_calls:
            call_id = call.get("tool_call_id") or call.get("tool_id")
            tool_name = call.get("tool_name", "unknown")
            args = call.get("args") or call.get("tool_input") or {}
            result = results_by_id.get(call_id, {})

            # Skip if explicitly marked as error
            if result.get("is_error"):
                continue

            content = result.get("content", "")
            if isinstance(content, list):
                # Handle list of content blocks
                content = " ".join(str(c) for c in content)
            content_str = str(content).lower()

            # Filter based on tool type and result content
            if tool_name == "Read":
                file_path = args.get("file_path", "")
                # Skip if file doesn't exist
                if "does not exist" in content_str or "file not found" in content_str:
                    continue
                # Skip if empty file
                if "empty contents" in content_str or not content.strip():
                    continue
                # Success - track unique file path
                if file_path not in read_files:
                    read_files.append(file_path)

            elif tool_name == "Grep":
                pattern = args.get("pattern", "")
                # Skip if no matches
                if "no matches found" in content_str or "found 0" in content_str:
                    continue
                if not content.strip():
                    continue
                # Extract file count if possible
                if "found" in content_str:
                    result_summary = content.split(chr(10))[0]
                else:
                    result_summary = "found matches"
                # Track unique patterns
                if not any(p == pattern for p, _ in grep_results):
                    grep_results.append((pattern, result_summary))

            elif tool_name == "Glob":
                pattern = args.get("pattern", "")
                # Skip if no matches
                if not content.strip() or "no matches" in content_str:
                    continue
                # Count files found
                file_count = len([l for l in content.split("\n") if l.strip()])
                # Track unique patterns
                if not any(p == pattern for p, _ in glob_results):
                    glob_results.append((pattern, file_count))

            elif tool_name == "Bash":
                command = args.get("command", "")[:50]  # Truncate long commands
                # Skip if command failed (common error indicators)
                if "error:" in content_str or "command not found" in content_str:
                    continue
                if "permission denied" in content_str:
                    continue
                # Skip if exited with non-zero (if we can detect it)
                if "exit code" in content_str and "exit code 0" not in content_str:
                    continue
                if command not in bash_commands:
                    bash_commands.append(command)

            elif tool_name == "Edit":
                file_path = args.get("file_path", "")
                # Skip if edit failed
                if "error" in content_str or "failed" in content_str:
                    continue
                if file_path not in edited_files:
                    edited_files.append(file_path)

            elif tool_name == "Write":
                file_path = args.get("file_path", "")
                if "error" in content_str or "failed" in content_str:
                    continue
                if file_path not in wrote_files:
                    wrote_files.append(file_path)

            elif tool_name == "WebFetch":
                url = args.get("url", "")[:50]
                if "error" in content_str or "failed" in content_str:
                    continue
                if url not in web_fetches:
                    web_fetches.append(url)

            elif tool_name == "WebSearch":
                query = args.get("query", "")[:30]
                if "error" in content_str or "no results" in content_str:
                    continue
                if query not in web_searches:
                    web_searches.append(query)

            # For other tools, include if not an error
            elif not result.get("is_error") and content.strip():
                # Generic: just note the tool was used
                other_tools.add(tool_name)

        # Build the final summary with deduped actions
        successful_actions = []

        if read_files:
            successful_actions.append(f"Read: {' | '.join(read_files)}")
        if edited_files:
            successful_actions.append(f"Edited: {' | '.join(edited_files)}")
        if wrote_files:
            successful_actions.append(f"Wrote: {' | '.join(wrote_files)}")
        for pattern, result_summary in grep_results:
            successful_actions.append(f"Grep '{pattern}': {result_summary}")
        for pattern, file_count in glob_results:
            successful_actions.append(f"Glob '{pattern}': {file_count} files")
        for command in bash_commands:
            successful_actions.append(f"Bash: {command}")
        for url in web_fetches:
            successful_actions.append(f"Fetched: {url}")
        for query in web_searches:
            successful_actions.append(f"Searched: '{query}'")
        for tool_name in sorted(other_tools):
            successful_actions.append(f"{tool_name}")

        if not successful_actions:
            return ""

        return " | ".join(successful_actions)

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

        # Calculate which nodes are close to compression
        # FULL nodes that will be compressed next (oldest ones above threshold)
        full_nodes_to_compress_count = max(0, len(full_nodes) - self.hierarchy_thresholds.summary_threshold)
        # SUMMARY nodes that will be grouped into META next
        summary_nodes_to_compress_count = max(0, len(summary_nodes) - self.hierarchy_thresholds.meta_threshold)
        # META nodes that will be archived next
        meta_nodes_to_archive_count = max(0, len(meta_nodes) - self.hierarchy_thresholds.archive_threshold)

        # Add archive context (highest level summary) - aggregated into batches
        if archive_nodes:
            context_parts.append("=== ARCHIVED CONTEXT ===")
            # Group archives into batches of ~20 to reduce token count
            batch_size = 20
            for i in range(0, len(archive_nodes), batch_size):
                batch = archive_nodes[i:i + batch_size]
                if not batch:
                    continue

                # Get node range
                first_node = batch[0]
                last_node = batch[-1]

                # Collect all topics from the batch
                all_topics = []
                total_lines = 0
                for node in batch:
                    if node.topics:
                        all_topics.extend(node.topics)
                    total_lines += node.line_count or 1

                # Get top 5 most common topics
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                top_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]
                topics_str = ", ".join(top_topics) if top_topics else "general"

                # Create compact batch summary
                context_parts.append(
                    f"Nodes {first_node.node_id}-{last_node.node_id}:"
                    f"({len(batch)} nodes, {total_lines} lines) [Topics: {topics_str}]"
                )

        # Add meta-level summaries
        if meta_nodes:
            context_parts.append("\n=== META SUMMARIES ===")
            for i, node in enumerate(meta_nodes):
                topics = ", ".join(node.topics) if node.topics else "general"
                # Add hint for nodes that will be archived next
                hint = " ⚠️ (will be archived next)" if i < meta_nodes_to_archive_count else ""
                context_parts.append(f"[Meta: {topics}] {node.summary or node.content[:300]}{hint}")

        # Add summary-level nodes
        if summary_nodes:
            context_parts.append("\n=== CONVERSATION SUMMARIES ===")
            displayed_summaries = summary_nodes[-10:]  # Last 10 summaries
            # Calculate which of the displayed summaries are close to META compression
            # The oldest summaries (first in the full list) are compressed first
            for node in displayed_summaries:
                role = "User" if node.node_type == NodeType.USER else "Assistant"
                content = node.summary or node.content[:200]
                # Check if this node is among those to be compressed
                node_index = summary_nodes.index(node)
                hint = " ⚠️ (will be grouped into META next)" if node_index < summary_nodes_to_compress_count else ""
                context_parts.append(f"[{role}] ID {node.node_id}: {content}{hint}")

        # Add full recent nodes
        if full_nodes:
            context_parts.append("\n=== RECENT CONVERSATION ===")
            displayed_full = full_nodes[-self.config.recent_node_limit:]
            for node in displayed_full:
                role = "User" if node.node_type == NodeType.USER else "Assistant"
                # Check if this node is among those to be compressed to SUMMARY
                node_index = full_nodes.index(node)
                hint = " ⚠️ (will be summarized next)" if node_index < full_nodes_to_compress_count else ""

                # For AI nodes, include successful tool actions summary
                tool_summary = ""
                if node.node_type == NodeType.AI and node.ai_components:
                    tool_summary = self._format_successful_tool_actions(node.ai_components)
                    if tool_summary:
                        tool_summary = f"\n  [Tools: {tool_summary}]"

                context_parts.append(f"[{role}] {node.content}{tool_summary}{hint}")

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
                    # Save token usage
                    if self.conversation_id:
                        usage = message.usage or {}
                        await self.storage.save_token_usage(
                            conversation_id=self.conversation_id,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                            cost_usd=message.total_cost_usd,
                            duration_ms=message.duration_ms,
                            model=self.model_config.model_name,
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

        # Track tool calls that we've already sent start events for
        started_tool_ids = set()

        try:
            async for message in query(prompt=user_message, options=options):
                # Handle streaming events (partial messages)
                if isinstance(message, StreamEvent):
                    event = message.event
                    event_type = event.get("type")

                    # Detect content_block_start for tool_use - this is when tool call begins
                    if event_type == "content_block_start" and include_tool_events:
                        content_block = event.get("content_block", {})
                        if content_block.get("type") == "tool_use":
                            tool_id = content_block.get("id", "")
                            tool_name = content_block.get("name", "")
                            if tool_id and tool_name and tool_id not in started_tool_ids:
                                started_tool_ids.add(tool_id)
                                yield ToolCallStartEvent(
                                    tool_id=tool_id,
                                    tool_name=tool_name,
                                )

                    elif event_type == "content_block_delta":
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
                            # Use field names that match conversation_manager expectations
                            tool_call = {
                                "tool_call_id": block.id,
                                "tool_id": block.id,  # Keep for backward compatibility
                                "tool_name": block.name,
                                "args": block.input,
                                "tool_input": block.input,  # Keep for backward compatibility
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
                            # Use field names that match conversation_manager expectations
                            tool_result = {
                                "tool_call_id": block.tool_use_id,
                                "tool_id": block.tool_use_id,  # Keep for backward compatibility
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
                    # Parse and yield usage data
                    usage = message.usage or {}
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    cache_creation = usage.get("cache_creation_input_tokens", 0)
                    total_tokens = input_tokens + output_tokens

                    # Save token usage to storage
                    if self.conversation_id:
                        await self.storage.save_token_usage(
                            conversation_id=self.conversation_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cache_read_tokens=cache_read,
                            cache_creation_tokens=cache_creation,
                            cost_usd=message.total_cost_usd,
                            duration_ms=message.duration_ms,
                            model=self.model_config.model_name,
                        )

                    # Yield usage event if tool events are enabled
                    if include_tool_events:
                        yield UsageEvent(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cache_read_tokens=cache_read,
                            cache_creation_tokens=cache_creation,
                            total_tokens=total_tokens,
                            cost_usd=message.total_cost_usd or 0.0,
                            duration_ms=message.duration_ms,
                            model=self.model_config.model_name,
                        )
        except Exception as e:
            # Check if this is a SIGINT (Ctrl+C) interrupt
            # Exit codes: -2 (Python internal), 130 (128+2, standard SIGINT)
            error_str = str(e).lower()
            is_sigint = (
                "exit code -2" in error_str
                or "exit code 130" in error_str
                or "exit code: -2" in error_str
                or "exit code: 130" in error_str
                or "sigint" in error_str
            )
            if is_sigint:
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

        logger.debug(
            f"Saving partial response ({len(partial_response)} chars), "
            f"tool_calls={len(self._current_tool_calls)}, "
            f"tool_results={len(self._current_tool_results)}"
        )

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
        # Get current working directory for context
        cwd = os.getcwd()

        base_prompt = f"""You are a helpful AI assistant with access to conversation memory.

**Working Directory:** {cwd}

You are running within a hierarchical memory system. Only recent messages are shown in full detail - older messages are progressively compressed into summaries, meta-summaries, and archived context. Use the memory tools (search_memory, expand_node, get_recent_nodes) when you need to recall specific details from earlier in the conversation.

The conversation history below is organized hierarchically:
- ARCHIVED CONTEXT: Very old conversations, highly compressed
- META SUMMARIES: Groups of related conversations, summarized by topic
- CONVERSATION SUMMARIES: Recent conversations in summary form
- RECENT CONVERSATION: The most recent exchanges in full detail

Use this context to maintain continuity and reference past discussions when relevant.

**Tip:** Save important context (like working directory, project details, user preferences) to your scratchpad using the `set_system_prompt` or `append_to_system_prompt` tools for persistence across conversations.
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

        # Add custom instructions if provided
        if self.custom_instructions:
            parts.append(f"\n=== CUSTOM INSTRUCTIONS ===\n{self.custom_instructions}\n")

        # Add user's scratchpad/system prompt if set
        if self.conversation_id:
            scratchpad = await self.storage.get_system_prompt(self.conversation_id)
            if scratchpad:
                parts.append(f"\n=== YOUR SCRATCHPAD / NOTES ===\n{scratchpad}\n")

        # Add memory context
        if memory_context:
            parts.append(f"\n{memory_context}")

        system_prompt = "".join(parts)

        # Store for HTML export
        self._last_system_prompt = system_prompt

        return system_prompt

    def enable_html_export(self, export_dir: str) -> None:
        """Enable HTML context view export.

        Args:
            export_dir: Directory to save HTML files (e.g., ".conversations")
        """
        self._html_export_dir = export_dir
        logger.debug(f"HTML context export enabled: {export_dir}")

    async def export_context_html(
        self,
        current_user_message: Optional[str] = None,
        token_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Export the current AI context as an HTML file.

        Args:
            current_user_message: The current user message being processed
            token_stats: Optional token usage statistics

        Returns:
            Path to the saved HTML file, or None if export is disabled
        """
        if not self._html_export_dir or not self.conversation_id:
            return None

        try:
            from .context_html_view import save_context_html

            # Get all nodes
            nodes = await self.storage.get_conversation_nodes(self.conversation_id)

            # Get scratchpad
            scratchpad = await self.storage.get_system_prompt(self.conversation_id)

            # Build fresh system prompt to show what would be sent
            # This ensures the HTML always shows the current prompt, even before first message
            memory_context = await self._build_memory_context()
            system_prompt = await self._build_system_prompt(memory_context)

            output_path = save_context_html(
                output_dir=self._html_export_dir,
                conversation_id=self.conversation_id,
                system_prompt=system_prompt,
                nodes=nodes,
                scratchpad=scratchpad,
                custom_instructions=self.custom_instructions,
                agentic_mode=self.agentic_mode,
                current_user_message=current_user_message,
                token_stats=token_stats,
            )

            logger.debug(f"Exported HTML context view to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting HTML context: {e}")
            return None

    async def _check_and_compress(self) -> None:
        """Check if advanced hierarchical compression is needed and perform it."""
        if not self.conversation_id:
            return

        try:
            # Get all nodes for this conversation
            all_nodes = await self.storage.get_conversation_nodes(self.conversation_id)

            if not all_nodes:
                return

            logger.debug(f"Checking hierarchy compression for {len(all_nodes)} nodes")

            # Use advanced hierarchy compression system (handles FULL→SUMMARY, SUMMARY→META, META→ARCHIVE)
            compression_results = (
                await self.compression_manager.process_hierarchy_compression(
                    nodes=all_nodes, storage=self.storage
                )
            )

            # Log the results
            if compression_results.get("error"):
                logger.error(
                    f"Hierarchy compression error: {compression_results['error']}"
                )
                return

            total_processed = compression_results.get("total_processed", 0)
            if total_processed > 0:
                logger.debug(
                    f"Advanced hierarchy compression completed: "
                    f"{compression_results.get('summary_compressed', 0)} summary compressions, "
                    f"{compression_results.get('meta_groups_created', 0)} META groups created, "
                    f"{compression_results.get('archive_compressed', 0)} archive compressions"
                )
            else:
                logger.debug("No hierarchy compression needed at this time")

        except Exception as e:
            logger.exception(
                f"Error during advanced hierarchy compression: {str(e)}", exc_info=True
            )

            # Fallback to simple compression if advanced fails
            try:
                logger.debug("Falling back to simple compression system")
                nodes = await self.storage.get_conversation_nodes(self.conversation_id)
                full_nodes = [n for n in nodes if n.level == CompressionLevel.FULL]

                # Compress if we exceed the threshold
                if len(full_nodes) > self.hierarchy_thresholds.summary_threshold:
                    await self._compress_old_nodes(full_nodes)
            except Exception as fallback_error:
                logger.error(f"Fallback compression also failed: {str(fallback_error)}")

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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token for Claude models)."""
        if not text:
            return 0
        return len(text) // 4

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation.

        Returns:
            Dictionary with conversation statistics including token counts
        """
        if not self.conversation_id:
            return {}

        nodes = await self.storage.get_conversation_nodes(self.conversation_id)

        # Group nodes by compression level
        nodes_by_level = {
            CompressionLevel.FULL: [],
            CompressionLevel.SUMMARY: [],
            CompressionLevel.META: [],
            CompressionLevel.ARCHIVE: [],
        }
        for n in nodes:
            nodes_by_level[n.level].append(n)

        # Calculate token counts for each level
        # For compressed nodes, we count what's actually used in context (summary/content)
        def get_context_text(node: ConversationNode) -> str:
            """Get the text that would actually be used in context for this node."""
            if node.level == CompressionLevel.FULL:
                return node.content
            elif node.summary:
                return node.summary
            else:
                return node.content

        def get_original_text(node: ConversationNode) -> str:
            """Get the original full content of the node."""
            return node.content

        def estimate_archive_batch_tokens(archive_nodes: List[ConversationNode]) -> int:
            """Estimate tokens for archive nodes as they're actually rendered in context.

            Archive nodes are batched into groups of 20 and only show:
            'Nodes X-Y:(N nodes, M lines) [Topics: topic1, topic2, ...]'
            """
            if not archive_nodes:
                return 0

            batch_size = 20
            total_tokens = 0

            for i in range(0, len(archive_nodes), batch_size):
                batch = archive_nodes[i:i + batch_size]
                if not batch:
                    continue

                # Collect topics from batch
                all_topics = []
                total_lines = 0
                for node in batch:
                    if node.topics:
                        all_topics.extend(node.topics)
                    total_lines += node.line_count or 1

                # Get top 5 topics
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                top_topics = sorted(topic_counts.keys(), key=lambda t: topic_counts[t], reverse=True)[:5]
                topics_str = ", ".join(top_topics) if top_topics else "general"

                # This is the actual text that appears in context
                batch_text = f"Nodes {batch[0].node_id}-{batch[-1].node_id}:({len(batch)} nodes, {total_lines} lines) [Topics: {topics_str}]"
                total_tokens += self._estimate_tokens(batch_text)

            return total_tokens

        token_stats = {}
        total_current_tokens = 0
        total_original_tokens = 0

        for level, level_nodes in nodes_by_level.items():
            level_name = level.name.lower()

            # Special handling for ARCHIVE - count batched representation
            if level == CompressionLevel.ARCHIVE:
                current_tokens = estimate_archive_batch_tokens(level_nodes)
            else:
                current_tokens = sum(self._estimate_tokens(get_context_text(n)) for n in level_nodes)

            original_tokens = sum(self._estimate_tokens(get_original_text(n)) for n in level_nodes)

            token_stats[level_name] = {
                "count": len(level_nodes),
                "current_tokens": current_tokens,
                "original_tokens": original_tokens,
                "compression_ratio": round(original_tokens / current_tokens, 2) if current_tokens > 0 else 0,
            }
            total_current_tokens += current_tokens
            total_original_tokens += original_tokens

        stats = {
            "conversation_id": self.conversation_id,
            "total_nodes": len(nodes),
            "compression_levels": {
                "full": len(nodes_by_level[CompressionLevel.FULL]),
                "summary": len(nodes_by_level[CompressionLevel.SUMMARY]),
                "meta": len(nodes_by_level[CompressionLevel.META]),
                "archive": len(nodes_by_level[CompressionLevel.ARCHIVE]),
            },
            "token_stats": {
                "by_level": token_stats,
                "total_current_tokens": total_current_tokens,
                "total_original_tokens": total_original_tokens,
                "overall_compression_ratio": round(total_original_tokens / total_current_tokens, 2) if total_current_tokens > 0 else 0,
                "tokens_saved": total_original_tokens - total_current_tokens,
                "tokens_saved_percent": round((1 - total_current_tokens / total_original_tokens) * 100, 1) if total_original_tokens > 0 else 0,
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
                "node_type": r.node.node_type.value,
                "content": r.node.content[:500],
                "summary": r.node.summary,
                "timestamp": r.node.timestamp.isoformat() if r.node.timestamp else "",
                "relevance_score": r.relevance_score,
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
        if not self.conversation_id:
            return None
        node = await self.storage.get_node(node_id, self.conversation_id)
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
        node = await self.storage.get_node(node_id, conversation_id)
        if not node:
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
