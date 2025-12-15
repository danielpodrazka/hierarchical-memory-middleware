"""Middleware components for hierarchical memory system."""

from typing import Optional, Union, List

from .conversation_manager import HierarchicalConversationManager
from .claude_agent_sdk_manager import (
    ClaudeAgentSDKConversationManager,
    StreamChunk,
    ToolCallStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ..config import Config
from ..storage import DuckDBStorage
from ..model_manager import ModelManager, ClaudeAgentSDKMarker


def create_conversation_manager(
    config: Optional[Config] = None,
    storage: Optional[DuckDBStorage] = None,
    mcp_server_url: Optional[str] = None,
    external_mcp_servers: Optional[List] = None,
    enable_memory_tools: bool = True,
    agentic_mode: bool = False,
) -> Union[HierarchicalConversationManager, ClaudeAgentSDKConversationManager]:
    """Factory function to create the appropriate conversation manager.

    Automatically selects between:
    - ClaudeAgentSDKConversationManager: For claude-agent-* models (uses CLI auth)
    - HierarchicalConversationManager: For all other models (uses API keys)

    Args:
        config: Application configuration (loads from env if not provided)
        storage: Optional storage instance
        mcp_server_url: URL for the MCP memory server (for non-SDK managers)
        external_mcp_servers: Additional MCP servers (for non-SDK managers)
        enable_memory_tools: Enable memory tools for Claude Agent SDK (default: True)
        agentic_mode: Enable agentic mode with auto-continue and yield_to_human (default: False)

    Returns:
        The appropriate conversation manager instance

    Example:
        # Auto-selects based on WORK_MODEL in config
        manager = create_conversation_manager()

        # Or with explicit config
        config = Config(work_model="claude-agent-sonnet")
        manager = create_conversation_manager(config=config)
    """
    if config is None:
        config = Config.from_env()

    # Check if the model uses Claude Agent SDK
    if ModelManager.is_claude_agent_sdk_model(config.work_model):
        # Get the model config for Claude Agent SDK
        model_config = ModelManager.get_model_config(config.work_model)

        # Parse allowed tools from config
        allowed_tools = []
        if config.agent_allowed_tools:
            allowed_tools = [
                t.strip() for t in config.agent_allowed_tools.split(",") if t.strip()
            ]

        return ClaudeAgentSDKConversationManager(
            config=config,
            model_config=model_config,
            storage=storage,
            allowed_tools=allowed_tools,
            permission_mode=config.agent_permission_mode,
            enable_memory_tools=enable_memory_tools,
            agentic_mode=agentic_mode,
        )
    else:
        # Use the standard PydanticAI-based manager
        return HierarchicalConversationManager(
            config=config,
            storage=storage,
            mcp_server_url=mcp_server_url,
            external_mcp_servers=external_mcp_servers,
        )


__all__ = [
    "HierarchicalConversationManager",
    "ClaudeAgentSDKConversationManager",
    "create_conversation_manager",
    "StreamChunk",
    "ToolCallStartEvent",
    "ToolCallEvent",
    "ToolResultEvent",
]
