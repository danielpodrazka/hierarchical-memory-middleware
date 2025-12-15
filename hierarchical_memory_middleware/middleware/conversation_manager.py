"""Main conversation manager for hierarchical memory system."""

import asyncio
import json
import uuid
import logging
from typing import Optional, List, Dict, Any


import httpx
import httpcore

from pydantic_ai import Agent, usage
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
)
from pydantic_ai.mcp import MCPServerStreamableHTTP

from ..config import Config
from ..storage import DuckDBStorage
from ..compression import TfidfCompressor, CompressionManager
from ..advanced_hierarchy import AdvancedCompressionManager
from ..models import CompressionLevel, NodeType, HierarchyThresholds, ConversationNode
from ..model_manager import ModelManager


logger = logging.getLogger(__name__)


class HierarchicalConversationManager:
    """Manages conversations with hierarchical memory compression."""

    def __init__(
        self,
        config: Config,
        storage: Optional[DuckDBStorage] = None,
        mcp_server_url: Optional[str] = None,
        external_mcp_servers: Optional[List[MCPServerStreamableHTTP]] = None,
    ):
        """Initialize the conversation manager."""
        self.config = config
        self.conversation_id: Optional[str] = None

        config.setup_logging()

        # Initialize storage
        self.storage = storage or DuckDBStorage(config.db_path)

        # Initialize advanced hierarchical compression system
        self.compressor = TfidfCompressor(max_words=8)

        # Create hierarchy thresholds (configurable)
        self.hierarchy_thresholds = HierarchyThresholds(
            summary_threshold=config.recent_node_limit,  # Use config setting
            meta_threshold=50,  # SUMMARY nodes before META grouping
            archive_threshold=200,  # META groups before ARCHIVE
            meta_group_size=20,  # Minimum nodes per META group
            meta_group_max=40,  # Maximum nodes per META group
        )

        # Initialize advanced compression manager
        self.compression_manager = AdvancedCompressionManager(
            base_compressor=self.compressor, thresholds=self.hierarchy_thresholds
        )

        # Keep the simple compression manager for backward compatibility
        self.simple_compression_manager = CompressionManager(
            compressor=self.compressor, recent_node_limit=config.recent_node_limit
        )

        # Initialize PydanticAI agent with optional MCP server integration
        system_prompt = """You are a helpful AI assistant. You provide accurate and helpful responses."""

        # Build MCP server list
        mcp_servers = []

        # Add memory server if provided
        if mcp_server_url:
            memory_server = MCPServerStreamableHTTP(
                url=mcp_server_url,
                tool_prefix="memory",
                process_tool_call=self._log_tool_call
                if config.log_tool_calls
                else None,
            )
            mcp_servers.append(memory_server)

        # Add external servers
        if external_mcp_servers:
            for server in external_mcp_servers:
                # Set log_tool_call attribute on each external server
                server.process_tool_call = (
                    self._log_tool_call if config.log_tool_calls else None
                )
                mcp_servers.append(server)

        # Create agent with model manager
        try:
            # Use model manager to create the appropriate model instance
            model_instance = ModelManager.create_model(config.work_model)
            logger.debug(
                f"Successfully created model instance for: {config.work_model}"
            )
        except Exception as e:
            logger.exception(f"Failed to create model {config.work_model}: {str(e)}")
            raise ValueError(
                f"Unable to initialize model '{config.work_model}': {str(e)}"
            )

        agent_kwargs = {
            "model": model_instance,
            "system_prompt": system_prompt,
            "history_processors": [self._hierarchical_memory_processor],
        }

        if mcp_servers:
            agent_kwargs["mcp_servers"] = mcp_servers

        self.work_agent = Agent(**agent_kwargs)
        self.has_mcp_tools = bool(mcp_servers)

        # Track tool calls and results during execution
        self._current_tool_calls = []
        self._current_tool_results = []

        # Track what the AI actually sees for debugging/visualization
        self._last_ai_view_data = None

        logger.debug(
            f"Initialized HierarchicalConversationManager with model: {config.work_model} and MCP tools"
        )

    async def _log_tool_call(
        self, ctx, call_tool, tool_name: str, args: Dict[str, Any]
    ):
        """Custom tool call processor that logs everything and captures tool calls/results"""
        logger.info(f"🔧 TOOL CALL: {tool_name}")
        logger.info(f"📥 ARGS: {json.dumps(args, indent=2)}")

        # Generate a unique ID for this tool call
        tool_call_id = f"tool_call_{len(self._current_tool_calls)}_{tool_name}"

        # Store the tool call
        self._current_tool_calls.append(
            {"tool_name": tool_name, "tool_call_id": tool_call_id, "args": args}
        )

        try:
            # Call the actual tool
            result = await call_tool(tool_name, args)

            # Log the result (truncate if too long)
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"

            logger.info(f"📤 RESULT: {result_str}")
            logger.debug(f"✅ TOOL CALL COMPLETED: {tool_name}")

            # Store the tool result
            self._current_tool_results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                    "timestamp": None,  # Could add timestamp if needed
                }
            )

            return result

        except Exception as e:
            logger.exception(f"❌ TOOL CALL FAILED: {tool_name} - {str(e)}")
            raise

    def _reconstruct_ai_message_from_node(
        self, node, use_summary: bool = False
    ) -> ModelResponse:
        """Reconstruct a ModelResponse from a stored AI node as text-only to avoid tool call pairing issues."""
        try:
            # Always use text-only reconstruction to avoid tool_use/tool_result pairing issues
            # The final text content contains the relevant information from tool executions
            content = node.summary if use_summary else node.content
            return ModelResponse(parts=[TextPart(content=content)])

        except Exception as e:
            logger.debug(
                f"Error reconstructing AI message from node {node.node_id}: {e}"
            )
            # Ultimate fallback
            content = node.summary if use_summary else node.content
            return ModelResponse(parts=[TextPart(content=content)])

    def _has_active_tool_calls(self, messages: List[ModelMessage]) -> bool:
        """Check if there are active tool calls in progress that we shouldn't interfere with."""
        # Look for recent tool calls or tool results that suggest active tool execution
        recent_messages = messages[-3:] if len(messages) > 3 else messages

        for msg in recent_messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    # Check for tool call parts by looking at part_kind attribute
                    part_kind = getattr(part, "part_kind", None)
                    if part_kind in ["tool-call", "tool-return"]:
                        return True
                    # Also check attributes as backup
                    if hasattr(part, "tool_name") and (
                        hasattr(part, "args") or hasattr(part, "content")
                    ):
                        return True
        return False

    async def _hierarchical_memory_processor(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """Process message history using hierarchical memory system."""
        # Reset AI view data
        self._last_ai_view_data = None

        if not self.conversation_id:
            # No conversation started yet, return messages as-is
            return messages

        # CRITICAL: If there are active tool calls, don't interfere with the flow
        if self._has_active_tool_calls(messages):
            logger.debug(
                "Active tool calls detected, skipping memory processing but capturing basic AI view"
            )
            # Still capture basic AI view data for tool call scenarios
            self._last_ai_view_data = {
                "compressed_nodes": [],
                "recent_nodes": [],
                "recent_messages_from_input": [
                    {
                        "message_type": msg.__class__.__name__,
                        "content": str(msg)[:200]
                        + ("..." if len(str(msg)) > 200 else ""),
                    }
                    for msg in messages[-3:]  # Last 3 messages for context
                ],
                "total_messages_sent_to_ai": len(messages),
                "note": "AI view captured during active tool calls - limited data available",
            }
            return messages

        try:
            # Get recent uncompressed nodes
            recent_nodes = await self.storage.get_recent_nodes(
                conversation_id=self.conversation_id,
                limit=self.config.recent_node_limit,
            )

            # Get the most recent compressed nodes from all hierarchy levels (SUMMARY, META, ARCHIVE)
            compressed_nodes = await self.storage.get_recent_hierarchical_nodes(
                conversation_id=self.conversation_id,
                limit=max(
                    1, self.config.summary_threshold - self.config.recent_node_limit
                ),
            )

            # Initialize AI view data
            ai_view_data = {
                "compressed_nodes": [],
                "recent_nodes": [],
                "recent_messages_from_input": [],
                "total_messages_sent_to_ai": 0,
            }

            # Build message history from hierarchical memory
            memory_messages = []

            # Add conversation context as the first message
            context_message = f"[Context: Conversation ID {self.conversation_id}]"
            memory_messages.append(
                ModelRequest(parts=[UserPromptPart(content=context_message)])
            )

            # Add compressed context (already the most recent from get_recent_compressed_nodes)
            for node in compressed_nodes:
                if node.node_type == NodeType.USER:
                    # For META group nodes, use content (clear instructions); for others, use summary (with ID prefix)
                    if node.level == CompressionLevel.META:
                        content = node.content  # META groups have instructional content
                    else:
                        content = (
                            node.summary or node.content
                        )  # Regular compressed nodes use summary
                    memory_messages.append(
                        ModelRequest(parts=[UserPromptPart(content=content)])
                    )
                    ai_view_data["compressed_nodes"].append(
                        {
                            "node_id": node.node_id,
                            "node_type": "user",
                            "content": content,
                            "is_summary": bool(node.summary),
                            "sequence_number": node.sequence_number,
                            "topics": node.topics or [],
                        }
                    )
                elif node.node_type == NodeType.AI:
                    # Try to reconstruct full message structure for compressed nodes
                    reconstructed_msg = self._reconstruct_ai_message_from_node(
                        node, use_summary=True
                    )
                    memory_messages.append(reconstructed_msg)
                    ai_view_data["compressed_nodes"].append(
                        {
                            "node_id": node.node_id,
                            "node_type": "ai",
                            "content": (
                                node.content
                                if node.level == CompressionLevel.META
                                else (node.summary or node.content)
                            ),
                            "is_summary": True,
                            "sequence_number": node.sequence_number,
                            "topics": node.topics or [],
                        }
                    )

            # Add recent full messages
            for node in recent_nodes[-8:]:  # Last 8 recent nodes
                if node.node_type == NodeType.USER:
                    memory_messages.append(
                        ModelRequest(parts=[UserPromptPart(content=node.content)])
                    )
                    ai_view_data["recent_nodes"].append(
                        {
                            "node_id": node.node_id,
                            "node_type": "user",
                            "content": node.content,
                            "sequence_number": node.sequence_number,
                            "topics": node.topics or [],
                        }
                    )
                elif node.node_type == NodeType.AI:
                    # Try to reconstruct full message structure for recent nodes
                    reconstructed_msg = self._reconstruct_ai_message_from_node(
                        node, use_summary=False
                    )
                    memory_messages.append(reconstructed_msg)
                    ai_view_data["recent_nodes"].append(
                        {
                            "node_id": node.node_id,
                            "node_type": "ai",
                            "content": node.content,
                            "sequence_number": node.sequence_number,
                            "topics": node.topics or [],
                        }
                    )

            logger.debug(
                f"Memory processor: found {len(memory_messages)} memory messages, {len(messages)} incoming messages"
            )

            # Combine memory with current/new user message
            if len(memory_messages) > 0:
                # Start with conversation memory
                combined_messages = memory_messages.copy()

                # Add recent messages AS-IS to preserve active tool execution
                # Only the stored memory_messages above were cleaned, not the live conversation
                recent_message_limit = 5  # Keep last few messages
                recent_messages = (
                    messages[-recent_message_limit:]
                    if len(messages) > recent_message_limit
                    else messages
                )

                # Track recent messages from input
                for msg in recent_messages:
                    message_data = {
                        "message_type": msg.__class__.__name__,
                        "parts": [],
                    }

                    if hasattr(msg, "parts"):
                        # Extract detailed content from message parts - show what AI actually sees
                        for part in msg.parts:
                            part_data = {
                                "part_type": part.__class__.__name__,
                            }

                            if hasattr(part, "content") and part.content is not None:
                                # Regular content (text, system prompts, etc.)
                                part_data["content"] = str(part.content)[:500] + (
                                    "..." if len(str(part.content)) > 500 else ""
                                )
                            elif hasattr(part, "tool_name"):
                                # Tool call part - show what AI sees
                                part_data["tool_call"] = {
                                    "tool_name": getattr(part, "tool_name", None),
                                    "tool_call_id": getattr(part, "tool_call_id", None),
                                    "args": getattr(part, "args", None),
                                }
                            elif hasattr(part, "tool_call_id") and hasattr(
                                part, "content"
                            ):
                                # Tool result part - show what AI receives
                                part_data["tool_result"] = {
                                    "tool_call_id": getattr(part, "tool_call_id", None),
                                    "tool_name": getattr(part, "tool_name", None),
                                    "content": str(getattr(part, "content", ""))[:500]
                                    + (
                                        "..."
                                        if len(str(getattr(part, "content", ""))) > 500
                                        else ""
                                    ),
                                }
                            else:
                                # Other part types
                                part_data["raw_content"] = str(part)[:200] + (
                                    "..." if len(str(part)) > 200 else ""
                                )

                            message_data["parts"].append(part_data)

                        # Create a summary content for readability
                        content_summary = []
                        for part in msg.parts:
                            if hasattr(part, "content") and part.content:
                                content_summary.append(
                                    str(part.content)[:100]
                                    + ("..." if len(str(part.content)) > 100 else "")
                                )
                            elif hasattr(part, "tool_name"):
                                content_summary.append(f"[TOOL CALL: {part.tool_name}]")
                            elif hasattr(part, "tool_call_id"):
                                content_summary.append(
                                    f"[TOOL RESULT: {getattr(part, 'tool_name', 'unknown')}]"
                                )
                        message_data["content_summary"] = (
                            " | ".join(content_summary)
                            if content_summary
                            else str(msg)[:200]
                        )
                    else:
                        # No parts, just convert to string
                        message_data["content_summary"] = str(msg)[:200] + (
                            "..." if len(str(msg)) > 200 else ""
                        )
                        message_data["raw_message"] = str(msg)[:500] + (
                            "..." if len(str(msg)) > 500 else ""
                        )

                    ai_view_data["recent_messages_from_input"].append(message_data)

                # Add recent messages without cleaning to preserve tool call flows
                combined_messages.extend(recent_messages)

                # Update total count
                ai_view_data["total_messages_sent_to_ai"] = len(combined_messages)

                # Store the AI view data
                self._last_ai_view_data = ai_view_data

                logger.debug(
                    f"Memory processor: returning {len(combined_messages)} total messages ({len(memory_messages)} from memory + {len(recent_messages)} recent)"
                )
                return combined_messages
            else:
                # No memory yet, use provided messages as-is to preserve tool flows
                ai_view_data["recent_messages_from_input"] = []
                for msg in messages:
                    message_data = {
                        "message_type": msg.__class__.__name__,
                        "parts": [],
                    }

                    if hasattr(msg, "parts"):
                        # Extract detailed content from message parts - show what AI actually sees
                        for part in msg.parts:
                            part_data = {
                                "part_type": part.__class__.__name__,
                            }

                            if hasattr(part, "content") and part.content is not None:
                                # Regular content (text, system prompts, etc.)
                                part_data["content"] = str(part.content)[:500] + (
                                    "..." if len(str(part.content)) > 500 else ""
                                )
                            elif hasattr(part, "tool_name"):
                                # Tool call part - show what AI sees
                                part_data["tool_call"] = {
                                    "tool_name": getattr(part, "tool_name", None),
                                    "tool_call_id": getattr(part, "tool_call_id", None),
                                    "args": getattr(part, "args", None),
                                }
                            elif hasattr(part, "tool_call_id") and hasattr(
                                part, "content"
                            ):
                                # Tool result part - show what AI receives
                                part_data["tool_result"] = {
                                    "tool_call_id": getattr(part, "tool_call_id", None),
                                    "tool_name": getattr(part, "tool_name", None),
                                    "content": str(getattr(part, "content", ""))[:500]
                                    + (
                                        "..."
                                        if len(str(getattr(part, "content", ""))) > 500
                                        else ""
                                    ),
                                }
                            else:
                                # Other part types
                                part_data["raw_content"] = str(part)[:200] + (
                                    "..." if len(str(part)) > 200 else ""
                                )

                            message_data["parts"].append(part_data)

                        # Create a summary content for readability
                        content_summary = []
                        for part in msg.parts:
                            if hasattr(part, "content") and part.content:
                                content_summary.append(
                                    str(part.content)[:100]
                                    + ("..." if len(str(part.content)) > 100 else "")
                                )
                            elif hasattr(part, "tool_name"):
                                content_summary.append(f"[TOOL CALL: {part.tool_name}]")
                            elif hasattr(part, "tool_call_id"):
                                content_summary.append(
                                    f"[TOOL RESULT: {getattr(part, 'tool_name', 'unknown')}]"
                                )
                        message_data["content_summary"] = (
                            " | ".join(content_summary)
                            if content_summary
                            else str(msg)[:200]
                        )
                    else:
                        # No parts, just convert to string
                        message_data["content_summary"] = str(msg)[:200] + (
                            "..." if len(str(msg)) > 200 else ""
                        )
                        message_data["raw_message"] = str(msg)[:500] + (
                            "..." if len(str(msg)) > 500 else ""
                        )

                    ai_view_data["recent_messages_from_input"].append(message_data)

                ai_view_data["total_messages_sent_to_ai"] = len(messages)
                self._last_ai_view_data = ai_view_data
                return messages

        except Exception as e:
            logger.exception(f"Error in history processor: {str(e)}", exc_info=True)
            # Fallback to provided messages on error to preserve tool flows
            return messages

    def get_last_ai_view_data(self) -> Optional[Dict[str, Any]]:
        """Get the last captured AI view data from the memory processor.

        Returns:
            Dictionary containing what the AI actually saw in the last message processing,
            or None if no data is available (e.g., no conversation started, tool calls active).
        """
        return self._last_ai_view_data

    async def _save_ai_view_data(self) -> None:
        """Save the current AI view data to a JSON file for debugging."""
        if not self.conversation_id:
            logger.debug("No conversation ID available for saving AI view")
            return

        if not self._last_ai_view_data:
            logger.debug("No AI view data available to save")
            return

        try:
            import json
            import os
            from datetime import datetime

            # Create .conversations directory if it doesn't exist
            conversations_dir = ".conversations"
            os.makedirs(conversations_dir, exist_ok=True)

            # Create AI view file path
            ai_view_file = os.path.join(
                conversations_dir, f"{self.conversation_id}_ai_view.json"
            )

            # Create enhanced view data with summary counts
            view_data = self._last_ai_view_data or {}

            enriched_view_data = {
                "conversation_id": self.conversation_id,
                "description": "This shows exactly what the AI agent saw in the last message processing",
                "last_updated": datetime.now().isoformat(),
                "model_used": self.config.work_model,
                # Summary counts for quick overview
                "compressed_nodes_count": len(view_data.get("compressed_nodes", [])),
                "recent_nodes_count": len(view_data.get("recent_nodes", [])),
                "recent_messages_from_input_count": len(
                    view_data.get("recent_messages_from_input", [])
                ),
                "total_messages_sent_to_ai": view_data.get(
                    "total_messages_sent_to_ai", 0
                ),
                # Detailed data - what the AI actually sees
                "compressed_nodes": view_data.get("compressed_nodes", []),
                "recent_nodes": view_data.get("recent_nodes", []),
                "recent_messages_from_input": view_data.get(
                    "recent_messages_from_input", []
                ),
                # Additional metadata
                "note": "Compressed nodes show summaries only (what AI sees), recent nodes show full content",
            }

            # Add fallback reason if present
            if "fallback_reason" in view_data:
                enriched_view_data["fallback_reason"] = view_data["fallback_reason"]

            # Save to file
            with open(ai_view_file, "w", encoding="utf-8") as f:
                json.dump(enriched_view_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"AI view data saved to {ai_view_file}")
        except Exception as e:
            logger.exception(f"Failed to save AI view data to file: {e}")

    async def _capture_current_ai_view(self) -> None:
        """Capture the current AI view by querying stored conversation data.

        This builds the AI view data structure by actually looking at what's stored
        in the database, showing exactly what the AI sees (summaries for compressed,
        full content for recent nodes).
        """
        if not self.conversation_id:
            return

        try:
            # Get recent uncompressed nodes (what the AI sees as "recent")
            recent_nodes = await self.storage.get_recent_nodes(
                conversation_id=self.conversation_id,
                limit=self.config.recent_node_limit,
            )

            # Get compressed nodes from all hierarchy levels
            compressed_nodes = await self.storage.get_recent_hierarchical_nodes(
                conversation_id=self.conversation_id,
                limit=max(
                    1, self.config.summary_threshold - self.config.recent_node_limit
                ),
            )

            # Build AI view data structure
            ai_view_data = {
                "compressed_nodes": [],
                "recent_nodes": [],
                "recent_messages_from_input": [],
                "total_messages_sent_to_ai": 0,
            }

            # Process compressed nodes (AI sees summaries only)
            for node in compressed_nodes:
                node_data = {
                    "node_id": node.node_id,
                    "node_type": node.node_type.name.lower(),
                    "sequence_number": node.sequence_number,
                    "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                    "level": node.level.name if node.level else "FULL",
                    "topics": node.topics or [],
                }

                # For compressed nodes, AI only sees the summary
                if node.node_type.name == "USER":
                    # For META group nodes, use content; for others, use summary
                    if hasattr(node.level, "name") and node.level.name == "META":
                        content = node.content  # META groups have instructional content
                    else:
                        content = (
                            node.summary or node.content
                        )  # Regular compressed nodes use summary
                    node_data["content_shown_to_ai"] = content
                elif node.node_type.name == "AI":
                    # AI nodes: always use summary for compressed
                    content = node.summary or node.content
                    node_data["content_shown_to_ai"] = content

                ai_view_data["compressed_nodes"].append(node_data)

            # Process recent nodes (AI sees full content)
            for node in recent_nodes[
                -8:
            ]:  # Last 8 recent nodes (same as memory processor)
                node_data = {
                    "node_id": node.node_id,
                    "node_type": node.node_type.name.lower(),
                    "sequence_number": node.sequence_number,
                    "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                    "level": "FULL",  # Recent nodes are always full
                    "topics": node.topics or [],
                    "content_shown_to_ai": node.content,  # AI sees full content for recent nodes
                }

                # Add AI components for AI nodes - show what the AI actually saw
                if node.node_type.name == "AI" and node.ai_components:
                    import json

                    try:
                        ai_components = (
                            json.loads(node.ai_components)
                            if isinstance(node.ai_components, str)
                            else node.ai_components
                        )

                        # Extract tool calls and results - what the AI actually saw
                        tool_calls = ai_components.get("tool_calls", [])
                        tool_results = ai_components.get("tool_results", [])

                        node_data["ai_components"] = {
                            "model_used": ai_components.get("model_used"),
                            "assistant_text": ai_components.get("assistant_text", ""),
                            "tool_calls_count": len(tool_calls),
                            "tool_results_count": len(tool_results),
                            "has_tools": len(tool_calls) > 0,
                        }

                        # Show the actual tool calls the AI made
                        if tool_calls:
                            node_data["tool_calls_ai_made"] = []
                            for call in tool_calls:
                                call_info = {
                                    "tool_name": call.get("tool_name"),
                                    "tool_call_id": call.get("tool_call_id"),
                                    "args": call.get("args"),
                                }
                                node_data["tool_calls_ai_made"].append(call_info)

                        # Show the actual tool results the AI received
                        if tool_results:
                            node_data["tool_results_ai_received"] = []
                            for result in tool_results:
                                result_info = {
                                    "tool_call_id": result.get("tool_call_id"),
                                    "content": result.get("content"),
                                    "timestamp": result.get("timestamp"),
                                }
                                node_data["tool_results_ai_received"].append(
                                    result_info
                                )

                    except Exception as e:
                        logger.debug(
                            f"Failed to parse AI components for node {node.node_id}: {e}"
                        )
                        # Fallback to basic info
                        node_data["ai_components"] = {
                            "error": f"Failed to parse AI components: {str(e)}",
                            "raw_components": str(node.ai_components)[:200] + "..."
                            if len(str(node.ai_components)) > 200
                            else str(node.ai_components),
                        }

                ai_view_data["recent_nodes"].append(node_data)

            # Calculate total messages the AI would see
            # This includes context message + compressed + recent
            total_messages = 1  # Context message
            total_messages += len(compressed_nodes)
            total_messages += len(recent_nodes[-8:])
            ai_view_data["total_messages_sent_to_ai"] = total_messages

            # Store the captured data
            self._last_ai_view_data = ai_view_data
            logger.debug(
                f"AI view captured: {len(compressed_nodes)} compressed, {len(recent_nodes)} recent nodes"
            )

        except Exception as e:
            logger.exception(f"Failed to capture current AI view: {e}")
            # Create fallback data
            self._last_ai_view_data = {
                "compressed_nodes": [],
                "recent_nodes": [],
                "recent_messages_from_input": [],
                "total_messages_sent_to_ai": 0,
                "fallback_reason": f"Failed to capture AI view: {str(e)}",
            }

    async def _ensure_ai_view_captured(self) -> None:
        """Ensure AI view data is captured after conversation nodes are saved."""
        if not self.conversation_id:
            return

        # Always capture fresh AI view data after nodes are saved
        # This ensures we see the current state of compressed/recent nodes
        await self._capture_current_ai_view()

    async def get_ai_view_data(self) -> Optional[Dict[str, Any]]:
        """Get the AI view data from the JSON file.

        Returns:
            Dictionary containing the AI view data from file, or fallback data if not available.
        """
        if not self.conversation_id:
            return {
                "conversation_id": None,
                "description": "No active conversation - start a conversation to see AI view data",
                "last_updated": None,
                "note": "The AI view is captured during message processing.",
            }

        try:
            import json
            import os

            # Try to read from file first
            conversations_dir = ".conversations"
            ai_view_file = os.path.join(
                conversations_dir, f"{self.conversation_id}_ai_view.json"
            )

            if os.path.exists(ai_view_file):
                with open(ai_view_file, "r", encoding="utf-8") as f:
                    stored_data = json.load(f)
                return stored_data

            # If not in file but we have current session data, return that
            if self._last_ai_view_data:
                return {
                    "conversation_id": self.conversation_id,
                    "view_data": self._last_ai_view_data,
                    "description": "Current session AI view data (not yet saved to file)",
                    "last_updated": None,
                    "note": "The AI view is captured during message processing.",
                }

            # Nothing available
            return {
                "conversation_id": self.conversation_id,
                "description": "No AI view data available yet - send a message to see what the AI sees",
                "last_updated": None,
                "note": "The AI view is captured during message processing.",
            }

        except Exception as e:
            logger.exception(f"Failed to get AI view data from file: {e}")
            return {
                "conversation_id": self.conversation_id,
                "description": f"Error retrieving AI view data: {str(e)}",
                "last_updated": None,
                "note": "The AI view is captured during message processing.",
            }

    async def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Initialize or resume a conversation."""
        if conversation_id:
            # Check if conversation exists
            if await self.storage.conversation_exists(conversation_id):
                self.conversation_id = conversation_id
                logger.info(f"Resuming conversation: {conversation_id}")
            else:
                logger.warning(
                    f"Conversation {conversation_id} not found, creating new one"
                )
                self.conversation_id = str(uuid.uuid4())
                # Create the conversation entry in the database
                await self.storage._ensure_conversation_exists(self.conversation_id)
        else:
            self.conversation_id = str(uuid.uuid4())
            logger.info(f"Starting new conversation: {self.conversation_id}")
            # Create the conversation entry in the database
            await self.storage._ensure_conversation_exists(self.conversation_id)

        return self.conversation_id

    async def set_conversation_name(self, conversation_id: str, name: str) -> bool:
        """Set the name of a conversation."""
        return await self.storage.set_conversation_name(conversation_id, name)

    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """Return tool calls captured during execution."""
        # Return the tool calls that were captured during execution
        return self._current_tool_calls.copy()

    def _extract_tool_results(self, response) -> List[Dict[str, Any]]:
        """Return tool results captured during execution."""
        # Return the tool results that were captured during execution
        return self._current_tool_results.copy()

    def _build_comprehensive_content(self, response) -> str:
        """Build comprehensive content including tool calls, results, and final response."""
        content_parts = []

        # Extract tool calls and results
        tool_calls = self._extract_tool_calls(response)
        tool_results = self._extract_tool_results(response)

        # Add tool calls to content
        if tool_calls:
            content_parts.append("=== TOOL CALLS ===")
            for i, call in enumerate(tool_calls, 1):
                content_parts.append(f"Tool Call {i}:")
                content_parts.append(f"  Name: {call['tool_name']}")
                content_parts.append(f"  ID: {call['tool_call_id']}")
                if call["args"]:
                    content_parts.append(
                        f"  Arguments: {json.dumps(call['args'], indent=4)}"
                    )
                content_parts.append("")

        # Add tool results to content
        if tool_results:
            content_parts.append("=== TOOL RESULTS ===")
            for i, result in enumerate(tool_results, 1):
                content_parts.append(f"Tool Result {i}:")
                content_parts.append(f"  Call ID: {result['tool_call_id']}")
                content_parts.append(f"  Content: {result['content']}")
                if result["timestamp"]:
                    content_parts.append(f"  Timestamp: {result['timestamp']}")
                content_parts.append("")

        # Add the final assistant response
        if content_parts:  # Only add separator if there were tool calls/results
            content_parts.append("=== ASSISTANT RESPONSE ===")
        content_parts.append(response.output)

        return "\n".join(content_parts)

    async def chat_stream(self, user_message: str):
        """Streaming version of chat with hierarchical memory integration and proper tool handling."""
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        try:
            # Clear tool tracking for new conversation turn
            self._current_tool_calls = []
            self._current_tool_results = []

            full_response_text = ""
            usage_info = None

            # Use proper pydantic-ai streaming with graph iteration
            # MCP tools are essential to the architecture, always use them
            logger.debug("Entering MCP context manager")
            async with self.work_agent.run_mcp_servers():
                logger.debug("MCP context manager entered, starting graph iteration")

                # Use agent.iter() for proper streaming with tool calls
                custom_limits = usage.UsageLimits(
                    request_limit=self.config.request_limit
                )
                async with self.work_agent.iter(
                    user_message, usage_limits=custom_limits
                ) as run:
                    content_streamed = False

                    async for node in run:
                        if self.work_agent.is_user_prompt_node(node):
                            logger.debug(
                                f"UserPromptNode: {getattr(node, 'user_prompt', str(node))}"
                            )

                        elif self.work_agent.is_model_request_node(node):
                            logger.debug(
                                "ModelRequestNode: streaming partial request tokens"
                            )
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    # Handle different event types for streaming
                                    content = self._extract_streamable_content(event)
                                    if content:
                                        content_streamed = True
                                        full_response_text += content
                                        yield content

                        elif self.work_agent.is_call_tools_node(node):
                            logger.debug("ToolCallNode: processing tool calls")
                            async with node.stream(run.ctx) as tool_stream:
                                async for event in tool_stream:
                                    # Track tool calls and results for memory
                                    self._track_tool_events(event)

                        elif self.work_agent.is_end_node(node):
                            logger.debug("EndNode: run completed")
                            if content_streamed:
                                logger.debug(f"Final result: {run.result.data}")
                            else:
                                # If no content was streamed, yield the final result
                                final_result = str(run.result.data)
                                full_response_text = final_result
                                yield final_result
                        else:
                            logger.debug(f"Unknown Node: {type(node)}")

                    # Get usage info from the run
                    usage_info = run.usage()

            # Save nodes after streaming is complete
            user_node = await self.storage.save_conversation_node(
                conversation_id=self.conversation_id,
                node_type=NodeType.USER,
                content=user_message,
            )

            # Extract token usage
            tokens_used = None
            try:
                if usage_info:
                    tokens_used = getattr(usage_info, "total_tokens", None)
            except Exception as e:
                logger.debug(f"Could not extract token usage: {e}")
                tokens_used = None

            # Save assistant response node
            ai_node = await self.storage.save_conversation_node(
                conversation_id=self.conversation_id,
                node_type=NodeType.AI,
                content=full_response_text,
                tokens_used=tokens_used,
                ai_components={
                    "assistant_text": full_response_text,
                    "model_used": self.config.work_model,
                    "tool_calls": self._current_tool_calls,
                    "tool_results": self._current_tool_results,
                },
            )

            await self._check_and_compress()

            # Ensure AI view data is captured and saved
            await self._ensure_ai_view_captured()
            await self._save_ai_view_data()
            logger.debug("chat_stream completed successfully")

        except Exception as e:
            # Check if we have partial response content to save
            has_partial_content = bool(full_response_text.strip())

            if has_partial_content:
                # Determine error type for better messaging
                error_type = self._categorize_streaming_error(e)

                logger.warning(
                    f"Streaming failed ({error_type}), but saving partial response ({len(full_response_text)} chars): {str(e)[:200]}..."
                )

                # Save the user message first if we haven't yet
                try:
                    user_node = await self.storage.save_conversation_node(
                        conversation_id=self.conversation_id,
                        node_type=NodeType.USER,
                        content=user_message,
                    )
                except Exception as save_error:
                    logger.warning(
                        f"Failed to save user message during streaming failure recovery: {save_error}"
                    )

                # Save the partial AI response with appropriate error note
                try:
                    error_note = self._get_error_note(error_type)
                    partial_content = f"{full_response_text}\n\n{error_note}"

                    ai_node = await self.storage.save_conversation_node(
                        conversation_id=self.conversation_id,
                        node_type=NodeType.AI,
                        content=partial_content,
                        tokens_used=None,  # Unknown due to interruption
                        ai_components={
                            "assistant_text": partial_content,
                            "model_used": self.config.work_model,
                            "tool_calls": self._current_tool_calls,
                            "tool_results": self._current_tool_results,
                            "interrupted_by_streaming_failure": True,
                            "error_type": error_type,
                            "original_error": str(e)[:500],  # Truncated error
                        },
                    )

                    logger.info(
                        f"Successfully saved partial response after {error_type}. Response length: {len(full_response_text)} characters"
                    )

                    # Try to run compression and AI view capture, but don't fail if they error
                    try:
                        await self._check_and_compress()
                        await self._ensure_ai_view_captured()
                        await self._save_ai_view_data()
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Cleanup operations failed after streaming failure (non-critical): {cleanup_error}"
                        )

                    # Return early - don't re-raise the streaming exception
                    logger.info(
                        f"{error_type} handled gracefully, conversation can continue from partial response"
                    )
                    return

                except Exception as save_error:
                    logger.error(
                        f"Failed to save partial response during streaming failure recovery: {save_error}"
                    )
                    # Fall through to re-raise original exception

            else:
                # No partial content to save
                error_type = self._categorize_streaming_error(e)
                logger.warning(
                    f"Streaming failed ({error_type}) with no partial response to save: {str(e)[:200]}..."
                )

            # For cases where we couldn't save partial content, re-raise the original exception
            logger.exception(f"Error in chat_stream: {e}", exc_info=True)
            raise

    async def chat(self, user_message: str) -> str:
        """Main conversation interface with hierarchical memory integration."""
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        try:
            # Clear tool tracking for new conversation turn
            self._current_tool_calls = []
            self._current_tool_results = []

            # Generate AI response using PydanticAI with history processor
            # The history processor will automatically manage conversation memory
            # MCP tools are essential to the architecture, always use them
            async with self.work_agent.run_mcp_servers():
                custom_limits = usage.UsageLimits(
                    request_limit=self.config.request_limit
                )
                response = await self.work_agent.run(
                    user_prompt=user_message, usage_limits=custom_limits
                )

            # Save the user message as a node
            user_node = await self.storage.save_conversation_node(
                conversation_id=self.conversation_id,
                node_type=NodeType.USER,
                content=user_message,
            )

            # Save the AI response as a node
            # Extract token usage from response
            tokens_used = None
            try:
                if hasattr(response, "usage"):
                    if callable(response.usage):
                        usage_info = response.usage()
                        tokens_used = getattr(usage_info, "total_tokens", None)
                    else:
                        # usage is a dict
                        usage_info = response.usage
                        tokens_used = usage_info.get("total_tokens", None)
            except Exception as e:
                logger.debug(f"Could not extract token usage: {e}")
                tokens_used = None

            # Build comprehensive content including tool calls and results
            comprehensive_content = self._build_comprehensive_content(response)

            ai_node = await self.storage.save_conversation_node(
                conversation_id=self.conversation_id,
                node_type=NodeType.AI,
                content=comprehensive_content,
                tokens_used=tokens_used,
                ai_components={
                    "assistant_text": response.output,
                    "model_used": self.config.work_model,
                    "tool_calls": self._extract_tool_calls(response),
                    "tool_results": self._extract_tool_results(response),
                },
            )

            # Check if compression is needed
            await self._check_and_compress()

            # Ensure AI view data is captured and saved
            await self._ensure_ai_view_captured()
            await self._save_ai_view_data()

            logger.debug(
                f"Processed conversation turn (user: {user_node.node_id}, ai: {ai_node.node_id}) in conversation {self.conversation_id}"
            )
            return response.output

        except Exception as e:
            logger.exception(f"Error in chat: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an error processing your message: {str(e)}"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token for Claude models)."""
        if not text:
            return 0
        return len(text) // 4

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state with token statistics."""
        if not self.conversation_id:
            return {"error": "No active conversation"}

        try:
            stats = await self.storage.get_conversation_stats(self.conversation_id)
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

            token_stats = {}
            total_current_tokens = 0
            total_original_tokens = 0

            for level, level_nodes in nodes_by_level.items():
                level_name = level.name.lower()
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

            return {
                "conversation_id": self.conversation_id,
                "total_nodes": len(nodes),
                "recent_nodes": len(nodes_by_level[CompressionLevel.FULL]),
                "compressed_nodes": len(nodes_by_level[CompressionLevel.SUMMARY]),
                "compression_stats": stats.compression_stats if stats else {},
                "token_stats": {
                    "by_level": token_stats,
                    "total_current_tokens": total_current_tokens,
                    "total_original_tokens": total_original_tokens,
                    "overall_compression_ratio": round(total_original_tokens / total_current_tokens, 2) if total_current_tokens > 0 else 0,
                    "tokens_saved": total_original_tokens - total_current_tokens,
                    "tokens_saved_percent": round((1 - total_current_tokens / total_original_tokens) * 100, 1) if total_original_tokens > 0 else 0,
                },
                "last_updated": stats.last_updated.isoformat()
                if stats and stats.last_updated
                else None,
            }

        except Exception as e:
            logger.exception(
                f"Error getting conversation summary: {str(e)}", exc_info=True
            )
            return {"error": str(e)}

    async def find(
        self, query: str, limit: int = 10, regex: bool = False
    ) -> List[Dict[str, Any]]:
        """Search conversation memory."""
        if not self.conversation_id:
            return []

        try:
            results = await self.storage.search_nodes(
                conversation_id=self.conversation_id,
                query=query,
                limit=limit,
                regex=regex,
            )

            return [
                {
                    "node_id": result.node.node_id,
                    "content": result.node.content[:200] + "..."
                    if len(result.node.content) > 200
                    else result.node.content,
                    "summary": result.node.summary,
                    "relevance_score": result.relevance_score,
                    "match_type": result.match_type,
                    "timestamp": result.node.timestamp.isoformat(),
                    "node_type": result.node.node_type.value,
                }
                for result in results
            ]

        except Exception as e:
            logger.exception(f"Error searching memory: {str(e)}", exc_info=True)
            return []

    async def _check_and_compress(self) -> None:
        """Check if advanced hierarchical compression is needed and perform it."""
        try:
            # Get all nodes for this conversation
            all_nodes = await self.storage.get_conversation_nodes(
                conversation_id=self.conversation_id
            )

            if not all_nodes:
                return

            logger.debug(f"Checking hierarchy compression for {len(all_nodes)} nodes")

            # Use advanced hierarchy compression system
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

                # Use the simple compression manager as fallback
                nodes_to_compress = (
                    self.simple_compression_manager.identify_nodes_to_compress(
                        all_nodes
                    )
                )

                if nodes_to_compress:
                    compression_results = (
                        self.simple_compression_manager.compress_nodes(
                            nodes_to_compress
                        )
                    )

                    # Update nodes in storage using simple compression
                    for result in compression_results:
                        await self.storage.compress_node(
                            node_id=result.original_node_id,
                            conversation_id=self.conversation_id,
                            compression_level=CompressionLevel.SUMMARY,
                            summary=result.compressed_content,
                            metadata=result.metadata,
                        )

                    logger.debug(
                        f"Fallback compression completed: {len(compression_results)} nodes"
                    )

            except Exception as fallback_error:
                logger.exception(
                    f"Fallback compression also failed: {str(fallback_error)}",
                    exc_info=True,
                )

    async def get_node_details(
        self, node_id: int, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get full details of a specific node (for expansion)."""
        try:
            node = await self.storage.get_node(node_id, conversation_id)
            if not node:
                return None

            return {
                "node_id": node.node_id,
                "conversation_id": node.conversation_id,
                "node_type": node.node_type.value,
                "content": node.content,
                "summary": node.summary,
                "timestamp": node.timestamp.isoformat(),
                "sequence_number": node.sequence_number,
                "line_count": node.line_count,
                "level": node.level.name,
                "tokens_used": node.tokens_used,
                "topics": node.topics,
                "ai_components": node.ai_components,
            }

        except Exception as e:
            logger.exception(f"Error getting node details: {str(e)}", exc_info=True)
            return None

    def _extract_streamable_content(self, event) -> str:
        """Extract streamable content from pydantic-ai events."""
        # Content we care about streaming
        if isinstance(event, PartStartEvent) and hasattr(event.part, "content"):
            return event.part.content
        elif (
            isinstance(event, PartDeltaEvent)
            and hasattr(event, "delta")
            and isinstance(event.delta, TextPartDelta)
        ):
            return event.delta.content_delta

        return ""

    def _track_tool_events(self, event):
        """Track tool calls and results for memory storage."""
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
            # Track tool call
            self._current_tool_calls.append(
                {
                    "tool_name": event.part.tool_name,
                    "tool_call_id": getattr(
                        event.part,
                        "tool_call_id",
                        f"tool_call_{len(self._current_tool_calls)}",
                    ),
                    "args": getattr(event.part, "args", {}),
                }
            )
            logger.debug(f"Tracked tool call: {event.part.tool_name}")

        elif isinstance(event, FunctionToolCallEvent):
            # Log tool call event - check what attributes are available
            logger.debug(
                f"Tool call event: {type(event).__name__} - {getattr(event, 'tool_call_id', 'no_id')}"
            )

        elif isinstance(event, FunctionToolResultEvent):
            # Track tool result - use available attributes defensively
            tool_call_id = getattr(event, "tool_call_id", "unknown")
            result_str = str(getattr(event, "result", "no_result"))
            self._current_tool_results.append(
                {
                    "tool_name": "unknown",  # FunctionToolResultEvent might not have tool_name
                    "tool_call_id": tool_call_id,
                    "result": result_str,
                }
            )
            logger.debug(f"Tracked tool result: {tool_call_id}")

    def _categorize_streaming_error(self, exception: Exception) -> str:
        """Categorize a streaming exception into a more specific error type."""
        exception_str = str(exception).lower()

        # Check for specific exception types and error patterns
        current_exception = exception
        while current_exception:
            # Timeout-related errors
            if self._is_timeout_error(current_exception, exception_str):
                return "timeout"

            # Rate limiting / quota errors
            if self._is_rate_limit_error(current_exception, exception_str):
                return "rate_limit"

            # Server overload errors
            if self._is_server_overload_error(current_exception, exception_str):
                return "server_overload"

            # Context/length limit errors
            if self._is_context_limit_error(current_exception, exception_str):
                return "context_limit"

            # Authentication errors
            if self._is_auth_error(current_exception, exception_str):
                return "auth_error"

            # Network/connection errors
            if self._is_network_error(current_exception, exception_str):
                return "network_error"

            # Check the exception chain
            current_exception = getattr(current_exception, "__cause__", None)
            if not current_exception:
                current_exception = getattr(exception, "__context__", None)
                break

        # Default fallback
        return "unknown_error"

    def _is_timeout_error(self, exception: Exception, exception_str: str) -> bool:
        """Check if exception is timeout-related."""
        # Check for specific timeout exception types
        timeout_exceptions = []

        if httpx:
            timeout_exceptions.extend(
                [
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.ConnectTimeout,
                    httpx.PoolTimeout,
                    httpx.TimeoutException,
                ]
            )

        if httpcore:
            timeout_exceptions.extend(
                [
                    httpcore.ReadTimeout,
                    httpcore.WriteTimeout,
                    httpcore.ConnectTimeout,
                    httpcore.PoolTimeout,
                ]
            )

        timeout_exceptions.append(asyncio.TimeoutError)

        # Check exception type
        if any(isinstance(exception, t) for t in timeout_exceptions):
            return True

        # Check error message
        timeout_keywords = ["timeout", "timed out", "read timeout", "write timeout"]
        return any(keyword in exception_str for keyword in timeout_keywords)

    def _is_rate_limit_error(self, exception: Exception, exception_str: str) -> bool:
        """Check if exception is rate limiting related."""
        rate_limit_keywords = [
            "rate limit",
            "quota",
            "credits",
            "billing",
            "usage limit",
            "requests per",
            "too many requests",
            "quota exceeded",
            "insufficient credits",
            "rate exceeded",
        ]
        return any(keyword in exception_str for keyword in rate_limit_keywords)

    def _is_server_overload_error(
        self, exception: Exception, exception_str: str
    ) -> bool:
        """Check if exception is server overload related."""
        overload_keywords = [
            "overloaded",
            "capacity",
            "unavailable",
            "503",
            "502",
            "500",
            "server error",
            "internal error",
            "service unavailable",
            "temporarily unavailable",
            "high demand",
        ]
        return any(keyword in exception_str for keyword in overload_keywords)

    def _is_context_limit_error(self, exception: Exception, exception_str: str) -> bool:
        """Check if exception is context length related."""
        context_keywords = [
            "context",
            "token limit",
            "maximum context",
            "context window",
            "context length",
            "token count",
            "input too long",
            "sequence length",
        ]
        return any(keyword in exception_str for keyword in context_keywords)

    def _is_auth_error(self, exception: Exception, exception_str: str) -> bool:
        """Check if exception is authentication related."""
        auth_keywords = [
            "authentication",
            "unauthorized",
            "401",
            "403",
            "api key",
            "invalid key",
            "permission",
            "access denied",
            "forbidden",
        ]
        return any(keyword in exception_str for keyword in auth_keywords)

    def _is_network_error(self, exception: Exception, exception_str: str) -> bool:
        """Check if exception is network related."""
        network_keywords = [
            "connection",
            "network",
            "dns",
            "resolve",
            "unreachable",
            "connection refused",
            "connection failed",
            "no route",
        ]
        return any(keyword in exception_str for keyword in network_keywords)

    def _get_error_note(self, error_type: str) -> str:
        """Get an appropriate note to append to partial responses based on error type."""
        error_notes = {
            "timeout": "[Note: Response was interrupted by a network timeout]",
            "rate_limit": "[Note: Response was interrupted due to rate limiting or quota exceeded]",
            "server_overload": "[Note: Response was interrupted due to server overload]",
            "context_limit": "[Note: Response was interrupted due to context length limits]",
            "auth_error": "[Note: Response was interrupted due to authentication issues]",
            "network_error": "[Note: Response was interrupted due to network connectivity issues]",
            "unknown_error": "[Note: Response was interrupted by an unexpected error]",
        }
        return error_notes.get(
            error_type, f"[Note: Response was interrupted by {error_type}]"
        )

    async def remove_node(self, node_id: int, conversation_id: Optional[str] = None) -> bool:
        """Remove a specific node from the conversation.
        
        Args:
            node_id: The node ID to remove
            conversation_id: The conversation ID (optional, uses current conversation if not provided)
        
        Returns:
            True if the node was successfully removed, False if the node was not found
        """
        # Use provided conversation_id or current conversation
        conv_id = conversation_id or self.conversation_id
        if not conv_id:
            raise ValueError("No conversation ID provided and no current conversation set")
        
        # Delegate to storage layer
        return await self.storage.remove_node(node_id, conv_id)

