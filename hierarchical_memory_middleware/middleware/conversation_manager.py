"""Main conversation manager for hierarchical memory system."""

import json
import uuid
import logging
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai.mcp import MCPServerStreamableHTTP

from ..config import Config
from ..storage import DuckDBStorage
from ..compression import SimpleCompressor, CompressionManager
from ..advanced_hierarchy import AdvancedCompressionManager
from ..models import CompressionLevel, NodeType, HierarchyThresholds
from ..model_manager import ModelManager


logger = logging.getLogger(__name__)


class HierarchicalConversationManager:
    """Manages conversations with hierarchical memory compression."""

    def __init__(
        self,
        config: Config,
        storage: Optional[DuckDBStorage] = None,
        mcp_server_url: Optional[str] = None,
    ):
        """Initialize the conversation manager."""
        self.config = config
        self.conversation_id: Optional[str] = None

        # Initialize storage
        self.storage = storage or DuckDBStorage(config.db_path)

        # Initialize advanced hierarchical compression system
        self.compressor = SimpleCompressor(max_words=8)

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

        # Create MCP server client if URL provided
        mcp_servers = []
        if mcp_server_url:
            mcp_server = MCPServerStreamableHTTP(
                url=mcp_server_url,
                tool_prefix="memory",  # Prefix tools with 'memory_'
                process_tool_call=self._log_tool_call
                if config.log_tool_calls
                else None,
            )
            mcp_servers.append(mcp_server)

        # Create agent with model manager
        try:
            # Use model manager to create the appropriate model instance
            model_instance = ModelManager.create_model(config.work_model)
            logger.info(f"Successfully created model instance for: {config.work_model}")
        except Exception as e:
            logger.error(f"Failed to create model {config.work_model}: {str(e)}")
            raise ValueError(f"Unable to initialize model '{config.work_model}': {str(e)}")
        
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

        logger.info(
            f"Initialized HierarchicalConversationManager with model: {config.work_model}"
            + (" and MCP tools" if self.has_mcp_tools else "")
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
            logger.info(f"✅ TOOL CALL COMPLETED: {tool_name}")

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
            logger.error(f"❌ TOOL CALL FAILED: {tool_name} - {str(e)}")
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
            logger.debug("Active tool calls detected, skipping memory processing")
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
                                node.content if node.level == CompressionLevel.META
                                else (node.summary or node.content)
                            ),
                            "is_summary": True,
                            "sequence_number": node.sequence_number,
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
                    if hasattr(msg, "parts"):
                        # Extract content from message parts
                        content_parts = []
                        for part in msg.parts:
                            if hasattr(part, "content"):
                                content_parts.append(part.content)
                            elif hasattr(part, "tool_name"):
                                content_parts.append(f"[Tool call: {part.tool_name}]")
                        content = " ".join(content_parts) if content_parts else str(msg)
                    else:
                        content = str(msg)

                    ai_view_data["recent_messages_from_input"].append(
                        {
                            "message_type": msg.__class__.__name__,
                            "content": content,
                        }
                    )

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
                    if hasattr(msg, "parts"):
                        content_parts = []
                        for part in msg.parts:
                            if hasattr(part, "content"):
                                content_parts.append(part.content)
                            elif hasattr(part, "tool_name"):
                                content_parts.append(f"[Tool call: {part.tool_name}]")
                        content = " ".join(content_parts) if content_parts else str(msg)
                    else:
                        content = str(msg)

                    ai_view_data["recent_messages_from_input"].append(
                        {
                            "message_type": msg.__class__.__name__,
                            "content": content,
                        }
                    )

                ai_view_data["total_messages_sent_to_ai"] = len(messages)
                self._last_ai_view_data = ai_view_data
                return messages

        except Exception as e:
            logger.error(f"Error in history processor: {str(e)}", exc_info=True)
            # Fallback to provided messages on error to preserve tool flows
            return messages

    def get_last_ai_view_data(self) -> Optional[Dict[str, Any]]:
        """Get the last captured AI view data from the memory processor.

        Returns:
            Dictionary containing what the AI actually saw in the last message processing,
            or None if no data is available (e.g., no conversation started, tool calls active).
        """
        return self._last_ai_view_data

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
            if self.has_mcp_tools:
                # Use MCP context manager for tools
                async with self.work_agent.run_mcp_servers():
                    response = await self.work_agent.run(user_prompt=user_message)
            else:
                # No MCP tools, run directly
                response = await self.work_agent.run(user_prompt=user_message)

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

            logger.info(
                f"Processed conversation turn (user: {user_node.node_id}, ai: {ai_node.node_id}) in conversation {self.conversation_id}"
            )
            return response.output

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an error processing your message: {str(e)}"

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        if not self.conversation_id:
            return {"error": "No active conversation"}

        try:
            stats = await self.storage.get_conversation_stats(self.conversation_id)
            nodes = await self.storage.get_conversation_nodes(self.conversation_id)

            return {
                "conversation_id": self.conversation_id,
                "total_nodes": len(nodes),
                "recent_nodes": len(
                    [n for n in nodes if n.level == CompressionLevel.FULL]
                ),
                "compressed_nodes": len(
                    [n for n in nodes if n.level == CompressionLevel.SUMMARY]
                ),
                "compression_stats": stats.compression_stats if stats else {},
                "last_updated": stats.last_updated.isoformat()
                if stats and stats.last_updated
                else None,
            }

        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}", exc_info=True)
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
            logger.error(f"Error searching memory: {str(e)}", exc_info=True)
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

            logger.info(f"Checking hierarchy compression for {len(all_nodes)} nodes")

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
                logger.info(
                    f"Advanced hierarchy compression completed: "
                    f"{compression_results.get('summary_compressed', 0)} summary compressions, "
                    f"{compression_results.get('meta_groups_created', 0)} META groups created, "
                    f"{compression_results.get('archive_compressed', 0)} archive compressions"
                )
            else:
                logger.debug("No hierarchy compression needed at this time")

        except Exception as e:
            logger.error(
                f"Error during advanced hierarchy compression: {str(e)}", exc_info=True
            )

            # Fallback to simple compression if advanced fails
            try:
                logger.info("Falling back to simple compression system")

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

                    logger.info(
                        f"Fallback compression completed: {len(compression_results)} nodes"
                    )

            except Exception as fallback_error:
                logger.error(
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
            logger.error(f"Error getting node details: {str(e)}", exc_info=True)
            return None
