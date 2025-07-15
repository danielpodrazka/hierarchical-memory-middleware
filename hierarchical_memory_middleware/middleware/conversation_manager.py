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
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.mcp import MCPServerStreamableHTTP

from ..config import Config
from ..storage import DuckDBStorage
from ..compression import SimpleCompressor, CompressionManager
from ..models import CompressionLevel, NodeType


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

        # Initialize compression system
        self.compressor = SimpleCompressor(max_words=8)
        self.compression_manager = CompressionManager(
            compressor=self.compressor, recent_node_limit=config.recent_node_limit
        )

        # Initialize PydanticAI agent with optional MCP server integration
        system_prompt = """You are a helpful AI assistant. You provide accurate and helpful responses.

            You have access to conversation memory that allows you to remember previous interactions.
            When you need to reference earlier parts of the conversation, you can do so naturally."""

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

        # Create agent
        agent_kwargs = {
            "model": config.work_model,
            "system_prompt": system_prompt,
            "history_processors": [self._hierarchical_memory_processor],
        }

        if mcp_servers:
            agent_kwargs["mcp_servers"] = mcp_servers

        self.work_agent = Agent(**agent_kwargs)
        self.has_mcp_tools = bool(mcp_servers)

        logger.info(
            f"Initialized HierarchicalConversationManager with model: {config.work_model}"
            + (" and MCP tools" if self.has_mcp_tools else "")
        )

    async def _log_tool_call(
        self, ctx, call_tool, tool_name: str, args: Dict[str, Any]
    ):
        """Custom tool call processor that logs everything"""
        logger.info(f"🔧 TOOL CALL: {tool_name}")
        logger.info(f"📥 ARGS: {json.dumps(args, indent=2)}")

        try:
            # Call the actual tool
            result = await call_tool(tool_name, args)

            # Log the result (truncate if too long)
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"

            logger.info(f"📤 RESULT: {result_str}")
            logger.info(f"✅ TOOL CALL COMPLETED: {tool_name}")

            return result

        except Exception as e:
            logger.error(f"❌ TOOL CALL FAILED: {tool_name} - {str(e)}")
            raise

    def _reconstruct_ai_message_from_node(self, node, use_summary: bool = False) -> ModelResponse:
        """Reconstruct a ModelResponse from a stored AI node, preserving tool call structure."""
        try:
            # Check if we have saved full message structure
            if node.ai_components and 'full_messages' in node.ai_components:
                full_messages = node.ai_components['full_messages']
                if full_messages:
                    # Reconstruct from the first complete message
                    msg_data = full_messages[0]
                    parts = []
                    
                    for part_data in msg_data.get('parts', []):
                        if part_data['type'] == 'TextPart':
                            parts.append(TextPart(content=part_data['content']))
                        elif part_data['type'] == 'ToolCallPart':
                            parts.append(ToolCallPart(
                                tool_name=part_data['tool_name'],
                                args=part_data['args'],
                                tool_call_id=part_data['tool_call_id']
                            ))
                        elif part_data['type'] == 'ToolReturnPart':
                            parts.append(ToolReturnPart(
                                tool_name=part_data['tool_name'],
                                content=part_data['content'],
                                tool_call_id=part_data['tool_call_id']
                            ))
                    
                    if parts:
                        return ModelResponse(parts=parts)
            
            # Fallback to simple text response
            content = node.summary if use_summary else node.content
            return ModelResponse(parts=[TextPart(content=content)])
            
        except Exception as e:
            logger.debug(f"Error reconstructing AI message from node {node.node_id}: {e}")
            # Ultimate fallback
            content = node.summary if use_summary else node.content
            return ModelResponse(parts=[TextPart(content=content)])

    async def _hierarchical_memory_processor(
        self, messages: List[ModelMessage]
    ) -> List[ModelMessage]:
        """Process message history using hierarchical memory system."""
        if not self.conversation_id:
            # No conversation started yet, return messages as-is
            return messages

        try:
            # Get recent uncompressed nodes
            recent_nodes = await self.storage.get_recent_nodes(
                conversation_id=self.conversation_id,
                limit=self.config.recent_node_limit,
            )

            # Get some compressed nodes for broader context
            compressed_nodes = await self.storage.get_conversation_nodes(
                conversation_id=self.conversation_id,
                limit=10,
                level=CompressionLevel.SUMMARY,
            )

            # Build message history from hierarchical memory
            memory_messages = []

            # Add compressed context first (older messages)
            for node in compressed_nodes[:5]:  # Limit compressed context
                if node.node_type == NodeType.USER:
                    memory_messages.append(
                        ModelRequest(
                            parts=[UserPromptPart(content=node.summary or node.content)]
                        )
                    )
                elif node.node_type == NodeType.AI:
                    # Try to reconstruct full message structure for compressed nodes
                    reconstructed_msg = self._reconstruct_ai_message_from_node(node, use_summary=True)
                    memory_messages.append(reconstructed_msg)

            # Add recent full messages
            for node in recent_nodes[-8:]:  # Last 8 recent nodes
                if node.node_type == NodeType.USER:
                    memory_messages.append(
                        ModelRequest(parts=[UserPromptPart(content=node.content)])
                    )
                elif node.node_type == NodeType.AI:
                    # Try to reconstruct full message structure for recent nodes
                    reconstructed_msg = self._reconstruct_ai_message_from_node(node, use_summary=False)
                    memory_messages.append(reconstructed_msg)

            logger.debug(
                f"Memory processor: found {len(memory_messages)} memory messages, {len(messages)} incoming messages"
            )

            # Combine memory with current/new user message
            if len(memory_messages) > 0:
                # Start with conversation memory
                combined_messages = memory_messages.copy()

                # Add recent messages to preserve tool use/result pairs and current context
                # We need to preserve the complete recent conversation including tool interactions
                recent_message_limit = (
                    5  # Keep last few messages to preserve tool chains
                )
                recent_messages = (
                    messages[-recent_message_limit:]
                    if len(messages) > recent_message_limit
                    else messages
                )

                for msg in recent_messages:
                    # Add all recent messages to preserve tool use/result pairing
                    # This includes ModelRequest (user), ModelResponse (assistant), and any tool messages
                    combined_messages.append(msg)

                logger.debug(
                    f"Memory processor: returning {len(combined_messages)} total messages ({len(memory_messages)} from memory + {len(recent_messages)} recent)"
                )
                return combined_messages
            else:
                # No memory yet, use provided messages as-is
                return messages

        except Exception as e:
            logger.error(f"Error in history processor: {str(e)}", exc_info=True)
            # Fallback to provided messages on error
            return messages

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

    async def chat(self, user_message: str) -> str:
        """Main conversation interface with hierarchical memory integration."""
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        try:
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

            # Extract full message structure from response
            full_messages = []
            try:
                # The response should contain the full conversation turn
                # including tool calls and results
                if hasattr(response, 'all_messages') and response.all_messages:
                    # Get the AI messages from this turn (excluding the user message)
                    for msg in response.all_messages():
                        if isinstance(msg, ModelResponse):
                            # Serialize the full message structure
                            msg_dict = {
                                'type': 'ModelResponse',
                                'parts': []
                            }
                            for part in msg.parts:
                                if isinstance(part, TextPart):
                                    msg_dict['parts'].append({
                                        'type': 'TextPart',
                                        'content': part.content
                                    })
                                elif isinstance(part, ToolCallPart):
                                    msg_dict['parts'].append({
                                        'type': 'ToolCallPart',
                                        'tool_name': part.tool_name,
                                        'args': part.args,
                                        'tool_call_id': part.tool_call_id
                                    })
                                elif isinstance(part, ToolReturnPart):
                                    msg_dict['parts'].append({
                                        'type': 'ToolReturnPart',
                                        'tool_name': part.tool_name,
                                        'content': part.content,
                                        'tool_call_id': part.tool_call_id
                                    })
                            full_messages.append(msg_dict)
            except Exception as e:
                logger.debug(f"Could not extract full message structure: {e}")

            ai_node = await self.storage.save_conversation_node(
                conversation_id=self.conversation_id,
                node_type=NodeType.AI,
                content=response.output,
                tokens_used=tokens_used,
                ai_components={
                    "assistant_text": response.output,
                    "model_used": self.config.work_model,
                    "full_messages": full_messages,  # Save complete message structure
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

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversation memory."""
        if not self.conversation_id:
            return []

        try:
            results = await self.storage.search_nodes(
                conversation_id=self.conversation_id, query=query, limit=limit
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
        """Check if compression is needed and perform it."""
        try:
            # Get all nodes for this conversation
            all_nodes = await self.storage.get_conversation_nodes(
                conversation_id=self.conversation_id
            )

            # Identify nodes that need compression
            nodes_to_compress = self.compression_manager.identify_nodes_to_compress(
                all_nodes
            )

            if not nodes_to_compress:
                return

            logger.info(f"Compressing {len(nodes_to_compress)} nodes")

            # Compress the nodes
            compression_results = self.compression_manager.compress_nodes(
                nodes_to_compress
            )

            # Update nodes in storage
            for result in compression_results:
                await self.storage.compress_node(
                    node_id=result.original_node_id,
                    conversation_id=self.conversation_id,
                    compression_level=CompressionLevel.SUMMARY,
                    summary=result.compressed_content,
                    metadata=result.metadata,
                )

            logger.info(f"Successfully compressed {len(compression_results)} nodes")

        except Exception as e:
            logger.error(f"Error during compression: {str(e)}", exc_info=True)

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
