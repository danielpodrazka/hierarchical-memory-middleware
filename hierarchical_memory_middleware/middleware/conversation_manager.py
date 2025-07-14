"""Main conversation manager for hierarchical memory system."""

import uuid
import logging
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart

from ..config import Config
from ..storage import DuckDBStorage
from ..compression import SimpleCompressor, CompressionManager
from ..models import CompressionLevel, NodeType


logger = logging.getLogger(__name__)


class HierarchicalConversationManager:
    """Manages conversations with hierarchical memory compression."""

    def __init__(self, config: Config):
        """Initialize the conversation manager."""
        self.config = config
        self.conversation_id: Optional[str] = None

        # Initialize storage
        self.storage = DuckDBStorage(config.db_path)

        # Initialize compression system
        self.compressor = SimpleCompressor(max_words=8)
        self.compression_manager = CompressionManager(
            compressor=self.compressor,
            recent_node_limit=config.recent_node_limit
        )

        # Initialize PydanticAI agents with history processors
        self.work_agent = Agent(
            model=config.work_model,
            system_prompt="""You are a helpful AI assistant. You provide accurate and helpful responses.

            You have access to conversation memory that allows you to remember previous interactions.
            When you need to reference earlier parts of the conversation, you can do so naturally.
            """,
            history_processors=[self._hierarchical_memory_processor]
        )

        logger.info(f"Initialized HierarchicalConversationManager with model: {config.work_model}")

    async def _hierarchical_memory_processor(self, messages: List[ModelMessage]) -> List[ModelMessage]:
        """Process message history using hierarchical memory system."""
        if not self.conversation_id:
            # No conversation started yet, return messages as-is
            return messages

        try:
            # Get recent uncompressed nodes
            recent_nodes = await self.storage.get_recent_nodes(
                conversation_id=self.conversation_id,
                limit=self.config.recent_node_limit
            )

            # Get some compressed nodes for broader context
            compressed_nodes = await self.storage.get_conversation_nodes(
                conversation_id=self.conversation_id,
                limit=10,
                level=CompressionLevel.SUMMARY
            )

            # Build message history from hierarchical memory
            memory_messages = []

            # Add compressed context first (older messages)
            for node in compressed_nodes[:5]:  # Limit compressed context
                if node.node_type == NodeType.USER:
                    memory_messages.append(ModelRequest(parts=[UserPromptPart(content=node.summary or node.content)]))
                elif node.node_type == NodeType.AI:
                    memory_messages.append(ModelResponse(parts=[TextPart(content=node.summary or node.content)]))

            # Add recent full messages
            for node in recent_nodes[-8:]:  # Last 8 recent nodes
                if node.node_type == NodeType.USER:
                    memory_messages.append(ModelRequest(parts=[UserPromptPart(content=node.content)]))
                elif node.node_type == NodeType.AI:
                    memory_messages.append(ModelResponse(parts=[TextPart(content=node.content)]))

            # Combine memory with incoming messages, preferring memory over incoming history
            # Keep only the most recent message from incoming if it's new
            if messages and len(memory_messages) > 0:
                # Use memory instead of provided history
                return memory_messages
            else:
                # No memory yet or fallback to provided messages
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
                logger.warning(f"Conversation {conversation_id} not found, creating new one")
                self.conversation_id = str(uuid.uuid4())
        else:
            self.conversation_id = str(uuid.uuid4())
            logger.info(f"Starting new conversation: {self.conversation_id}")

        return self.conversation_id

    async def chat(self, user_message: str) -> str:
        """Main conversation interface with hierarchical memory integration."""
        if not self.conversation_id:
            raise ValueError("No active conversation. Call start_conversation() first.")

        try:
            # Generate AI response using PydanticAI with history processor
            # The history processor will automatically manage conversation memory
            response = await self.work_agent.run(
                user_prompt=user_message
            )

            # Save the conversation turn
            turn = await self.storage.save_conversation_turn(
                conversation_id=self.conversation_id,
                user_message=user_message,
                ai_response=response.output,
                tokens_used=getattr(response, 'usage', {}).get('total_tokens') if hasattr(response, 'usage') else None,
                ai_components={
                    "assistant_text": response.output,
                    "model_used": self.config.work_model
                }
            )

            # Check if compression is needed
            await self._check_and_compress()

            logger.info(f"Processed conversation turn {turn.turn_id} in conversation {self.conversation_id}")
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
                "recent_nodes": len([n for n in nodes if n.level == CompressionLevel.FULL]),
                "compressed_nodes": len([n for n in nodes if n.level == CompressionLevel.SUMMARY]),
                "compression_stats": stats.compression_stats if stats else {},
                "last_updated": stats.last_updated.isoformat() if stats and stats.last_updated else None
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
                conversation_id=self.conversation_id,
                query=query,
                limit=limit
            )

            return [
                {
                    "node_id": result.node.id,
                    "content": result.node.content[:200] + "..." if len(result.node.content) > 200 else result.node.content,
                    "summary": result.node.summary,
                    "relevance_score": result.relevance_score,
                    "match_type": result.match_type,
                    "timestamp": result.node.timestamp.isoformat(),
                    "node_type": result.node.node_type.value
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
            nodes_to_compress = self.compression_manager.identify_nodes_to_compress(all_nodes)

            if not nodes_to_compress:
                return

            logger.info(f"Compressing {len(nodes_to_compress)} nodes")

            # Compress the nodes
            compression_results = self.compression_manager.compress_nodes(nodes_to_compress)

            # Update nodes in storage
            for result in compression_results:
                await self.storage.compress_node(
                    node_id=result.original_node_id,
                    compression_level=CompressionLevel.SUMMARY,
                    summary=result.compressed_content,
                    metadata=result.metadata
                )

            logger.info(f"Successfully compressed {len(compression_results)} nodes")

        except Exception as e:
            logger.error(f"Error during compression: {str(e)}", exc_info=True)

    async def get_node_details(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get full details of a specific node (for expansion)."""
        try:
            node = await self.storage.get_node(node_id)
            if not node:
                return None

            return {
                "id": node.id,
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
                "ai_components": node.ai_components
            }

        except Exception as e:
            logger.error(f"Error getting node details: {str(e)}", exc_info=True)
            return None
