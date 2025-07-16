#!/usr/bin/env python3
"""Tests for TF-IDF AI view functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import (
    HierarchicalConversationManager,
)
from hierarchical_memory_middleware.storage import DuckDBStorage
from hierarchical_memory_middleware.models import (
    ConversationNode,
    NodeType,
    CompressionLevel,
)
from hierarchical_memory_middleware.compression import TfidfCompressor


class TestTfidfAiView:
    """Test TF-IDF functionality with AI view."""

    @pytest.fixture
    def config(self, monkeypatch):
        """Create test configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")
        return Config(
            db_path=":memory:",
            recent_node_limit=3,
            summary_threshold=5,
            work_model="claude-4-haiku",
        )

    @pytest.fixture
    def storage(self):
        """Create test storage."""
        return DuckDBStorage(":memory:")

    @pytest.fixture
    def conversation_manager(self, config, storage):
        """Create test conversation manager."""
        return HierarchicalConversationManager(config, storage=storage)

    @pytest.mark.asyncio
    async def test_tfidf_compression_with_ai_view(self, conversation_manager):
        """Test that TF-IDF compression works with AI view."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add some nodes manually to trigger compression and AI view
        # This simulates what happens during normal chat operations
        test_messages = [
            "Hello, I want to learn about machine learning",
            "Can you explain neural networks?",
            "What about deep learning algorithms?",
            "Tell me about natural language processing",
            "How do transformers work?",
            "What is attention mechanism?",
        ]

        # Add nodes to storage to simulate chat history
        for i, message in enumerate(test_messages):
            # Add user message
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.USER,
                content=message,
            )
            
            # Add AI response
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.AI,
                content=f"I'd be happy to help you learn about {message.split()[-1] if message.split() else 'that topic'}!",
                ai_components={
                    "assistant_text": f"Response to: {message}",
                    "model_used": "claude-4-haiku",
                    "tool_calls": [],
                    "tool_results": [],
                },
            )

        # Trigger compression manually
        await conversation_manager._check_and_compress()

        # Now simulate what happens during message processing
        # Create some mock messages to pass to the memory processor
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        
        mock_messages = [
            ModelRequest(parts=[UserPromptPart(content="Tell me about AI")])
        ]

        # Call the memory processor directly to test AI view functionality
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)

        # Check that AI view data is available
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None, "AI view data should be available after memory processing"
        assert 'compressed_nodes' in ai_view_data
        assert 'recent_nodes' in ai_view_data
        assert 'total_messages_sent_to_ai' in ai_view_data

        # Check that compressed nodes have properly formatted content
        compressed_nodes = ai_view_data.get('compressed_nodes', [])
        for node in compressed_nodes:
            assert 'node_id' in node
            assert 'content' in node
            # Content should be a string
            assert isinstance(node['content'], str)
            assert len(node['content']) > 0

        # Check that recent nodes are also properly formatted
        recent_nodes = ai_view_data.get('recent_nodes', [])
        for node in recent_nodes:
            assert 'node_id' in node
            assert 'content' in node
            assert isinstance(node['content'], str)
            assert len(node['content']) > 0

        # Verify that the memory processor returned the expected messages
        assert len(processed_messages) > 0
        assert ai_view_data['total_messages_sent_to_ai'] == len(processed_messages)

    # Additional test method to verify TF-IDF topics are handled correctly
    @pytest.mark.asyncio
    async def test_tfidf_topics_in_ai_view(self, conversation_manager):
        """Test that TF-IDF topics are properly handled in AI view."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add a node with TF-IDF topics
        user_node = await conversation_manager.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.USER,
            content="Tell me about machine learning algorithms",
        )

        ai_node = await conversation_manager.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content="Machine learning algorithms are computational methods...",
            topics=["machine", "learning", "algorithms", "computational"],
        )

        # Compress the node
        await conversation_manager.storage.compress_node(
            node_id=ai_node.node_id,
            conversation_id=conversation_id,
            compression_level=CompressionLevel.SUMMARY,
            summary="Discussion about machine learning algorithms",
            topics=["machine", "learning", "algorithms"],
        )

        # Test the memory processor with compressed nodes
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        
        mock_messages = [
            ModelRequest(parts=[UserPromptPart(content="Tell me more")])
        ]

        # Call the memory processor
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)

        # Check that AI view data includes topics
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None
        
        # Check that compressed nodes contain topics information
        compressed_nodes = ai_view_data.get('compressed_nodes', [])
        found_topics = False
        for node in compressed_nodes:
            if 'Topics:' in node.get('content', ''):
                found_topics = True
                assert 'machine' in node['content']
                assert 'learning' in node['content']
                break
        
        # Topics should be present in enhanced summaries
        assert found_topics, "TF-IDF topics should be present in enhanced summaries"

    @pytest.mark.asyncio
    async def test_enhance_node_summary_with_topics(self, storage):
        """Test that node summary enhancement works with TF-IDF topics."""
        # Create a test node with TF-IDF topics
        node = ConversationNode(
            node_id=1,
            conversation_id="test",
            node_type=NodeType.AI,
            content="Machine learning is a subset of artificial intelligence.",
            timestamp=datetime.now(),
            sequence_number=1,
            line_count=1,
            level=CompressionLevel.SUMMARY,
            summary="Discussion about machine learning concepts",
            topics=["machine", "learning", "artificial", "intelligence"],
        )

        # Test enhancement with valid topics
        enhanced_node = storage._enhance_node_summary(node)
        assert enhanced_node.summary is not None
        assert "machine, learning, artificial" in enhanced_node.summary
        assert "[Topics:" in enhanced_node.summary

        # Test enhancement with empty topics
        node_empty_topics = ConversationNode(
            node_id=2,
            conversation_id="test",
            node_type=NodeType.AI,
            content="Another test message.",
            timestamp=datetime.now(),
            sequence_number=2,
            line_count=1,
            level=CompressionLevel.SUMMARY,
            summary="Another discussion",
            topics=[],
        )

        enhanced_node_empty = storage._enhance_node_summary(node_empty_topics)
        assert enhanced_node_empty.summary is not None
        assert "[Topics:" not in enhanced_node_empty.summary

        # Test enhancement with None topics
        node_none_topics = ConversationNode(
            node_id=3,
            conversation_id="test",
            node_type=NodeType.AI,
            content="Yet another test message.",
            timestamp=datetime.now(),
            sequence_number=3,
            line_count=1,
            level=CompressionLevel.SUMMARY,
            summary="Yet another discussion",
            topics=None,
        )

        enhanced_node_none = storage._enhance_node_summary(node_none_topics)
        assert enhanced_node_none.summary is not None
        assert "[Topics:" not in enhanced_node_none.summary

    @pytest.mark.asyncio
    async def test_json_serialization_with_topics(self, storage):
        """Test that JSON serialization works with TF-IDF topics."""
        # Create nodes with various topic configurations
        nodes = [
            ConversationNode(
                node_id=1,
                conversation_id="test",
                node_type=NodeType.AI,
                content="Content 1",
                timestamp=datetime.now(),
                sequence_number=1,
                line_count=1,
                topics=["topic1", "topic2"],
            ),
            ConversationNode(
                node_id=2,
                conversation_id="test",
                node_type=NodeType.AI,
                content="Content 2",
                timestamp=datetime.now(),
                sequence_number=2,
                line_count=1,
                topics=[],
            ),
            ConversationNode(
                node_id=3,
                conversation_id="test",
                node_type=NodeType.AI,
                content="Content 3",
                timestamp=datetime.now(),
                sequence_number=3,
                line_count=1,
                topics=None,
            ),
        ]

        # Test that all nodes can be processed without errors
        for node in nodes:
            # Test model validation
            assert node.topics is not None  # Should be converted to [] by validator
            assert isinstance(node.topics, list)

            # Test JSON serialization (this simulates what happens in save_conversation_to_json)
            import json

            topics_json = json.dumps(
                node.topics if node.topics and isinstance(node.topics, list) else []
            )
            assert topics_json is not None

            # Test deserialization
            topics_back = json.loads(topics_json)
            assert isinstance(topics_back, list)

    @pytest.mark.asyncio
    async def test_tfidf_compressor_error_handling(self):
        """Test TF-IDF compressor error handling."""
        compressor = TfidfCompressor(max_words=8)

        # Test with valid node
        node = ConversationNode(
            node_id=1,
            conversation_id="test",
            node_type=NodeType.USER,
            content="This is a test message about machine learning algorithms.",
            timestamp=datetime.now(),
            sequence_number=1,
            line_count=1,
        )

        # Ensure corpus is fitted
        compressor.topic_extractor.add_document(node.content)
        compressor.topic_extractor.fit()

        # Test compression
        result = compressor.compress_node(node)
        assert result.compressed_content is not None
        assert result.topics_extracted is not None
        assert isinstance(result.topics_extracted, list)

        # Test with empty content
        empty_node = ConversationNode(
            node_id=2,
            conversation_id="test",
            node_type=NodeType.USER,
            content="",
            timestamp=datetime.now(),
            sequence_number=2,
            line_count=1,
        )

        result_empty = compressor.compress_node(empty_node)
        assert result_empty.compressed_content is not None
        assert result_empty.topics_extracted is not None
        assert isinstance(result_empty.topics_extracted, list)
