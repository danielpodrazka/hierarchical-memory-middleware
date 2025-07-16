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
from hierarchical_memory_middleware.advanced_hierarchy import AdvancedCompressionManager


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

    # === TF-IDF Corpus and Topic Extraction Tests ===

    @pytest.mark.asyncio
    async def test_tfidf_corpus_building_during_compression(self, conversation_manager):
        """Verify that TF-IDF compressor builds corpus from conversation nodes and extracts topics during compression."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Create nodes with machine learning content
        test_nodes = [
            (NodeType.USER, "Tell me about machine learning algorithms"),
            (NodeType.AI, "Machine learning algorithms are computational methods that enable systems to learn from data without being explicitly programmed. They include supervised learning, unsupervised learning, and reinforcement learning."),
            (NodeType.USER, "What about neural networks?"),
            (NodeType.AI, "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information through layers."),
            (NodeType.USER, "How do transformers work?"),
            (NodeType.AI, "Transformers are a type of neural network architecture that uses attention mechanisms to process sequences of data efficiently."),
        ]

        # Add nodes to storage
        for node_type, content in test_nodes:
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=node_type,
                content=content,
            )

        # Get the compressor and verify corpus is built
        compressor = conversation_manager.compressor
        assert hasattr(compressor, 'topic_extractor')
        assert compressor.topic_extractor is not None

        # Build TF-IDF corpus from all nodes before compression
        all_nodes = await conversation_manager.storage.get_conversation_nodes(conversation_id)
        compressor.build_corpus(all_nodes)

        # Trigger compression to build corpus
        await conversation_manager._check_and_compress()

        # Verify corpus was built
        assert compressor.topic_extractor.vectorizer is not None  # Vectorizer should be fitted
        assert len(compressor.topic_extractor.document_corpus) > 0

        # Verify topics can be extracted
        test_content = "Machine learning algorithms are powerful computational methods"
        topics = compressor.topic_extractor.extract_topics(test_content)
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert any('machine' in topic or 'learning' in topic for topic in topics)

    @pytest.mark.asyncio
    async def test_topic_extraction_integration_with_compression_manager(self, conversation_manager):
        """Verify that the compression manager properly calls TF-IDF topic extraction and stores topics in nodes."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add content with clear topics
        content_with_topics = "Natural language processing uses machine learning algorithms to analyze text. Deep learning models like transformers and neural networks are commonly used for language tasks."

        # Add enough nodes to exceed recent_node_limit (3) to trigger compression
        nodes_to_add = [
            (NodeType.USER, "Tell me about NLP"),
            (NodeType.AI, content_with_topics),
            (NodeType.USER, "What about machine learning?"),
            (NodeType.AI, "Machine learning involves algorithms that learn from data"),
            (NodeType.USER, "Explain deep learning"),
            (NodeType.AI, "Deep learning uses neural networks with multiple layers"),
        ]

        for node_type, content in nodes_to_add:
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=node_type,
                content=content,
            )

        # Build TF-IDF corpus
        compressor = conversation_manager.compressor
        # Add all content to the corpus
        for node_type, content in nodes_to_add:
            compressor.topic_extractor.add_document(content)
        compressor.topic_extractor.fit()

        # Test compression manager
        all_nodes = await conversation_manager.storage.get_conversation_nodes(conversation_id)
        compression_manager = conversation_manager.simple_compression_manager
        
        # Identify nodes to compress
        nodes_to_compress = compression_manager.identify_nodes_to_compress(all_nodes)
        assert len(nodes_to_compress) > 0

        # Compress nodes (this should extract topics)
        results = compression_manager.compress_nodes(nodes_to_compress, all_nodes=all_nodes)
        assert len(results) > 0

        # Verify topics were extracted
        topics_found = False
        for result in results:
            assert hasattr(result, 'topics_extracted')
            assert isinstance(result.topics_extracted, list)
            # Should have some topics for content with clear technical terms
            if len(result.topics_extracted) > 0:
                topics_found = True
                topic_words = ' '.join(result.topics_extracted)
                # Should contain some meaningful words (not just short/common words)
                assert any(len(word) > 3 for word in result.topics_extracted), f"Topics should contain meaningful words, got: {result.topics_extracted}"
        
        # At least one result should have topics
        assert topics_found, "At least one compression result should have topics extracted"

    # === Topic Storage and Retrieval Tests ===

    @pytest.mark.asyncio
    async def test_topics_stored_in_database_during_compression(self, conversation_manager):
        """Verify that topics extracted by TF-IDF are properly stored in the database when nodes are compressed."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add a node with content
        ai_node = await conversation_manager.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content="Machine learning algorithms use statistical methods to learn patterns from data. Deep learning is a subset of machine learning that uses neural networks.",
        )

        # Manually compress with topics
        test_topics = ["machine", "learning", "algorithms", "statistical"]
        await conversation_manager.storage.compress_node(
            node_id=ai_node.node_id,
            conversation_id=conversation_id,
            compression_level=CompressionLevel.SUMMARY,
            summary="Discussion about machine learning algorithms",
            topics=test_topics,
        )

        # Retrieve the compressed node
        compressed_node = await conversation_manager.storage.get_node(ai_node.node_id, conversation_id)
        assert compressed_node is not None
        assert compressed_node.level == CompressionLevel.SUMMARY
        assert compressed_node.topics is not None
        assert isinstance(compressed_node.topics, list)
        assert len(compressed_node.topics) > 0
        assert all(topic in compressed_node.topics for topic in test_topics)

    @pytest.mark.asyncio
    async def test_enhanced_summary_includes_topics_in_ai_view(self, conversation_manager):
        """Verify that enhanced summaries with topics are properly displayed in AI view compressed nodes."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add nodes and compress with topics
        ai_node = await conversation_manager.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content="Natural language processing involves computational linguistics and machine learning techniques.",
        )

        # Compress with topics
        topics = ["natural", "language", "processing", "computational"]
        await conversation_manager.storage.compress_node(
            node_id=ai_node.node_id,
            conversation_id=conversation_id,
            compression_level=CompressionLevel.SUMMARY,
            summary="Discussion about natural language processing",
            topics=topics,
        )

        # Test memory processor to get AI view
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        mock_messages = [ModelRequest(parts=[UserPromptPart(content="Tell me more")])]

        # Call memory processor
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)
        
        # Get AI view data
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None

        # Check that compressed nodes have enhanced summaries with topics
        compressed_nodes = ai_view_data.get('compressed_nodes', [])
        found_enhanced_summary = False
        for node in compressed_nodes:
            if '[Topics:' in node.get('content', ''):
                found_enhanced_summary = True
                content = node['content']
                assert 'natural' in content
                assert 'language' in content
                assert 'processing' in content
                break

        assert found_enhanced_summary, "Enhanced summary with topics should be present in AI view"

    # === Debug and Diagnostic Tests ===

    @pytest.mark.asyncio
    async def test_compression_manager_topic_extraction_debug(self, conversation_manager):
        """Debug test to verify each step of topic extraction during compression."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add node with clear technical content
        test_content = "Machine learning algorithms use statistical methods to learn patterns from data. Neural networks are computational models inspired by biological brain networks."
        
        ai_node = await conversation_manager.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content=test_content,
        )

        # Debug: Check TF-IDF compressor setup
        compressor = conversation_manager.compressor
        assert isinstance(compressor, TfidfCompressor)
        assert hasattr(compressor, 'topic_extractor')

        # Debug: Build corpus manually
        compressor.topic_extractor.add_document(test_content)
        compressor.topic_extractor.fit()
        assert compressor.topic_extractor.vectorizer is not None  # Vectorizer should be fitted

        # Debug: Test topic extraction
        topics = compressor.topic_extractor.extract_topics(test_content)
        assert isinstance(topics, list)
        assert len(topics) > 0

        # Debug: Test compression with topic extraction
        result = compressor.compress_node(ai_node)
        assert result is not None
        assert hasattr(result, 'topics_extracted')
        assert isinstance(result.topics_extracted, list)
        assert len(result.topics_extracted) > 0

        # Debug: Verify topics are meaningful
        topic_text = ' '.join(result.topics_extracted)
        assert any(word in topic_text for word in ['machine', 'learning', 'neural', 'network', 'data'])

    @pytest.mark.asyncio
    async def test_memory_processor_ai_view_data_debug(self, conversation_manager):
        """Debug test to verify AI view data is set at each step of memory processing."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add some test data
        for i in range(3):
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.USER,
                content=f"User message {i}",
            )
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.AI,
                content=f"AI response {i} with technical content",
            )

        # Debug: Check initial state
        initial_ai_view = conversation_manager.get_last_ai_view_data()
        assert initial_ai_view is None  # Should be None before processing

        # Debug: Test memory processor
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        mock_messages = [ModelRequest(parts=[UserPromptPart(content="Debug test")])]

        # Debug: Call memory processor
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)
        
        # Debug: Check AI view data is set
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None, "AI view data should be set after memory processing"

        # Debug: Verify all required fields
        required_fields = ['compressed_nodes', 'recent_nodes', 'recent_messages_from_input', 'total_messages_sent_to_ai']
        for field in required_fields:
            assert field in ai_view_data, f"Missing required field: {field}"

        # Debug: Verify data structure
        assert isinstance(ai_view_data['compressed_nodes'], list)
        assert isinstance(ai_view_data['recent_nodes'], list)
        assert isinstance(ai_view_data['recent_messages_from_input'], list)
        assert isinstance(ai_view_data['total_messages_sent_to_ai'], int)
        assert ai_view_data['total_messages_sent_to_ai'] > 0

        # Debug: Verify processed messages
        assert len(processed_messages) > 0
        assert len(processed_messages) == ai_view_data['total_messages_sent_to_ai']

        # Debug: Verify node structure
        all_nodes = ai_view_data['compressed_nodes'] + ai_view_data['recent_nodes']
        for node in all_nodes:
            assert 'node_id' in node
            assert 'node_type' in node
            assert 'content' in node
            assert node['node_type'] in ['user', 'ai']
            assert isinstance(node['content'], str)
            assert len(node['content']) > 0

    @pytest.mark.asyncio
    async def test_chat_to_compression_to_ai_view_flow(self, conversation_manager):
        """Test complete flow: chat messages → compression with topics → AI view shows topics."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Simulate chat messages with rich content
        chat_messages = [
            "What are machine learning algorithms?",
            "Can you explain neural networks?",
            "How do transformers work in NLP?",
            "What about deep learning?",
            "Tell me about computer vision",
            "How does reinforcement learning work?",
        ]

        # Add messages to storage (simulating chat flow)
        for i, msg in enumerate(chat_messages):
            # User message
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.USER,
                content=msg,
            )

            # AI response with technical content
            ai_response = f"Response {i}: {msg} involves computational methods and algorithms that process data using statistical techniques and neural network architectures."
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.AI,
                content=ai_response,
            )

        # Trigger compression (this should extract topics)
        await conversation_manager._check_and_compress()

        # Test memory processor to get AI view
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        mock_messages = [ModelRequest(parts=[UserPromptPart(content="What have we discussed?")])]

        # Call memory processor
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)
        
        # Get AI view data
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None

        # Verify AI view contains proper data
        assert 'compressed_nodes' in ai_view_data
        assert 'recent_nodes' in ai_view_data
        assert 'total_messages_sent_to_ai' in ai_view_data
        assert ai_view_data['total_messages_sent_to_ai'] > 0

        # Check that some nodes have content
        all_nodes = ai_view_data.get('compressed_nodes', []) + ai_view_data.get('recent_nodes', [])
        assert len(all_nodes) > 0

        # Verify nodes have proper structure
        for node in all_nodes:
            assert 'node_id' in node
            assert 'node_type' in node
            assert 'content' in node
            assert isinstance(node['content'], str)
            assert len(node['content']) > 0

    @pytest.mark.asyncio
    async def test_ai_view_with_meta_level_compressions(self, conversation_manager):
        """Verify that AI view works correctly when META level compressions are present."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Create enough nodes to trigger META level compression
        for i in range(25):  # More than meta_threshold to trigger META
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.USER,
                content=f"User message {i} about machine learning and AI",
            )
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.AI,
                content=f"AI response {i} discussing machine learning algorithms and neural networks",
            )

        # Trigger advanced compression
        await conversation_manager._check_and_compress()

        # Test memory processor with META level nodes
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        mock_messages = [ModelRequest(parts=[UserPromptPart(content="What did we discuss?")])]

        # Call memory processor
        processed_messages = await conversation_manager._hierarchical_memory_processor(mock_messages)
        
        # Get AI view data
        ai_view_data = conversation_manager.get_last_ai_view_data()
        assert ai_view_data is not None

        # Check that compressed nodes include META level nodes
        compressed_nodes = ai_view_data.get('compressed_nodes', [])
        assert len(compressed_nodes) > 0

        # The AI view should work with META level nodes
        assert 'total_messages_sent_to_ai' in ai_view_data
        assert ai_view_data['total_messages_sent_to_ai'] > 0

    @pytest.mark.asyncio
    async def test_advanced_compression_manager_uses_tfidf(self, conversation_manager):
        """Verify that AdvancedCompressionManager properly integrates with TF-IDF compressor."""
        # Start conversation
        conversation_id = await conversation_manager.start_conversation()

        # Add content with clear topics
        content_nodes = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are inspired by biological brain networks",
            "Deep learning uses multiple layers of neural networks",
            "Natural language processing analyzes human language",
            "Computer vision processes and analyzes visual data",
        ]

        for content in content_nodes:
            await conversation_manager.storage.save_conversation_node(
                conversation_id=conversation_id,
                node_type=NodeType.AI,
                content=content,
            )

        # Get advanced compression manager
        advanced_manager = conversation_manager.compression_manager
        assert isinstance(advanced_manager, AdvancedCompressionManager)

        # Verify it uses TF-IDF compressor
        assert hasattr(advanced_manager, 'hierarchy_manager')
        assert hasattr(advanced_manager.hierarchy_manager, 'base_compressor')
        assert isinstance(advanced_manager.hierarchy_manager.base_compressor, TfidfCompressor)

        # Get all nodes
        all_nodes = await conversation_manager.storage.get_conversation_nodes(conversation_id)
        assert len(all_nodes) > 0

        # Test compression processing
        compression_results = await advanced_manager.process_hierarchy_compression(
            nodes=all_nodes,
            storage=conversation_manager.storage
        )

        # Verify compression results
        assert compression_results is not None
        assert isinstance(compression_results, dict)
        assert 'total_processed' in compression_results
