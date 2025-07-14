"""Tests for compression system."""

import pytest
from datetime import datetime

from hierarchical_memory_middleware.compression import SimpleCompressor, CompressionManager
from hierarchical_memory_middleware.models import (
    ConversationNode,
    CompressionLevel,
    NodeType
)


@pytest.fixture
def sample_user_node():
    """Create a sample user node for testing."""
    return ConversationNode(
        id=1,
        conversation_id="test-conv",
        node_type=NodeType.USER,
        content="Hello, can you help me understand machine learning algorithms?",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1
    )


@pytest.fixture
def sample_ai_node():
    """Create a sample AI node for testing."""
    return ConversationNode(
        id=2,
        conversation_id="test-conv",
        node_type=NodeType.AI,
        content="""Machine learning algorithms are computational methods that enable computers to learn from data without being explicitly programmed.

There are several types:
1. Supervised learning algorithms
2. Unsupervised learning algorithms
3. Reinforcement learning algorithms

Each type has specific use cases and approaches to solving problems.""",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=8,
        ai_components={
            "assistant_text": "Detailed explanation...",
            "tool_calls": []
        }
    )


@pytest.fixture
def compressor():
    """Create a SimpleCompressor instance."""
    return SimpleCompressor(max_words=8)


def test_compress_short_content(compressor, sample_user_node):
    """Test compression of short content."""
    result = compressor.compress_node(sample_user_node)

    assert result.original_node_id == sample_user_node.id
    assert "Hello, can you help me understand" in result.compressed_content
    assert result.compressed_content.endswith("...")
    assert result.compression_ratio < 1.0
    assert len(result.topics_extracted) >= 0


def test_compress_long_content(compressor, sample_ai_node):
    """Test compression of longer content."""
    result = compressor.compress_node(sample_ai_node)

    # Should truncate to first 8 words
    words = result.compressed_content.replace("...", "").strip().split()
    assert len(words) == 8
    assert result.compressed_content.startswith("Machine learning algorithms are computational methods that")
    assert result.compressed_content.endswith("...")

    # Check metadata
    assert result.metadata["truncated"] is True
    assert result.metadata["compression_method"] == "first_n_words"
    assert result.metadata["max_words"] == 8


def test_extract_topics(compressor):
    """Test topic extraction from content."""
    text = "Machine learning algorithms are used in artificial intelligence applications for data science"
    topics = compressor._extract_simple_topics(text)

    # Should extract meaningful keywords
    assert "machine" in topics
    assert "learning" in topics
    assert "algorithms" in topics
    # Should not include stopwords
    assert "are" not in topics
    assert "in" not in topics


def test_should_compress(compressor, sample_user_node):
    """Test compression decision logic."""
    # Node at FULL level should be compressible
    assert compressor.should_compress(sample_user_node, 10)

    # Node already compressed should not be compressible
    sample_user_node.level = CompressionLevel.SUMMARY
    assert not compressor.should_compress(sample_user_node, 10)


@pytest.fixture
def compression_manager():
    """Create a CompressionManager instance."""
    compressor = SimpleCompressor(max_words=8)
    return CompressionManager(compressor, recent_node_limit=3)


def test_identify_nodes_to_compress(compression_manager):
    """Test identifying which nodes should be compressed."""
    # Create a list of nodes
    nodes = []
    for i in range(4):  # 4 conversation turns = 8 nodes
        # User node
        nodes.append(ConversationNode(
            id=i*2 + 1,
            conversation_id="test",
            node_type=NodeType.USER,
            content=f"User message {i}",
            timestamp=datetime.now(),
            sequence_number=i*2 + 1,
            line_count=1
        ))
        # AI node
        nodes.append(ConversationNode(
            id=i*2 + 2,
            conversation_id="test",
            node_type=NodeType.AI,
            content=f"AI response {i}",
            timestamp=datetime.now(),
            sequence_number=i*2 + 2,
            line_count=1
        ))

    # With recent_limit=3, should compress 8-3=5 oldest nodes
    nodes_to_compress = compression_manager.identify_nodes_to_compress(nodes)

    assert len(nodes_to_compress) == 5
    # Should be the oldest nodes (lowest sequence numbers)
    assert all(node.sequence_number <= 5 for node in nodes_to_compress)


def test_identify_nodes_under_limit(compression_manager):
    """Test when there are fewer nodes than the limit."""
    # Create only 2 nodes (under the limit of 3)
    nodes = [
        ConversationNode(
            id=1,
            conversation_id="test",
            node_type=NodeType.USER,
            content="Hello",
            timestamp=datetime.now(),
            sequence_number=1,
            line_count=1
        ),
        ConversationNode(
            id=2,
            conversation_id="test",
            node_type=NodeType.AI,
            content="Hi there",
            timestamp=datetime.now(),
            sequence_number=2,
            line_count=1
        )
    ]

    nodes_to_compress = compression_manager.identify_nodes_to_compress(nodes)
    assert len(nodes_to_compress) == 0  # Nothing to compress


def test_compress_nodes(compression_manager):
    """Test compressing a list of nodes."""
    nodes = [
        ConversationNode(
            id=1,
            conversation_id="test",
            node_type=NodeType.USER,
            content="Tell me about machine learning algorithms and their applications",
            timestamp=datetime.now(),
            sequence_number=1,
            line_count=1
        ),
        ConversationNode(
            id=2,
            conversation_id="test",
            node_type=NodeType.AI,
            content="Machine learning algorithms are computational methods for data analysis",
            timestamp=datetime.now(),
            sequence_number=2,
            line_count=1
        )
    ]

    results = compression_manager.compress_nodes(nodes)

    assert len(results) == 2
    assert all(result.original_node_id in [1, 2] for result in results)
    assert all("..." in result.compressed_content for result in results)


def test_extract_words_edge_cases(compressor):
    """Test word extraction with edge cases."""
    # Empty string
    assert compressor._extract_words("") == []

    # Only whitespace
    assert compressor._extract_words("   \n\t  ") == []

    # Multiple spaces
    words = compressor._extract_words("hello    world   test")
    assert words == ["hello", "world", "test"]

    # Single word
    assert compressor._extract_words("hello") == ["hello"]
