"""Tests for storage layer."""

import pytest
import tempfile
import os
from datetime import datetime

from hierarchical_memory_middleware.storage import DuckDBStorage
from hierarchical_memory_middleware.models import CompressionLevel, NodeType


@pytest.fixture
def storage():
    """Create a storage instance for testing."""
    return DuckDBStorage(":memory:")


@pytest.mark.asyncio
async def test_save_conversation_node(storage):
    """Test saving conversation nodes."""
    conversation_id = "test-conv-1"
    user_message = "Hello, how are you?"
    ai_response = "I'm doing well, thank you for asking!"

    # Save user node
    user_node = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content=user_message,
    )

    # Save AI node
    ai_node = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content=ai_response,
        tokens_used=100,
        ai_components={
            "assistant_text": ai_response,
            "model_used": "test-model",
        },
    )

    # Test user node
    assert user_node.conversation_id == conversation_id
    assert user_node.content == user_message
    assert user_node.node_type == NodeType.USER
    assert user_node.sequence_number == 0
    assert user_node.node_id is not None

    # Test AI node
    assert ai_node.conversation_id == conversation_id
    assert ai_node.content == ai_response
    assert ai_node.node_type == NodeType.AI
    assert ai_node.sequence_number == 1
    assert ai_node.tokens_used == 100
    assert ai_node.ai_components["assistant_text"] == ai_response
    assert ai_node.ai_components["model_used"] == "test-model"
    assert ai_node.node_id is not None


@pytest.mark.asyncio
async def test_get_conversation_nodes(storage):
    """Test retrieving conversation nodes."""
    conversation_id = "test-conv-2"

    # Save conversation nodes
    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="What is Python?",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Python is a programming language.",
    )

    # Get nodes
    nodes = await storage.get_conversation_nodes(conversation_id)

    assert len(nodes) == 2
    assert nodes[0].node_type == NodeType.USER
    assert nodes[1].node_type == NodeType.AI
    assert nodes[0].content == "What is Python?"
    assert nodes[1].content == "Python is a programming language."


@pytest.mark.asyncio
async def test_compress_node(storage):
    """Test node compression."""
    conversation_id = "test-conv-3"

    # Save conversation nodes
    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="Tell me about machine learning algorithms",
    )

    ai_node = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Machine learning algorithms are computational methods that learn from data.",
    )

    # Compress the AI node
    success = await storage.compress_node(
        node_id=ai_node.node_id,
        conversation_id=conversation_id,
        compression_level=CompressionLevel.SUMMARY,
        summary="Brief explanation of machine learning...",
        metadata={"compression_method": "test"},
    )

    assert success

    # Verify compression
    node = await storage.get_node(ai_node.node_id, conversation_id)
    assert node.level == CompressionLevel.SUMMARY
    assert node.summary == "Brief explanation of machine learning..."
    assert node.summary_metadata["compression_method"] == "test"


@pytest.mark.asyncio
async def test_search_nodes(storage):
    """Test basic node searching."""
    conversation_id = "test-conv-4"

    # Save multiple conversation nodes
    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="What is Python?",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Python is a programming language.",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="Tell me about Java",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Java is also a programming language.",
    )

    # Search for "Python"
    results = await storage.search_nodes(conversation_id, "Python")

    assert len(results) >= 1
    assert any("Python" in result.node.content for result in results)


@pytest.mark.asyncio
async def test_conversation_exists(storage):
    """Test conversation existence check."""
    conversation_id = "test-conv-5"

    # Check non-existent conversation
    exists = await storage.conversation_exists(conversation_id)
    assert not exists

    # Create conversation
    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="Hello",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Hi there!",
    )

    # Check existing conversation
    exists = await storage.conversation_exists(conversation_id)
    assert exists


@pytest.mark.asyncio
async def test_get_recent_nodes(storage):
    """Test getting recent nodes."""
    conversation_id = "test-conv-6"

    # Create multiple turns
    for i in range(5):
        await storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.USER,
            content=f"Message {i}",
        )

        await storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content=f"Response {i}",
        )

    # Get recent nodes (should be at FULL compression level)
    recent_nodes = await storage.get_recent_nodes(conversation_id, limit=6)

    assert len(recent_nodes) == 6  # Limited to 6 nodes
    assert all(node.level == CompressionLevel.FULL for node in recent_nodes)


@pytest.mark.asyncio
async def test_get_conversation_stats(storage):
    """Test getting conversation statistics."""
    conversation_id = "test-conv-7"

    # Save some conversation nodes
    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="Hello",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Hi there!",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="How are you?",
    )

    await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="I'm doing well!",
    )

    # Get stats
    stats = await storage.get_conversation_stats(conversation_id)

    assert stats is not None
    assert stats.conversation_id == conversation_id
    assert stats.total_nodes == 4  # 2 turns * 2 nodes each
    assert CompressionLevel.FULL in stats.compression_stats
    assert stats.compression_stats[CompressionLevel.FULL] == 4
