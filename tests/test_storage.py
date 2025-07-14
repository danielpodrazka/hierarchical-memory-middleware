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
async def test_save_conversation_turn(storage):
    """Test saving a conversation turn."""
    conversation_id = "test-conv-1"
    user_message = "Hello, how are you?"
    ai_response = "I'm doing well, thank you for asking!"

    turn = await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message=user_message,
        ai_response=ai_response,
        tokens_used=100
    )

    assert turn.conversation_id == conversation_id
    assert turn.user_message == user_message
    assert turn.ai_response == ai_response
    assert turn.tokens_used == 100
    assert turn.user_node_id is not None
    assert turn.ai_node_id is not None


@pytest.mark.asyncio
async def test_get_conversation_nodes(storage):
    """Test retrieving conversation nodes."""
    conversation_id = "test-conv-2"

    # Save a conversation turn
    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="What is Python?",
        ai_response="Python is a programming language."
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
    
    # Save a conversation turn
    turn = await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="Tell me about machine learning algorithms",
        ai_response="Machine learning algorithms are computational methods that learn from data."
    )

    # Compress the AI node
    success = await storage.compress_node(
        node_id=turn.ai_node_id,
        compression_level=CompressionLevel.SUMMARY,
        summary="Brief explanation of machine learning...",
        metadata={"compression_method": "test"}
    )

    assert success

    # Verify compression
    node = await storage.get_node(turn.ai_node_id)
    assert node.level == CompressionLevel.SUMMARY
    assert node.summary == "Brief explanation of machine learning..."
    assert node.summary_metadata["compression_method"] == "test"


@pytest.mark.asyncio
async def test_search_nodes(storage):
    """Test basic node searching."""
    conversation_id = "test-conv-4"
    
    # Save multiple conversation turns
    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="What is Python?",
        ai_response="Python is a programming language."
    )
    
    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="Tell me about Java",
        ai_response="Java is also a programming language."
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
    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="Hello",
        ai_response="Hi there!"
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
        await storage.save_conversation_turn(
            conversation_id=conversation_id,
            user_message=f"Message {i}",
            ai_response=f"Response {i}"
        )

    # Get recent nodes (should be at FULL compression level)
    recent_nodes = await storage.get_recent_nodes(conversation_id, limit=6)

    assert len(recent_nodes) == 6  # 3 turns * 2 nodes each
    assert all(node.level == CompressionLevel.FULL for node in recent_nodes)


@pytest.mark.asyncio
async def test_get_conversation_stats(storage):
    """Test getting conversation statistics."""
    conversation_id = "test-conv-7"

    # Save some conversation turns
    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="Hello",
        ai_response="Hi there!"
    )

    await storage.save_conversation_turn(
        conversation_id=conversation_id,
        user_message="How are you?",
        ai_response="I'm doing well!"
    )

    # Get stats
    stats = await storage.get_conversation_stats(conversation_id)

    assert stats is not None
    assert stats.conversation_id == conversation_id
    assert stats.total_nodes == 4  # 2 turns * 2 nodes each
    assert CompressionLevel.FULL in stats.compression_stats
    assert stats.compression_stats[CompressionLevel.FULL] == 4
