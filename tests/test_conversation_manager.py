"""Tests for conversation manager."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import HierarchicalConversationManager
from hierarchical_memory_middleware.models import (
    ConversationNode,
    ConversationTurn,
    ConversationState,
    CompressionLevel,
    NodeType,
    CompressionResult,
    SearchResult
)
from hierarchical_memory_middleware.storage import DuckDBStorage
from hierarchical_memory_middleware.compression import SimpleCompressor, CompressionManager


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        work_model="test",
        summary_model="test",
        db_path=":memory:",
        recent_node_limit=10,
        summary_threshold=20
    )


@pytest.fixture
def sample_conversation_node():
    """Create a sample conversation node for testing."""
    return ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="Hello, how are you?",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.FULL
    )


@pytest.fixture
def sample_conversation_turn():
    """Create a sample conversation turn for testing."""
    return ConversationTurn(
        turn_id=1,
        conversation_id="test-conv-1",
        user_message="Hello, how are you?",
        ai_response="I'm doing well, thank you!",
        timestamp=datetime.now(),
        tokens_used=50,
        user_node_id=1,
        ai_node_id=2
    )


@pytest.fixture
def sample_search_result(sample_conversation_node):
    """Create a sample search result for testing."""
    return SearchResult(
        node=sample_conversation_node,
        relevance_score=0.95,
        match_type="content",
        matched_text="Hello, how are you?"
    )


@pytest.fixture
def sample_compression_result():
    """Create a sample compression result for testing."""
    return CompressionResult(
        original_node_id=1,
        compressed_content="User greeted",
        compression_ratio=0.5,
        topics_extracted=["greeting"],
        metadata={"original_length": 20}
    )


# Test 1: Constructor & Initialization Tests
@pytest.mark.asyncio
async def test_conversation_manager_initialization(mock_config):
    """Test that all components are properly initialized."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Check that all components are initialized
    assert manager.config == mock_config
    assert manager.conversation_id is None
    assert isinstance(manager.storage, DuckDBStorage)
    assert isinstance(manager.compressor, SimpleCompressor)
    assert isinstance(manager.compression_manager, CompressionManager)
    assert isinstance(manager.work_agent, Agent)
    
    # Check that the agent has the history processor
    assert len(manager.work_agent.history_processors) == 1
    assert manager.work_agent.history_processors[0] == manager._hierarchical_memory_processor


# Test 2: Constructor & Initialization Tests
@pytest.mark.asyncio
async def test_conversation_manager_with_custom_config():
    """Test initialization with different config values."""
    custom_config = Config(
        work_model="test",
        summary_model="test",
        db_path="/tmp/custom.db",
        recent_node_limit=15,
        summary_threshold=30
    )
    
    manager = HierarchicalConversationManager(custom_config)
    
    # Check that config values are used correctly
    assert manager.config.recent_node_limit == 15
    assert manager.config.summary_threshold == 30
    assert manager.config.db_path == "/tmp/custom.db"
    assert manager.config.work_model == "test"
    
    # Check that compression manager uses the custom config
    assert manager.compression_manager.recent_node_limit == 15
    
    # Check that storage uses the custom db_path (though we can't easily test this without mocking)
    assert isinstance(manager.storage, DuckDBStorage)
    
    # Check agent is initialized with custom model
    assert manager.work_agent.model._model_name == "test"  # Accessing test model name


# Test 3: Conversation Lifecycle Tests
@pytest.mark.asyncio
async def test_start_new_conversation(mock_config):
    """Test starting a conversation without providing conversation_id."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Initially no conversation is active
    assert manager.conversation_id is None
    
    # Start a new conversation
    conversation_id = await manager.start_conversation()
    
    # Verify a new UUID was generated and set
    assert manager.conversation_id is not None
    assert manager.conversation_id == conversation_id
    
    # Verify it's a valid UUID
    try:
        uuid_obj = uuid.UUID(conversation_id)
        assert str(uuid_obj) == conversation_id
    except ValueError:
        pytest.fail("conversation_id is not a valid UUID")
    
    # Verify it's a version 4 UUID (random)
    uuid_obj = uuid.UUID(conversation_id)
    assert uuid_obj.version == 4


# Test 4: Start conversation with existing ID
@pytest.mark.asyncio
async def test_start_conversation_with_existing_id(mock_config):
    """Test resuming an existing conversation by providing valid conversation_id."""
    manager = HierarchicalConversationManager(mock_config)
    existing_id = "existing-conv-123"
    
    # Mock storage.conversation_exists to return True
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists:
        mock_exists.return_value = True
        
        # Start conversation with existing ID
        conversation_id = await manager.start_conversation(existing_id)
        
        # Verify it uses the provided ID
        assert manager.conversation_id == existing_id
        assert conversation_id == existing_id
        
        # Verify storage.conversation_exists was called
        mock_exists.assert_called_once_with(existing_id)


# Test 5: Start conversation with nonexistent ID
@pytest.mark.asyncio
async def test_start_conversation_with_nonexistent_id(mock_config):
    """Test providing a conversation_id that doesn't exist."""
    manager = HierarchicalConversationManager(mock_config)
    nonexistent_id = "nonexistent-conv-456"
    
    # Mock storage.conversation_exists to return False
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists:
        mock_exists.return_value = False
        
        # Start conversation with nonexistent ID
        conversation_id = await manager.start_conversation(nonexistent_id)
        
        # Verify it creates a new conversation instead
        assert manager.conversation_id != nonexistent_id
        assert conversation_id != nonexistent_id
        
        # Verify it's a valid UUID
        uuid_obj = uuid.UUID(conversation_id)
        assert uuid_obj.version == 4
        
        # Verify storage.conversation_exists was called
        mock_exists.assert_called_once_with(nonexistent_id)


# Test 6: Start conversation generates valid UUID
@pytest.mark.asyncio
async def test_start_conversation_generates_valid_uuid(mock_config):
    """Test that generated conversation IDs are valid UUIDs and unique."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Generate multiple conversation IDs
    conversation_ids = []
    for _ in range(5):
        # Reset conversation_id for each iteration
        manager.conversation_id = None
        conv_id = await manager.start_conversation()
        conversation_ids.append(conv_id)
        
        # Verify it's a valid UUID
        uuid_obj = uuid.UUID(conv_id)
        assert uuid_obj.version == 4
    
    # Verify uniqueness
    assert len(set(conversation_ids)) == 5
    
    # Verify all are different from each other
    for i, conv_id in enumerate(conversation_ids):
        for j, other_id in enumerate(conversation_ids):
            if i != j:
                assert conv_id != other_id


# Test 7: Chat without active conversation
@pytest.mark.asyncio
async def test_chat_without_active_conversation(mock_config):
    """Test calling chat() before start_conversation()."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Try to chat without starting a conversation
    with pytest.raises(ValueError, match="No active conversation"):
        await manager.chat("Hello")


# Test 8: Chat successful response
@pytest.mark.asyncio
async def test_chat_successful_response(mock_config, sample_conversation_turn):
    """Test successful chat interaction."""
    manager = HierarchicalConversationManager(mock_config)

    # Mock storage.conversation_exists to return True
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists:
        mock_exists.return_value = True
        
        # Start a conversation
        await manager.start_conversation("test-conv-1")

        # Mock the agent response
        mock_response = Mock()
        mock_response.output = "Hello! I'm doing well, thank you!"
        mock_response.usage = {"total_tokens": 25}

        # Mock work_agent.run and storage operations
        with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
             patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock) as mock_save, \
             patch.object(manager, '_check_and_compress', new_callable=AsyncMock) as mock_compress:

            mock_run.return_value = mock_response
            mock_save.return_value = sample_conversation_turn

            # Send a chat message
            response = await manager.chat("Hello, how are you?")

            # Verify response
            assert response == "Hello! I'm doing well, thank you!"

            # Verify work_agent.run was called
            mock_run.assert_called_once_with(user_prompt="Hello, how are you?")

            # Verify storage.save_conversation_turn was called
            mock_save.assert_called_once()
            call_args = mock_save.call_args
            assert call_args[1]['conversation_id'] == "test-conv-1"
            assert call_args[1]['user_message'] == "Hello, how are you?"
            assert call_args[1]['ai_response'] == "Hello! I'm doing well, thank you!"
            assert call_args[1]['tokens_used'] == 25

            # Verify compression check was triggered
            mock_compress.assert_called_once()


# Test 9: Chat handles agent exceptions
@pytest.mark.asyncio
async def test_chat_handles_agent_exceptions(mock_config):
    """Test behavior when work_agent.run raises an exception."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Start a conversation
    await manager.start_conversation("test-conv-1")
    
    # Mock work_agent.run to raise an exception
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
         patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock) as mock_save:
        
        mock_run.side_effect = Exception("Agent error")
        
        # Send a chat message
        response = await manager.chat("Hello")
        
        # Verify graceful error handling
        assert "I apologize, but I encountered an error" in response
        assert "Agent error" in response
        
        # Verify no conversation turn was saved on failure
        mock_save.assert_not_called()


# Test 10: Chat saves conversation turn correctly
@pytest.mark.asyncio
async def test_chat_saves_conversation_turn_correctly(mock_config):
    """Test that conversation turns are saved with all metadata."""
    manager = HierarchicalConversationManager(mock_config)

    # Mock storage.conversation_exists to return True
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists:
        mock_exists.return_value = True
        
        # Start a conversation
        await manager.start_conversation("test-conv-1")

        # Mock the agent response with usage information
        mock_response = Mock()
        mock_response.output = "This is my response"
        mock_response.usage = {"total_tokens": 42, "prompt_tokens": 20, "completion_tokens": 22}

        mock_turn = Mock()
        mock_turn.turn_id = 1

        with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
             patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock) as mock_save, \
             patch.object(manager, '_check_and_compress', new_callable=AsyncMock):

            mock_run.return_value = mock_response
            mock_save.return_value = mock_turn

            # Send a chat message
            await manager.chat("Test message")

            # Verify storage.save_conversation_turn was called with all metadata
            mock_save.assert_called_once()
            call_args = mock_save.call_args

            assert call_args[1]['conversation_id'] == "test-conv-1"
            assert call_args[1]['user_message'] == "Test message"
            assert call_args[1]['ai_response'] == "This is my response"
            assert call_args[1]['tokens_used'] == 42
            assert call_args[1]['ai_components']['assistant_text'] == "This is my response"
            assert call_args[1]['ai_components']['model_used'] == "test"


# Test 11: Chat triggers compression check
@pytest.mark.asyncio
async def test_chat_triggers_compression_check(mock_config):
    """Test that _check_and_compress is called after successful chat."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Start a conversation
    await manager.start_conversation("test-conv-1")
    
    # Mock the agent response
    mock_response = Mock()
    mock_response.output = "Response"
    
    mock_turn = Mock()
    mock_turn.turn_id = 1
    
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
         patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock) as mock_save, \
         patch.object(manager, '_check_and_compress', new_callable=AsyncMock) as mock_compress:
        
        mock_run.return_value = mock_response
        mock_save.return_value = mock_turn
        
        # Send a chat message
        await manager.chat("Test")
        
        # Verify compression check was triggered
        mock_compress.assert_called_once()
        
        # Verify it was called after saving the turn
        assert mock_save.called
        assert mock_compress.called


# Test 12: History processor without conversation
@pytest.mark.asyncio
async def test_hierarchical_memory_processor_no_conversation(mock_config):
    """Test history processor when conversation_id is None."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Create some sample messages
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")])
    ]
    
    # Call the history processor without a conversation
    result = await manager._hierarchical_memory_processor(messages)
    
    # Verify it returns messages unchanged
    assert result == messages


# Test 13: History processor with memory
@pytest.mark.asyncio
async def test_hierarchical_memory_processor_with_memory(mock_config, sample_conversation_node):
    """Test history processor with existing conversation memory."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Create sample nodes
    recent_node = ConversationNode(
        id=2,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="Recent message",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=1,
        level=CompressionLevel.FULL
    )
    
    compressed_node = ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Original content",
        summary="Compressed AI response",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.SUMMARY
    )
    
    messages = [ModelRequest(parts=[UserPromptPart(content="New message")])]
    
    # Mock storage calls
    with patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_compressed:
        
        mock_recent.return_value = [recent_node]
        mock_compressed.return_value = [compressed_node]
        
        result = await manager._hierarchical_memory_processor(messages)
        
        # Verify storage methods were called
        mock_recent.assert_called_once_with(conversation_id="test-conv-1", limit=10)
        mock_compressed.assert_called_once_with(conversation_id="test-conv-1", limit=10, level=CompressionLevel.SUMMARY)
        
        # Verify proper message format and content
        assert len(result) >= 1
        assert any(isinstance(msg, ModelResponse) for msg in result)
        assert any("Compressed AI response" in str(msg) for msg in result)


# Test 14: History processor with empty memory
@pytest.mark.asyncio
async def test_hierarchical_memory_processor_empty_memory(mock_config):
    """Test history processor when no memory exists for conversation."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
    
    # Mock storage calls to return empty results
    with patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_compressed:
        
        mock_recent.return_value = []
        mock_compressed.return_value = []
        
        result = await manager._hierarchical_memory_processor(messages)
        
        # Verify it falls back to provided messages
        assert result == messages


# Test 15: History processor handles storage errors
@pytest.mark.asyncio
async def test_hierarchical_memory_processor_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail in history processor."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
    
    # Mock storage calls to raise exceptions
    with patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent:
        mock_recent.side_effect = Exception("Storage error")
        
        result = await manager._hierarchical_memory_processor(messages)
        
        # Verify it falls back to provided messages gracefully
        assert result == messages


# Test 16: History processor message format
@pytest.mark.asyncio
async def test_hierarchical_memory_processor_message_format(mock_config):
    """Test that messages are properly formatted as ModelRequest/ModelResponse."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Create sample nodes with different types
    user_node = ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="User message",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.FULL
    )
    
    ai_node = ConversationNode(
        id=2,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="AI response",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=1,
        level=CompressionLevel.FULL
    )
    
    messages = [ModelRequest(parts=[UserPromptPart(content="Current message")])]
    
    # Mock storage calls
    with patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_compressed:
        
        mock_recent.return_value = [user_node, ai_node]
        mock_compressed.return_value = []
        
        result = await manager._hierarchical_memory_processor(messages)
        
        # Verify message types and content
        user_messages = [msg for msg in result if isinstance(msg, ModelRequest)]
        ai_messages = [msg for msg in result if isinstance(msg, ModelResponse)]
        
        assert len(user_messages) >= 1
        assert len(ai_messages) >= 1
        
        # Verify content mapping
        user_content = user_messages[0].parts[0].content
        assert user_content == "User message"
        
        ai_content = ai_messages[0].parts[0].content
        assert ai_content == "AI response"


# Test 17: Memory search without conversation
@pytest.mark.asyncio
async def test_search_memory_without_conversation(mock_config):
    """Test search_memory when no conversation is active."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Search memory without an active conversation
    results = await manager.search_memory("test query")
    
    # Verify it returns empty list
    assert results == []


# Test 18: Memory search successful
@pytest.mark.asyncio
async def test_search_memory_successful(mock_config, sample_search_result):
    """Test successful memory search with results."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Mock storage.search_nodes to return sample results
    with patch.object(manager.storage, 'search_nodes', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [sample_search_result]
        
        results = await manager.search_memory("test query", limit=5)
        
        # Verify search was called correctly
        mock_search.assert_called_once_with(conversation_id="test-conv-1", query="test query", limit=5)
        
        # Verify response format
        assert len(results) == 1
        result = results[0]
        assert "node_id" in result
        assert "content" in result
        assert "relevance_score" in result
        assert "timestamp" in result


# Test 19: Memory search handles storage errors
@pytest.mark.asyncio
async def test_search_memory_handles_storage_errors(mock_config):
    """Test error handling when storage search fails."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Mock storage.search_nodes to raise exception
    with patch.object(manager.storage, 'search_nodes', new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = Exception("Search error")
        
        results = await manager.search_memory("test query")
        
        # Verify it returns empty list gracefully
        assert results == []


# Test 20: Memory search response format
@pytest.mark.asyncio
async def test_search_memory_response_format(mock_config):
    """Test that search results are properly formatted."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Create a node with long content to test truncation
    long_content = "A" * 250  # Longer than 200 chars
    search_node = ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content=long_content,
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1
    )
    
    search_result = SearchResult(
        node=search_node,
        relevance_score=0.85,
        match_type="content",
        matched_text="AAA"
    )
    
    with patch.object(manager.storage, 'search_nodes', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [search_result]
        
        results = await manager.search_memory("test")
        
        result = results[0]
        # Verify content truncation
        assert len(result["content"]) <= 203  # 200 + "..."
        assert result["content"].endswith("...")
        
        # Verify all required fields
        assert "node_id" in result
        assert "summary" in result
        assert "relevance_score" in result
        assert "match_type" in result
        assert "timestamp" in result
        assert "node_type" in result


# Test 21: Get conversation summary without conversation
@pytest.mark.asyncio
async def test_get_conversation_summary_without_conversation(mock_config):
    """Test getting summary when no conversation is active."""
    manager = HierarchicalConversationManager(mock_config)
    
    summary = await manager.get_conversation_summary()
    
    # Verify it returns error message
    assert "error" in summary
    assert summary["error"] == "No active conversation"


# Test 22: Get conversation summary successful
@pytest.mark.asyncio
async def test_get_conversation_summary_successful(mock_config):
    """Test successful summary retrieval."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Create mock stats and nodes
    mock_stats = Mock()
    mock_stats.compression_stats = {CompressionLevel.FULL: 5, CompressionLevel.SUMMARY: 3}
    mock_stats.last_updated = datetime.now()
    
    mock_nodes = [
        Mock(level=CompressionLevel.FULL),
        Mock(level=CompressionLevel.FULL),
        Mock(level=CompressionLevel.SUMMARY)
    ]
    
    with patch.object(manager.storage, 'get_conversation_stats', new_callable=AsyncMock) as mock_get_stats, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes:
        
        mock_get_stats.return_value = mock_stats
        mock_get_nodes.return_value = mock_nodes
        
        summary = await manager.get_conversation_summary()
        
        # Verify response contains all expected fields
        assert summary["conversation_id"] == "test-conv-1"
        assert summary["total_nodes"] == 3
        assert summary["recent_nodes"] == 2
        assert summary["compressed_nodes"] == 1
        assert "compression_stats" in summary
        assert "last_updated" in summary


# Test 23: Get conversation summary handles storage errors
@pytest.mark.asyncio
async def test_get_conversation_summary_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    with patch.object(manager.storage, 'get_conversation_stats', new_callable=AsyncMock) as mock_get_stats:
        mock_get_stats.side_effect = Exception("Storage error")
        
        summary = await manager.get_conversation_summary()
        
        # Verify graceful error response
        assert "error" in summary
        assert "Storage error" in summary["error"]


# Test 24: Get conversation summary compression stats
@pytest.mark.asyncio
async def test_get_conversation_summary_compression_stats(mock_config):
    """Test that compression statistics are properly calculated."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Create mock data with specific compression levels
    current_time = datetime.now()
    mock_stats = Mock()
    mock_stats.compression_stats = {
        CompressionLevel.FULL: 10,
        CompressionLevel.SUMMARY: 5
    }
    mock_stats.last_updated = current_time
    
    # Create nodes with different compression levels
    mock_nodes = []
    for i in range(8):  # 8 FULL nodes
        mock_nodes.append(Mock(level=CompressionLevel.FULL))
    for i in range(3):  # 3 SUMMARY nodes
        mock_nodes.append(Mock(level=CompressionLevel.SUMMARY))
    
    with patch.object(manager.storage, 'get_conversation_stats', new_callable=AsyncMock) as mock_get_stats, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes:
        
        mock_get_stats.return_value = mock_stats
        mock_get_nodes.return_value = mock_nodes
        
        summary = await manager.get_conversation_summary()
        
        # Verify compression statistics
        assert summary["total_nodes"] == 11
        assert summary["recent_nodes"] == 8  # FULL level nodes
        assert summary["compressed_nodes"] == 3  # SUMMARY level nodes
        
        # Verify timestamp formatting
        assert summary["last_updated"] == current_time.isoformat()
        
        # Verify compression stats passthrough
        assert summary["compression_stats"] == mock_stats.compression_stats


# Test 25: Check and compress no compression needed
@pytest.mark.asyncio
async def test_check_and_compress_no_compression_needed(mock_config):
    """Test compression check when no nodes need compression."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    with patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes, \
         patch.object(manager.compression_manager, 'identify_nodes_to_compress') as mock_identify:
        
        mock_get_nodes.return_value = []
        mock_identify.return_value = []
        
        # Should complete without error and no compression operations
        await manager._check_and_compress()
        
        mock_identify.assert_called_once_with([])


# Test 26: Check and compress successful compression
@pytest.mark.asyncio
async def test_check_and_compress_successful_compression(mock_config, sample_compression_result):
    """Test successful compression of nodes."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    mock_nodes = [Mock(id=1), Mock(id=2)]
    
    with patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes, \
         patch.object(manager.compression_manager, 'identify_nodes_to_compress') as mock_identify, \
         patch.object(manager.compression_manager, 'compress_nodes') as mock_compress_nodes, \
         patch.object(manager.storage, 'compress_node', new_callable=AsyncMock) as mock_compress_node:
        
        mock_get_nodes.return_value = mock_nodes
        mock_identify.return_value = [mock_nodes[0]]
        mock_compress_nodes.return_value = [sample_compression_result]
        
        await manager._check_and_compress()
        
        mock_compress_node.assert_called_once_with(
            node_id=sample_compression_result.original_node_id,
            compression_level=CompressionLevel.SUMMARY,
            summary=sample_compression_result.compressed_content,
            metadata=sample_compression_result.metadata
        )


# Test 27: Check and compress handles compression errors
@pytest.mark.asyncio
async def test_check_and_compress_handles_compression_errors(mock_config):
    """Test error handling during compression operations."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    with patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes:
        mock_get_nodes.side_effect = Exception("Storage error")
        
        # Should handle error gracefully
        await manager._check_and_compress()
        
        # No assertions needed, just verify it doesn't crash


# Test 28: Check and compress compression workflow
@pytest.mark.asyncio
async def test_check_and_compress_compression_workflow(mock_config, sample_compression_result):
    """Test complete compression workflow."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    mock_nodes = [Mock(id=1, content="Node content")]
    compression_results = [sample_compression_result]
    
    with patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_get_nodes, \
         patch.object(manager.compression_manager, 'identify_nodes_to_compress') as mock_identify, \
         patch.object(manager.compression_manager, 'compress_nodes') as mock_compress_nodes, \
         patch.object(manager.storage, 'compress_node', new_callable=AsyncMock) as mock_compress_node:
        
        mock_get_nodes.return_value = mock_nodes
        mock_identify.return_value = mock_nodes
        mock_compress_nodes.return_value = compression_results
        
        await manager._check_and_compress()
        
        # Verify workflow
        mock_get_nodes.assert_called_once_with(conversation_id="test-conv-1")
        mock_identify.assert_called_once_with(mock_nodes)
        mock_compress_nodes.assert_called_once_with(mock_nodes)
        mock_compress_node.assert_called_once()


# Test 29: Get node details existing node
@pytest.mark.asyncio
async def test_get_node_details_existing_node(mock_config, sample_conversation_node):
    """Test retrieving details for an existing node."""
    manager = HierarchicalConversationManager(mock_config)
    
    with patch.object(manager.storage, 'get_node', new_callable=AsyncMock) as mock_get_node:
        mock_get_node.return_value = sample_conversation_node
        
        details = await manager.get_node_details(1)
        
        mock_get_node.assert_called_once_with(1)
        
        # Verify response format
        assert details is not None
        assert details["id"] == sample_conversation_node.id
        assert details["conversation_id"] == sample_conversation_node.conversation_id
        assert details["content"] == sample_conversation_node.content
        assert details["level"] == sample_conversation_node.level.name


# Test 30: Get node details nonexistent node
@pytest.mark.asyncio
async def test_get_node_details_nonexistent_node(mock_config):
    """Test retrieving details for non-existent node."""
    manager = HierarchicalConversationManager(mock_config)
    
    with patch.object(manager.storage, 'get_node', new_callable=AsyncMock) as mock_get_node:
        mock_get_node.return_value = None
        
        details = await manager.get_node_details(999)
        
        assert details is None


# Test 31: Get node details handles storage errors
@pytest.mark.asyncio
async def test_get_node_details_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail."""
    manager = HierarchicalConversationManager(mock_config)
    
    with patch.object(manager.storage, 'get_node', new_callable=AsyncMock) as mock_get_node:
        mock_get_node.side_effect = Exception("Storage error")
        
        details = await manager.get_node_details(1)
        
        assert details is None


# Test 32: Get node details response format
@pytest.mark.asyncio
async def test_get_node_details_response_format(mock_config):
    """Test that node details response contains all expected fields."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Create a complete node with all fields
    complete_node = ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Complete node content",
        summary="Node summary",
        timestamp=datetime.now(),
        sequence_number=5,
        line_count=3,
        level=CompressionLevel.SUMMARY,
        tokens_used=42,
        topics=["topic1", "topic2"],
        ai_components={"model": "test", "tokens": 42}
    )
    
    with patch.object(manager.storage, 'get_node', new_callable=AsyncMock) as mock_get_node:
        mock_get_node.return_value = complete_node
        
        details = await manager.get_node_details(1)
        
        # Verify all expected fields
        expected_fields = [
            "id", "conversation_id", "node_type", "content", "summary",
            "timestamp", "sequence_number", "line_count", "level",
            "tokens_used", "topics", "ai_components"
        ]
        
        for field in expected_fields:
            assert field in details
        
        # Verify enum conversion
        assert details["node_type"] == "ai"
        assert details["level"] == "SUMMARY"


# Test 33: Full conversation workflow (Integration)
@pytest.mark.asyncio
async def test_full_conversation_workflow(mock_config):
    """Integration test of complete conversation workflow."""
    # Use real storage (in-memory)
    config = Config(
        work_model="test",
        db_path=":memory:",
        recent_node_limit=5
    )
    
    manager = HierarchicalConversationManager(config)
    
    # Start conversation
    conv_id = await manager.start_conversation()
    assert manager.conversation_id == conv_id
    
    # Mock agent for chat
    mock_response = Mock()
    mock_response.output = "Hello there!"
    mock_response.usage = {"total_tokens": 20}
    
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
         patch.object(manager, '_check_and_compress', new_callable=AsyncMock):
        
        mock_run.return_value = mock_response
        
        # Have a conversation
        response = await manager.chat("Hello")
        assert response == "Hello there!"
        
        # Get summary
        summary = await manager.get_conversation_summary()
        assert summary["conversation_id"] == conv_id
        assert summary["total_nodes"] >= 0


# Test 34: Conversation with compression cycle
@pytest.mark.asyncio
async def test_conversation_with_compression_cycle(mock_config):
    """Test conversation that triggers compression."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Mock compression being triggered
    mock_response = Mock()
    mock_response.output = "Response"
    
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
         patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock), \
         patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists, \
         patch.object(manager, '_check_and_compress', new_callable=AsyncMock) as mock_compress:
        
        mock_exists.return_value = True
        mock_run.return_value = mock_response
        
        # Multiple chat turns
        for i in range(3):
            await manager.chat(f"Message {i}")
        
        # Verify compression was checked multiple times
        assert mock_compress.call_count == 3


# Test 35: Conversation resume after compression
@pytest.mark.asyncio
async def test_conversation_resume_after_compression(mock_config):
    """Test resuming conversation that has compressed nodes."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Mock existing conversation with compressed nodes
    compressed_node = ConversationNode(
        id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Original long content",
        summary="Compressed summary",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.SUMMARY
    )
    
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists, \
         patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_compressed:
        
        mock_exists.return_value = True
        mock_recent.return_value = []
        mock_compressed.return_value = [compressed_node]
        
        # Resume conversation
        await manager.start_conversation("test-conv-1")
        
        # Test history processor with compressed nodes
        messages = [ModelRequest(parts=[UserPromptPart(content="New message")])]
        result = await manager._hierarchical_memory_processor(messages)
        
        # Should include compressed content
        assert len(result) >= 1


# Test 36: Agent initialization with history processor
@pytest.mark.asyncio
async def test_agent_initialization_with_history_processor(mock_config):
    """Test that PydanticAI agent is initialized with history processor."""
    manager = HierarchicalConversationManager(mock_config)
    
    # Already tested in initialization tests, but verify again
    assert len(manager.work_agent.history_processors) == 1
    assert manager.work_agent.history_processors[0] == manager._hierarchical_memory_processor


# Test 37: Conversation manager with invalid config
@pytest.mark.asyncio
async def test_conversation_manager_with_invalid_config():
    """Test initialization with missing or invalid config values."""
    # Test with None config should raise error
    with pytest.raises(Exception):
        HierarchicalConversationManager(None)


# Test 38: Chat with empty message
@pytest.mark.asyncio
async def test_chat_with_empty_message(mock_config):
    """Test chat with empty or whitespace-only user message."""
    manager = HierarchicalConversationManager(mock_config)
    
    with patch.object(manager.storage, 'conversation_exists', new_callable=AsyncMock) as mock_exists:
        mock_exists.return_value = True
        await manager.start_conversation("test-conv-1")
        
        mock_response = Mock()
        mock_response.output = "I received an empty message"
        
        with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
             patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock), \
             patch.object(manager, '_check_and_compress', new_callable=AsyncMock):
            
            mock_run.return_value = mock_response
            
            # Test empty message
            response = await manager.chat("")
            assert "I received an empty message" == response
            
            # Test whitespace message
            response = await manager.chat("   ")
            assert "I received an empty message" == response


# Test 39: Memory processor with mixed node types
@pytest.mark.asyncio
async def test_memory_processor_with_mixed_node_types(mock_config):
    """Test history processor with different node types and compression levels."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"
    
    # Mixed nodes with different types and compression levels
    mixed_nodes = [
        ConversationNode(
            id=1, conversation_id="test-conv-1", node_type=NodeType.USER,
            content="User message", timestamp=datetime.now(), sequence_number=1,
            line_count=1, level=CompressionLevel.FULL
        ),
        ConversationNode(
            id=2, conversation_id="test-conv-1", node_type=NodeType.AI,
            content="Original AI response", summary="Compressed AI",
            timestamp=datetime.now(), sequence_number=2, line_count=1,
            level=CompressionLevel.SUMMARY
        )
    ]
    
    with patch.object(manager.storage, 'get_recent_nodes', new_callable=AsyncMock) as mock_recent, \
         patch.object(manager.storage, 'get_conversation_nodes', new_callable=AsyncMock) as mock_compressed:
        
        mock_recent.return_value = [mixed_nodes[0]]
        mock_compressed.return_value = [mixed_nodes[1]]
        
        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
        result = await manager._hierarchical_memory_processor(messages)
        
        # Should handle mixed types correctly
        assert len(result) >= 1


# Test 40: Conversation manager logging
@pytest.mark.asyncio
async def test_conversation_manager_logging(mock_config, caplog):
    """Test that appropriate log messages are generated."""
    import logging
    caplog.set_level(logging.INFO)
    
    manager = HierarchicalConversationManager(mock_config)
    
    # Test initialization logging
    assert "Initialized HierarchicalConversationManager" in caplog.text
    
    # Test conversation start logging
    conv_id = await manager.start_conversation()
    assert f"Starting new conversation: {conv_id}" in caplog.text
    
    # Test chat logging
    mock_response = Mock()
    mock_response.output = "Response"
    mock_turn = Mock()
    mock_turn.turn_id = 1
    
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run, \
         patch.object(manager.storage, 'save_conversation_turn', new_callable=AsyncMock) as mock_save, \
         patch.object(manager, '_check_and_compress', new_callable=AsyncMock):
        
        mock_run.return_value = mock_response
        mock_save.return_value = mock_turn
        
        await manager.chat("Test message")
        
        assert f"Processed conversation turn 1 in conversation {conv_id}" in caplog.text
    
    # Test error logging
    with patch.object(manager.work_agent, 'run', new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = Exception("Test error")
        
        await manager.chat("Error message")
        
        assert "Error in chat" in caplog.text
