"""Tests for conversation manager."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from pydantic_ai import Agent, usage
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.middleware.conversation_manager import (
    HierarchicalConversationManager,
)
from hierarchical_memory_middleware.models import (
    ConversationNode,
    CompressionLevel,
    NodeType,
    CompressionResult,
    SearchResult,
)
from hierarchical_memory_middleware.storage import DuckDBStorage
from hierarchical_memory_middleware.compression import (
    SimpleCompressor,
)
from hierarchical_memory_middleware.advanced_hierarchy import (
    AdvancedCompressionManager,
)


@pytest.fixture
def mock_config(monkeypatch):
    """Create a mock config for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")
    return Config(
        work_model="claude-sonnet-4",
        summary_model="claude-sonnet-4",
        db_path=":memory:",
        recent_node_limit=10,
        summary_threshold=20,
    )


@pytest.fixture
def sample_conversation_node():
    """Create a sample conversation node for testing."""
    return ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="Hello, how are you?",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.FULL,
    )


@pytest.fixture
def sample_search_result(sample_conversation_node):
    """Create a sample search result for testing."""
    return SearchResult(
        node=sample_conversation_node,
        relevance_score=0.95,
        match_type="content",
        matched_text="Hello, how are you?",
    )


@pytest.fixture
def sample_compression_result():
    """Create a sample compression result for testing."""
    return CompressionResult(
        original_node_id=1,
        compressed_content="User greeted",
        compression_ratio=0.5,
        topics_extracted=["greeting"],
        metadata={"original_length": 20},
    )


@pytest.mark.asyncio
async def test_conversation_manager_initialization(mock_config):
    """Test that all components are properly initialized."""
    manager = HierarchicalConversationManager(mock_config)

    assert manager.config == mock_config
    assert manager.conversation_id is None
    assert isinstance(manager.storage, DuckDBStorage)
    assert isinstance(manager.compressor, SimpleCompressor)
    assert isinstance(manager.compression_manager, AdvancedCompressionManager)
    assert isinstance(manager.work_agent, Agent)

    assert len(manager.work_agent.history_processors) == 1
    assert (
        manager.work_agent.history_processors[0]
        == manager._hierarchical_memory_processor
    )


@pytest.mark.asyncio
async def test_conversation_manager_with_custom_config(monkeypatch):
    """Test initialization with different config values."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")
    custom_config = Config(
        work_model="claude-3-5-haiku",
        summary_model="claude-3-5-haiku",
        db_path="/tmp/custom.db",
        recent_node_limit=15,
        summary_threshold=30,
    )

    manager = HierarchicalConversationManager(custom_config)

    assert manager.config.recent_node_limit == 15
    assert manager.config.summary_threshold == 30
    assert manager.config.db_path == "/tmp/custom.db"
    assert manager.config.work_model == "claude-3-5-haiku"

    assert manager.compression_manager.recent_node_limit == 15

    assert isinstance(manager.storage, DuckDBStorage)

    assert manager.work_agent.model._model_name == "claude-3-5-haiku-20241022"


@pytest.mark.asyncio
async def test_start_new_conversation(mock_config):
    """Test starting a conversation without providing conversation_id."""
    manager = HierarchicalConversationManager(mock_config)

    assert manager.conversation_id is None

    conversation_id = await manager.start_conversation()

    assert manager.conversation_id is not None
    assert manager.conversation_id == conversation_id

    try:
        uuid_obj = uuid.UUID(conversation_id)
        assert str(uuid_obj) == conversation_id
    except ValueError:
        pytest.fail("conversation_id is not a valid UUID")

    uuid_obj = uuid.UUID(conversation_id)
    assert uuid_obj.version == 4


@pytest.mark.asyncio
async def test_start_conversation_with_existing_id(mock_config):
    """Test resuming an existing conversation by providing valid conversation_id."""
    manager = HierarchicalConversationManager(mock_config)
    existing_id = "existing-conv-123"

    with patch.object(
        manager.storage, "conversation_exists", new_callable=AsyncMock
    ) as mock_exists:
        mock_exists.return_value = True

        conversation_id = await manager.start_conversation(existing_id)

        assert manager.conversation_id == existing_id
        assert conversation_id == existing_id

        mock_exists.assert_called_once_with(existing_id)


@pytest.mark.asyncio
async def test_start_conversation_with_nonexistent_id(mock_config):
    """Test providing a conversation_id that doesn't exist."""
    manager = HierarchicalConversationManager(mock_config)
    nonexistent_id = "nonexistent-conv-456"

    with patch.object(
        manager.storage, "conversation_exists", new_callable=AsyncMock
    ) as mock_exists:
        mock_exists.return_value = False

        conversation_id = await manager.start_conversation(nonexistent_id)

        assert manager.conversation_id != nonexistent_id
        assert conversation_id != nonexistent_id

        uuid_obj = uuid.UUID(conversation_id)
        assert uuid_obj.version == 4

        mock_exists.assert_called_once_with(nonexistent_id)


@pytest.mark.asyncio
async def test_start_conversation_generates_valid_uuid(mock_config):
    """Test that generated conversation IDs are valid UUIDs and unique."""
    manager = HierarchicalConversationManager(mock_config)

    conversation_ids = []
    for _ in range(5):
        manager.conversation_id = None
        conv_id = await manager.start_conversation()
        conversation_ids.append(conv_id)

        uuid_obj = uuid.UUID(conv_id)
        assert uuid_obj.version == 4

    assert len(set(conversation_ids)) == 5

    for i, conv_id in enumerate(conversation_ids):
        for j, other_id in enumerate(conversation_ids):
            if i != j:
                assert conv_id != other_id


@pytest.mark.asyncio
async def test_chat_without_active_conversation(mock_config):
    """Test calling chat() before start_conversation()."""
    manager = HierarchicalConversationManager(mock_config)

    with pytest.raises(ValueError, match="No active conversation"):
        await manager.chat("Hello")


@pytest.mark.asyncio
async def test_chat_successful_response(mock_config):
    """Test successful chat interaction."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "conversation_exists", new_callable=AsyncMock
    ) as mock_exists:
        mock_exists.return_value = True

        await manager.start_conversation("test-conv-1")

        mock_response = Mock()
        mock_response.output = "Hello! I'm doing well, thank you!"
        mock_response.usage = {"total_tokens": 25}

        # Create mock nodes to return
        mock_user_node = Mock()
        mock_user_node.node_id = 1
        mock_ai_node = Mock()
        mock_ai_node.node_id = 2

        with (
            patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
            patch.object(
                manager.storage, "save_conversation_node", new_callable=AsyncMock
            ) as mock_save,
            patch.object(
                manager, "_check_and_compress", new_callable=AsyncMock
            ) as mock_compress,
        ):
            mock_run.return_value = mock_response
            # Configure save_conversation_node to return different nodes for user/AI
            mock_save.side_effect = [mock_user_node, mock_ai_node]

            response = await manager.chat("Hello, how are you?")

            assert response == "Hello! I'm doing well, thank you!"

            mock_run.assert_called_once_with(
                user_prompt="Hello, how are you?",
                usage_limits=usage.UsageLimits(request_limit=500),
            )

            # Should be called twice: once for user node, once for AI node
            assert mock_save.call_count == 2

            # Check first call (user node)
            first_call = mock_save.call_args_list[0]
            assert first_call[1]["conversation_id"] == "test-conv-1"
            assert first_call[1]["node_type"].value == "user"
            assert first_call[1]["content"] == "Hello, how are you?"

            # Check second call (AI node)
            second_call = mock_save.call_args_list[1]
            assert second_call[1]["conversation_id"] == "test-conv-1"
            assert second_call[1]["node_type"].value == "ai"
            assert second_call[1]["content"] == "Hello! I'm doing well, thank you!"
            assert second_call[1]["tokens_used"] == 25

            mock_compress.assert_called_once()


@pytest.mark.asyncio
async def test_chat_handles_agent_exceptions(mock_config):
    """Test behavior when work_agent.run raises an exception."""
    manager = HierarchicalConversationManager(mock_config)

    await manager.start_conversation("test-conv-1")

    with (
        patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
        patch.object(
            manager.storage, "save_conversation_node", new_callable=AsyncMock
        ) as mock_save,
    ):
        mock_run.side_effect = Exception("Agent error")

        response = await manager.chat("Hello")

        assert "I apologize, but I encountered an error" in response
        assert "Agent error" in response

        mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_chat_saves_conversation_nodes_correctly(mock_config):
    """Test that conversation nodes are saved with all metadata."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "conversation_exists", new_callable=AsyncMock
    ) as mock_exists:
        mock_exists.return_value = True

        await manager.start_conversation("test-conv-1")

        mock_response = Mock()
        mock_response.output = "This is my response"
        mock_response.usage = {
            "total_tokens": 42,
            "prompt_tokens": 20,
            "completion_tokens": 22,
        }

        # Create mock nodes to return
        mock_user_node = Mock()
        mock_user_node.node_id = 1
        mock_ai_node = Mock()
        mock_ai_node.node_id = 2

        with (
            patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
            patch.object(
                manager.storage, "save_conversation_node", new_callable=AsyncMock
            ) as mock_save,
            patch.object(manager, "_check_and_compress", new_callable=AsyncMock),
        ):
            mock_run.return_value = mock_response
            mock_save.side_effect = [mock_user_node, mock_ai_node]

            await manager.chat("Test message")

            # Should be called twice: once for user node, once for AI node
            assert mock_save.call_count == 2

            # Check first call (user node)
            first_call = mock_save.call_args_list[0]
            assert first_call[1]["conversation_id"] == "test-conv-1"
            assert first_call[1]["node_type"].value == "user"
            assert first_call[1]["content"] == "Test message"

            # Check second call (AI node)
            second_call = mock_save.call_args_list[1]
            assert second_call[1]["conversation_id"] == "test-conv-1"
            assert second_call[1]["node_type"].value == "ai"
            assert second_call[1]["content"] == "This is my response"
            assert second_call[1]["tokens_used"] == 42
            assert (
                second_call[1]["ai_components"]["assistant_text"]
                == "This is my response"
            )
            assert second_call[1]["ai_components"]["model_used"] == "claude-sonnet-4"


@pytest.mark.asyncio
async def test_chat_triggers_compression_check(mock_config):
    """Test that _check_and_compress is called after successful chat."""
    manager = HierarchicalConversationManager(mock_config)

    await manager.start_conversation("test-conv-1")

    mock_response = Mock()
    mock_response.output = "Response"

    # Create mock nodes to return
    mock_user_node = Mock()
    mock_user_node.node_id = 1
    mock_ai_node = Mock()
    mock_ai_node.node_id = 2

    with (
        patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
        patch.object(
            manager.storage, "save_conversation_node", new_callable=AsyncMock
        ) as mock_save,
        patch.object(
            manager, "_check_and_compress", new_callable=AsyncMock
        ) as mock_compress,
    ):
        mock_run.return_value = mock_response
        mock_save.side_effect = [mock_user_node, mock_ai_node]

        await manager.chat("Test")

        mock_compress.assert_called_once()

        assert mock_save.called
        assert mock_compress.called


@pytest.mark.asyncio
async def test_hierarchical_memory_processor_no_conversation(mock_config):
    """Test history processor when conversation_id is None."""
    manager = HierarchicalConversationManager(mock_config)

    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    result = await manager._hierarchical_memory_processor(messages)

    assert result == messages


@pytest.mark.asyncio
async def test_hierarchical_memory_processor_with_memory(
    mock_config, sample_conversation_node
):
    """Test history processor with existing conversation memory."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    recent_node = ConversationNode(
        node_id=2,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="Recent message",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=1,
        level=CompressionLevel.FULL,
    )

    compressed_node = ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Original content",
        summary="Compressed AI response",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.SUMMARY,
    )

    messages = [ModelRequest(parts=[UserPromptPart(content="New message")])]

    with (
        patch.object(
            manager.storage, "get_recent_nodes", new_callable=AsyncMock
        ) as mock_recent,
        patch.object(
            manager.storage, "get_recent_hierarchical_nodes", new_callable=AsyncMock
        ) as mock_compressed,
    ):
        mock_recent.return_value = [recent_node]
        mock_compressed.return_value = [compressed_node]

        result = await manager._hierarchical_memory_processor(messages)

        mock_recent.assert_called_once_with(conversation_id="test-conv-1", limit=10)
        mock_compressed.assert_called_once_with(conversation_id="test-conv-1", limit=10)

        assert len(result) >= 1
        assert any(isinstance(msg, ModelResponse) for msg in result)
        assert any("Compressed AI response" in str(msg) for msg in result)


@pytest.mark.asyncio
async def test_hierarchical_memory_processor_empty_memory(mock_config):
    """Test history processor when no memory exists for conversation."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    with (
        patch.object(
            manager.storage, "get_recent_nodes", new_callable=AsyncMock
        ) as mock_recent,
        patch.object(
            manager.storage, "get_recent_hierarchical_nodes", new_callable=AsyncMock
        ) as mock_compressed,
    ):
        mock_recent.return_value = []
        mock_compressed.return_value = []

        result = await manager._hierarchical_memory_processor(messages)

        # With current implementation, context message is always added when there's a conversation_id
        # So result should contain: [context_message] + original_messages
        assert len(result) == 2  # context + original message
        assert result[0].parts[0].content == "[Context: Conversation ID test-conv-1]"
        assert result[1].parts[0].content == "Hello"


@pytest.mark.asyncio
async def test_hierarchical_memory_processor_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail in history processor."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]

    with patch.object(
        manager.storage, "get_recent_nodes", new_callable=AsyncMock
    ) as mock_recent:
        mock_recent.side_effect = Exception("Storage error")

        result = await manager._hierarchical_memory_processor(messages)

        assert result == messages


@pytest.mark.asyncio
async def test_hierarchical_memory_processor_message_format(mock_config):
    """Test that messages are properly formatted as ModelRequest/ModelResponse."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    user_node = ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="User message",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.FULL,
    )

    ai_node = ConversationNode(
        node_id=2,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="AI response",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=1,
        level=CompressionLevel.FULL,
    )

    messages = [ModelRequest(parts=[UserPromptPart(content="Current message")])]

    with (
        patch.object(
            manager.storage, "get_recent_nodes", new_callable=AsyncMock
        ) as mock_recent,
        patch.object(
            manager.storage, "get_recent_hierarchical_nodes", new_callable=AsyncMock
        ) as mock_compressed,
    ):
        mock_recent.return_value = [user_node, ai_node]
        mock_compressed.return_value = []

        result = await manager._hierarchical_memory_processor(messages)

        user_messages = [msg for msg in result if isinstance(msg, ModelRequest)]
        ai_messages = [msg for msg in result if isinstance(msg, ModelResponse)]

        assert len(user_messages) >= 3  # context + user_node + current message
        assert len(ai_messages) >= 1  # ai_node

        # First user message should be the context
        assert (
            user_messages[0].parts[0].content
            == "[Context: Conversation ID test-conv-1]"
        )
        # Second user message should be from memory
        assert user_messages[1].parts[0].content == "User message"

        ai_content = ai_messages[0].parts[0].content
        assert ai_content == "AI response"


@pytest.mark.asyncio
async def test_find_without_conversation(mock_config):
    """Test find when no conversation is active."""
    manager = HierarchicalConversationManager(mock_config)

    results = await manager.find("test query")

    assert results == []


@pytest.mark.asyncio
async def test_find_successful(mock_config, sample_search_result):
    """Test successful memory search with results."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [sample_search_result]

        results = await manager.find("test query", limit=5)

        mock_search.assert_called_once_with(
            conversation_id="test-conv-1", query="test query", limit=5, regex=False
        )

        assert len(results) == 1
        result = results[0]
        assert "node_id" in result
        assert "content" in result
        assert "relevance_score" in result
        assert "timestamp" in result


@pytest.mark.asyncio
async def test_find_handles_storage_errors(mock_config):
    """Test error handling when storage search fails."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.side_effect = Exception("Search error")

        results = await manager.find("test query")

        assert results == []


@pytest.mark.asyncio
async def test_find_response_format(mock_config):
    """Test that search results are properly formatted."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    long_content = "A" * 250
    search_node = ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content=long_content,
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
    )

    search_result = SearchResult(
        node=search_node, relevance_score=0.85, match_type="content", matched_text="AAA"
    )

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [search_result]

        results = await manager.find("test")

        result = results[0]

        assert len(result["content"]) <= 203
        assert result["content"].endswith("...")

        assert "node_id" in result
        assert "summary" in result
        assert "relevance_score" in result
        assert "match_type" in result
        assert "timestamp" in result
        assert "node_type" in result


@pytest.mark.asyncio
async def test_find_with_regex_false(mock_config, sample_search_result):
    """Test find method with regex=False (default behavior)."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [sample_search_result]

        results = await manager.find("test query", limit=5, regex=False)

        mock_search.assert_called_once_with(
            conversation_id="test-conv-1", query="test query", limit=5, regex=False
        )

        assert len(results) == 1


@pytest.mark.asyncio
async def test_find_with_regex_true(mock_config, sample_search_result):
    """Test find method with regex=True."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [sample_search_result]

        results = await manager.find(r"\d+", limit=10, regex=True)

        mock_search.assert_called_once_with(
            conversation_id="test-conv-1", query=r"\d+", limit=10, regex=True
        )

        assert len(results) == 1


@pytest.mark.asyncio
async def test_find_regex_default_parameter(mock_config, sample_search_result):
    """Test that regex parameter defaults to False."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [sample_search_result]

        # Call without regex parameter
        results = await manager.find("test query")

        # Should default to regex=False
        mock_search.assert_called_once_with(
            conversation_id="test-conv-1", query="test query", limit=10, regex=False
        )

        assert len(results) == 1


@pytest.mark.asyncio
async def test_find_regex_with_different_patterns(mock_config):
    """Test find method with various regex patterns."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    # Create multiple test search results
    email_node = ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.USER,
        content="My email is john@example.com",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
    )

    phone_node = ConversationNode(
        node_id=2,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Call me at 123-456-7890",
        timestamp=datetime.now(),
        sequence_number=2,
        line_count=1,
    )

    email_result = SearchResult(
        node=email_node,
        relevance_score=0.9,
        match_type="content",
        matched_text="john@example.com",
    )

    phone_result = SearchResult(
        node=phone_node,
        relevance_score=0.85,
        match_type="content",
        matched_text="123-456-7890",
    )

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        # Test email regex
        mock_search.return_value = [email_result]
        results = await manager.find(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", regex=True
        )

        mock_search.assert_called_with(
            conversation_id="test-conv-1",
            query=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            limit=10,
            regex=True,
        )

        assert len(results) == 1
        assert "email" in results[0]["content"].lower()

        # Test phone number regex
        mock_search.return_value = [phone_result]
        results = await manager.find(r"\d{3}-\d{3}-\d{4}", regex=True)

        assert len(results) == 1
        assert "123-456-7890" in results[0]["content"]


@pytest.mark.asyncio
async def test_find_regex_handles_invalid_patterns(mock_config):
    """Test find method gracefully handles invalid regex patterns."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "search_nodes", new_callable=AsyncMock
    ) as mock_search:
        # Return empty list for invalid regex (handled by storage layer)
        mock_search.return_value = []

        results = await manager.find("[invalid regex", regex=True)

        mock_search.assert_called_once_with(
            conversation_id="test-conv-1", query="[invalid regex", limit=10, regex=True
        )

        assert results == []


@pytest.mark.asyncio
async def test_get_conversation_summary_without_conversation(mock_config):
    """Test getting summary when no conversation is active."""
    manager = HierarchicalConversationManager(mock_config)

    summary = await manager.get_conversation_summary()

    assert "error" in summary
    assert summary["error"] == "No active conversation"


@pytest.mark.asyncio
async def test_get_conversation_summary_successful(mock_config):
    """Test successful summary retrieval."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    mock_stats = Mock()
    mock_stats.compression_stats = {
        CompressionLevel.FULL: 5,
        CompressionLevel.SUMMARY: 3,
    }
    mock_stats.last_updated = datetime.now()

    mock_nodes = [
        Mock(level=CompressionLevel.FULL),
        Mock(level=CompressionLevel.FULL),
        Mock(level=CompressionLevel.SUMMARY),
    ]

    with (
        patch.object(
            manager.storage, "get_conversation_stats", new_callable=AsyncMock
        ) as mock_get_stats,
        patch.object(
            manager.storage, "get_conversation_nodes", new_callable=AsyncMock
        ) as mock_get_nodes,
    ):
        mock_get_stats.return_value = mock_stats
        mock_get_nodes.return_value = mock_nodes

        summary = await manager.get_conversation_summary()

        assert summary["conversation_id"] == "test-conv-1"
        assert summary["total_nodes"] == 3
        assert summary["recent_nodes"] == 2
        assert summary["compressed_nodes"] == 1
        assert "compression_stats" in summary
        assert "last_updated" in summary


@pytest.mark.asyncio
async def test_get_conversation_summary_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "get_conversation_stats", new_callable=AsyncMock
    ) as mock_get_stats:
        mock_get_stats.side_effect = Exception("Storage error")

        summary = await manager.get_conversation_summary()

        assert "error" in summary
        assert "Storage error" in summary["error"]


@pytest.mark.asyncio
async def test_get_conversation_summary_compression_stats(mock_config):
    """Test that compression statistics are properly calculated."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    current_time = datetime.now()
    mock_stats = Mock()
    mock_stats.compression_stats = {
        CompressionLevel.FULL: 10,
        CompressionLevel.SUMMARY: 5,
    }
    mock_stats.last_updated = current_time

    mock_nodes = []
    for i in range(8):
        mock_nodes.append(Mock(level=CompressionLevel.FULL))
    for i in range(3):
        mock_nodes.append(Mock(level=CompressionLevel.SUMMARY))

    with (
        patch.object(
            manager.storage, "get_conversation_stats", new_callable=AsyncMock
        ) as mock_get_stats,
        patch.object(
            manager.storage, "get_conversation_nodes", new_callable=AsyncMock
        ) as mock_get_nodes,
    ):
        mock_get_stats.return_value = mock_stats
        mock_get_nodes.return_value = mock_nodes

        summary = await manager.get_conversation_summary()

        assert summary["total_nodes"] == 11
        assert summary["recent_nodes"] == 8
        assert summary["compressed_nodes"] == 3

        assert summary["last_updated"] == current_time.isoformat()

        assert summary["compression_stats"] == mock_stats.compression_stats


@pytest.mark.asyncio
async def test_check_and_compress_handles_compression_errors(mock_config):
    """Test error handling during compression operations."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    with patch.object(
        manager.storage, "get_conversation_nodes", new_callable=AsyncMock
    ) as mock_get_nodes:
        mock_get_nodes.side_effect = Exception("Storage error")

        await manager._check_and_compress()


@pytest.mark.asyncio
async def test_get_node_details_existing_node(mock_config, sample_conversation_node):
    """Test retrieving details for an existing node."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "get_node", new_callable=AsyncMock
    ) as mock_get_node:
        mock_get_node.return_value = sample_conversation_node

        details = await manager.get_node_details(1, "test-conv-1")

        mock_get_node.assert_called_once_with(1, "test-conv-1")

        assert details is not None
        assert details["node_id"] == sample_conversation_node.node_id
        assert details["conversation_id"] == sample_conversation_node.conversation_id
        assert details["content"] == sample_conversation_node.content
        assert details["level"] == sample_conversation_node.level.name


@pytest.mark.asyncio
async def test_get_node_details_nonexistent_node(mock_config):
    """Test retrieving details for non-existent node."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "get_node", new_callable=AsyncMock
    ) as mock_get_node:
        mock_get_node.return_value = None

        details = await manager.get_node_details(999, "test-conv-1")

        assert details is None


@pytest.mark.asyncio
async def test_get_node_details_handles_storage_errors(mock_config):
    """Test error handling when storage operations fail."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "get_node", new_callable=AsyncMock
    ) as mock_get_node:
        mock_get_node.side_effect = Exception("Storage error")

        details = await manager.get_node_details(1, "test-conv-1")

        assert details is None


@pytest.mark.asyncio
async def test_get_node_details_response_format(mock_config):
    """Test that node details response contains all expected fields."""
    manager = HierarchicalConversationManager(mock_config)

    complete_node = ConversationNode(
        node_id=1,
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
        ai_components={"model": "test", "tokens": 42},
    )

    with patch.object(
        manager.storage, "get_node", new_callable=AsyncMock
    ) as mock_get_node:
        mock_get_node.return_value = complete_node

        details = await manager.get_node_details(1, "test-conv-1")

        expected_fields = [
            "node_id",
            "conversation_id",
            "node_type",
            "content",
            "summary",
            "timestamp",
            "sequence_number",
            "line_count",
            "level",
            "tokens_used",
            "topics",
            "ai_components",
        ]

        for field in expected_fields:
            assert field in details

        assert details["node_type"] == "ai"
        assert details["level"] == "SUMMARY"


@pytest.mark.asyncio
async def test_full_conversation_workflow(mock_config):
    """Integration test of complete conversation workflow."""

    config = Config(
        work_model="claude-sonnet-4", db_path=":memory:", recent_node_limit=5
    )

    manager = HierarchicalConversationManager(config)

    conv_id = await manager.start_conversation()
    assert manager.conversation_id == conv_id

    mock_response = Mock()
    mock_response.output = "Hello there!"
    mock_response.usage = {"total_tokens": 20}

    with (
        patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
        patch.object(manager, "_check_and_compress", new_callable=AsyncMock),
    ):
        mock_run.return_value = mock_response

        response = await manager.chat("Hello")
        assert response == "Hello there!"

        summary = await manager.get_conversation_summary()
        assert summary["conversation_id"] == conv_id
        assert summary["total_nodes"] >= 0


@pytest.mark.asyncio
async def test_conversation_with_compression_cycle(mock_config):
    """Test conversation that triggers compression."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    mock_response = Mock()
    mock_response.output = "Response"

    # Create mock nodes to return
    mock_user_node = Mock()
    mock_user_node.node_id = 1
    mock_ai_node = Mock()
    mock_ai_node.node_id = 2

    with (
        patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
        patch.object(
            manager.storage, "save_conversation_node", new_callable=AsyncMock
        ) as mock_save,
        patch.object(
            manager.storage, "conversation_exists", new_callable=AsyncMock
        ) as mock_exists,
        patch.object(
            manager, "_check_and_compress", new_callable=AsyncMock
        ) as mock_compress,
    ):
        mock_exists.return_value = True
        mock_run.return_value = mock_response
        mock_save.side_effect = [
            mock_user_node,
            mock_ai_node,
        ] * 3  # 3 iterations, 2 nodes each

        for i in range(3):
            await manager.chat(f"Message {i}")

        assert mock_compress.call_count == 3


@pytest.mark.asyncio
async def test_conversation_resume_after_compression(mock_config):
    """Test resuming conversation that has compressed nodes."""
    manager = HierarchicalConversationManager(mock_config)

    compressed_node = ConversationNode(
        node_id=1,
        conversation_id="test-conv-1",
        node_type=NodeType.AI,
        content="Original long content",
        summary="Compressed summary",
        timestamp=datetime.now(),
        sequence_number=1,
        line_count=1,
        level=CompressionLevel.SUMMARY,
    )

    with (
        patch.object(
            manager.storage, "conversation_exists", new_callable=AsyncMock
        ) as mock_exists,
        patch.object(
            manager.storage, "get_recent_nodes", new_callable=AsyncMock
        ) as mock_recent,
        patch.object(
            manager.storage, "get_conversation_nodes", new_callable=AsyncMock
        ) as mock_compressed,
    ):
        mock_exists.return_value = True
        mock_recent.return_value = []
        mock_compressed.return_value = [compressed_node]

        await manager.start_conversation("test-conv-1")

        messages = [ModelRequest(parts=[UserPromptPart(content="New message")])]
        result = await manager._hierarchical_memory_processor(messages)

        assert len(result) >= 1


@pytest.mark.asyncio
async def test_agent_initialization_with_history_processor(mock_config):
    """Test that PydanticAI agent is initialized with history processor."""
    manager = HierarchicalConversationManager(mock_config)

    assert len(manager.work_agent.history_processors) == 1
    assert (
        manager.work_agent.history_processors[0]
        == manager._hierarchical_memory_processor
    )


@pytest.mark.asyncio
async def test_conversation_manager_with_invalid_config():
    """Test initialization with missing or invalid config values."""

    with pytest.raises(Exception):
        HierarchicalConversationManager(None)


@pytest.mark.asyncio
async def test_chat_with_empty_message(mock_config):
    """Test chat with empty or whitespace-only user message."""
    manager = HierarchicalConversationManager(mock_config)

    with patch.object(
        manager.storage, "conversation_exists", new_callable=AsyncMock
    ) as mock_exists:
        mock_exists.return_value = True
        await manager.start_conversation("test-conv-1")

        mock_response = Mock()
        mock_response.output = "I received an empty message"

        # Create mock nodes to return
        mock_user_node = Mock()
        mock_user_node.node_id = 1
        mock_ai_node = Mock()
        mock_ai_node.node_id = 2

        with (
            patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
            patch.object(
                manager.storage, "save_conversation_node", new_callable=AsyncMock
            ) as mock_save,
            patch.object(manager, "_check_and_compress", new_callable=AsyncMock),
        ):
            mock_run.return_value = mock_response
            mock_save.side_effect = [
                mock_user_node,
                mock_ai_node,
            ] * 2  # 2 calls, 2 nodes each

            response = await manager.chat("")
            assert "I received an empty message" == response

            response = await manager.chat("   ")
            assert "I received an empty message" == response


@pytest.mark.asyncio
async def test_memory_processor_with_mixed_node_types(mock_config):
    """Test history processor with different node types and compression levels."""
    manager = HierarchicalConversationManager(mock_config)
    manager.conversation_id = "test-conv-1"

    mixed_nodes = [
        ConversationNode(
            node_id=1,
            conversation_id="test-conv-1",
            node_type=NodeType.USER,
            content="User message",
            timestamp=datetime.now(),
            sequence_number=1,
            line_count=1,
            level=CompressionLevel.FULL,
        ),
        ConversationNode(
            node_id=2,
            conversation_id="test-conv-1",
            node_type=NodeType.AI,
            content="Original AI response",
            summary="Compressed AI",
            timestamp=datetime.now(),
            sequence_number=2,
            line_count=1,
            level=CompressionLevel.SUMMARY,
        ),
    ]

    with (
        patch.object(
            manager.storage, "get_recent_nodes", new_callable=AsyncMock
        ) as mock_recent,
        patch.object(
            manager.storage, "get_conversation_nodes", new_callable=AsyncMock
        ) as mock_compressed,
    ):
        mock_recent.return_value = [mixed_nodes[0]]
        mock_compressed.return_value = [mixed_nodes[1]]

        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
        result = await manager._hierarchical_memory_processor(messages)

        assert len(result) >= 1


@pytest.mark.asyncio
async def test_conversation_manager_logging(mock_config, caplog):
    """Test that appropriate log messages are generated."""
    import logging

    caplog.set_level(logging.DEBUG)  # Set to DEBUG to capture debug level logs

    # Mock setup_logging to prevent it from interfering with caplog
    with patch.object(mock_config, "setup_logging"):
        manager = HierarchicalConversationManager(mock_config)

        assert "Initialized HierarchicalConversationManager" in caplog.text

        conv_id = await manager.start_conversation()
        assert f"Starting new conversation: {conv_id}" in caplog.text

        mock_response = Mock()
        mock_response.output = "Response"
        # Create mock nodes to return
        mock_user_node = Mock()
        mock_user_node.node_id = 1
        mock_ai_node = Mock()
        mock_ai_node.node_id = 2

        with (
            patch.object(manager.work_agent, "run", new_callable=AsyncMock) as mock_run,
            patch.object(
                manager.storage, "save_conversation_node", new_callable=AsyncMock
            ) as mock_save,
            patch.object(manager, "_check_and_compress", new_callable=AsyncMock),
        ):
            mock_run.return_value = mock_response
            mock_save.side_effect = [mock_user_node, mock_ai_node]

            await manager.chat("Test message")

            # The log message is at debug level in current implementation
            assert (
                f"Processed conversation turn (user: 1, ai: 2) in conversation {conv_id}"
                in caplog.text
            )

        with patch.object(
            manager.work_agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = Exception("Test error")

            await manager.chat("Error message")

            assert "Error in chat" in caplog.text
