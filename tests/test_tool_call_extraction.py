"""Unit tests for tool call extraction in conversation manager."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json

from hierarchical_memory_middleware.middleware.conversation_manager import (
    HierarchicalConversationManager,
)
from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.storage import DuckDBStorage


class TestToolCallExtraction:
    """Test tool call and result extraction functionality."""

    @pytest.fixture
    async def manager(self, tmp_path, monkeypatch):
        """Create a conversation manager for testing."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")
        config = Config(
            db_path=str(tmp_path / "test.db"),
            work_model="claude-sonnet-4",
            log_tool_calls=True,
        )
        storage = DuckDBStorage(config.db_path)

        # Mock the Agent to avoid API key requirement
        with patch(
            "hierarchical_memory_middleware.middleware.conversation_manager.Agent"
        ) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            manager = HierarchicalConversationManager(
                config=config,
                storage=storage,
                mcp_server_url=None,  # No actual MCP server for unit tests
            )

            # Ensure the manager has the mocked agent
            assert manager.work_agent == mock_agent_instance

            await manager.start_conversation()
            return manager

    @pytest.mark.asyncio
    async def test_log_tool_call_captures_calls_and_results(self, manager):
        """Test that _log_tool_call properly captures tool calls and results."""
        # Mock the call_tool function
        mock_call_tool = AsyncMock(
            return_value={"status": "success", "data": "test result"}
        )

        # Test tool call
        tool_name = "memory_expand_node"
        args = {"node_id": "12345", "include_metadata": True}

        # Call the _log_tool_call method
        result = await manager._log_tool_call(
            ctx=None,  # Context not used in our implementation
            call_tool=mock_call_tool,
            tool_name=tool_name,
            args=args,
        )

        # Verify tool call was captured
        assert len(manager._current_tool_calls) == 1
        tool_call = manager._current_tool_calls[0]
        assert tool_call["tool_name"] == tool_name
        assert tool_call["args"] == args
        assert "tool_call_id" in tool_call
        assert tool_call["tool_call_id"].startswith("tool_call_0_")

        # Verify tool result was captured
        assert len(manager._current_tool_results) == 1
        tool_result = manager._current_tool_results[0]
        assert tool_result["tool_call_id"] == tool_call["tool_call_id"]
        assert tool_result["content"] == str(
            {"status": "success", "data": "test result"}
        )
        assert tool_result["timestamp"] is None

        # Verify the actual tool was called
        mock_call_tool.assert_called_once_with(tool_name, args)
        assert result == {"status": "success", "data": "test result"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, manager):
        """Test capturing multiple tool calls in sequence."""
        mock_call_tool = AsyncMock()
        mock_call_tool.side_effect = [
            {"result": "first result"},
            {"result": "second result"},
            {"result": "third result"},
        ]

        # Make three tool calls
        tools = [
            ("memory_find", {"query": "test"}),
            ("memory_expand_node", {"node_id": "123"}),
            ("memory_get_context", {"limit": 10}),
        ]

        for i, (tool_name, args) in enumerate(tools):
            await manager._log_tool_call(None, mock_call_tool, tool_name, args)

        # Verify all calls were captured
        assert len(manager._current_tool_calls) == 3
        assert len(manager._current_tool_results) == 3

        # Verify each call
        for i, (tool_name, args) in enumerate(tools):
            assert manager._current_tool_calls[i]["tool_name"] == tool_name
            assert manager._current_tool_calls[i]["args"] == args
            assert manager._current_tool_results[i]["content"] == str(
                {"result": f"{['first', 'second', 'third'][i]} result"}
            )

    @pytest.mark.asyncio
    async def test_extract_methods_return_captured_data(self, manager):
        """Test that extraction methods return the captured data."""
        # Manually populate tool calls and results
        manager._current_tool_calls = [
            {
                "tool_name": "test_tool",
                "tool_call_id": "test_id_1",
                "args": {"param": "value"},
            }
        ]
        manager._current_tool_results = [
            {
                "tool_call_id": "test_id_1",
                "content": "test result content",
                "timestamp": None,
            }
        ]

        # Mock response object (not used by our new implementation)
        mock_response = Mock()

        # Test extraction methods
        tool_calls = manager._extract_tool_calls(mock_response)
        tool_results = manager._extract_tool_results(mock_response)

        # Verify we get copies, not references
        assert tool_calls == manager._current_tool_calls
        assert tool_results == manager._current_tool_results
        assert tool_calls is not manager._current_tool_calls
        assert tool_results is not manager._current_tool_results

    @pytest.mark.asyncio
    async def test_tool_tracking_cleared_between_turns(self, manager):
        """Test that tool calls/results are cleared between conversation turns."""
        # Mock the PydanticAI agent
        mock_response = Mock()
        mock_response.output = "Test response"
        mock_response.usage = Mock(return_value=Mock(total_tokens=100))

        manager.work_agent = Mock()
        manager.work_agent.run = AsyncMock(return_value=mock_response)

        # Manually add some tool calls from a previous turn
        manager._current_tool_calls = [{"old": "call"}]
        manager._current_tool_results = [{"old": "result"}]

        # Process a new message
        await manager.chat("Test message")

        # Verify tool tracking was cleared (empty after chat since no tools were called)
        assert manager._current_tool_calls == []
        assert manager._current_tool_results == []

    @pytest.mark.asyncio
    async def test_comprehensive_content_includes_tools(self, manager):
        """Test that comprehensive content includes tool calls and results."""
        # Set up tool calls and results
        manager._current_tool_calls = [
            {
                "tool_name": "memory_expand_node",
                "tool_call_id": "call_123",
                "args": {"node_id": "node_456", "include_metadata": True},
            }
        ]
        manager._current_tool_results = [
            {
                "tool_call_id": "call_123",
                "content": "Expanded node content with 500 tokens",
                "timestamp": None,
            }
        ]

        # Mock response
        mock_response = Mock()
        mock_response.output = "Based on the expanded context, here's my answer..."

        # Build comprehensive content
        content = manager._build_comprehensive_content(mock_response)

        # Verify tool information is included
        assert "=== TOOL CALLS ===" in content
        assert "Tool Call 1:" in content
        assert "Name: memory_expand_node" in content
        assert "ID: call_123" in content
        assert '"node_id": "node_456"' in content
        assert '"include_metadata": true' in content

        assert "=== TOOL RESULTS ===" in content
        assert "Tool Result 1:" in content
        assert "Call ID: call_123" in content
        assert "Content: Expanded node content with 500 tokens" in content

        assert "Based on the expanded context, here's my answer..." in content

    @pytest.mark.asyncio
    async def test_tool_call_error_handling(self, manager):
        """Test error handling in tool calls."""
        # Mock a failing tool call
        mock_call_tool = AsyncMock(side_effect=Exception("Tool execution failed"))

        # Test that the error is propagated
        with pytest.raises(Exception, match="Tool execution failed"):
            await manager._log_tool_call(
                ctx=None,
                call_tool=mock_call_tool,
                tool_name="failing_tool",
                args={"will": "fail"},
            )

        # Verify the tool call was still captured before the error
        assert len(manager._current_tool_calls) == 1
        assert manager._current_tool_calls[0]["tool_name"] == "failing_tool"
        # But no result should be captured due to the error
        assert len(manager._current_tool_results) == 0

    @pytest.mark.asyncio
    async def test_empty_tool_calls(self, manager):
        """Test behavior when no tools are called."""
        mock_response = Mock()
        mock_response.output = "Simple response without tools"

        # Ensure no tool calls are stored
        manager._current_tool_calls = []
        manager._current_tool_results = []

        # Build comprehensive content
        content = manager._build_comprehensive_content(mock_response)

        # Verify no tool sections are included
        assert "=== TOOL CALLS ===" not in content
        assert "=== TOOL RESULTS ===" not in content
        assert content.strip() == "Simple response without tools"
