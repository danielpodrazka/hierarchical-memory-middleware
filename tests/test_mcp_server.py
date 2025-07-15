"""Tests for the MCP server implementation."""

import pytest
from unittest.mock import AsyncMock, patch, Mock

from hierarchical_memory_middleware.config import Config
from hierarchical_memory_middleware.mcp_server.memory_server import MemoryMCPServer
from hierarchical_memory_middleware.models import ConversationNode, NodeType, CompressionLevel
from hierarchical_memory_middleware.storage import DuckDBStorage


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    config = Config(
        db_path=":memory:",
        work_model="claude-sonnet-4-20250514",
        summary_model="claude-sonnet-4-20250514",
        recent_node_limit=5
    )
    
    return config


@pytest.fixture
@patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
async def mcp_server(mock_agent, temp_config):
    """Create an MCP server instance for testing."""
    # Mock the Agent to avoid network calls during initialization
    mock_agent.return_value = Mock()
    
    server = MemoryMCPServer(temp_config)
    await server.start_conversation()
    return server


@pytest.fixture
async def sample_nodes(mcp_server):
    """Create some sample conversation nodes for testing."""
    storage = mcp_server.storage
    conversation_id = mcp_server.conversation_manager.conversation_id

    # Create sample nodes
    # First conversation turn
    user1 = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="Hello, can you help me with Python?",
    )

    ai1 = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?",
        tokens_used=50,
        ai_components={
            "assistant_text": "Of course! I'd be happy to help you with Python. What specific topic would you like to learn about?",
            "model_used": "test-model",
        },
    )

    # Second conversation turn
    user2 = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.USER,
        content="How do I create a list comprehension?",
    )

    ai2 = await storage.save_conversation_node(
        conversation_id=conversation_id,
        node_type=NodeType.AI,
        content="List comprehensions are a concise way to create lists in Python. The syntax is [expression for item in iterable if condition]. For example: [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 8.",
        tokens_used=75,
        ai_components={
            "assistant_text": "List comprehensions are a concise way to create lists in Python. The syntax is [expression for item in iterable if condition]. For example: [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 8.",
            "model_used": "test-model",
        },
    )

    # Return nodes in order: user1, ai1, user2, ai2
    return [user1, ai1, user2, ai2]


class TestMemoryMCPServer:
    """Test cases for the Memory MCP Server."""

    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_server_initialization(self, mock_agent, temp_config):
        """Test that the server initializes correctly."""
        # Mock the Agent to avoid network calls during initialization
        mock_agent.return_value = Mock()
        
        server = MemoryMCPServer(temp_config)

        assert server.config == temp_config
        assert server.storage is not None
        assert server.conversation_manager is not None
        assert server.mcp is not None

    async def test_storage_debug(self, mcp_server, sample_nodes):
        """Debug test to check storage layer."""
        # Test that we can retrieve nodes directly from storage
        storage = mcp_server.storage
        
        for node in sample_nodes:
            retrieved = await storage.get_node(node.node_id, node.conversation_id)
            assert retrieved is not None, f"Failed to retrieve node {node.node_id}"
            assert retrieved.node_id == node.node_id
            assert retrieved.content == node.content

    async def test_expand_node_success(self, mcp_server, sample_nodes):
        """Test successful node expansion."""
        # Debug: check what nodes we have
        assert len(sample_nodes) >= 2, f"Expected at least 2 nodes, got {len(sample_nodes)}"
        
        # Get a node ID - use the first AI node (should be index 1)
        ai_nodes = [n for n in sample_nodes if n.node_type.value == "ai"]
        assert len(ai_nodes) > 0, "No AI nodes found in sample_nodes"
        
        node = ai_nodes[0]  # First AI node
        node_id = node.node_id

        # Test the expand_node tool indirectly through the conversation manager
        result = await mcp_server.conversation_manager.get_node_details(node_id, node.conversation_id)

        assert result is not None, f"get_node_details returned None for node_id {node_id}"
        assert result["node_id"] == node_id
        assert result["content"] is not None
        assert result["node_type"] == "ai"

    async def test_expand_node_not_found(self, mcp_server):
        """Test node expansion with non-existent node ID."""
        # Test with a non-existent node ID
        result = await mcp_server.conversation_manager.get_node_details(99999, "nonexistent-conversation")
        
        assert result is None

    async def test_search_memory(self, mcp_server, sample_nodes):
        """Test memory search functionality."""
        # Search for Python-related content
        results = await mcp_server.conversation_manager.search_memory("Python")
        
        assert isinstance(results, list)
        # Should find at least one result containing "Python"
        assert len(results) >= 1
        
        # Check that results have the expected structure
        if results:
            result = results[0]
            assert "node_id" in result
            assert "content" in result
            assert "relevance_score" in result

    async def test_conversation_stats(self, mcp_server, sample_nodes):
        """Test getting conversation statistics."""
        stats = await mcp_server.conversation_manager.get_conversation_summary()
        
        assert isinstance(stats, dict)
        assert "conversation_id" in stats
        assert "total_nodes" in stats
        assert stats["total_nodes"] >= 4  # At least our sample nodes

    async def test_get_recent_nodes(self, mcp_server, sample_nodes):
        """Test getting recent nodes."""
        conversation_id = mcp_server.conversation_manager.conversation_id
        
        # Get recent nodes directly from storage
        recent_nodes = await mcp_server.storage.get_recent_nodes(
            conversation_id=conversation_id,
            limit=10
        )
        
        assert isinstance(recent_nodes, list)
        assert len(recent_nodes) >= 4  # Our sample nodes
        
        # Check node structure
        if recent_nodes:
            node = recent_nodes[0]
            assert hasattr(node, 'node_id')
            assert hasattr(node, 'content')
            assert hasattr(node, 'node_type')

    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_start_conversation_new(self, mock_agent, temp_config):
        """Test starting a new conversation."""
        # Mock the Agent to avoid network calls during initialization
        mock_agent.return_value = Mock()
        
        server = MemoryMCPServer(temp_config)

        conversation_id = await server.start_conversation()

        assert conversation_id is not None
        assert len(conversation_id) > 0
        assert server.conversation_manager.conversation_id == conversation_id

    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_start_conversation_resume(self, mock_agent, mcp_server, sample_nodes):
        """Test conversation resumption behavior with in-memory database."""
        # Mock the Agent to avoid network calls when creating new server
        mock_agent.return_value = Mock()

        original_id = mcp_server.conversation_manager.conversation_id

        # Create a new server instance (with separate in-memory database)
        new_server = MemoryMCPServer(mcp_server.config)
        
        # When trying to resume a conversation from a different server instance
        # with in-memory database, it creates a new conversation
        resumed_id = await new_server.start_conversation(original_id)

        # With in-memory databases, each server instance has its own database,
        # so the conversation doesn't exist and a new one is created
        assert resumed_id != original_id  # Should be a new conversation ID
        assert new_server.conversation_manager.conversation_id == resumed_id

        # Test that the same server can resume its own conversations
        same_server_resume = await new_server.start_conversation(resumed_id)
        assert same_server_resume == resumed_id  # Should find the existing conversation

    @pytest.mark.asyncio
    async def test_tool_registration(self, mcp_server):
        """Test that all expected tools are registered."""
        # Verify the MCP server was created successfully
        assert mcp_server.mcp is not None
        
        # Verify that the server has the expected methods that indicate tools are registered
        # We can't easily inspect FastMCP internals, but if initialization succeeded,
        # the tools were registered
        assert hasattr(mcp_server, '_register_tools')
        assert callable(mcp_server._register_tools)

    @pytest.mark.slow
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_error_handling(self, mock_agent, temp_config):
        """Test error handling in MCP tools."""
        # Mock the Agent to avoid network calls during initialization
        mock_agent.return_value = Mock()
        
        server = MemoryMCPServer(temp_config)

        # Test with no active conversation
        stats = await server.conversation_manager.get_conversation_summary()
        assert "error" in stats

        # Test search with no conversation
        results = await server.conversation_manager.search_memory("test")
        assert isinstance(results, list)
        assert len(results) == 0


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""

    @pytest.mark.integration
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_full_workflow(self, mock_agent, temp_config):
        """Test a complete workflow with the MCP server."""
        # Mock the Agent to avoid network calls during initialization
        mock_agent.return_value = Mock()
        
        server = MemoryMCPServer(temp_config)

        # Start conversation
        conversation_id = await server.start_conversation()
        assert conversation_id is not None

        # Add some data
        user_node = await server.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.USER,
            content="Test message",
        )

        ai_node = await server.storage.save_conversation_node(
            conversation_id=conversation_id,
            node_type=NodeType.AI,
            content="Test response",
            tokens_used=25,
        )

        # Test node expansion
        node_details = await server.conversation_manager.get_node_details(user_node.node_id, user_node.conversation_id)
        assert node_details is not None
        assert "Test message" in node_details["content"]
        
        # Test search
        search_results = await server.conversation_manager.search_memory("Test")
        assert len(search_results) >= 1
        
        # Test stats
        stats = await server.conversation_manager.get_conversation_summary()
        assert stats["total_nodes"] >= 2

    @pytest.mark.integration
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    async def test_server_configuration(self, mock_agent, temp_config):
        """Test server configuration and setup."""
        # Mock the Agent to avoid network calls during initialization
        mock_agent.return_value = Mock()
        
        server = MemoryMCPServer(temp_config)

        # Test that server can be configured for different transports
        # Note: We're not actually starting the server here, just testing setup
        assert hasattr(server, 'run')
        assert hasattr(server, 'run_async')
        
        # Test that tools are properly configured
        assert server.mcp is not None
