"""Tests for external MCP server integration."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from hierarchical_memory_middleware.config import Config, ExternalMCPServer
from hierarchical_memory_middleware.mcp_manager import SimpleMCPManager
from hierarchical_memory_middleware.middleware.conversation_manager import HierarchicalConversationManager


class TestExternalMCPServerConfig:
    """Test external MCP server configuration loading."""

    def test_external_mcp_server_model_creation(self):
        """Test ExternalMCPServer model can be created with valid data."""
        server_config = ExternalMCPServer(
            command="/path/to/python",
            args=["server.py", "--port", "8001"],
            env={"TEST_ENV": "value"},
            port=8001,
            tool_prefix="test",
            enabled=True
        )
        
        assert server_config.command == "/path/to/python"
        assert server_config.args == ["server.py", "--port", "8001"]
        assert server_config.env == {"TEST_ENV": "value"}
        assert server_config.port == 8001
        assert server_config.tool_prefix == "test"
        assert server_config.enabled is True

    def test_external_mcp_server_defaults(self):
        """Test ExternalMCPServer model defaults."""
        server_config = ExternalMCPServer(
            command="/path/to/python",
            args=["server.py"],
            port=8001
        )
        
        assert server_config.env == {}
        assert server_config.tool_prefix == ""
        assert server_config.enabled is True

    def test_load_external_mcp_servers_no_file(self):
        """Test loading external servers when config file doesn't exist."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path('/fake/home')
            servers = Config.load_external_mcp_servers()
            assert servers == {}

    def test_load_external_mcp_servers_with_file(self):
        """Test loading external servers from config file."""
        config_data = {
            "text-editor": {
                "command": "/path/to/python",
                "args": ["server.py", "--port", "8001"],
                "env": {"TEST_VAR": "value"},
                "port": 8001,
                "tool_prefix": "text-editor",
                "enabled": True
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".config" / "hierarchical_memory_middleware"
            config_dir.mkdir(parents=True)
            config_file = config_dir / "mcp_servers.json"
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            with patch('pathlib.Path.home') as mock_home:
                mock_home.return_value = Path(tmpdir)
                servers = Config.load_external_mcp_servers()
                
                assert len(servers) == 1
                assert "text-editor" in servers
                server = servers["text-editor"]
                assert server.command == "/path/to/python"
                assert server.port == 8001
                assert server.tool_prefix == "text-editor"
                assert server.enabled is True


class TestSimpleMCPManager:
    """Test SimpleMCPManager functionality."""

    def test_mcp_manager_initialization(self):
        """Test SimpleMCPManager initializes correctly."""
        manager = SimpleMCPManager()
        assert manager.processes == {}
        assert manager.clients == {}
        assert manager.logger is not None

    async def test_start_server_disabled(self):
        """Test starting a disabled server returns None."""
        manager = SimpleMCPManager()
        config = ExternalMCPServer(
            command="/path/to/python",
            args=["server.py"],
            port=8001,
            enabled=False
        )
        
        result = await manager.start_server("test-server", config)
        assert result is None

    async def test_start_server_enabled(self):
        """Test starting an enabled server."""
        manager = SimpleMCPManager()
        config = ExternalMCPServer(
            command="/bin/echo",  # Use a simple command that will work
            args=["test"],
            port=8001,
            enabled=True
        )
        
        with patch('subprocess.Popen') as mock_popen, \
             patch('hierarchical_memory_middleware.mcp_manager.MCPServerStreamableHTTP') as mock_client:
            
            mock_process = Mock()
            mock_popen.return_value = mock_process
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            result = await manager.start_server("test-server", config)
            
            # Verify subprocess was called correctly
            mock_popen.assert_called_once()
            args, kwargs = mock_popen.call_args
            assert args[0] == ["/bin/echo", "test"]
            
            # Verify client was created
            mock_client.assert_called_once()
            assert result == mock_client_instance
            assert "test-server" in manager.clients
            assert "test-server" in manager.processes

    def test_stop_all(self):
        """Test stopping all servers."""
        manager = SimpleMCPManager()
        
        # Add mock processes
        mock_process1 = Mock()
        mock_process2 = Mock()
        manager.processes = {
            "server1": mock_process1,
            "server2": mock_process2
        }
        manager.clients = {"server1": Mock(), "server2": Mock()}
        
        manager.stop_all()
        
        # Verify all processes were terminated
        mock_process1.terminate.assert_called_once()
        mock_process2.terminate.assert_called_once()
        
        # Verify dictionaries were cleared
        assert manager.processes == {}
        assert manager.clients == {}

    def test_get_clients(self):
        """Test getting all active clients."""
        manager = SimpleMCPManager()
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        manager.clients = {
            "server1": mock_client1,
            "server2": mock_client2
        }
        
        clients = manager.get_clients()
        assert len(clients) == 2
        assert mock_client1 in clients
        assert mock_client2 in clients


class TestConversationManagerIntegration:
    """Test integration with HierarchicalConversationManager."""

    @patch('hierarchical_memory_middleware.middleware.conversation_manager.ModelManager')
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    def test_conversation_manager_with_external_servers(self, mock_agent, mock_model_manager):
        """Test conversation manager accepts external MCP servers."""
        # Mock the model manager and agent
        mock_model_instance = Mock()
        mock_model_manager.create_model.return_value = mock_model_instance
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance

        # Create mock external clients
        mock_external_client = Mock()
        external_clients = [mock_external_client]

        # Create config
        config = Config()

        # Initialize conversation manager with external servers
        manager = HierarchicalConversationManager(
            config=config,
            external_mcp_servers=external_clients
        )

        # Verify manager was initialized
        assert manager is not None
        assert manager.config == config
        assert manager.work_agent == mock_agent_instance

        # Verify Agent was called with external servers
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert 'mcp_servers' in call_args.kwargs
        assert len(call_args.kwargs['mcp_servers']) == 1
        assert call_args.kwargs['mcp_servers'][0] == mock_external_client

    @patch('hierarchical_memory_middleware.middleware.conversation_manager.ModelManager')
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.Agent')
    @patch('hierarchical_memory_middleware.middleware.conversation_manager.MCPServerStreamableHTTP')
    def test_conversation_manager_with_memory_and_external_servers(self, mock_mcp_client, mock_agent, mock_model_manager):
        """Test conversation manager with both memory and external servers."""
        # Mock the model manager and agent
        mock_model_instance = Mock()
        mock_model_manager.create_model.return_value = mock_model_instance
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        mock_memory_client = Mock()
        mock_mcp_client.return_value = mock_memory_client

        # Create mock external clients
        mock_external_client = Mock()
        external_clients = [mock_external_client]

        # Create config
        config = Config()

        # Initialize conversation manager with both memory and external servers
        manager = HierarchicalConversationManager(
            config=config,
            mcp_server_url="http://localhost:8000/mcp",
            external_mcp_servers=external_clients
        )

        # Verify manager was initialized
        assert manager is not None
        assert manager.config == config
        assert manager.work_agent == mock_agent_instance

        # Verify Agent was called with both memory and external servers
        mock_agent.assert_called_once()
        call_args = mock_agent.call_args
        assert 'mcp_servers' in call_args.kwargs
        assert len(call_args.kwargs['mcp_servers']) == 2  # memory + external
        assert mock_memory_client in call_args.kwargs['mcp_servers']
        assert mock_external_client in call_args.kwargs['mcp_servers']

        # Verify has_mcp_tools is True when servers are provided
        assert manager.has_mcp_tools is True
