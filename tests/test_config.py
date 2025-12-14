"""Tests for configuration."""

import os
from unittest.mock import patch
from hierarchical_memory_middleware.config import Config


def test_config_defaults():
    """Test that Config has expected default values."""
    config = Config()

    # Default model is now Claude Agent SDK (uses CLI auth, Pro/Max subscription)
    assert config.work_model == "claude-agent-sonnet"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.db_path == "./conversations.db"
    assert config.recent_node_limit == 10
    assert config.summary_threshold == 20
    assert config.meta_summary_threshold == 50
    assert config.archive_threshold == 200
    assert config.backup_interval_minutes == 5
    assert config.mcp_port == 8000
    assert config.enable_mcp_tools is True
    # Claude Agent SDK specific defaults
    assert config.agent_permission_mode == "default"
    assert config.agent_allowed_tools == ""


def test_config_customization():
    """Test that Config can be customized."""
    config = Config(
        work_model="custom-model",
        db_path="/custom/path.db",
        recent_node_limit=15,
        enable_mcp_tools=False,
    )

    assert config.work_model == "custom-model"
    assert config.db_path == "/custom/path.db"
    assert config.recent_node_limit == 15
    assert config.enable_mcp_tools is False


def test_from_env_with_environment_variables():
    """Test loading config from environment variables."""
    env_vars = {
        "WORK_MODEL": "claude-opus",
        "DB_PATH": "/tmp/test.db",
        "RECENT_NODE_LIMIT": "25",
        "ENABLE_MCP_TOOLS": "false",
        "MCP_PORT": "9000",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = Config.from_env(env_file=None)  # Don't load .env file

        assert config.work_model == "claude-opus"
        assert config.db_path == "/tmp/test.db"
        assert config.recent_node_limit == 25
        assert config.enable_mcp_tools is False
        assert config.mcp_port == 9000
        # Test defaults for unset vars


def test_from_env_or_default_no_env_file():
    """Test from_env_or_default returns defaults when no .env file exists."""
    config = Config.from_env_or_default(env_file="nonexistent.env")

    # Should have default values (claude-agent-sonnet is new default)
    assert config.work_model == "claude-agent-sonnet"
    assert config.db_path == "./conversations.db"


def test_env_parsing_edge_cases():
    """Test environment variable parsing edge cases."""
    env_vars = {
        "ENABLE_MCP_TOOLS": "TRUE",  # uppercase
        "RECENT_NODE_LIMIT": "invalid",  # invalid int
        "SUMMARY_THRESHOLD": "30",  # valid int
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = Config.from_env(env_file=None)

        assert config.enable_mcp_tools is True  # TRUE should parse as true
        assert config.recent_node_limit == 10  # invalid int should use default
        assert config.summary_threshold == 30  # valid int should parse
