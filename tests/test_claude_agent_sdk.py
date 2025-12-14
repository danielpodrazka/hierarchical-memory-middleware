"""Tests for Claude Agent SDK integration."""

import pytest
from unittest.mock import patch, MagicMock

from hierarchical_memory_middleware import (
    Config,
    ModelManager,
    ModelProvider,
    ClaudeAgentSDKMarker,
    create_conversation_manager,
    ClaudeAgentSDKConversationManager,
    HierarchicalConversationManager,
)


class TestClaudeAgentSDKModelManager:
    """Tests for Claude Agent SDK model manager functionality."""

    def test_claude_agent_sdk_models_registered(self):
        """Test that Claude Agent SDK models are in the registry."""
        models = ModelManager.list_available_models()

        assert "claude-agent-opus" in models
        assert "claude-agent-sonnet" in models
        assert "claude-agent-haiku" in models

    def test_claude_agent_sdk_provider_type(self):
        """Test that Claude Agent SDK models have correct provider."""
        for model_name in ["claude-agent-opus", "claude-agent-sonnet", "claude-agent-haiku"]:
            config = ModelManager.get_model_config(model_name)
            assert config is not None
            assert config.provider == ModelProvider.CLAUDE_AGENT_SDK

    def test_is_claude_agent_sdk_model(self):
        """Test model type detection."""
        # Agent SDK models
        assert ModelManager.is_claude_agent_sdk_model("claude-agent-opus") is True
        assert ModelManager.is_claude_agent_sdk_model("claude-agent-sonnet") is True
        assert ModelManager.is_claude_agent_sdk_model("claude-agent-haiku") is True

        # Non-Agent SDK models
        assert ModelManager.is_claude_agent_sdk_model("claude-sonnet-4") is False
        assert ModelManager.is_claude_agent_sdk_model("gpt-4o") is False
        assert ModelManager.is_claude_agent_sdk_model("nonexistent") is False

    def test_create_model_returns_marker_for_agent_sdk(self):
        """Test that create_model returns ClaudeAgentSDKMarker for Agent SDK models."""
        result = ModelManager.create_model("claude-agent-sonnet")

        assert isinstance(result, ClaudeAgentSDKMarker)
        assert result.model_name == "claude-agent-sonnet"
        assert result.config.provider == ModelProvider.CLAUDE_AGENT_SDK

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_claude_cli_available_check(self, mock_run, mock_which):
        """Test Claude CLI availability check."""
        # CLI available
        mock_which.return_value = "/usr/bin/claude"
        mock_run.return_value = MagicMock(returncode=0)

        assert ModelManager._check_claude_cli_available() is True

        # CLI not installed
        mock_which.return_value = None
        assert ModelManager._check_claude_cli_available() is False

    def test_model_metadata_contains_built_in_tools(self):
        """Test that Agent SDK models have built_in_tools in metadata."""
        config = ModelManager.get_model_config("claude-agent-sonnet")

        assert "built_in_tools" in config.metadata
        assert "Read" in config.metadata["built_in_tools"]
        assert "Write" in config.metadata["built_in_tools"]
        assert "Bash" in config.metadata["built_in_tools"]


class TestClaudeAgentSDKConfig:
    """Tests for Claude Agent SDK configuration."""

    def test_default_model_is_agent_sdk(self):
        """Test that default model is Claude Agent SDK."""
        config = Config()
        assert config.work_model == "claude-agent-sonnet"

    def test_agent_permission_mode_default(self):
        """Test default permission mode."""
        config = Config()
        assert config.agent_permission_mode == "default"

    def test_agent_allowed_tools_default(self):
        """Test default allowed tools (empty)."""
        config = Config()
        assert config.agent_allowed_tools == ""

    def test_agent_allowed_tools_parsing(self):
        """Test allowed tools configuration."""
        config = Config(agent_allowed_tools="Read,Glob,Grep")
        tools = [t.strip() for t in config.agent_allowed_tools.split(",")]
        assert tools == ["Read", "Glob", "Grep"]


class TestConversationManagerFactory:
    """Tests for the conversation manager factory function."""

    def test_factory_creates_agent_sdk_manager_for_agent_models(self):
        """Test that factory creates ClaudeAgentSDKConversationManager for Agent SDK models."""
        config = Config(work_model="claude-agent-sonnet")
        manager = create_conversation_manager(config=config)

        assert isinstance(manager, ClaudeAgentSDKConversationManager)

    def test_factory_creates_standard_manager_for_other_models(self):
        """Test that factory creates HierarchicalConversationManager for other models."""
        # This test requires an API key to be set, so we mock the model creation
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = Config(work_model="claude-sonnet-4")
            manager = create_conversation_manager(config=config)

            assert isinstance(manager, HierarchicalConversationManager)

    def test_factory_passes_permission_mode(self):
        """Test that factory passes permission mode to Agent SDK manager."""
        config = Config(
            work_model="claude-agent-sonnet",
            agent_permission_mode="bypassPermissions",
        )
        manager = create_conversation_manager(config=config)

        assert isinstance(manager, ClaudeAgentSDKConversationManager)
        assert manager.permission_mode == "bypassPermissions"

    def test_factory_parses_allowed_tools(self):
        """Test that factory parses allowed tools correctly."""
        config = Config(
            work_model="claude-agent-sonnet",
            agent_allowed_tools="Read,Glob,Grep",
        )
        manager = create_conversation_manager(config=config)

        assert isinstance(manager, ClaudeAgentSDKConversationManager)
        assert manager.allowed_tools == ["Read", "Glob", "Grep"]


class TestClaudeAgentSDKConversationManager:
    """Tests for ClaudeAgentSDKConversationManager."""

    def test_manager_initialization(self):
        """Test basic manager initialization."""
        config = Config(work_model="claude-agent-sonnet")
        model_config = ModelManager.get_model_config("claude-agent-sonnet")

        manager = ClaudeAgentSDKConversationManager(
            config=config,
            model_config=model_config,
        )

        assert manager.config == config
        assert manager.model_config == model_config
        assert manager.conversation_id is None
        assert manager.permission_mode == "default"

    @pytest.mark.asyncio
    async def test_start_conversation_creates_id(self):
        """Test that starting a conversation creates an ID."""
        config = Config(work_model="claude-agent-sonnet")
        model_config = ModelManager.get_model_config("claude-agent-sonnet")

        manager = ClaudeAgentSDKConversationManager(
            config=config,
            model_config=model_config,
        )

        conversation_id = await manager.start_conversation()

        assert conversation_id is not None
        assert manager.conversation_id == conversation_id

    @pytest.mark.asyncio
    async def test_start_conversation_with_existing_id(self):
        """Test resuming a conversation with existing ID."""
        config = Config(work_model="claude-agent-sonnet")
        model_config = ModelManager.get_model_config("claude-agent-sonnet")

        manager = ClaudeAgentSDKConversationManager(
            config=config,
            model_config=model_config,
        )

        conversation_id = await manager.start_conversation(
            conversation_id="test-conv-123"
        )

        assert conversation_id == "test-conv-123"
        assert manager.conversation_id == "test-conv-123"

    @pytest.mark.asyncio
    async def test_chat_requires_active_conversation(self):
        """Test that chat raises error without active conversation."""
        config = Config(work_model="claude-agent-sonnet")
        model_config = ModelManager.get_model_config("claude-agent-sonnet")

        manager = ClaudeAgentSDKConversationManager(
            config=config,
            model_config=model_config,
        )

        with pytest.raises(ValueError, match="No active conversation"):
            await manager.chat("Hello")
