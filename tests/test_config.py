"""Tests for configuration."""

from hierarchical_memory_middleware.config import Config


def test_config_defaults():
    """Test that Config has expected default values."""
    config = Config()
    
    assert config.work_model == "claude-4-sonnet"
    assert config.summary_model == "claude-4-sonnet"
    assert config.db_path == "./conversations.db"
    assert config.max_context_tokens == 30000
    assert config.use_smart_context is True


def test_config_customization():
    """Test that Config can be customized."""
    config = Config(
        work_model="custom-model",
        db_path="/custom/path.db",
        max_context_tokens=50000
    )
    
    assert config.work_model == "custom-model"
    assert config.db_path == "/custom/path.db"
    assert config.max_context_tokens == 50000
