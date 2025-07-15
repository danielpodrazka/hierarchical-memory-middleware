"""Configuration settings for hierarchical memory middleware."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Main configuration for the hierarchical memory system."""

    # Model configuration
    work_model: str = "claude-4-sonnet"
    summary_model: str = "claude-4-sonnet"
    embedding_model: str = "text-embedding-3-small"

    # Hierarchy configuration
    recent_node_limit: int = 10
    summary_threshold: int = 20
    meta_summary_threshold: int = 50
    archive_threshold: int = 200

    # Storage
    db_path: str = "./conversations.db"
    backup_interval_minutes: int = 5

    # MCP server
    mcp_port: int = 8000
    enable_mcp_tools: bool = True
    log_tool_calls: bool = True

    @classmethod
    def from_env(cls, env_file: Optional[str] = ".env") -> "Config":
        """Load configuration from environment variables and .env file.

        Args:
            env_file: Path to .env file. If None, only system env vars are used.

        Returns:
            Config instance with values from environment variables.
        """
        # Load .env file if available and path provided
        if env_file:
            if os.path.exists(env_file):
                load_dotenv(env_file)

        def get_env_bool(key: str, default: bool) -> bool:
            """Helper to parse boolean environment variables."""
            value = os.getenv(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")

        def get_env_int(key: str, default: int) -> int:
            """Helper to parse integer environment variables."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        return cls(
            # Model configuration
            work_model=os.getenv("WORK_MODEL", "claude-4-sonnet"),
            summary_model=os.getenv("SUMMARY_MODEL", "claude-4-sonnet"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            # Hierarchy configuration
            recent_node_limit=get_env_int("RECENT_NODE_LIMIT", 10),
            summary_threshold=get_env_int("SUMMARY_THRESHOLD", 20),
            meta_summary_threshold=get_env_int("META_SUMMARY_THRESHOLD", 50),
            archive_threshold=get_env_int("ARCHIVE_THRESHOLD", 200),
            # Storage
            db_path=os.getenv("DB_PATH", "./conversations.db"),
            backup_interval_minutes=get_env_int("BACKUP_INTERVAL_MINUTES", 5),
            # MCP server
            mcp_port=get_env_int("MCP_PORT", 8000),
            enable_mcp_tools=get_env_bool("ENABLE_MCP_TOOLS", True),
        )

    @classmethod
    def from_env_or_default(cls, env_file: Optional[str] = ".env") -> "Config":
        """Load from environment or return default config if .env doesn't exist.

        This is a convenience method that won't fail if .env file is missing.
        """
        if env_file and os.path.exists(env_file):
            return cls.from_env(env_file)
        return cls()
