"""Configuration settings for hierarchical memory middleware."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Main configuration for the hierarchical memory system."""

    # Model configuration
    work_model: str = "claude-sonnet-4"
    summary_model: str = "claude-sonnet-4"
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

    log_level: str = "INFO"
    log_file: Optional[str] = "hierarchical_memory.log"
    enable_file_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    cli_log_format: str = "%(asctime)s - %(message)s"
    debug_mode: bool = False

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
            work_model=os.getenv("WORK_MODEL", "claude-sonnet-4"),
            summary_model=os.getenv("SUMMARY_MODEL", "claude-sonnet-4"),
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
            log_tool_calls=get_env_bool("LOG_TOOL_CALLS", True),
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "hierarchical_memory.log"),
            enable_file_logging=get_env_bool("ENABLE_FILE_LOGGING", True),
            log_format=os.getenv(
                "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            cli_log_format=os.getenv(
                "CLI_LOG_FORMAT", "%(asctime)s - %(message)s"
            ),
            debug_mode=get_env_bool("DEBUG_MODE", False),
        )

    @classmethod
    def from_env_or_default(cls, env_file: Optional[str] = ".env") -> "Config":
        """Load from environment or return default config if .env doesn't exist.

        This is a convenience method that won't fail if .env file is missing.
        """
        if env_file and os.path.exists(env_file):
            return cls.from_env(env_file)
        return cls()

    def setup_logging(self) -> None:
        """Set up logging configuration based on config settings."""
        import logging
        from logging.handlers import RotatingFileHandler

        # Set root logger level
        logging.getLogger().setLevel(
            getattr(logging, self.log_level.upper(), logging.INFO)
        )

        # Create formatter
        formatter = logging.Formatter(self.log_format)

        # Clear existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.enable_file_logging and self.log_file:
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Configure third-party library logging levels based on debug mode
        if not self.debug_mode:
            # In non-debug mode, suppress verbose logging from third-party libraries
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("mcp").setLevel(logging.WARNING)
            logging.getLogger("mcp.client").setLevel(logging.WARNING)
            logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
        else:
            # In debug mode, allow all logging
            logging.getLogger("httpx").setLevel(logging.DEBUG)
            logging.getLogger("mcp").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client.streamable_http").setLevel(logging.DEBUG)

        logging.info(
            f"Logging configured - Level: {self.log_level}, Debug mode: {self.debug_mode}, File: {self.log_file if self.enable_file_logging else 'None'}"
        )

    def setup_cli_logging(self) -> None:
        """Set up CLI-specific logging configuration with simplified format."""
        import logging
        from logging.handlers import RotatingFileHandler

        # Set root logger level
        logging.getLogger().setLevel(
            getattr(logging, self.log_level.upper(), logging.INFO)
        )

        # Create CLI-specific formatter (simplified format)
        cli_formatter = logging.Formatter(self.cli_log_format)

        # Clear existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Always add console handler with CLI format
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(cli_formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if enabled (uses standard format for file logging)
        if self.enable_file_logging and self.log_file:
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            # Use standard format for file logging
            file_formatter = logging.Formatter(self.log_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Configure third-party library logging levels based on debug mode
        if not self.debug_mode:
            # In non-debug mode, suppress verbose logging from third-party libraries
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("mcp").setLevel(logging.WARNING)
            logging.getLogger("mcp.client").setLevel(logging.WARNING)
            logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
        else:
            # In debug mode, allow all logging
            logging.getLogger("httpx").setLevel(logging.DEBUG)
            logging.getLogger("mcp").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client").setLevel(logging.DEBUG)
            logging.getLogger("mcp.client.streamable_http").setLevel(logging.DEBUG)

        logging.info(
            f"CLI logging configured - Level: {self.log_level}, Debug mode: {self.debug_mode}, File: {self.log_file if self.enable_file_logging else 'None'}"
        )
