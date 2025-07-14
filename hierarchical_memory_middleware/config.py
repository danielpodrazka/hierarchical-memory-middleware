"""Configuration settings for hierarchical memory middleware."""

from dataclasses import dataclass


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
