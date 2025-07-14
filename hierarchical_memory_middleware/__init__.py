"""
Hierarchical Memory Middleware

A middleware system that enables infinite AI agent conversations
through intelligent hierarchical compression.
"""

__version__ = "0.1.0"

# Core exports
from .config import Config
from .middleware.conversation_manager import HierarchicalConversationManager
from .storage import DuckDBStorage
from .compression import SimpleCompressor, CompressionManager
from .models import (
    ConversationNode,
    ConversationState,
    CompressionLevel,
    NodeType,
    CompressionResult,
    SearchResult,
)

__all__ = [
    "Config",
    "HierarchicalConversationManager",
    "DuckDBStorage",
    "SimpleCompressor",
    "CompressionManager",
    "ConversationNode",
    "ConversationState",
    "CompressionLevel",
    "NodeType",
    "CompressionResult",
    "SearchResult",
    "__version__",
]
